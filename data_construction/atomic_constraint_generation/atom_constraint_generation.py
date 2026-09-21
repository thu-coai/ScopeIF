#!/usr/bin/env python3
"""Add one sampled numeric format constraint to each seed prompt.

One pass over input/input_examples.jsonl writes both outputs in output/:

    python atom_constraint_generation.py

generated_examples/ holds the raw API rows (one JSON file per repeat and
iteration, reused on a later run unless --overwrite); output_examples.json holds
those rows flattened into one record per constraint, with each category
translated into the released English taxonomy. It is what constraint_validation
consumes.
"""

import argparse
import glob
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from traceback import print_exc
from typing import Any, Dict, Iterable, List, Tuple

from prompts.atom_constraint_generation_prompts_final import (
    numerical_constraint_generation_prompt as GENERATION_PROMPT,
)
from utils import (
    as_single_field_item,
    dump_json_atomic,
    dump_jsonl_atomic,
    dumps_json,
    english_record,
    load_json,
    load_jsonl,
    strip_json_fences,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
GENERATION_BACKEND = "doubao"
DEFAULT_MODEL_NAME = "doubao-seed-2-0-pro-260215"
FORMAT_ERROR_FILENAME = "format_error_examples.jsonl"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


file_lock = threading.Lock()
client = None

REQUIRED_INSTRUCTION_FIELDS = [
    "原用户指令",
    "具体作用域",
    "二级作用目标",
    "是否计数式表示",
    "新的用户指令",
    "新添加的格式要求",
]
COUNT_EXPRESSION_FIELD = "是否计数式表示"
COUNT_EXPRESSION_FIELD_ALIASES = ("是否计数式",)
# Carried in the payload for generate_one's worker thread; never written to disk.
INTERNAL_PAYLOAD_KEYS = ("output_path",)

SCOPE_UNITS = ["回复全局", "结构块", "结构元素", "段落", "句子", "单词", "字符"]
SCOPE_UNIT_SET = frozenset(SCOPE_UNITS)
SCOPE_MODES = ["回复全局", "遍历", "索引", "条件"]
TARGET_FAMILIES = ["文本容量", "字面量", "结构量", "清单命中量", "模式匹配量", "关系量"]
RANGE_TYPES = ["上下界关系", "禁止关系", "等于关系", "区间关系", "比较关系", "运算关系"]

SECONDARY_TARGETS_BY_PRIMARY = {
    "文本容量": ["文本长度", "文本单元数"],
    "字面量": ["指定字面量", "字面量类别"],
    "结构量": ["结构容器数", "结构成员数", "结构层级数"],
    "清单命中量": ["指定项命中", "规范项命中"],
    "模式匹配量": ["边界模板", "骨架模板", "写法模板"],
    "关系量": ["排列关系", "链式关系", "定位关系"],
}

# 文本容量 is the one family that cannot take 禁止关系.
VALID_RANGES = {
    family: (RANGE_TYPES if family != "文本容量" else
             [r for r in RANGE_TYPES if r != "禁止关系"])
    for family in TARGET_FAMILIES
}

SCOPE_MODE_REPEATS = {"回复全局": 1, "遍历": 1, "索引": 2, "条件": 1}
TARGET_FAMILY_REPEATS = {
    "文本容量": 2, "字面量": 2, "结构量": 1,
    "清单命中量": 1, "模式匹配量": 1, "关系量": 1,
}
RANGE_TYPE_REPEATS = {
    "上下界关系": 2, "禁止关系": 1, "等于关系": 2,
    "区间关系": 2, "比较关系": 1, "运算关系": 1,
}

SCOPE_UNIT_ALIASES = {
    "字符/符号": "字符", "符号": "字符",
    "词语/短语": "单词", "词语": "单词", "短语": "单词",
    "列表项": "结构元素", "表格行": "结构元素", "表格列": "结构元素",
    "表格单元格": "结构元素", "CSV字段": "结构元素", "JSON字段": "结构元素",
    "JSON对象": "结构元素", "JSON数组": "结构元素", "XML标签": "结构元素",
    "HTML标签": "结构元素", "属性": "结构元素", "字段": "结构元素",
    "代码块": "结构块", "列表": "结构块", "表格": "结构块",
}

RESULT_BLOCK_RE = re.compile(r"\[改造结果-开始\](.*?)\[改造结果-结束\]", re.DOTALL)
ANALYSIS_BLOCK_RE = re.compile(r"\[分析-开始\](.*?)\[分析-结束\]", re.DOTALL)
SCOPE_LAYER_RE = re.compile(r"^\s*(回复全局|遍历|索引|条件)\(([^()]*)\)\s*$")
FIRST_LAYER_RE = re.compile(r"^\s*[^()]+\(([^()]+)\)\s*$")


def get_response(messages, model_name):
    global client
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ["GENERATION_API_BASE_URL"],
            api_key=os.environ["GENERATION_API_KEY"],
        )
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=65536,
        extra_body={
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        },
    )


def normalize_new_instruction_fields(inst):
    if not isinstance(inst, dict):
        return inst
    normalized = dict(inst)
    if COUNT_EXPRESSION_FIELD not in normalized:
        for alias in COUNT_EXPRESSION_FIELD_ALIASES:
            alias_value = normalized.get(alias)
            if isinstance(alias_value, str) and alias_value.strip():
                normalized[COUNT_EXPRESSION_FIELD] = alias_value
                break
    for alias in COUNT_EXPRESSION_FIELD_ALIASES:
        normalized.pop(alias, None)
    return normalized


def normalize_new_instructions(new_instructions):
    if not isinstance(new_instructions, list):
        return new_instructions
    return [normalize_new_instruction_fields(inst) for inst in new_instructions]


def get_existing_constraints(existing_constraints):
    return dumps_json(existing_constraints) if existing_constraints else ""


def extract_analysis_and_result(text):
    result_match = RESULT_BLOCK_RE.search(text)
    if not result_match:
        return "", []
    analysis_match = ANALYSIS_BLOCK_RE.search(text)
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    try:
        rows = json.loads(strip_json_fences(result_match.group(1).strip()))
    except json.JSONDecodeError:
        print_exc()
        return "", []
    if not isinstance(rows, list):
        return "", []
    return analysis, normalize_new_instructions(rows)


def normalize_scope_unit_for_prompt(unit):
    if not isinstance(unit, str):
        return None
    unit = re.sub(r"[（(].*?[）)]", "", unit).strip()
    return SCOPE_UNIT_ALIASES.get(unit, unit)


def is_valid_specific_scope(scope_text):
    if not isinstance(scope_text, str) or not scope_text.strip():
        return False
    for raw_segment in scope_text.split("->"):
        match = SCOPE_LAYER_RE.match(raw_segment.strip())
        if not match or match.group(2).strip() not in SCOPE_UNIT_SET:
            return False
    return True


def extract_first_scope_unit(scope_text):
    if not isinstance(scope_text, str):
        return None
    match = FIRST_LAYER_RE.match(scope_text.split("->", 1)[0].strip())
    return normalize_scope_unit_for_prompt(match.group(1).strip()) if match else None


def count_existing_scope_units(existing_constraints):
    counts = {unit: 0 for unit in SCOPE_UNITS}
    for inst in existing_constraints or []:
        if not isinstance(inst, dict):
            continue
        for scope in as_single_field_item(inst.get("具体作用域", "")):
            unit = extract_first_scope_unit(scope)
            if unit in counts:
                counts[unit] += 1
    return counts


def count_existing_secondary_targets(existing_constraints, primary_target):
    counts = {
        target: 0 for target in SECONDARY_TARGETS_BY_PRIMARY.get(primary_target, [])
    }
    for inst in existing_constraints or []:
        if not isinstance(inst, dict):
            continue
        for target in as_single_field_item(inst.get("二级作用目标", "")):
            if target in counts:
                counts[target] += 1
    return counts


def diagnose_result_parse_error(text):
    if not isinstance(text, str):
        return ["response_content_not_string"]
    result_match = RESULT_BLOCK_RE.search(text)
    if not result_match:
        return ["missing_result_block"]
    result_content = strip_json_fences(result_match.group(1).strip())
    try:
        json_data = json.loads(result_content)
    except json.JSONDecodeError as exc:
        return [f"invalid_json:{exc.msg}@line_{exc.lineno}_column_{exc.colno}"]
    if not isinstance(json_data, list):
        return ["result_block_not_json_array"]
    if not json_data:
        return ["empty_result_array"]
    return ["unclassified_parse_error"]


def save_format_error_example(
    output_path, item, error_type, error_details, raw_output=None,
    input_prompt=None, attempt=None,
):
    record = {
        "backend": GENERATION_BACKEND,
        "error_type": error_type,
        "error_details": error_details,
        "attempt": attempt,
        "input_prompt": (
            item.get("input_prompt", "") if input_prompt is None else input_prompt
        ),
        "raw_output": item.get("raw_output", "") if raw_output is None else raw_output,
        "new_instructions": item.get("new_instructions", []),
        **{
            key: item.get(key)
            for key in (
                "repeat_id", "iter_id", "config_id", "instruction",
                "config", "base_config", "category", "existing_constraints",
            )
        },
    }
    with file_lock:
        path = os.path.join(output_path, FORMAT_ERROR_FILENAME)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_one(item):
    existing = item["existing_constraints"]
    user_prompt = GENERATION_PROMPT.format(
        instruction=item["instruction"],
        category=item["category"],
        existing_constraints=get_existing_constraints(existing),
        existing_scope_unit_distribution=dumps_json(
            count_existing_scope_units(existing)
        ),
        existing_secondary_target_distribution=dumps_json(
            count_existing_secondary_targets(existing, item["config"].get("作用目标"))
        ),
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    result = None
    retry = 3
    attempt = 0
    while retry:
        attempt += 1
        try:
            result = get_response(messages, item["model_name"])
            content = result.choices[0].message.content
            if extract_analysis_and_result(content)[1] == []:
                save_format_error_example(
                    item["output_path"], item, "parse_error",
                    diagnose_result_parse_error(content),
                    raw_output=content, input_prompt=user_prompt, attempt=attempt,
                )
                retry -= 1
                continue
            break
        except Exception:
            print_exc()
            retry -= 1

    item["input_prompt"] = user_prompt
    try:
        content = result.choices[0].message.content
        item["analysis"], item["new_instructions"] = extract_analysis_and_result(content)
        item["raw_output"] = content
        item["completion_tokens"] = result.usage.completion_tokens
        item["prompt_tokens"] = result.usage.prompt_tokens
    except Exception:
        item["analysis"] = ""
        item["raw_output"] = ""
        item["new_instructions"] = []
        item["completion_tokens"] = 0
        item["prompt_tokens"] = 0

    return item


def render_config_iter():
    """The taxonomy sweep: each axis repeated per its *_REPEATS weight."""
    return [
        {"作用域选择逻辑": mode, "作用目标": target, "计量范围": rng}
        for mode in SCOPE_MODES
        for _ in range(SCOPE_MODE_REPEATS[mode])
        for target in TARGET_FAMILIES
        for _ in range(TARGET_FAMILY_REPEATS[target])
        for rng in VALID_RANGES[target]
        for _ in range(RANGE_TYPE_REPEATS[rng])
    ]


def sample_multilayer(scope_mode):
    return False if scope_mode == "回复全局" else random.random() < 0.5


def build_category_config(base_config, multilayer):
    return {
        "作用域选择逻辑": base_config["作用域选择逻辑"],
        "多层作用域": multilayer,
        "作用目标": base_config["作用目标"],
        "计量范围": base_config["计量范围"],
    }


def seed_prompt_filter(item):
    tags = item.get("tags")
    if tags is None:
        return True
    if not isinstance(tags, list) or len(tags) < 2:
        return False
    if tags[0] in ["客观问答"]:
        return False
    if tags[1] in ["文本编辑校对", "翻译", "汉语翻译"]:
        return False
    return True


def get_iteration_json_path(output_path, repeat_id, iter_id, suffix="json"):
    return f"{output_path}/repeat_{repeat_id}_iteration_{iter_id}_results.{suffix}"


def get_iteration_jsonl_path(output_path, repeat_id, iter_id):
    return get_iteration_json_path(output_path, repeat_id, iter_id, "jsonl")


def clear_existing_result_files(output_path):
    patterns = [
        "repeat_*_iteration_*_results.json", "repeat_*_iteration_*_results.jsonl",
        "iteration_*_results.json", "iteration_*_results.jsonl",
        FORMAT_ERROR_FILENAME,
    ]
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_path, pattern)):
            os.remove(path)


def get_new_instruction_format_errors(inst, index=None):
    prefix = f"new_instructions[{index}]" if index is not None else "new_instruction"
    if not isinstance(inst, dict):
        return [f"{prefix}:not_object"]
    inst = normalize_new_instruction_fields(inst)
    errors = [
        f"{prefix}:missing_or_invalid_field:{field}"
        for field in REQUIRED_INSTRUCTION_FIELDS
        if not isinstance(inst.get(field), str) or not inst.get(field, "").strip()
    ]
    label = inst.get(COUNT_EXPRESSION_FIELD)
    if isinstance(label, str) and label.strip() and label not in {"是", "否"}:
        errors.append(f"{prefix}:invalid_count_expression_label")
    scope = inst.get("具体作用域")
    if isinstance(scope, str) and scope.strip() and not is_valid_specific_scope(scope):
        errors.append(f"{prefix}:invalid_specific_scope")
    return errors


def get_result_row_format_errors(row, repeat_id, iter_id, config_id):
    if not isinstance(row, dict):
        return ["result_row:not_object"]
    errors = [
        f"result_row:{field}_mismatch"
        for field, want in (
            ("repeat_id", repeat_id), ("iter_id", iter_id), ("config_id", config_id)
        )
        if row.get(field) != want
    ]
    new_instructions = row.get("new_instructions")
    if not isinstance(new_instructions, list):
        return errors + ["result_row:new_instructions_not_array"]
    if len(new_instructions) != 5:
        errors.append(f"result_row:new_instructions_length:{len(new_instructions)}")
    for index, inst in enumerate(new_instructions):
        errors.extend(get_new_instruction_format_errors(inst, index))
    return errors


def load_existing_result_rows(output_path, repeat_id, iter_id, num_configs):
    """Rows from a previous run that can be reused instead of re-generated."""
    path = get_iteration_json_path(output_path, repeat_id, iter_id)
    if not os.path.exists(path):
        return {}, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        print_exc()
        return {}, 1
    if not isinstance(rows, list):
        return {}, 1
    rows_by_config = {}
    invalid_count = 0
    for row in rows:
        config_id = row.get("config_id") if isinstance(row, dict) else None
        if not isinstance(config_id, int) or not (0 <= config_id < num_configs):
            invalid_count += 1
            continue
        row["new_instructions"] = normalize_new_instructions(
            row.get("new_instructions")
        )
        if get_result_row_format_errors(row, repeat_id, iter_id, config_id):
            invalid_count += 1
        else:
            rows_by_config[config_id] = row
    return rows_by_config, invalid_count


def dump_iteration_results(output_path, repeat_id, iter_id, rows):
    # Drop the worker-thread plumbing so no local path reaches the files.
    rows = [
        {k: v for k, v in row.items() if k not in INTERNAL_PAYLOAD_KEYS}
        for row in sorted(rows, key=lambda x: x["config_id"])
    ]
    dump_json_atomic(get_iteration_json_path(output_path, repeat_id, iter_id), rows)
    dump_jsonl_atomic(get_iteration_jsonl_path(output_path, repeat_id, iter_id), rows)


def generation_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
    return tuple(int(row.get(k, -1)) for k in ("repeat_id", "iter_id", "config_id"))


def iter_generation_rows(source_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    result_paths = sorted(source_dir.glob("repeat_*_iteration_*_results.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No repeat_*_iteration_*_results.json files found in {source_dir}"
        )
    for path in result_paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Generation result must be a JSON list: {path}")
        for row_index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: row {row_index} is not an object")
            yield path, row


def flatten(source_dir: Path):
    rows_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for path, row in iter_generation_rows(source_dir):
        key = generation_key(row)
        if key in rows_by_key:
            raise ValueError(f"{path}: duplicate generation key {key}")
        rows_by_key[key] = row

    records: List[Dict[str, Any]] = []
    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        for inst in row["new_instructions"]:
            category = {
                "具体作用域": inst["具体作用域"],
                "二级作用目标": inst["二级作用目标"],
                "是否计数式表示": inst["是否计数式表示"],
                "config": row.get("config") or {},
                "base_config": row.get("base_config") or {},
            }
            records.append(
                {
                    "id": len(records),
                    "seed_prompt_id": row["config_id"],
                    "original_prompt": inst["原用户指令"],
                    "prompt": inst["新的用户指令"],
                    "constraints": [
                        {"constraint": inst["新添加的格式要求"], "category": category}
                    ],
                }
            )

    return records, {"generation_rows": len(rows_by_key)}


def generate(args):
    """Step 1: rewrite seed prompts into constrained instructions."""
    os.makedirs(args.generated_dir, exist_ok=True)
    if args.overwrite and not args.dry_run:
        clear_existing_result_files(args.generated_dir)

    seeds = load_jsonl(args.input_file)
    seeds = [p for p in seeds if p.get("history", []) == [] and seed_prompt_filter(p)]
    if not seeds:
        raise ValueError("No seed prompts left after filtering.")
    random.seed(args.seed)
    random.shuffle(seeds)

    configs = render_config_iter()
    if args.num_configs is not None:
        configs = configs[: args.num_configs]
    planned_seed_uses = len(configs) * args.num_iterations * args.num_repeats
    print(
        f"model={args.model_name} configs={len(configs)} "
        f"iterations={args.num_iterations} repeats={args.num_repeats} "
        f"seed_pool={len(seeds)} planned_rewrites={planned_seed_uses * 5}"
    )
    print(f"output_path={args.generated_dir} resume={not args.overwrite}")
    if planned_seed_uses > len(seeds):
        print("seed_pool_wrap=True")

    if args.dry_run:
        for i, config in enumerate(configs[:10]):
            print(f"config_id={i}")
            category = build_category_config(
                config, sample_multilayer(config["作用域选择逻辑"])
            )
            print(json.dumps(category, ensure_ascii=False, indent=4))
        return

    cnt = 0
    # One 多层作用域 draw per (repeat, config), fixed across iterations; each
    # repeat accumulates its own history so the prompt can ask for diversity.
    constraint_history = [{} for _ in range(args.num_repeats)]
    dirty_configs = [set() for _ in range(args.num_repeats)]
    category_configs = [
        [
            build_category_config(c, sample_multilayer(c["作用域选择逻辑"]))
            for c in configs
        ]
        for _ in range(args.num_repeats)
    ]

    for i in range(args.num_iterations):
        rows_by_repeat = {r: {} for r in range(args.num_repeats)}
        invalid_existing_rows = 0
        payloads = []
        for repeat_id in range(args.num_repeats):
            history = constraint_history[repeat_id]
            existing_rows, invalid_count = load_existing_result_rows(
                args.generated_dir, repeat_id, i, len(configs)
            )
            invalid_existing_rows += invalid_count
            for j, config in enumerate(configs):
                if cnt >= len(seeds):
                    random.shuffle(seeds)
                    cnt = 0
                category = category_configs[repeat_id][j]
                if j in existing_rows and j not in dirty_configs[repeat_id]:
                    rows_by_repeat[repeat_id][j] = existing_rows[j]
                else:
                    dirty_configs[repeat_id].add(j)
                    payloads.append(
                        {
                            "repeat_id": repeat_id,
                            "iter_id": i,
                            "config_id": j,
                            "instruction": seeds[cnt]["prompt"],
                            "config": category,
                            "base_config": config,
                            "category": json.dumps(
                                category, ensure_ascii=False, indent=4
                            ),
                            "existing_constraints": history.get(j, []),
                            # generate_one runs in a worker thread and only
                            # receives this payload, so it carries the two
                            # settings the request needs.
                            "model_name": args.model_name,
                            "output_path": str(args.generated_dir),
                        }
                    )
                cnt += 1

        reused = sum(len(rows) for rows in rows_by_repeat.values())
        if invalid_existing_rows:
            print(f"iteration={i} invalid_existing_rows={invalid_existing_rows}")
        print(
            f"iteration={i} reused_payloads={reused} "
            f"scheduled_payloads={len(payloads)}"
        )

        if payloads:
            for repeat_id, iter_id in {(p["repeat_id"], p["iter_id"]) for p in payloads}:
                path = get_iteration_jsonl_path(args.generated_dir, repeat_id, iter_id)
                if os.path.exists(path):
                    os.remove(path)
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                payloads = list(
                    tqdm(executor.map(generate_one, payloads), total=len(payloads))
                )

        invalid_generated_rows = 0
        for p in sorted(payloads, key=lambda x: (x["repeat_id"], x["config_id"])):
            errors = get_result_row_format_errors(p, p["repeat_id"], i, p["config_id"])
            if not errors:
                rows_by_repeat[p["repeat_id"]][p["config_id"]] = p
            else:
                invalid_generated_rows += 1
                if p.get("new_instructions"):
                    save_format_error_example(
                        args.generated_dir, p, "validation_error", errors
                    )
        if invalid_generated_rows:
            print(f"iteration={i} invalid_generated_rows={invalid_generated_rows}")

        for repeat_id, rows_by_config in rows_by_repeat.items():
            rows = [rows_by_config[j] for j in sorted(rows_by_config)]
            history = constraint_history[repeat_id]
            for row in rows:
                history[row["config_id"]] = (
                    history.get(row["config_id"], []) + row["new_instructions"]
                )
            dump_iteration_results(args.generated_dir, repeat_id, i, rows)
            if len(rows) != len(configs):
                print(
                    f"WARNING: repeat={repeat_id} iteration={i} has "
                    f"{len(rows)}/{len(configs)} valid payloads; "
                    "only valid payloads were written."
                )


def build(args):
    """Step 2: flatten the generated files into one dataset."""
    flattened, load_summary = flatten(args.generated_dir)
    records = [english_record(record) for record in flattened]
    dump_json_atomic(args.output_file, records)
    print(
        f"generation_rows={load_summary['generation_rows']} "
        f"flattened_records={len(flattened)} output_records={len(records)}"
    )
    print(f"Wrote samples: {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add = parser.add_argument
    add("--input_file", type=Path, default=INPUT_DIR / "input_examples.jsonl")
    add("--generated_dir", type=Path, default=OUTPUT_DIR / "generated_examples")
    add("--output_file", type=Path, default=OUTPUT_DIR / "output_examples.json")
    add("--num_configs", type=int, default=None,
        help="Sampled configs to use. Defaults to the full taxonomy sweep.")
    add("--num_iterations", type=int, default=10)
    add("--num_repeats", type=int, default=2)
    add("--max_workers", type=int, default=100)
    add("--model_name", default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
    add("--seed", type=int, default=233)
    add("--dry_run", action="store_true")
    add("--overwrite", action="store_true",
        help="Discard existing results instead of resuming from them.")
    add("--build_only", action="store_true",
        help="Re-flatten an existing generated_dir instead of calling the API.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.build_only:
        generate(args)
        if args.dry_run:
            return
    build(args)


if __name__ == "__main__":
    main()
