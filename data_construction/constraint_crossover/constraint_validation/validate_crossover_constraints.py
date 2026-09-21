#!/usr/bin/env python3
"""LLM-judge filtering for crossover constraint samples (extraction + quality).

One pass over input/input_examples.json writes both files in output/:

    python validate_crossover_constraints.py

output_examples.json holds the samples that passed and is what the next stage
consumes; judged_examples.json holds every sample with its verdicts.
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from utils import dump_json_atomic, load_json


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_MODEL_PATH = os.environ.get("GPT_OSS_MODEL_PATH", "/path/to/gpt-oss-120b")

STAGES = ["extraction", "quality"]
PROMPT_MODULES = {
    "extraction": (
        "extraction_completeness_prompts.py",
        "extraction_completeness_prompt",
    ),
    "quality": ("quality_evaluation_prompts.py", "quality_evaluation_prompt"),
}
STAGE_AGGREGATION = {stage: "all_runs_must_pass" for stage in STAGES}
FILTER_STAGES = list(STAGES)

STAGE_VERDICTS = {
    "extraction": {"通过": True, "不通过": False},
    "quality": {"高质量": True, "低质量": False},
}
VERDICT_RE = re.compile(r"\[\[\s*([^\[\]]+?)\s*\]\]")


def load_prompt(prompt_dir: Path, stage: str) -> str:
    filename, variable = PROMPT_MODULES[stage]
    path = Path(prompt_dir) / filename
    spec = importlib.util.spec_from_file_location(f"crossover_prompt_{stage}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = getattr(module, variable)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: {variable} is empty")
    return prompt


def final_message(text: str) -> str:
    """Keep only the final answer from a gpt-oss completion."""
    text = str(text or "")
    for marker in (
        "<|start|>assistant<|channel|>final<|message|>",
        "<|channel|>final<|message|>",
        "assistantfinal",
    ):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    for stop_token in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(stop_token, 1)[0]
    return text.strip()


def parse_verdict(stage: str, text: str) -> Dict[str, Any]:
    labels = STAGE_VERDICTS[stage]
    raw_text = str(text or "")
    answer = final_message(raw_text)
    for candidate in (answer, raw_text):
        matches = [m for m in VERDICT_RE.findall(candidate) if m in labels]
        if matches:
            verdict = matches[-1]
            return {
                "pass": labels[verdict],
                "parse_ok": True,
                "label": verdict,
                "reasoning": answer,
            }
    return {"pass": False, "parse_ok": False, "label": None, "reasoning": answer}


def aggregate_runs(stage: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    rule = STAGE_AGGREGATION[stage]
    combine = {"all_runs_must_pass": all, "not_all_runs_fail": any}[rule]
    run_passes = [run["pass"] and run["parse_ok"] for run in runs]
    return {
        "pass": combine(run_passes),
        "aggregation": rule,
        "num_runs": len(runs),
        "runs": runs,
    }


def is_valid_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
        return False
    constraints = item.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        return False
    return all(
        isinstance(entry, dict)
        and isinstance(entry.get("constraint"), str)
        and entry["constraint"].strip()
        for entry in constraints
    )


def build_messages(template: str, item: Dict[str, Any]) -> List[Dict[str, str]]:
    constraints = [
        {"id": index, "constraint": entry["constraint"]}
        for index, entry in enumerate(item["constraints"], start=1)
    ]
    # The quality prompt uses {prompt}; the extraction prompt uses {new_prompt}.
    content = template.format(
        prompt=item["prompt"],
        new_prompt=item["prompt"],
        constraints_json=json.dumps(constraints, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


def apply_chat_template(tokenizer, messages, reasoning_effort: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def run_stage(
    args,
    tokenizer,
    llm,
    stage: str,
    items: List[Dict[str, Any]],
    template: str,
) -> List[Dict[str, Any]]:
    from vllm import SamplingParams

    prompts = [
        apply_chat_template(
            tokenizer, build_messages(template, item), args.reasoning_effort
        )
        for item in items
    ]
    sampling_params = SamplingParams(
        n=args.num_repeats,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed + STAGES.index(stage) * 1000,
        skip_special_tokens=True,
    )
    print(f"{stage}: {len(prompts)} prompts x {args.num_repeats} runs")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    runs_per_item = [
        [parse_verdict(stage, candidate.text) for candidate in output.outputs]
        for output in outputs
    ]
    run_counts = {len(runs) for runs in runs_per_item}
    if len(runs_per_item) != len(items) or run_counts - {args.num_repeats}:
        raise RuntimeError(
            f"{stage}: expected {len(items)} items x {args.num_repeats} runs, "
            f"got {len(runs_per_item)} items with run counts {sorted(run_counts)}"
        )
    return [aggregate_runs(stage, runs) for runs in runs_per_item]


def load_model(args):
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        **({"max_model_len": args.max_model_len} if args.max_model_len else {}),
    )
    return tokenizer, llm


def passing_records(judged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for record in judged:
        if not record["validation"]["all_pass"]:
            continue
        kept.append(
            {
                "id": len(kept),
                "prompt": record["prompt"],
                "constraints": record["constraints"],
            }
        )
    return kept


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input_file", type=Path, default=INPUT_DIR / "input_examples.json"
    )
    parser.add_argument(
        "--judged_file", type=Path, default=OUTPUT_DIR / "judged_examples.json"
    )
    parser.add_argument(
        "--output_file", type=Path, default=OUTPUT_DIR / "output_examples.json"
    )
    parser.add_argument("--prompt_dir", type=Path, default=SCRIPT_DIR / "prompts")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--reasoning_effort",
        choices=["none", "low", "medium", "high"],
        default="high",
    )
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.model_path).exists():
        raise SystemExit(
            f"Model path not found: {args.model_path}. Set --model_path or "
            "GPT_OSS_MODEL_PATH to a local gpt-oss-120b checkout."
        )
    if args.judged_file.exists() and args.output_file.exists() and not args.overwrite:
        print(f"Skip existing output: {args.output_file}; pass --overwrite to redo")
        return

    data = load_json(args.input_file)
    if not isinstance(data, list) or not data:
        raise ValueError(
            f"input_file must contain a non-empty JSON list: {args.input_file}"
        )
    malformed = [index for index, item in enumerate(data) if not is_valid_item(item)]
    if malformed:
        raise ValueError(
            f"{args.input_file}: records {malformed[:20]} lack a prompt or "
            "constraints"
        )

    templates = {stage: load_prompt(args.prompt_dir, stage) for stage in STAGES}
    # Fail on a placeholder mismatch before paying for model loading.
    for stage in STAGES:
        build_messages(templates[stage], data[0])

    print(f"Judging {len(data)} samples")
    tokenizer, llm = load_model(args)
    stage_outputs = {
        stage: run_stage(args, tokenizer, llm, stage, data, templates[stage])
        for stage in STAGES
    }

    judged = []
    for index, item in enumerate(data):
        stages = {stage: stage_outputs[stage][index] for stage in STAGES}
        judged.append(
            {
                **item,
                "validation": {
                    "stages": stages,
                    "all_pass": all(
                        stages[stage]["pass"] for stage in FILTER_STAGES
                    ),
                },
            }
        )

    kept = passing_records(judged)
    dump_json_atomic(args.judged_file, judged)
    dump_json_atomic(args.output_file, kept)
    print(f"Wrote {len(judged)} judged samples: {args.judged_file}")
    print(f"Wrote {len(kept)} passing samples: {args.output_file}")


if __name__ == "__main__":
    main()
