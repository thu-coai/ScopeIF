"""ScopeIF-Test: generate with the policy, then score with the code-assisted judge.

    python run_scopeif.py --stage generate --model_path <policy> --model_name <name>
    python run_scopeif.py --stage score    --judge_model_path <judge> --model_name <name>

Two processes on purpose: the policy and judge models cannot share GPU memory.
Reports ISR (all constraints satisfied) and CSR (mean satisfied share) into
results/<model_name>/{responses.jsonl, responses_scored.json, metrics.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from prompts.code_evaluation import code_evaluation_prompt
from prompts.llm_evaluation import llm_evaluation_prompt
from utils import (
    execute_helper_code,
    extract_code_and_params,
    has_any_exact_judge,
    has_single_exact_judge,
    is_gpt_oss_model,
    is_intentional_no_code,
    objective_satisfied,
    parse_judge,
    parse_objective_results,
    parse_response_with_reasoning,
    parse_solution,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "data" / "test.json"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_JUDGE_MODEL_PATH = "openai/gpt-oss-120b"

RESULT_FIELDS = (
    ("code", "codes"), ("params", "params"), ("result", "results"),
    ("code_valid", "code_valids"), ("critique", "critiques"),
    ("judge_label", "judge_labels"), ("objective_result", "objective_results"),
    ("critique_valid", "critique_valids"),
)
EVALUATION_FIELDS = tuple(field for _, field in RESULT_FIELDS) + ("labels",)

STAGE_ALL_MESSAGE = (
    "--stage all is not supported: the policy model and the judge model cannot "
    "share GPU memory, and freeing a vLLM engine inside a live process is not "
    "reliable enough to reuse the GPU. Run two processes instead:\n"
    "    python run_scopeif.py --stage generate --model_path <policy> --model_name <name>\n"
    "    python run_scopeif.py --stage score    --judge_model_path <judge> --model_name <name>"
)


def chat(user_content: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}]


def truncate_response(response: str, max_chars: int) -> str:
    return response[:max_chars] if max_chars and max_chars > 0 else response


def constraint_text(constraint: Any) -> str:
    if not isinstance(constraint, dict):
        return str(constraint)
    value = constraint.get("constraint")
    if value is None:
        raise ValueError(f"Constraint object has no 'constraint' field: {constraint!r}")
    return str(value)


def render_plan(plan: Any) -> str:
    return plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False, indent=4)


def mean(values: Sequence[bool | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_write(path: Path, write: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            write(file)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, lambda f: json.dump(value, f, ensure_ascii=False, indent=4))


def atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write(path, lambda f: f.write(body))


def validate_dataset(records: Sequence[Dict[str, Any]]) -> None:
    """Validate the schema: {id, prompt, constraints, targets}."""
    if not records:
        raise ValueError("Dataset is empty.")
    seen_ids = set()
    for index, item in enumerate(records):
        constraints, targets, item_id = item.get("constraints"), item.get("targets"), item.get("id")
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"Item {index} has no non-empty prompt.")
        if not isinstance(constraints, list) or not constraints:
            raise ValueError(f"Item {index} has no non-empty constraints list.")
        if not isinstance(targets, list) or len(targets) != len(constraints):
            raise ValueError(
                f"Item {index} has {len(constraints)} constraints but "
                f"{len(targets) if isinstance(targets, list) else 'no'} targets."
            )
        if item_id is None:
            raise ValueError(f"Item {index} has no id.")
        if item_id in seen_ids:
            raise ValueError(f"Item {index} has a duplicate id: {item_id!r}")
        seen_ids.add(item_id)
        for constraint in constraints:
            constraint_text(constraint)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Invalid dataset (expected a JSON array of objects): {path}")
    validate_dataset(records)
    return records


def invalid_judgement_rate(metrics: Dict[str, Any]) -> float:
    """Share of constraints the judge never resolved; 1.0 if unstated."""
    rate = metrics.get("Invalid_Judgement_Rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return 1.0
    return float(rate)


def valid_metrics(path: Path, expected_count: int, max_invalid_rate: float) -> bool:
    """Resume guard: a judge outage also writes a parsable metrics.json, so health is checked too."""
    try:
        metrics = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        isinstance(metrics, dict)
        and metrics.get("Num_Instructions") == expected_count
        and isinstance(metrics.get("Num_Constraints"), int)
        and invalid_judgement_rate(metrics) <= max_invalid_rate
    )


def load_complete_inference(
    response_file: Path, dataset: Sequence[Dict[str, Any]], *, raise_on_error: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """Return responses.jsonl if it is complete and aligned with the dataset."""
    try:
        records = load_jsonl(response_file)
        if len(records) != len(dataset):
            raise ValueError(f"expected {len(dataset)} responses, found {len(records)}")
        validate_dataset(records)
        for index, (item, reference) in enumerate(zip(records, dataset)):
            if item.get("id") != reference.get("id"):
                raise ValueError(
                    f"record {index} id {item.get('id')!r} != dataset id {reference.get('id')!r}"
                )
            if not isinstance(item.get("response"), str):
                raise ValueError(f"record {index} has no string response")
        return records
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if raise_on_error:
            raise ValueError(f"Incomplete inference: {response_file}: {exc}") from exc
        return None


def greedy_sampling(max_tokens: int, skip_special_tokens: bool) -> Any:
    from vllm import SamplingParams

    return SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                          max_tokens=max_tokens, skip_special_tokens=skip_special_tokens)


def build_engine(model_path: str, args: argparse.Namespace) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    return tokenizer, llm


def run_generation(dataset: List[Dict[str, Any]], response_file: Path, args: argparse.Namespace) -> None:
    tokenizer, llm = build_engine(args.model_path, args)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for item in dataset
    ]
    sampling = greedy_sampling(args.max_tokens, not is_gpt_oss_model(args.model_path))
    for item, output in zip(dataset, llm.generate(texts, sampling), strict=True):
        item["response"], item["reasoning"] = parse_response_with_reasoning(output.outputs[0].text)
    atomic_write_jsonl(response_file, dataset)


def parse_gpt_oss_final(text: str) -> str:
    text = str(text or "")
    marker_index, marker_length = -1, 0
    for marker in ("<|start|>assistant<|channel|>final<|message|>",
                   "<|channel|>final<|message|>", "assistantfinal"):
        index = text.rfind(marker)
        if index > marker_index:
            marker_index, marker_length = index, len(marker)
    if marker_index >= 0:
        text = text[marker_index + marker_length :]
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    for stop_token in ("<|return|>", "<|end|>", "<|call|>"):
        if stop_token in text:
            text = text.split(stop_token, 1)[0]
    return parse_solution(text)


def apply_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]], effort: str) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if effort and effort != "none":
        kwargs["reasoning_effort"] = effort
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class JudgeVLLMClient:
    """Local gpt-oss judge served by vLLM in this same process."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.max_retries = args.judge_empty_output_retries
        self.max_tokens = args.judge_max_tokens
        self.reasoning_effort = args.judge_reasoning_effort
        self.tokenizer, self.llm = build_engine(args.judge_model_path, args)

    def generate(self, messages_list: Sequence[List[Dict[str, str]]], desc: str) -> List[str]:
        """One judge text per prompt; "" where none came back, which scores as an invalid judgement."""
        results = [""] * len(messages_list)
        pending = list(range(len(messages_list)))
        last_error: Optional[BaseException] = None
        attempts = self.max_retries + 1

        for retry_index in range(attempts):
            if not pending:
                break
            prompts = [
                apply_chat_template(self.tokenizer, messages_list[index], self.reasoning_effort)
                for index in pending
            ]
            print(f"{desc}: prompts={len(prompts)} local_retry={retry_index + 1}/{attempts}", flush=True)
            try:
                outputs = self.llm.generate(
                    prompts, greedy_sampling(self.max_tokens, True), use_tqdm=True
                )
                if len(outputs) != len(pending):
                    raise RuntimeError(f"{desc}: expected {len(pending)} outputs, got {len(outputs)}")
                still_pending = []
                for index, output in zip(pending, outputs):
                    parsed = parse_gpt_oss_final(output.outputs[0].text if output.outputs else "")
                    results[index] = parsed
                    if not parsed:
                        still_pending.append(index)
                pending = still_pending
                if pending:
                    last_error = RuntimeError(f"{desc}: judge returned {len(pending)} empty response(s)")
            except Exception as exc:  # noqa: BLE001 - retried below, then degraded
                last_error = exc

        if pending:
            print(
                f"[WARN] {desc}: {len(pending)} prompt(s) produced no usable judge text after "
                f"{attempts} attempts; last_error={last_error!r}; treating them as invalid",
                file=sys.stderr,
                flush=True,
            )
        return results


def flatten_constraints(records: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    return [
        (item_index, constraint_index)
        for item_index, item in enumerate(records)
        for constraint_index in range(len(item["constraints"]))
    ]


def initialize_evaluation_fields(records: Sequence[Dict[str, Any]]) -> None:
    for item in records:
        for field in EVALUATION_FIELDS:
            item[field] = [None] * len(item["constraints"])


def objective_label(objective: Optional[List[Dict[str, Any]]]) -> Optional[bool]:
    if objective is None:
        return None
    value = objective_satisfied(objective, default_on_invalid=None)
    return None if value is None else bool(value)


def judge_rounds(
    records: List[Dict[str, Any]],
    judge: JudgeVLLMClient,
    max_iters: int,
    label: str,
    build: Callable[[Dict[str, Any], int], str],
    handle: Callable[[Dict[str, Any], int, str], bool],
) -> None:
    """Re-ask the judge for constraints that did not resolve, up to max_iters rounds."""
    pending = flatten_constraints(records)
    for attempt in range(max_iters):
        if not pending:
            break
        messages_list = [chat(build(records[i], c)) for i, c in pending]
        outputs = judge.generate(messages_list, desc=f"{label} {attempt + 1}/{max_iters}")
        pending = [
            (i, c) for (i, c), output in zip(pending, outputs) if not handle(records[i], c, output)
        ]


def run_code_generation(
    records: List[Dict[str, Any]], judge: JudgeVLLMClient, args: argparse.Namespace
) -> None:
    def build(item: Dict[str, Any], index: int) -> str:
        return code_evaluation_prompt.format(
            prompt=item["prompt"],
            checklist=constraint_text(item["constraints"][index]),
            response=truncate_response(item["response"], args.judge_max_response_chars),
            plan=render_plan(item["targets"][index]),
        )

    def handle(item: Dict[str, Any], index: int, output: str) -> bool:
        code, params = extract_code_and_params(output)
        no_code = is_intentional_no_code(output)
        result, valid = ("", True) if no_code else execute_helper_code(
            code, params, item["response"],
            timeout_seconds=args.judge_exec_timeout,
            memory_limit_mb=args.judge_exec_memory_mb,
        )
        if len(result) > args.judge_max_result_chars:
            result, valid = "", False
        item["codes"][index], item["params"][index] = code, params
        item["results"][index], item["code_valids"][index] = result, valid
        return valid or no_code

    judge_rounds(records, judge, args.judge_max_iters, "Code generation", build, handle)


def run_critique_generation(
    records: List[Dict[str, Any]], judge: JudgeVLLMClient, args: argparse.Namespace
) -> None:
    def build(item: Dict[str, Any], index: int) -> str:
        return llm_evaluation_prompt.format(
            prompt=item["prompt"],
            checklist=constraint_text(item["constraints"][index]),
            results=item["results"][index] or "",
            response=truncate_response(item["response"], args.judge_max_response_chars),
            plan=render_plan(item["targets"][index]),
        )

    def handle(item: Dict[str, Any], index: int, output: str) -> bool:
        output = parse_solution(output)
        objective = parse_objective_results(output)
        valid = has_any_exact_judge(output) and objective_label(objective) is not None
        item["critiques"][index] = output
        item["judge_labels"][index] = parse_judge(output) if has_single_exact_judge(output) else None
        item["labels"][index] = objective_label(objective) if valid else None
        item["objective_results"][index] = objective
        item["critique_valids"][index] = valid
        return valid

    judge_rounds(records, judge, args.judge_max_iters, "Critique generation", build, handle)


def finalize_evaluation_records(records: List[Dict[str, Any]]) -> None:
    for item in records:
        labels = item["labels"]
        item["constraint_results"] = [
            {
                "constraint": constraint_text(item["constraints"][index]),
                "label": labels[index],
                **{key: item[field][index] for key, field in RESULT_FIELDS},
            }
            for index in range(len(item["constraints"]))
        ]
        item["instruction_score"] = mean([label is True for label in labels])
        item["follow_all_constraints"] = bool(labels) and all(label is True for label in labels)


def calculate_group_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    category_values: Dict[str, Dict[str, List[bool]]] = {}
    flat_tags: Dict[str, List[bool]] = {}
    scores: List[float] = []
    successes: List[bool] = []
    invalid_judgements = num_constraints = 0

    for item in records:
        labels = item["labels"]
        scores.append(mean([label is True for label in labels]))
        successes.append(bool(labels) and all(label is True for label in labels))
        num_constraints += len(labels)
        invalid_judgements += sum(label is None for label in labels)
        for index, constraint in enumerate(item["constraints"]):
            label = labels[index] is True
            category = constraint.get("category") if isinstance(constraint, dict) else None
            if not isinstance(category, dict):
                continue
            for field, value in category.items():
                for entry in ([] if value is None else value if isinstance(value, list) else [value]):
                    entry = str(entry)
                    category_values.setdefault(str(field), {}).setdefault(entry, []).append(label)
                    flat_tags.setdefault(entry, []).append(label)

    def summarize(groups: Dict[str, List[bool]]) -> Dict[str, Dict[str, Any]]:
        return {k: {"Count": len(v), "CSR": mean(v)} for k, v in sorted(groups.items())}

    isr = mean(successes)
    return {
        "main_score": isr,
        "Num_Instructions": len(records),
        "Num_Constraints": num_constraints,
        "ISR": isr,
        "CSR": mean(scores),
        "Num_Invalid_Judgements": invalid_judgements,
        "Invalid_Judgement_Rate": invalid_judgements / num_constraints if num_constraints else 0.0,
        "Metrics_per_Category_Field": {
            field: summarize(values) for field, values in sorted(category_values.items())
        },
        "Metrics_per_Tag": summarize(flat_tags),
    }


def print_metrics(model_name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n===== ScopeIF-Test: {model_name} =====")
    for key in ("Num_Instructions", "Num_Constraints"):
        print(f"{key:<22} : {metrics[key]}")
    for key in ("ISR", "CSR"):
        print(f"{key:<22} : {metrics[key] * 100:.2f}")
    print(f"{'Num_Invalid_Judgements':<22} : {metrics['Num_Invalid_Judgements']}")
    print(f"{'Invalid_Judgement_Rate':<22} : {metrics['Invalid_Judgement_Rate'] * 100:.2f}")
    print(f"{'main_score (= ISR)':<22} : {metrics['main_score']:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScopeIF-Test evaluation (generate / score)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_name", default="Qwen3-8B")
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Results root; <model_name> is appended. Default: ./results",
    )
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument(
        "--responses_file",
        default="",
        help="Override the responses.jsonl path; defaults to <output_dir>/<model_name>/.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument(
        "--stage",
        choices=["generate", "score", "all"],
        required=True,
        help="generate = policy model; score = gpt-oss judge.",
    )
    parser.add_argument("--judge_model_path", default=DEFAULT_JUDGE_MODEL_PATH)
    parser.add_argument("--judge_max_tokens", type=int, default=32768)
    parser.add_argument(
        "--judge_reasoning_effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--judge_max_iters", type=int, default=3)
    parser.add_argument("--judge_empty_output_retries", type=int, default=5)
    parser.add_argument("--judge_exec_timeout", type=int, default=5)
    parser.add_argument("--judge_exec_memory_mb", type=int, default=1024)
    parser.add_argument("--judge_max_response_chars", type=int, default=20000)
    parser.add_argument("--judge_max_result_chars", type=int, default=10000)
    parser.add_argument("--judge_max_invalid_rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "all":
        raise SystemExit(STAGE_ALL_MESSAGE)

    dataset = load_dataset(args.input_file)
    model_dir = Path(args.output_dir) / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    response_file = Path(args.responses_file or model_dir / "responses.jsonl")
    metrics_file = model_dir / "metrics.json"

    if valid_metrics(metrics_file, len(dataset), args.judge_max_invalid_rate):
        print(f"[SKIP] already scored: {metrics_file}")
        print_metrics(args.model_name, load_json(metrics_file))
        return

    if args.stage == "generate":
        if load_complete_inference(response_file, dataset) is not None:
            print(f"[SKIP] complete inference already present: {response_file}")
            return
        run_generation(dataset, response_file, args)
        print(f"[DONE] generate -> {response_file}")
        return

    records = load_complete_inference(response_file, dataset, raise_on_error=True)
    assert records is not None
    print(f"[RESUME] scoring existing responses: {response_file}")
    initialize_evaluation_fields(records)
    judge = JudgeVLLMClient(args)
    run_code_generation(records, judge, args)
    run_critique_generation(records, judge, args)
    finalize_evaluation_records(records)
    atomic_write_json(model_dir / "responses_scored.json", records)
    metrics = calculate_group_metrics(records)
    atomic_write_json(metrics_file, metrics)
    print(f"[DONE] score -> {model_dir / 'responses_scored.json'}, {metrics_file}")
    print_metrics(args.model_name, metrics)

    rate = invalid_judgement_rate(metrics)
    if rate > args.judge_max_invalid_rate:
        raise SystemExit(
            f"[FAIL] {metrics['Num_Invalid_Judgements']}/{metrics['Num_Constraints']} constraints "
            f"({rate * 100:.2f}%) never got a usable judgement, above the "
            f"{args.judge_max_invalid_rate * 100:.2f}% limit. ISR/CSR above are NOT reportable. "
            f"Re-run the score stage (this run is not cached), or raise --judge_max_invalid_rate."
        )


if __name__ == "__main__":
    main()
