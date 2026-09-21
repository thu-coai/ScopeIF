#!/usr/bin/env python3
"""Extract the counting targets of every constraint, keeping the ones a judge accepts.

    export GPT_OSS_MODEL_PATH=/path/to/gpt-oss-120b
    python extract_targets.py

Reads input/input_examples.json and writes output/output_examples.json (the
accepted samples, in the released schema) plus output/judged_examples.json
(every candidate with its verdict).
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from prompts.target_extraction_prompts import target_extraction_prompt
from prompts.target_validation_prompts import target_validation_prompt
from utils import (
    dump_json_atomic,
    extract_targets,
    final_message,
    load_json,
    translate_targets,
    validate_targets,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_MODEL_PATH = os.environ.get("GPT_OSS_MODEL_PATH", "/path/to/gpt-oss-120b")

VERDICT_RE = re.compile(r"\[\[\s*(高质量|非高质量|不高质量|低质量)\s*\]\]")
CONCLUSION_RE = re.compile(
    r"最终结论[^\[]*\[\[\s*(高质量|非高质量|不高质量|低质量)\s*\]\]", flags=re.DOTALL
)
PASS_VERDICT = "高质量"
SYSTEM_PROMPT = "You are a helpful assistant."


def parse_verdict(text: str) -> Dict[str, Any]:
    """Take the verdict on the 最终结论 line; the analysis may quote others.

    Only the final channel is searched. A completion that never got there is
    left unparsed rather than trusted, since its reasoning tends to rehearse
    the answer format verbatim.
    """
    answer = final_message(text)
    matches = CONCLUSION_RE.findall(answer) or VERDICT_RE.findall(answer)
    verdict = matches[-1] if matches else None
    return {
        "pass": verdict == PASS_VERDICT,
        "parse_ok": verdict is not None,
        "label": verdict,
        "reasoning": answer,
    }


def is_valid_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
        return False
    return isinstance(item.get("constraints"), list) and bool(item["constraints"])


def constraint_text(constraint: Any) -> str:
    if isinstance(constraint, dict):
        return str(constraint.get("constraint") or "")
    return str(constraint or "")


def build_messages(template: str, **fields) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template.format(**fields)},
    ]


def apply_chat_template(tokenizer, messages, reasoning_effort: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:  # chat template without a reasoning_effort parameter
        kwargs.pop("reasoning_effort", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


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


def generate(args, tokenizer, llm, messages_list, temperature, num_samples):
    """Return num_samples completions for every prompt."""
    from vllm import SamplingParams

    if not messages_list:
        return []
    prompts = [
        apply_chat_template(tokenizer, messages, args.reasoning_effort)
        for messages in messages_list
    ]
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    texts = [[candidate.text or "" for candidate in output.outputs] for output in outputs]
    counts = {len(item) for item in texts}
    if len(texts) != len(prompts) or counts - {num_samples}:
        raise RuntimeError(
            f"expected {len(prompts)} prompts x {num_samples} samples, "
            f"got {len(texts)} prompts with sample counts {sorted(counts)}"
        )
    return texts


def extraction_stage(args, tokenizer, llm, data):
    """Sample candidate target lists for every constraint of every sample."""
    messages_list = [
        build_messages(
            target_extraction_prompt,
            prompt=item["prompt"],
            checklist=constraint_text(constraint),
        )
        for item in data
        for constraint in item["constraints"]
    ]
    print(
        f"extraction: {len(messages_list)} constraints x {args.num_candidates} candidates"
    )
    texts = generate(
        args, tokenizer, llm, messages_list, args.extraction_temperature, args.num_candidates
    )
    per_item, cursor = [], 0
    for item in data:
        width = len(item["constraints"])
        per_item.append([[extract_targets(text) for text in texts[cursor + i]] for i in range(width)])
        cursor += width
    return per_item


def validation_stage(args, tokenizer, llm, data, candidates_per_item):
    """Judge every parsable candidate; drop the malformed ones without a call."""
    verdicts = [
        [[None] * len(group) for group in item_candidates]
        for item_candidates in candidates_per_item
    ]
    messages_list, slots, dropped = [], [], 0
    for item_index, (item, item_candidates) in enumerate(zip(data, candidates_per_item)):
        for constraint_index, group in enumerate(item_candidates):
            checklist = constraint_text(item["constraints"][constraint_index])
            for candidate_index, candidate in enumerate(group):
                try:
                    validate_targets(candidate)
                except ValueError as exc:
                    dropped += 1
                    verdicts[item_index][constraint_index][candidate_index] = {
                        "pass": False,
                        "parse_ok": False,
                        "label": None,
                        "reasoning": f"计数目标格式无效：{exc}",
                        "judged_by": "schema_check",
                    }
                    continue
                messages_list.append(
                    build_messages(
                        target_validation_prompt,
                        prompt=item["prompt"],
                        checklist=checklist,
                        targets=candidate,
                    )
                )
                slots.append((item_index, constraint_index, candidate_index))

    print(
        f"validation: {len(messages_list)} candidates judged"
        + (f" ({dropped} unparsable, dropped)" if dropped else "")
    )
    texts = generate(args, tokenizer, llm, messages_list, args.validation_temperature, 1)
    for (item_index, constraint_index, candidate_index), sampled in zip(slots, texts):
        verdict = parse_verdict(sampled[0])
        verdict["judged_by"] = "model"
        verdicts[item_index][constraint_index][candidate_index] = verdict
    return verdicts


def first_accepted(candidates, verdicts):
    """Index of the first candidate the judge accepted, or None."""
    for index, verdict in enumerate(verdicts):
        if verdict and verdict["pass"] and candidates[index]:
            return index
    return None


def passing_records(judged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Emit the released record schema for the fully-selected samples."""
    kept = []
    for record in judged:
        validation = record["target_validation"]
        if not validation["all_selected"]:
            continue
        kept.append(
            {
                "id": len(kept),
                "prompt": record["prompt"],
                "constraints": record["constraints"],
                "targets": [
                    translate_targets(entry["selected"])
                    for entry in validation["constraints"]
                ],
            }
        )
    return kept


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_file", type=Path, default=INPUT_DIR / "input_examples.json")
    parser.add_argument(
        "--judged_file", type=Path, default=OUTPUT_DIR / "judged_examples.json"
    )
    parser.add_argument(
        "--output_file", type=Path, default=OUTPUT_DIR / "output_examples.json"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    # "auto" honours the checkpoint's own quantization (gpt-oss-120b is mxfp4).
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--reasoning_effort", choices=["none", "low", "medium", "high"], default="high"
    )
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--extraction_temperature", type=float, default=1.0)
    parser.add_argument("--validation_temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_candidates <= 0:
        parser.error("--num_candidates must be positive")
    return args


def main():
    args = parse_args()
    if not Path(args.model_path).exists():
        raise SystemExit(
            f"Model path not found: {args.model_path}. Set --model_path or "
            "GPT_OSS_MODEL_PATH to a local gpt-oss-120b checkout."
        )
    existing = [path for path in (args.judged_file, args.output_file) if path.exists()]
    if existing and not args.overwrite:
        print(f"Skip existing output ({', '.join(map(str, existing))}); pass --overwrite")
        return

    data = load_json(args.input_file)
    if not isinstance(data, list) or not data:
        raise ValueError(f"input_file must contain a non-empty JSON list: {args.input_file}")
    malformed = [index for index, item in enumerate(data) if not is_valid_item(item)]
    if malformed:
        raise ValueError(
            f"{args.input_file}: records {malformed[:20]} lack a prompt or constraints"
        )

    checklist = constraint_text(data[0]["constraints"][0])
    build_messages(target_extraction_prompt, prompt=data[0]["prompt"], checklist=checklist)
    build_messages(
        target_validation_prompt,
        prompt=data[0]["prompt"],
        checklist=checklist,
        targets="[]",
    )

    print(f"Extracting targets for {len(data)} samples")
    tokenizer, llm = load_model(args)
    candidates_per_item = extraction_stage(args, tokenizer, llm, data)
    verdicts_per_item = validation_stage(args, tokenizer, llm, data, candidates_per_item)

    judged = []
    for item, item_candidates, item_verdicts in zip(
        data, candidates_per_item, verdicts_per_item
    ):
        per_constraint = []
        for candidates, verdicts in zip(item_candidates, item_verdicts):
            chosen = first_accepted(candidates, verdicts)
            per_constraint.append(
                {
                    "candidates": candidates,
                    "verdicts": verdicts,
                    "selected_index": chosen,
                    "selected": None if chosen is None else candidates[chosen],
                }
            )
        judged.append(
            {
                **item,
                "target_validation": {
                    "num_candidates": args.num_candidates,
                    "constraints": per_constraint,
                    "all_selected": all(
                        entry["selected_index"] is not None for entry in per_constraint
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
