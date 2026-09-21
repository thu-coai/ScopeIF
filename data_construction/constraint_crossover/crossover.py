#!/usr/bin/env python3
"""Cross a seed prompt with sampled atomic constraints into a new instruction.

One pass over input/ writes both files in output/:

    python crossover.py

Inputs are input_examples.jsonl (seed prompts) and atom.json (the validated
atomic constraints the crossover samples from).

generated_examples.json holds the raw API rows (one per seed prompt, each with
several candidate instructions); output_examples.json holds those candidates
flattened into standalone samples and is what constraint_validation consumes.
"""

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from traceback import print_exc
from typing import Any, Dict, List

from prompts.crossover_prompts_final import crossover_en_prompt as crossover_prompt
from tqdm import tqdm
from utils import check_category, dump_json_atomic, load_json, load_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
MODEL = "doubao-seed-2-0-pro-260215"

# Constraints sampled per prompt, and how many of them the rewrite must use.
CONSTRAINTS_PER_PROMPT = 16
CONSTRAINT_NUMBERS = [2, 4, 6, 8]
CANDIDATES_PER_PROMPT = 5

REQUIRED_INSTRUCTION_FIELDS = ("指令编号", "新用户指令", "新用户指令中的所有格式要求")
REQUIRED_REQUIREMENT_FIELDS = ("要求内容", "原始要求编号")
RESULT_RE = re.compile(r"\[构造结果-开始\](.*?)\[构造结果-结束\]", re.DOTALL)

client = None


def get_response(messages):
    global client
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url="<YOUR_API_BASE_URL>",
            api_key="<YOUR_API_KEY>",
        )
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=32768,
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )


def strip_trailing_commas(text: str) -> str:
    """Drop ",]" / ",}" commas the model sometimes emits, ignoring strings."""
    out = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ",":
            rest = text[index + 1 :].lstrip()
            if rest[:1] in ("}", "]"):
                continue
        out.append(char)
    return "".join(out)


def parse_candidates(text: str) -> List[Dict[str, Any]]:
    """Read the [构造结果] block into a list of candidate instructions."""
    match = RESULT_RE.search(text or "")
    if not match:
        return []
    body = match.group(1).strip().removeprefix("```json").strip("`").strip()
    # Newlines inside JSON strings are not legal; escape them before parsing.
    body = re.sub(
        r'"(?:[^"\\]|\\.)*"',
        lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
        body,
        flags=re.DOTALL,
    )
    try:
        candidates = json.loads(strip_trailing_commas(body))
    except json.JSONDecodeError:
        return []
    if not isinstance(candidates, list):
        return []
    for candidate in candidates:
        if isinstance(candidate, dict):
            # The model occasionally uses these older field names.
            for old, new in (
                ("新的用户指令", "新用户指令"),
                ("新用户指令中的格式要求", "新用户指令中的所有格式要求"),
            ):
                if old in candidate:
                    candidate.setdefault(new, candidate.pop(old))
            candidate.pop("原用户指令", None)
    return candidates


def prompt_constraints(constraints: List[Dict[str, Any]]) -> str:
    return json.dumps(
        [
            {key: entry[key] for key in ("要求编号", "要求内容", "要求分类")}
            for entry in constraints
        ],
        ensure_ascii=False,
        indent=3,
    )


def generate_one(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the model for CANDIDATES_PER_PROMPT rewrites; retry on bad output."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": crossover_prompt.format(
                instruction=row["prompt"],
                constraints=prompt_constraints(row["constraints"]),
                constraint_number=row["constraint_number"],
            ),
        },
    ]
    for attempt in range(1, 6):
        try:
            response = get_response(messages)
            candidates = parse_candidates(response.choices[0].message.content)
            if candidates:
                return {**row, "candidates": candidates}
            reason = "no_candidates_parsed"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print_exc()
        print(f"[retry] id={row['id']} attempt={attempt} {reason}", flush=True)
    print(f"[failed] id={row['id']} after 5 attempts", flush=True)
    return {**row, "candidates": [], "error": reason}


def atom_constraint_pool(path: Path) -> List[Dict[str, Any]]:
    """Flatten the validated atomic constraints into one sampling pool.

    Reads the atom stage's filter output, whose taxonomy already uses the
    released English names.
    """
    pool = []
    for record_index, record in enumerate(load_json(path)):
        for entry_index, entry in enumerate(record.get("constraints", [])):
            try:
                category = check_category(entry["category"])
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"{path}: record {record_index} "
                    f"(id={record.get('id')!r}) constraint {entry_index}: {exc}"
                ) from None
            pool.append({"要求内容": entry["constraint"], "要求分类": category})
    if not pool:
        raise ValueError(f"No atomic constraints found in {path}")
    return pool


def build_rows(seeds, pool, count) -> List[Dict[str, Any]]:
    """One row per seed prompt, each carrying its own sampled constraints."""
    rows = []
    for index, seed in enumerate(seeds[:count]):
        sampled = random.sample(pool, min(CONSTRAINTS_PER_PROMPT, len(pool)))
        rows.append(
            {
                "id": index,
                "seed_prompt_id": seed.get("id"),
                "prompt": seed["prompt"],
                "constraint_number": CONSTRAINT_NUMBERS[index % len(CONSTRAINT_NUMBERS)],
                "constraints": [
                    {"要求编号": number, **entry}
                    for number, entry in enumerate(sampled)
                ],
            }
        )
    return rows


def candidate_id(row: Dict[str, Any], instruction_id: Any, prompt: str) -> str:
    """Stable id, so re-flattening the same generation gives the same ids."""
    digest = hashlib.sha256(
        json.dumps(
            [row["seed_prompt_id"], instruction_id, prompt],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{row['id']}_item{instruction_id}_{digest}"


def flatten_candidate(
    row: Dict[str, Any], candidate: Any, by_number: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Turn one candidate into a sample, mapping requirements back to sources."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate is not an object")
    if any(field not in candidate for field in REQUIRED_INSTRUCTION_FIELDS):
        raise ValueError("candidate is missing required fields")
    prompt = candidate["新用户指令"]
    requirements = candidate["新用户指令中的所有格式要求"]
    if not isinstance(candidate["指令编号"], (int, str)) or isinstance(
        candidate["指令编号"], bool
    ):
        raise ValueError(f"指令编号 must be an int or string, got {candidate['指令编号']!r}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("新用户指令 must be a non-empty string")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("format requirement list must be non-empty")

    constraints = []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            raise ValueError(f"requirement {index} is not an object")
        if any(field not in requirement for field in REQUIRED_REQUIREMENT_FIELDS):
            raise ValueError(f"requirement {index} is incomplete")
        text = requirement["要求内容"]
        number = requirement["原始要求编号"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"requirement {index} has empty content")
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(
                f"requirement {index} has a non-integer source id {number!r}"
            )
        if number not in by_number:
            raise ValueError(f"requirement {index} has unknown source id {number}")
        source = by_number[number]
        constraints.append(
            {
                "constraint": text.strip(),
                "category": dict(source["要求分类"]),
                "source_constraint": source["要求内容"],
            }
        )
    return {
        "id": candidate_id(row, candidate["指令编号"], prompt),
        "seed_prompt_id": row["seed_prompt_id"],
        "original_prompt": row["prompt"],
        "prompt": prompt.strip(),
        "constraints": constraints,
    }


def flatten(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split every row's candidates into standalone samples, skipping bad ones."""
    samples = []
    skipped = 0
    for row in rows:
        by_number = {entry["要求编号"]: entry for entry in row["constraints"]}
        for candidate in row.get("candidates") or []:
            try:
                samples.append(flatten_candidate(row, candidate, by_number))
            except ValueError as exc:
                skipped += 1
                print(f"[skip] id={row['id']} {exc}", flush=True)
    if skipped:
        print(f"skipped_candidates={skipped}")
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input_file", type=Path, default=INPUT_DIR / "input_examples.jsonl"
    )
    parser.add_argument(
        "--constraint_file", type=Path, default=INPUT_DIR / "atom.json"
    )
    parser.add_argument(
        "--generated_file", type=Path, default=OUTPUT_DIR / "generated_examples.json"
    )
    parser.add_argument(
        "--output_file", type=Path, default=OUTPUT_DIR / "output_examples.json"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Seed prompts to use. Defaults to every prompt in the input file.",
    )
    parser.add_argument("--max_workers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument(
        "--flatten_only",
        action="store_true",
        help="Re-flatten an existing generated_file instead of calling the API.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.flatten_only:
        rows = load_json(args.generated_file)
    else:
        seeds = load_jsonl(args.input_file)
        pool = atom_constraint_pool(args.constraint_file)
        random.shuffle(seeds)
        rows = build_rows(seeds, pool, args.num_prompts or len(seeds))
        print(
            f"model={MODEL} seed_prompts={len(rows)} "
            f"constraint_pool={len(pool)} "
            f"candidates={len(rows) * CANDIDATES_PER_PROMPT}"
        )
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            rows = list(tqdm(executor.map(generate_one, rows), total=len(rows)))
        dump_json_atomic(args.generated_file, rows)
        print(f"Wrote generation rows: {args.generated_file}")

    samples = flatten(rows)
    if not samples:
        raise ValueError("No valid candidate instructions were produced")
    dump_json_atomic(args.output_file, samples)
    failed = sum(1 for row in rows if row.get("error"))
    print(
        f"generation_rows={len(rows)} failed_rows={failed} "
        f"candidates={sum(len(row.get('candidates') or []) for row in rows)} "
        f"output_samples={len(samples)}"
    )
    print(f"Wrote samples: {args.output_file}")


if __name__ == "__main__":
    main()
