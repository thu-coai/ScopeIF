import json
import os
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


TARGET_START = "[计数目标-开始]"
TARGET_END = "[计数目标-结束]"
TARGET_ITEM_KEYS = {"计数目标", "目标类型"}
TARGET_TYPE = {"特称": "Specific", "全称": "Universal"}


def final_message(text: str) -> str:
    text = str(text or "")
    for marker in (
        "<|start|>assistant<|channel|>final<|message|>",
        "<|channel|>final<|message|>",
        "assistantfinal",
    ):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break
    else:
        if text.lstrip().startswith("analysis"):
            return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    for stop_token in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(stop_token, 1)[0]
    return text.strip()


def extract_targets(text: str) -> str:
    text = final_message(text)
    start = text.rfind(TARGET_START)
    if start != -1:
        start += len(TARGET_START)
        end = text.find(TARGET_END, start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    text = text.strip(" \n\t*：:")

    candidates = []
    blocks = re.findall(r"```(?:json|JSON)?\s*.*?```", text, flags=re.DOTALL)
    if blocks:
        lines = blocks[-1].strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
        if body:
            candidates.append(f"```json\n{body}\n```")
    arrays = re.findall(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if arrays:
        candidates.append(f"```json\n{arrays[-1].strip()}\n```")
    if text:
        candidates.append(text)

    for candidate in candidates:
        try:
            validate_targets(candidate)
        except ValueError:
            continue
        return candidate
    return ""


def validate_targets(targets: Any):
    if isinstance(targets, list):
        parsed = targets
    elif isinstance(targets, str):
        text = targets.strip()
        fenced = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"not valid JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
    else:
        raise ValueError(f"must be a JSON array or string, got {type(targets).__name__}")

    if not isinstance(parsed, list):
        raise ValueError(f"root must be a JSON array, got {type(parsed).__name__}")
    if not parsed:
        raise ValueError("array must not be empty")

    seen = set()
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"target {index} must be an object, got {type(item).__name__}")

        keys = set(item)
        if keys != TARGET_ITEM_KEYS:
            details = []
            if TARGET_ITEM_KEYS - keys:
                details.append(f"missing keys {sorted(TARGET_ITEM_KEYS - keys)}")
            if keys - TARGET_ITEM_KEYS:
                details.append(f"extra keys {sorted(keys - TARGET_ITEM_KEYS)}")
            raise ValueError(f"target {index} has invalid fields: {', '.join(details)}")

        counting_object = item["计数目标"]
        if not isinstance(counting_object, str) or not counting_object.strip():
            raise ValueError(f'target {index} "计数目标" must be a non-empty string')

        if item["目标类型"] not in TARGET_TYPE:
            raise ValueError(
                f'target {index} "目标类型" must be one of {sorted(TARGET_TYPE)}, '
                f"got {item['目标类型']!r}"
            )

        normalized = re.sub(r"\s+", " ", counting_object).strip()
        if normalized in seen:
            raise ValueError(f"target {index} duplicates an earlier 计数目标: {counting_object!r}")
        seen.add(normalized)

    return parsed


def translate_targets(targets: Any) -> str:
    english = [
        {"counting_object": item["计数目标"], "type": TARGET_TYPE[item["目标类型"]]}
        for item in validate_targets(targets)
    ]
    body = json.dumps(english, ensure_ascii=False, indent=4)
    return f"```json\n{body}\n```"
