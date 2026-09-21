"""JSON helpers shared by the atomic constraint generation scripts.

The generation prompts are Chinese, so the model labels each constraint with the
Chinese taxonomy. SCOPE_LOGIC / SCOPE_UNIT / PRIMARY_TARGET / SECONDARY_TARGET /
MEASUREMENT_RANGE translate those labels into the released English names that
ScopeInstruct/*.json uses and that the next stage reads as-is.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def dump_jsonl_atomic(path: Path, rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=4)


def strip_json_fences(text: str) -> str:
    """Drop ``` fences and escape newlines that appear inside JSON strings."""
    if "```json" in text:
        text = text.replace("```json", "").replace("```", "")
    elif "```" in text:
        text = text.replace("```", "")
    return re.sub(
        r'"(?:[^"\\]|\\.)*"',
        lambda match: match.group(0).replace("\n", "\\n").replace("\r", ""),
        text.strip(),
        flags=re.DOTALL,
    )


def as_single_field_item(value: Any) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    return [value.strip()]


SCOPE_LOGIC = {
    "回复全局": "Global",
    "遍历": "Traversal",
    "索引": "Position",
    "条件": "Condition",
}
SCOPE_UNIT = {
    "回复全局": "Entire Response",
    "结构块": "Structural Block",
    "结构元素": "Structural Element",
    "段落": "Paragraph",
    "句子": "Sentence",
    "单词": "Word",
    "字符": "Character",
}
PRIMARY_TARGET = {
    "文本容量": "Capacity",
    "字面量": "Literals",
    "结构量": "Structure",
    "清单命中量": "Checklist",
    "模式匹配量": "Pattern",
    "关系量": "Relations",
}
SECONDARY_TARGET = {
    "文本长度": "Text Length",
    "文本单元数": "Unit Count",
    "指定字面量": "Specific Strings",
    "字面量类别": "Literal Classes",
    "结构容器数": "Containers",
    "结构成员数": "Members",
    "结构层级数": "Nesting Depth",
    "指定项命中": "Presence",
    "规范项命中": "Validity",
    "边界模板": "Boundary",
    "骨架模板": "Skeleton",
    "写法模板": "Surface",
    "排列关系": "Ordering Relations",
    "链式关系": "Chaining Relations",
    "定位关系": "Positional Relations",
}
MEASUREMENT_RANGE = {
    "上下界关系": "One-Sided Bounds",
    "禁止关系": "Prohibition",
    "等于关系": "Equality",
    "区间关系": "Intervals",
    "比较关系": "Comparison",
    "运算关系": "Arithmetic Relations",
}

# Accept the same full-width arrows and parentheses the step-2 scope filter
# accepts, so a record that passes the filter always translates.
SCOPE_LAYER_RE = re.compile(r"^\s*([^()（）]+?)\s*[（(]\s*([^()（）]+?)\s*[）)]\s*$")
SCOPE_ARROW_RE = re.compile(r"\s*(?:->|→)\s*")


def translate(table: Dict[str, str], value: Any, axis: str) -> str:
    try:
        return table[value]
    except KeyError:
        raise ValueError(f"Unknown {axis}: {value!r}") from None


def translate_scope(scope: Any) -> str:
    """"索引(段落)->遍历(句子)" becomes "Position(Paragraph) -> Traversal(Sentence)"."""
    layers = []
    for raw_layer in SCOPE_ARROW_RE.split(str(scope or "")):
        match = SCOPE_LAYER_RE.match(raw_layer)
        if match is None:
            raise ValueError(f"Cannot parse scope layer: {raw_layer!r}")
        logic = translate(SCOPE_LOGIC, match.group(1).strip(), "scope logic")
        unit = translate(SCOPE_UNIT, match.group(2).strip(), "scope unit")
        layers.append(f"{logic}({unit})")
    return " -> ".join(layers)


def english_category(category: Dict[str, Any]) -> Dict[str, str]:
    """Translate one constraint's Chinese category into the released names.

    Drops the sampling bookkeeping (config / base_config): the released schema is
    exactly the four axes below.
    """
    config = category.get("config") or {}
    return {
        "scope": translate_scope(category.get("具体作用域")),
        "primary_target": translate(
            PRIMARY_TARGET, config.get("作用目标"), "primary target"
        ),
        "secondary_target": translate(
            SECONDARY_TARGET, category.get("二级作用目标"), "secondary target"
        ),
        "range": translate(
            MEASUREMENT_RANGE, config.get("计量范围"), "measurement range"
        ),
    }


def english_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite one flattened record's constraints into the released schema."""
    return {
        **record,
        "constraints": [
            {
                "constraint": entry["constraint"],
                "category": english_category(entry["category"]),
            }
            for entry in record["constraints"]
        ],
    }
