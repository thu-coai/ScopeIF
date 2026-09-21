"""Shared helpers: JSON I/O and the constraint-taxonomy vocabulary.

The atom stage's filter emits the released English names, so this stage reads
them as-is. SCOPE_LOGIC / SCOPE_UNIT / PRIMARY_TARGET / SECONDARY_TARGET /
MEASUREMENT_RANGE are the accepted values for each axis.
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


SCOPE_LOGIC = frozenset(
    {"Global", "Traversal", "Position", "Condition"}
)
SCOPE_UNIT = frozenset(
    {
        "Entire Response",
        "Structural Block",
        "Structural Element",
        "Paragraph",
        "Sentence",
        "Word",
        "Character",
    }
)
PRIMARY_TARGET = frozenset(
    {"Capacity", "Literals", "Structure", "Checklist", "Pattern", "Relations"}
)
SECONDARY_TARGET = frozenset(
    {
        "Text Length",
        "Unit Count",
        "Specific Strings",
        "Literal Classes",
        "Containers",
        "Members",
        "Nesting Depth",
        "Presence",
        "Validity",
        "Boundary",
        "Skeleton",
        "Surface",
        "Ordering Relations",
        "Chaining Relations",
        "Positional Relations",
    }
)
MEASUREMENT_RANGE = frozenset(
    {
        "One-Sided Bounds",
        "Prohibition",
        "Equality",
        "Intervals",
        "Comparison",
        "Arithmetic Relations",
    }
)

SCOPE_LAYER_RE = re.compile(r"^\s*([^()]+?)\s*\(\s*([^()]+?)\s*\)\s*$")


CATEGORY_KEYS = ("scope", "primary_target", "secondary_target", "range")


def check_scope(scope: Any) -> None:
    """Every layer of "Position(Paragraph) -> Traversal(Sentence)" must be known."""
    for raw_layer in str(scope).split("->"):
        match = SCOPE_LAYER_RE.match(raw_layer)
        if match is None:
            raise ValueError(f"Cannot parse scope layer: {raw_layer!r}")
        logic, unit = match.group(1).strip(), match.group(2).strip()
        if logic not in SCOPE_LOGIC:
            raise ValueError(f"Unknown scope logic: {logic!r}")
        if unit not in SCOPE_UNIT:
            raise ValueError(f"Unknown scope unit: {unit!r}")


def check_category(category: Dict[str, Any]) -> Dict[str, str]:
    """Validate one pool constraint's category, as emitted by the atom filter."""
    if not isinstance(category, dict):
        raise ValueError(f"category must be an object, got {type(category).__name__}")
    missing = [key for key in CATEGORY_KEYS if not category.get(key)]
    if missing:
        raise ValueError(f"category is missing {missing}")
    for key, values, axis in (
        ("primary_target", PRIMARY_TARGET, "primary target"),
        ("secondary_target", SECONDARY_TARGET, "secondary target"),
        ("range", MEASUREMENT_RANGE, "measurement range"),
    ):
        if category[key] not in values:
            raise ValueError(f"Unknown {axis}: {category[key]!r}")
    check_scope(category["scope"])
    return {key: category[key] for key in CATEGORY_KEYS}
