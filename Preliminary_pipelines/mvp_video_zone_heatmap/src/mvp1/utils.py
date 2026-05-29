from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_project_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def display_path(project_root: Path, target_path: Path) -> str:
    try:
        return target_path.relative_to(project_root).as_posix()
    except ValueError:
        return target_path.as_posix()


def split_warnings(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def combine_warnings(*values: object) -> str:
    combined: list[str] = []
    seen: set[str] = set()
    for value in values:
        items: Iterable[str]
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = split_warnings(value)
        for item in items:
            if item not in seen:
                seen.add(item)
                combined.append(item)
    return ";".join(combined)


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def safe_float(value: object, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int | None = None) -> int | None:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_value(row: pd.Series | dict, key: str, default: object = "") -> object:
    if key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return value
