from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def combine_warnings(*values: object) -> str:
    combined: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            if not text or text.lower() == "nan":
                items = []
            else:
                items = [item.strip() for item in text.split(";") if item.strip()]
        for item in items:
            if item not in seen:
                seen.add(item)
                combined.append(item)
    return ";".join(combined)


def safe_float(value: object, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prepare_time_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_df = dataframe.copy()
    if "start_time" in working_df.columns:
        working_df["start_time_dt"] = pd.to_datetime(working_df["start_time"], errors="coerce", utc=False)
    else:
        working_df["start_time_dt"] = pd.NaT
    if "end_time" in working_df.columns:
        working_df["end_time_dt"] = pd.to_datetime(working_df["end_time"], errors="coerce", utc=False)
    else:
        working_df["end_time_dt"] = pd.NaT
    return working_df


def compute_overlap_seconds(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> float:
    if pd.isna(window_start) or pd.isna(window_end) or pd.isna(event_start) or pd.isna(event_end):
        return 0.0
    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)
    if overlap_end <= overlap_start:
        return 0.0
    return float((overlap_end - overlap_start).total_seconds())


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    minimum = float(valid.min())
    maximum = float(valid.max())
    if minimum == maximum:
        fill_value = 0.0 if minimum == 0.0 else 0.5
        return pd.Series([fill_value] * len(series), index=series.index, dtype=float)
    scaled = (numeric - minimum) / (maximum - minimum)
    return scaled.fillna(0.0).clip(0.0, 1.0)


def resolve_video_path(workspace_root: Path, configured_path: object) -> Path:
    text = str(configured_path or "").strip()
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute():
        return path

    direct = workspace_root / text
    if direct.exists():
        return direct

    first_week_path = workspace_root / "first_week_data_cleaning" / text
    if first_week_path.exists():
        return first_week_path

    return first_week_path


def ensure_required_columns(dataframe: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def safe_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def report_missing_counts(dataframe: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    return {column: int(dataframe[column].isna().sum()) for column in columns if column in dataframe.columns}


def distribution_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return np.repeat(1.0 / values.size, values.size)
    return values / total

