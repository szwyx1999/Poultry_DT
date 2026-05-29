from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Mvp1Config
from .utils import split_warnings


UNITY_JSON_FILENAME = "mvp1_zone_activity_timeline.json"


def export_unity_timeline(
    features_df: pd.DataFrame,
    zone_config: dict,
    config: Mvp1Config,
) -> Path:
    export_df = features_df.copy()
    export_df["activity_mean_numeric"] = pd.to_numeric(
        export_df.get("activity_mean"),
        errors="coerce",
    )
    export_df["activity_normalized"] = _normalize_activity(
        export_df["activity_mean_numeric"], config.normalize_activity
    )

    ordered_zones = sorted(zone_config["zones"], key=lambda item: (item["row"], item["col"]))
    timeline: list[dict] = []

    if not export_df.empty:
        sort_columns = [column for column in ("start_time", "window_id", "zone_id") if column in export_df.columns]
        grouped_df = export_df.sort_values(sort_columns, kind="stable")
        for _, window_group in grouped_df.groupby("window_id", sort=False):
            first_row = window_group.iloc[0]
            row_lookup = {
                str(row["zone_id"]): row
                for _, row in window_group.iterrows()
            }
            timeline.append(
                {
                    "window_id": str(first_row.get("window_id", "")),
                    "timestamp": str(first_row.get("start_time", "")),
                    "room_id": _safe_text(first_row.get("room_id", "")),
                    "session_id": _safe_text(first_row.get("session_id", "")),
                    "system_id": _safe_text(first_row.get("system_id", "")),
                    "warnings": _merge_group_warnings(window_group),
                    "zones": [
                        _build_zone_activity_entry(zone, row_lookup.get(zone["zone_id"]))
                        for zone in ordered_zones
                    ],
                }
            )

    payload = {
        "metadata": {
            "mvp": "MVP 1 - Video Zone Heatmap",
            "description": "Zone-level activity timeline generated from preprocessed MP4 windows",
            "zone_layout": config.zone_layout,
            "activity_normalized": config.normalize_activity,
        },
        "zones": ordered_zones,
        "timeline": timeline,
    }
    output_path = config.unity_json_dir / UNITY_JSON_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def _normalize_activity(activity_series: pd.Series, normalize: bool) -> pd.Series:
    normalized = pd.Series([None] * len(activity_series), index=activity_series.index, dtype=object)
    valid_series = activity_series.dropna()
    if not normalize or valid_series.empty:
        return normalized

    minimum = float(valid_series.min())
    maximum = float(valid_series.max())
    if minimum == maximum:
        fill_value = 0.0 if minimum == 0 else 1.0
        normalized.loc[valid_series.index] = [fill_value] * len(valid_series)
        return normalized

    scaled_values = (valid_series - minimum) / (maximum - minimum)
    normalized.loc[valid_series.index] = [float(value) for value in scaled_values.tolist()]
    return normalized


def _build_zone_activity_entry(zone: dict, row: pd.Series | None) -> dict:
    if row is None:
        activity_raw = None
        activity = None
    else:
        activity_raw = _optional_float(row.get("activity_mean_numeric"))
        activity = _optional_float(row.get("activity_normalized"))
    return {
        "zone_id": zone["zone_id"],
        "row": int(zone["row"]),
        "col": int(zone["col"]),
        "activity": activity,
        "activity_raw": activity_raw,
    }


def _merge_group_warnings(window_group: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for value in window_group.get("warnings", pd.Series(dtype=object)).tolist():
        for warning in split_warnings(value):
            if warning not in seen_set:
                seen_set.add(warning)
                seen.append(warning)
    return seen


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)
