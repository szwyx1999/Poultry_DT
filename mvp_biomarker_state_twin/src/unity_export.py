from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BiomarkerTwinConfig


UNITY_JSON_FILENAME = "poultry_twin_demo_timeline.json"


def export_unity_json(
    canonical_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    state_df: pd.DataFrame,
    zone_config: dict,
    config: BiomarkerTwinConfig,
    model_type: str,
) -> tuple[Path, list[Path]]:
    output_path = config.unity_json_dir / UNITY_JSON_FILENAME
    room_payload = _build_room_payload(canonical_df, zone_config)
    zone_activity_df = _prepare_zone_export_frame(canonical_df)
    merged_window_df = biomarker_df.merge(
        state_df[
            [
                "window_id",
                "room_id",
                "state_id",
                "state_label",
                "state_probability_max",
                "welfare_risk_score",
                "risk_level",
                "sustained_risk_flag",
            ]
        ],
        on=["window_id", "room_id"],
        how="left",
    )
    merged_window_df = merged_window_df.sort_values(["room_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)

    timeline_rows: list[dict] = []
    for frame_index, (_, row) in enumerate(merged_window_df.iterrows()):
        zone_rows = zone_activity_df[(zone_activity_df["window_id"] == row["window_id"]) & (zone_activity_df["room_id"] == row["room_id"])]
        timeline_rows.append(
            {
                "frame_index": frame_index,
                "window_id": row["window_id"],
                "room_id": row["room_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "metrics": {
                    "mobility_index": _optional_float(row.get("mobility_index")),
                    "spatial_freedom_index": _optional_float(row.get("spatial_freedom_index")),
                    "occupancy_imbalance_index": _optional_float(row.get("occupancy_imbalance_index")),
                    "activity_mean": _optional_float(row.get("activity_mean")),
                },
                "state": {
                    "state_id": _optional_int(row.get("state_id")),
                    "state_label": _optional_text(row.get("state_label")),
                    "state_probability": _optional_float(row.get("state_probability_max")),
                },
                "welfare": {
                    "risk_score": _optional_float(row.get("welfare_risk_score")),
                    "risk_level": _optional_text(row.get("risk_level")),
                    "sustained_risk_flag": bool(row.get("sustained_risk_flag", False)),
                },
                "zones": [
                    {
                        "zone_id": zone_row["zone_id"],
                        "activity": _optional_float(zone_row.get("activity_mean")),
                        "activity_norm": _optional_float(zone_row.get("activity_norm")),
                        "overlay_intensity": _optional_float(zone_row.get("overlay_intensity")),
                    }
                    for _, zone_row in zone_rows.sort_values("zone_id", kind="stable").iterrows()
                ],
                "event": {
                    "event_id": _optional_text(row.get("event_id"), allow_none=True),
                    "event_phase": _optional_text(row.get("event_phase")) or "normal",
                    "event_type": _optional_text(row.get("event_type"), allow_none=True),
                },
            }
        )

    payload = {
        "metadata": {
            "schema_version": "mvp_biomarker_state_twin_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_table": str(config.features_dir / "biomarker_window_table.csv"),
            "model_type": model_type,
            "notes": "prototype demo data, not validated welfare diagnosis",
        },
        "rooms": room_payload,
        "timeline": timeline_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    copied_paths: list[Path] = []
    for target_path in config.unity_copy_paths:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied_paths.append(target_path)
    return output_path, copied_paths


def _build_room_payload(canonical_df: pd.DataFrame, zone_config: dict) -> list[dict]:
    room_ids = sorted([room_id for room_id in canonical_df["room_id"].dropna().unique().tolist() if str(room_id)])
    ordered_zones = zone_config.get("zones", [])
    if not room_ids:
        room_ids = [zone_config.get("room_id", "room_1")]
    rooms = []
    for room_id in room_ids:
        rooms.append(
            {
                "room_id": room_id,
                "zones": [
                    {
                        "zone_id": zone["zone_id"],
                        "display_name": zone["zone_id"].replace("_", " ").title(),
                        "row": int(zone.get("row", 0)),
                        "col": int(zone.get("col", 0)),
                        "polygon": _zone_polygon(zone),
                    }
                    for zone in ordered_zones
                ],
            }
        )
    return rooms


def _prepare_zone_export_frame(canonical_df: pd.DataFrame) -> pd.DataFrame:
    export_df = canonical_df.copy()
    activity_series = pd.to_numeric(export_df["activity_mean"], errors="coerce")
    fallback_series = pd.to_numeric(export_df["activity_proxy_raw"], errors="coerce")
    export_df["activity_mean"] = activity_series.fillna(fallback_series)
    valid_activity = export_df["activity_mean"].dropna()
    if valid_activity.empty:
        export_df["activity_norm"] = 0.0
    else:
        minimum = float(valid_activity.min())
        maximum = float(valid_activity.max())
        if minimum == maximum:
            export_df["activity_norm"] = 1.0 if minimum > 0 else 0.0
        else:
            export_df["activity_norm"] = (export_df["activity_mean"] - minimum) / (maximum - minimum)
    export_df["activity_norm"] = export_df["activity_norm"].fillna(0.0).clip(lower=0.0, upper=1.0)
    export_df["overlay_intensity"] = export_df["activity_norm"]
    return export_df


def _zone_polygon(zone: dict) -> list[dict[str, float]]:
    row = float(zone.get("row", 0))
    col = float(zone.get("col", 0))
    return [
        {"x": col, "y": row},
        {"x": col + 1.0, "y": row},
        {"x": col + 1.0, "y": row + 1.0},
        {"x": col, "y": row + 1.0},
    ]


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_text(value: object, allow_none: bool = False) -> str | None:
    if value is None or pd.isna(value) or str(value) == "":
        return None if allow_none else ""
    return str(value)
