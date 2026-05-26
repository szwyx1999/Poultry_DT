from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BiomarkerTwinConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResilienceResult:
    biomarker_window_df: pd.DataFrame
    resilience_event_df: pd.DataFrame


def compute_resilience(window_df: pd.DataFrame, event_log_df: pd.DataFrame, config: BiomarkerTwinConfig) -> ResilienceResult:
    if window_df.empty:
        empty_events = pd.DataFrame(
            columns=[
                "event_id",
                "room_id",
                "event_type",
                "candidate_event_source",
                "event_start_time",
                "event_end_time",
                "peak_window_id",
                "peak_time",
                "baseline_activity",
                "peak_activity",
                "recovery_time_windows",
                "recovery_time_seconds",
                "recovery_slope",
                "area_under_recovery_curve",
                "recovered_flag",
            ]
        )
        return ResilienceResult(window_df.copy(), empty_events)

    working_df = window_df.copy().sort_values(["room_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)
    if event_log_df.empty:
        events_df = _detect_fallback_events(working_df, config)
    else:
        events_df = _normalize_event_log(event_log_df)

    event_rows: list[dict] = []
    annotations = pd.DataFrame(
        {
            "window_id": working_df["window_id"],
            "event_id": None,
            "event_phase": "normal",
            "event_type": None,
            "candidate_event_source": None,
            "resilience_pressure_index": 0.0,
        }
    )

    for _, event_row in events_df.iterrows():
        event_metrics, event_annotation_rows = _evaluate_single_event(working_df, event_row, config)
        event_rows.append(event_metrics)
        annotations = _merge_annotations(annotations, event_annotation_rows)

    enriched_df = working_df.merge(annotations, on="window_id", how="left")
    enriched_df["event_phase"] = enriched_df["event_phase"].fillna("normal")
    enriched_df["resilience_pressure_index"] = pd.to_numeric(enriched_df["resilience_pressure_index"], errors="coerce").fillna(0.0)
    enriched_df["candidate_event_source"] = enriched_df["candidate_event_source"].fillna("none")
    enriched_df["event_type"] = enriched_df["event_type"].fillna(pd.NA)
    event_df = pd.DataFrame(event_rows)
    return ResilienceResult(enriched_df, event_df)


def render_empty_resilience_plot(output_path: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.text(0.5, 0.5, "No resilience events available", ha="center", va="center", fontsize=12)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _normalize_event_log(event_log_df: pd.DataFrame) -> pd.DataFrame:
    normalized = event_log_df.copy()
    rename_map = {}
    for column in normalized.columns:
        normalized_name = "".join(character.lower() for character in str(column) if character.isalnum())
        if normalized_name == "eventid":
            rename_map[column] = "event_id"
        elif normalized_name == "roomid":
            rename_map[column] = "room_id"
        elif normalized_name == "starttime":
            rename_map[column] = "start_time"
        elif normalized_name == "endtime":
            rename_map[column] = "end_time"
        elif normalized_name == "eventtype":
            rename_map[column] = "event_type"
    normalized = normalized.rename(columns=rename_map)
    for column in ("event_id", "room_id", "start_time", "end_time", "event_type"):
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["event_id"] = normalized["event_id"].fillna(
        normalized.apply(lambda row: f"{row.get('room_id', 'room')}_event_{row.name:03d}", axis=1)
    )
    normalized["event_type"] = normalized["event_type"].fillna("unspecified_event")
    normalized["candidate_event_source"] = "event_log"
    normalized["start_time_dt"] = pd.to_datetime(normalized["start_time"], errors="coerce", utc=False)
    normalized["end_time_dt"] = pd.to_datetime(normalized["end_time"], errors="coerce", utc=False)
    return normalized[["event_id", "room_id", "start_time", "end_time", "event_type", "candidate_event_source", "start_time_dt", "end_time_dt"]]


def _detect_fallback_events(window_df: pd.DataFrame, config: BiomarkerTwinConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for room_id, room_group in window_df.groupby("room_id", sort=False):
        activity = pd.to_numeric(room_group["activity_mean"], errors="coerce").fillna(pd.to_numeric(room_group["activity_total"], errors="coerce")).fillna(0.0)
        if activity.empty:
            continue
        mean_value = float(activity.mean())
        std_value = float(activity.std(ddof=0))
        if std_value == 0:
            z_scores = pd.Series([0.0] * len(activity), index=activity.index)
        else:
            z_scores = (activity - mean_value) / std_value
        local_peak_candidates: list[int] = []
        indices = list(room_group.index)
        for idx_position, dataframe_index in enumerate(indices):
            current_value = float(activity.loc[dataframe_index])
            prev_value = float(activity.loc[indices[idx_position - 1]]) if idx_position > 0 else current_value
            next_value = float(activity.loc[indices[idx_position + 1]]) if idx_position < len(indices) - 1 else current_value
            if current_value >= prev_value and current_value >= next_value and float(z_scores.loc[dataframe_index]) >= config.disturbance_peak_zscore:
                local_peak_candidates.append(dataframe_index)
        if not local_peak_candidates:
            local_peak_candidates = [int(activity.idxmax())]

        ranked_candidates = sorted(
            local_peak_candidates,
            key=lambda dataframe_index: (float(z_scores.loc[dataframe_index]), float(activity.loc[dataframe_index])),
            reverse=True,
        )
        picked_candidates = ranked_candidates[: config.fallback_max_events_per_room]
        for rank, dataframe_index in enumerate(picked_candidates, start=1):
            row = room_group.loc[dataframe_index]
            rows.append(
                {
                    "event_id": f"{room_id}_candidate_peak_{rank:02d}",
                    "room_id": room_id,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "event_type": "candidate_activity_peak",
                    "candidate_event_source": "activity_peak_fallback",
                    "start_time_dt": row["start_time_dt"],
                    "end_time_dt": row["end_time_dt"],
                }
            )
    return pd.DataFrame(rows)


def _evaluate_single_event(window_df: pd.DataFrame, event_row: pd.Series, config: BiomarkerTwinConfig) -> tuple[dict, pd.DataFrame]:
    room_id = event_row.get("room_id")
    room_group = window_df[window_df["room_id"] == room_id].sort_values(["start_time_dt", "window_id"], kind="stable").reset_index(drop=True)
    start_time_dt = pd.to_datetime(event_row.get("start_time_dt") or event_row.get("start_time"), errors="coerce", utc=False)
    end_time_dt = pd.to_datetime(event_row.get("end_time_dt") or event_row.get("end_time"), errors="coerce", utc=False)

    overlap_mask = (room_group["start_time_dt"] <= end_time_dt) & (room_group["end_time_dt"] >= start_time_dt)
    overlap_group = room_group[overlap_mask]
    if overlap_group.empty:
        nearest_index = _find_nearest_time_index(room_group["start_time_dt"], start_time_dt)
        overlap_group = room_group.iloc[[nearest_index]]
    activity_series = pd.to_numeric(overlap_group["activity_mean"], errors="coerce").fillna(pd.to_numeric(overlap_group["activity_total"], errors="coerce")).fillna(0.0)
    peak_local_position = int(activity_series.idxmax())
    peak_row = overlap_group.loc[peak_local_position]
    peak_room_position = int(room_group.index[room_group["window_id"] == peak_row["window_id"]][0])

    baseline_start = max(0, peak_room_position - config.resilience_baseline_windows)
    baseline_group = room_group.iloc[baseline_start:peak_room_position]
    baseline_activity = float(
        pd.to_numeric(baseline_group["activity_mean"], errors="coerce")
        .fillna(pd.to_numeric(baseline_group["activity_total"], errors="coerce"))
        .fillna(0.0)
        .mean()
    ) if not baseline_group.empty else float(
        pd.to_numeric(room_group["activity_mean"], errors="coerce").fillna(pd.to_numeric(room_group["activity_total"], errors="coerce")).fillna(0.0).iloc[: peak_room_position + 1].median()
    )

    peak_activity = float(pd.to_numeric(peak_row.get("activity_mean"), errors="coerce"))
    if np.isnan(peak_activity):
        peak_activity = float(pd.to_numeric(peak_row.get("activity_total"), errors="coerce"))
    peak_excess = max(peak_activity - baseline_activity, 0.0)
    recovery_threshold = baseline_activity + (config.resilience_recovery_threshold_fraction * peak_excess)

    post_event_group = room_group.iloc[peak_room_position + 1 : peak_room_position + 1 + config.resilience_max_recovery_windows]
    recovered_flag = False
    recovery_end_position = peak_room_position
    recovery_time_windows: int | None = None
    recovery_time_seconds: float | None = None

    for local_offset, (_, candidate_row) in enumerate(post_event_group.iterrows(), start=1):
        candidate_activity = float(pd.to_numeric(candidate_row.get("activity_mean"), errors="coerce"))
        if np.isnan(candidate_activity):
            candidate_activity = float(pd.to_numeric(candidate_row.get("activity_total"), errors="coerce"))
        if candidate_activity <= recovery_threshold:
            recovered_flag = True
            recovery_end_position = peak_room_position + local_offset
            recovery_time_windows = local_offset
            recovery_time_seconds = _time_delta_seconds(
                peak_row["end_time_dt"],
                room_group.iloc[recovery_end_position]["end_time_dt"],
                fallback_seconds=float(peak_row.get("duration_seconds") or 0.0) * local_offset,
            )
            break

    if recovery_time_windows is None:
        recovery_end_position = min(len(room_group) - 1, peak_room_position + config.resilience_max_recovery_windows)
        recovery_time_windows = recovery_end_position - peak_room_position if recovery_end_position > peak_room_position else 0
        recovery_time_seconds = _time_delta_seconds(
            peak_row["end_time_dt"],
            room_group.iloc[recovery_end_position]["end_time_dt"],
            fallback_seconds=float(peak_row.get("duration_seconds") or 0.0) * recovery_time_windows,
        )

    recovery_slice = room_group.iloc[peak_room_position : recovery_end_position + 1]
    recovery_activity = pd.to_numeric(recovery_slice["activity_mean"], errors="coerce").fillna(pd.to_numeric(recovery_slice["activity_total"], errors="coerce")).fillna(0.0)
    area_under_curve = float(np.maximum(recovery_activity - baseline_activity, 0.0).sum() * float(peak_row.get("duration_seconds") or 0.0))
    recovery_slope = 0.0
    if recovery_time_seconds and recovery_time_seconds > 0:
        recovery_slope = float((recovery_activity.iloc[-1] - peak_activity) / recovery_time_seconds)

    annotation_rows = []
    event_type = event_row.get("event_type") or "unspecified_event"
    candidate_source = event_row.get("candidate_event_source") or "event_log"
    denominator = peak_excess if peak_excess > 0 else 1.0
    for room_position, (_, candidate_row) in enumerate(room_group.iterrows()):
        if baseline_start <= room_position < peak_room_position:
            phase = "baseline"
        elif peak_room_position <= room_position <= peak_room_position:
            phase = "during"
        elif peak_room_position < room_position <= recovery_end_position:
            phase = "recovery"
        else:
            continue
        candidate_activity = float(pd.to_numeric(candidate_row.get("activity_mean"), errors="coerce"))
        if np.isnan(candidate_activity):
            candidate_activity = float(pd.to_numeric(candidate_row.get("activity_total"), errors="coerce"))
        pressure = max(candidate_activity - baseline_activity, 0.0) / denominator
        annotation_rows.append(
            {
                "window_id": candidate_row["window_id"],
                "event_id": event_row["event_id"],
                "event_phase": phase,
                "event_type": event_type,
                "candidate_event_source": candidate_source,
                "resilience_pressure_index": float(np.clip(pressure, 0.0, 1.5)),
            }
        )

    metrics = {
        "event_id": event_row["event_id"],
        "room_id": room_id,
        "event_type": event_type,
        "candidate_event_source": candidate_source,
        "event_start_time": event_row.get("start_time"),
        "event_end_time": event_row.get("end_time"),
        "peak_window_id": peak_row["window_id"],
        "peak_time": peak_row["start_time"],
        "baseline_activity": baseline_activity,
        "peak_activity": peak_activity,
        "recovery_time_windows": recovery_time_windows,
        "recovery_time_seconds": recovery_time_seconds,
        "recovery_slope": recovery_slope,
        "area_under_recovery_curve": area_under_curve,
        "recovered_flag": bool(recovered_flag),
        "baseline_window_count": len(baseline_group),
        "recovery_end_time": room_group.iloc[recovery_end_position]["end_time"],
    }
    return metrics, pd.DataFrame(annotation_rows)


def _merge_annotations(base_annotations: pd.DataFrame, new_annotations: pd.DataFrame) -> pd.DataFrame:
    if new_annotations.empty:
        return base_annotations
    merged = base_annotations.merge(new_annotations, on="window_id", how="left", suffixes=("", "__new"))
    keep_new_mask = merged["event_phase__new"].notna()
    for column in ("event_id", "event_phase", "event_type", "candidate_event_source", "resilience_pressure_index"):
        new_column = f"{column}__new"
        if new_column not in merged.columns:
            continue
        if column == "resilience_pressure_index":
            merged[column] = np.where(keep_new_mask, pd.to_numeric(merged[new_column], errors="coerce"), merged[column])
        else:
            merged[column] = np.where(keep_new_mask, merged[new_column], merged[column])
    return merged[base_annotations.columns]


def _find_nearest_time_index(time_series: pd.Series, target_time: pd.Timestamp) -> int:
    if target_time is pd.NaT or pd.isna(target_time):
        return 0
    deltas = (time_series - target_time).abs()
    return int(deltas.idxmin())


def _time_delta_seconds(start_time: pd.Timestamp, end_time: pd.Timestamp, fallback_seconds: float) -> float:
    if pd.isna(start_time) or pd.isna(end_time):
        return float(fallback_seconds)
    return float((end_time - start_time).total_seconds())
