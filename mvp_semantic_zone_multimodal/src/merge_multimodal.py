from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SemanticZoneConfig
from .utils import compute_overlap_seconds, prepare_time_columns


@dataclass(frozen=True)
class SemanticMultimodalMergeResult:
    multimodal_df: pd.DataFrame
    output_path: Path
    report_path: Path
    audio_matched_windows: int
    environment_matched_windows: int
    event_labelled_windows: int


def merge_semantic_multimodal_table(
    config: SemanticZoneConfig,
    semantic_biomarker_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    env_df: pd.DataFrame,
    event_log_df: pd.DataFrame | None = None,
) -> SemanticMultimodalMergeResult:
    if semantic_biomarker_df.empty:
        raise ValueError("Semantic biomarker window table is empty; cannot build multimodal merge.")

    biomarker_df = prepare_time_columns(semantic_biomarker_df)
    merged_df = _merge_audio_features(biomarker_df, audio_df)
    merged_df = _attach_environment_context(merged_df, env_df, config)
    merged_df = attach_event_labels(merged_df, event_log_df if event_log_df is not None else pd.DataFrame(), config)
    merged_df = merged_df.sort_values(
        ["room_id", "session_id", "start_time_dt", "window_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_path = config.features_dir / "semantic_multimodal_window_table.csv"
    report_path = config.reports_dir / "semantic_multimodal_merge_report.md"
    merged_df.to_csv(output_path, index=False)

    audio_matched_windows = int(merged_df["audio_available"].fillna(False).astype(bool).sum())
    environment_matched_windows = int((merged_df["temp_context"].notna() | merged_df["rh_context"].notna()).sum())
    event_labelled_windows = int(merged_df["event_phase"].isin(["pre_entry_baseline", "during_entry", "post_entry_recovery"]).sum())
    report_path.write_text(
        _build_merge_report(
            merged_df=merged_df,
            audio_matched_windows=audio_matched_windows,
            environment_matched_windows=environment_matched_windows,
            event_labelled_windows=event_labelled_windows,
            event_log_df=event_log_df if event_log_df is not None else pd.DataFrame(),
        ),
        encoding="utf-8",
    )
    return SemanticMultimodalMergeResult(
        multimodal_df=merged_df,
        output_path=output_path,
        report_path=report_path,
        audio_matched_windows=audio_matched_windows,
        environment_matched_windows=environment_matched_windows,
        event_labelled_windows=event_labelled_windows,
    )


def attach_event_labels(
    merged_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    config: SemanticZoneConfig,
) -> pd.DataFrame:
    working_df = prepare_time_columns(merged_df)
    working_df["window_mid_time_dt"] = working_df["start_time_dt"] + (
        (working_df["end_time_dt"] - working_df["start_time_dt"]) / 2
    )
    working_df["event_id"] = pd.NA
    working_df["event_type"] = pd.NA
    working_df["event_phase"] = "outside_event_window"
    working_df["seconds_from_event_start"] = np.nan
    working_df["seconds_from_event_end"] = np.nan
    working_df["event_start_time"] = pd.NA
    working_df["event_end_time"] = pd.NA
    working_df["event_overlap_seconds"] = 0.0
    working_df["overlaps_event"] = False
    working_df["_event_priority"] = 0

    if event_log_df.empty:
        return working_df.drop(columns=["_event_priority"], errors="ignore")

    event_df = event_log_df.copy()
    event_df["event_start_time_dt"] = pd.to_datetime(event_df["event_start_time"], errors="coerce", utc=False)
    event_df["event_end_time_dt"] = pd.to_datetime(event_df["event_end_time"], errors="coerce", utc=False)

    for _, event_row in event_df.iterrows():
        event_start = event_row.get("event_start_time_dt")
        event_end = event_row.get("event_end_time_dt")
        if pd.isna(event_start) or pd.isna(event_end):
            continue
        room_mask = working_df["room_id"].astype(str) == str(event_row.get("room_id", "") or "")
        baseline_start = event_start - pd.to_timedelta(config.baseline_minutes_before_event, unit="m")
        recovery_end = event_end + pd.to_timedelta(config.recovery_minutes_after_event, unit="m")
        overlap_seconds = working_df.apply(
            lambda row: compute_overlap_seconds(row["start_time_dt"], row["end_time_dt"], event_start, event_end),
            axis=1,
        )

        during_mask = room_mask & (overlap_seconds >= config.event_overlap_min_seconds)
        baseline_mask = room_mask & (working_df["end_time_dt"] <= event_start) & (working_df["end_time_dt"] > baseline_start)
        recovery_mask = room_mask & (working_df["start_time_dt"] >= event_end) & (working_df["start_time_dt"] < recovery_end)

        _apply_event_assignment(working_df, during_mask, event_row, event_start, event_end, overlap_seconds, "during_entry", 3)
        _apply_event_assignment(working_df, baseline_mask, event_row, event_start, event_end, overlap_seconds, "pre_entry_baseline", 2)
        _apply_event_assignment(working_df, recovery_mask, event_row, event_start, event_end, overlap_seconds, "post_entry_recovery", 1)
    return working_df.drop(columns=["_event_priority"], errors="ignore")


def _merge_audio_features(working_df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = working_df.copy()
    if audio_df.empty:
        for column in (
            "audio_rms",
            "audio_short_time_energy",
            "audio_zero_crossing_rate",
            "audio_spectral_centroid",
            "audio_spectral_bandwidth",
            "audio_spectral_rolloff",
            "audio_spectral_flatness",
            "audio_duration_sec",
            "audio_sample_rate",
            "audio_available",
            "audio_quality_flag",
            "audio_warning",
        ):
            merged_df[column] = False if column == "audio_available" else np.nan
        return merged_df

    audio_working_df = prepare_time_columns(audio_df)
    deduped_audio_df = audio_working_df.drop_duplicates(subset=["window_id"], keep="first").copy()
    merged_df = merged_df.merge(
        deduped_audio_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore"),
        on="window_id",
        how="left",
        suffixes=("", "_audio"),
    )
    if "audio_available" not in merged_df.columns or merged_df["audio_available"].isna().all():
        fallback_audio_df = deduped_audio_df.drop_duplicates(subset=["room_id", "start_time"], keep="first")
        merge_columns = [column for column in deduped_audio_df.columns if column not in {"window_id", "media_id", "end_time", "start_time_dt", "end_time_dt"}]
        merged_df = merged_df.merge(
            fallback_audio_df[["room_id", "start_time"] + [column for column in merge_columns if column not in {"room_id", "start_time"}]],
            on=["room_id", "start_time"],
            how="left",
            suffixes=("", "_fallback_audio"),
        )
    merged_df["audio_available"] = merged_df["audio_available"].fillna(False).astype(bool)
    return merged_df


def _attach_environment_context(
    merged_df: pd.DataFrame,
    env_df: pd.DataFrame,
    config: SemanticZoneConfig,
) -> pd.DataFrame:
    working_df = merged_df.copy()
    working_df["local_date"] = working_df["start_time_dt"].dt.date
    working_df["env_period"] = np.where(working_df["start_time_dt"].dt.hour < config.env_am_hour_cutoff, "AM", "PM")

    if env_df.empty:
        for column in ("temp_context", "rh_context", "temp_daily_mean", "rh_daily_mean", "temp_daily_range", "env_quality_flag"):
            working_df[column] = np.nan
        return working_df

    join_df = env_df.copy()
    join_df["local_date"] = pd.to_datetime(join_df["date"], errors="coerce").dt.date
    working_df = working_df.merge(
        join_df[
            [
                "local_date",
                "temp_am_mean",
                "temp_pm",
                "temp_daily_mean",
                "rh_am",
                "rh_pm",
                "rh_daily_mean",
                "temp_daily_range",
                "env_quality_flag",
            ]
        ],
        on="local_date",
        how="left",
    )
    working_df["temp_context"] = np.where(working_df["env_period"] == "AM", working_df["temp_am_mean"], working_df["temp_pm"])
    working_df["temp_context"] = working_df["temp_context"].fillna(working_df["temp_daily_mean"])
    working_df["rh_context"] = np.where(working_df["env_period"] == "AM", working_df["rh_am"], working_df["rh_pm"])
    working_df["rh_context"] = working_df["rh_context"].fillna(working_df["rh_daily_mean"])
    return working_df


def _apply_event_assignment(
    working_df: pd.DataFrame,
    candidate_mask: pd.Series,
    event_row: pd.Series,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    overlap_seconds: pd.Series,
    phase_name: str,
    priority: int,
) -> None:
    update_mask = candidate_mask & (working_df["_event_priority"] < priority)
    if not bool(update_mask.any()):
        return
    working_df.loc[update_mask, "event_id"] = event_row.get("event_id")
    working_df.loc[update_mask, "event_type"] = event_row.get("event_type")
    working_df.loc[update_mask, "event_phase"] = phase_name
    working_df.loc[update_mask, "seconds_from_event_start"] = (
        working_df.loc[update_mask, "window_mid_time_dt"] - event_start
    ).dt.total_seconds()
    working_df.loc[update_mask, "seconds_from_event_end"] = (
        working_df.loc[update_mask, "window_mid_time_dt"] - event_end
    ).dt.total_seconds()
    working_df.loc[update_mask, "event_start_time"] = event_start.isoformat()
    working_df.loc[update_mask, "event_end_time"] = event_end.isoformat()
    working_df.loc[update_mask, "event_overlap_seconds"] = overlap_seconds.loc[update_mask]
    working_df.loc[update_mask, "overlaps_event"] = phase_name == "during_entry"
    working_df.loc[update_mask, "_event_priority"] = priority


def _build_merge_report(
    merged_df: pd.DataFrame,
    audio_matched_windows: int,
    environment_matched_windows: int,
    event_labelled_windows: int,
    event_log_df: pd.DataFrame,
) -> str:
    missing_summary = merged_df[
        [
            column
            for column in (
                "audio_rms",
                "audio_short_time_energy",
                "audio_spectral_centroid",
                "temp_context",
                "rh_context",
                "event_phase",
            )
            if column in merged_df.columns
        ]
    ].isna().sum().to_dict()
    lines = [
        "# Semantic Multimodal Merge Report",
        "",
        "- This table merges semantic-zone video biomarkers, embedded MP4 audio, daily Room 1 environment context, and caretaker-event labels when available.",
        f"- Number of semantic biomarker windows: {len(merged_df)}",
        f"- Number with audio features: {audio_matched_windows}",
        f"- Number with environment match: {environment_matched_windows}",
        f"- Event-labelled windows: {event_labelled_windows}",
        f"- Date coverage start: `{merged_df['start_time'].min() if not merged_df.empty else 'n/a'}`",
        f"- Date coverage end: `{merged_df['start_time'].max() if not merged_df.empty else 'n/a'}`",
        f"- Event log rows available: {len(event_log_df)}",
        "",
        "## Missingness Summary",
        "",
    ]
    for column, count in missing_summary.items():
        lines.append(f"- `{column}`: {int(count)}")
    return "\n".join(lines) + "\n"
