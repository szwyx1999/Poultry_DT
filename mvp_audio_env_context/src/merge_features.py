from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AudioEnvContextConfig


@dataclass(frozen=True)
class MultimodalMergeResult:
    multimodal_df: pd.DataFrame
    output_path: Path
    report_path: Path
    audio_matched_windows: int
    environment_matched_windows: int
    event_labelled_windows: int


def build_multimodal_window_table(
    config: AudioEnvContextConfig,
    env_df: pd.DataFrame,
    audio_df: pd.DataFrame,
) -> MultimodalMergeResult:
    biomarker_df = _read_csv(config.biomarker_window_table_csv)
    hmm_df = _read_csv(config.hmm_state_sequence_csv)
    event_log_df = _read_csv(config.event_log_csv)

    if biomarker_df.empty:
        raise ValueError(f"Biomarker window table not found or empty: {config.biomarker_window_table_csv}")
    if hmm_df.empty:
        raise ValueError(f"HMM state sequence not found or empty: {config.hmm_state_sequence_csv}")

    biomarker_df = _prepare_time_columns(biomarker_df)
    hmm_df = _prepare_time_columns(hmm_df)

    hmm_columns = [
        "window_id",
        "room_id",
        "session_id",
        "start_time",
        "state_id",
        "state_label",
        "state_probability_max",
        "welfare_risk_score",
        "risk_level",
        "sustained_risk_flag",
        "high_risk_flag",
    ]
    available_hmm_columns = [column for column in hmm_columns if column in hmm_df.columns]
    merged_df = biomarker_df.merge(
        hmm_df[available_hmm_columns].drop_duplicates(subset=["window_id"], keep="first"),
        on="window_id",
        how="left",
        suffixes=("", "_hmm"),
    )
    if "state_id" not in merged_df.columns and {"room_id", "start_time"} <= set(merged_df.columns):
        fallback_df = hmm_df.drop_duplicates(subset=["room_id", "start_time"], keep="first")
        merged_df = merged_df.drop(columns=[column for column in available_hmm_columns if column in merged_df.columns and column != "window_id"], errors="ignore")
        merged_df = merged_df.merge(
            fallback_df[available_hmm_columns],
            on=["room_id", "start_time"],
            how="left",
        )

    merged_df = _merge_audio_features(merged_df, audio_df)
    merged_df = _attach_environment_context(merged_df, env_df, config)
    merged_df = _attach_event_labels(merged_df, event_log_df, config)
    merged_df = merged_df.sort_values(
        [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in merged_df.columns],
        kind="stable",
    ).reset_index(drop=True)

    required_columns = [
        "window_id",
        "room_id",
        "session_id",
        "start_time",
        "end_time",
        "local_date",
        "event_id",
        "event_phase",
        "activity_mean",
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "state_id",
        "state_label",
        "welfare_risk_score",
        "audio_rms",
        "audio_short_time_energy",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_spectral_rolloff",
        "temp_context",
        "rh_context",
        "temp_daily_mean",
        "rh_daily_mean",
        "audio_available",
        "env_quality_flag",
    ]
    for column in required_columns:
        if column not in merged_df.columns:
            merged_df[column] = np.nan

    output_path = config.features_dir / "multimodal_window_table.csv"
    report_path = config.reports_dir / "multimodal_merge_report.md"
    merged_df.to_csv(output_path, index=False)

    audio_matched_windows = int(merged_df["audio_available"].fillna(False).astype(bool).sum())
    environment_matched_windows = int(merged_df["temp_context"].notna().sum() | merged_df["rh_context"].notna().sum()) if not merged_df.empty else 0
    if not merged_df.empty:
        environment_matched_windows = int(
            (merged_df["temp_context"].notna() | merged_df["rh_context"].notna()).sum()
        )
    event_labelled_windows = int(
        merged_df["event_phase"].astype(str).isin(
            ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
        ).sum()
    ) if not merged_df.empty else 0

    report_path.write_text(
        _build_merge_report(
            merged_df=merged_df,
            event_log_df=event_log_df,
            audio_matched_windows=audio_matched_windows,
            environment_matched_windows=environment_matched_windows,
            event_labelled_windows=event_labelled_windows,
        ),
        encoding="utf-8",
    )
    return MultimodalMergeResult(
        multimodal_df=merged_df,
        output_path=output_path,
        report_path=report_path,
        audio_matched_windows=audio_matched_windows,
        environment_matched_windows=environment_matched_windows,
        event_labelled_windows=event_labelled_windows,
    )


def _merge_audio_features(merged_df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        working_df = merged_df.copy()
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
            if column not in working_df.columns:
                working_df[column] = np.nan if column not in {"audio_available"} else False
        return working_df

    deduped_audio_df = audio_df.drop_duplicates(subset=["window_id"], keep="first").copy()
    working_df = merged_df.merge(deduped_audio_df, on="window_id", how="left", suffixes=("", "_audio"))

    if "audio_available" not in working_df.columns or working_df["audio_available"].isna().all():
        fallback_audio_df = deduped_audio_df.drop_duplicates(subset=["room_id", "start_time"], keep="first")
        merge_columns = [column for column in deduped_audio_df.columns if column not in {"window_id", "media_id", "end_time"}]
        working_df = working_df.drop(columns=[column for column in merge_columns if column in working_df.columns and column.startswith("audio_")], errors="ignore")
        working_df = working_df.merge(
            fallback_audio_df[["room_id", "start_time"] + [column for column in merge_columns if column not in {"room_id", "start_time"}]],
            on=["room_id", "start_time"],
            how="left",
        )

    if "audio_available" in working_df.columns:
        working_df["audio_available"] = working_df["audio_available"].fillna(False).astype(bool)
    return working_df


def _attach_environment_context(
    merged_df: pd.DataFrame,
    env_df: pd.DataFrame,
    config: AudioEnvContextConfig,
) -> pd.DataFrame:
    working_df = merged_df.copy()
    working_df["local_date"] = working_df["start_time_dt"].dt.date
    working_df["env_period"] = np.where(
        working_df["start_time_dt"].dt.hour < config.env_am_hour_cutoff,
        "AM",
        "PM",
    )
    if env_df.empty:
        for column in ("temp_context", "rh_context", "temp_daily_mean", "rh_daily_mean", "temp_daily_range", "env_quality_flag"):
            if column not in working_df.columns:
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

    working_df["temp_context"] = np.where(
        working_df["env_period"] == "AM",
        working_df["temp_am_mean"],
        working_df["temp_pm"],
    )
    working_df["temp_context"] = working_df["temp_context"].fillna(working_df["temp_daily_mean"])

    working_df["rh_context"] = np.where(
        working_df["env_period"] == "AM",
        working_df["rh_am"],
        working_df["rh_pm"],
    )
    working_df["rh_context"] = working_df["rh_context"].fillna(working_df["rh_daily_mean"])
    return working_df


def _attach_event_labels(
    merged_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    config: AudioEnvContextConfig,
) -> pd.DataFrame:
    working_df = merged_df.copy()
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

    event_log_df = event_log_df.copy()
    event_log_df["event_start_time_dt"] = pd.to_datetime(event_log_df["event_start_time"], errors="coerce", utc=False)
    event_log_df["event_end_time_dt"] = pd.to_datetime(event_log_df["event_end_time"], errors="coerce", utc=False)
    for _, event_row in event_log_df.iterrows():
        event_start = event_row.get("event_start_time_dt")
        event_end = event_row.get("event_end_time_dt")
        if pd.isna(event_start) or pd.isna(event_end):
            continue
        room_mask = working_df["room_id"].astype(str) == str(event_row.get("room_id", "") or "")
        baseline_start = event_start - pd.to_timedelta(config.baseline_minutes_before_event, unit="m")
        recovery_end = event_end + pd.to_timedelta(config.recovery_minutes_after_event, unit="m")
        overlap_seconds = working_df.apply(
            lambda row: _compute_overlap_seconds(row["start_time_dt"], row["end_time_dt"], event_start, event_end),
            axis=1,
        )

        during_mask = room_mask & (overlap_seconds >= config.event_overlap_min_seconds)
        baseline_mask = room_mask & (working_df["end_time_dt"] <= event_start) & (working_df["end_time_dt"] > baseline_start)
        recovery_mask = room_mask & (working_df["start_time_dt"] >= event_end) & (working_df["start_time_dt"] < recovery_end)

        _apply_event_assignment(working_df, during_mask, event_row, event_start, event_end, overlap_seconds, "during_entry", 3)
        _apply_event_assignment(working_df, baseline_mask, event_row, event_start, event_end, overlap_seconds, "pre_entry_baseline", 2)
        _apply_event_assignment(working_df, recovery_mask, event_row, event_start, event_end, overlap_seconds, "post_entry_recovery", 1)

    return working_df.drop(columns=["_event_priority"], errors="ignore")


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


def _compute_overlap_seconds(
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


def _prepare_time_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_df = dataframe.copy()
    working_df["start_time_dt"] = pd.to_datetime(working_df.get("start_time"), errors="coerce", utc=False)
    working_df["end_time_dt"] = pd.to_datetime(working_df.get("end_time"), errors="coerce", utc=False)
    return working_df


def _build_merge_report(
    merged_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    audio_matched_windows: int,
    environment_matched_windows: int,
    event_labelled_windows: int,
) -> str:
    missingness_df = (
        merged_df[
            [
                "audio_rms",
                "audio_short_time_energy",
                "audio_zero_crossing_rate",
                "temp_context",
                "rh_context",
                "state_id",
                "welfare_risk_score",
            ]
        ]
        .isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        if not merged_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# Multimodal Merge Report",
        "",
        f"- Number of video/HMM windows: {len(merged_df)}",
        f"- Number with audio features: {audio_matched_windows}",
        f"- Number with environment match: {environment_matched_windows}",
        f"- Date coverage start: `{merged_df['local_date'].min() if not merged_df.empty else 'n/a'}`",
        f"- Date coverage end: `{merged_df['local_date'].max() if not merged_df.empty else 'n/a'}`",
        f"- Event-labelled windows: {event_labelled_windows}",
        "",
        "## Missingness Summary",
        "",
        "```text",
        missingness_df.to_string() if not missingness_df.empty else "No multimodal rows available.",
        "```",
        "",
        "## Event Coverage",
        "",
    ]
    if event_log_df.empty:
        lines.append("- No event log was available; event coverage is absent.")
    else:
        lines.append(f"- Event log rows available: {len(event_log_df)}")
        lines.append(
            "- Event phases present in merged table: "
            + ", ".join(sorted(merged_df["event_phase"].astype(str).dropna().unique().tolist()))
        )
    return "\n".join(lines) + "\n"


def _read_csv(path_value: Path) -> pd.DataFrame:
    if not path_value.exists():
        return pd.DataFrame()
    return pd.read_csv(path_value)
