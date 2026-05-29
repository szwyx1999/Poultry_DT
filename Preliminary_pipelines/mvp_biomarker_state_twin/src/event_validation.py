from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .config import BiomarkerTwinConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_ORDER = ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
EVENT_LOG_COLUMNS = [
    "event_id",
    "event_type",
    "system_id",
    "room_id",
    "session_id",
    "source_video_path",
    "source_media_id",
    "source_video_start_time",
    "entry_offset_sec",
    "exit_offset_sec",
    "event_start_time",
    "event_end_time",
    "label_quality",
    "notes",
]


@dataclass(frozen=True)
class EventValidationResult:
    labelled_state_df: pd.DataFrame
    summary_df: pd.DataFrame
    report_markdown: str
    event_covered: bool
    labelled_window_count: int
    processed_start_time: str
    processed_end_time: str


def ensure_caretaker_event_log(
    media_manifest_df: pd.DataFrame,
    config: BiomarkerTwinConfig,
    report_path: Path,
) -> pd.DataFrame:
    reference_path = config.caretaker_reference_video_path.resolve(strict=False)
    if media_manifest_df.empty:
        return _write_empty_event_log(
            config=config,
            report_path=report_path,
            reason="media_manifest.csv is empty; caretaker reference timestamp could not be resolved.",
        )
    if "file_path" not in media_manifest_df.columns:
        return _write_empty_event_log(
            config=config,
            report_path=report_path,
            reason="media_manifest.csv does not include a file_path column for caretaker reference matching.",
        )

    manifest_df = media_manifest_df.copy()
    manifest_df["resolved_file_path"] = manifest_df["file_path"].astype(str).apply(
        lambda value: str((config.workspace_root / value).resolve(strict=False))
    )
    match_df = manifest_df[manifest_df["resolved_file_path"] == str(reference_path)].copy()
    if match_df.empty:
        match_df = manifest_df[
            manifest_df["file_name"].astype(str).str.casefold() == config.caretaker_reference_video_path.name.casefold()
        ].copy()

    if match_df.empty:
        return _write_empty_event_log(
            config=config,
            report_path=report_path,
            reason=(
                "Could not find the caretaker reference video in media_manifest.csv: "
                f"{reference_path.as_posix()}"
            ),
        )

    reference_row = match_df.sort_values("start_time", kind="stable").iloc[0]
    video_start_time = pd.to_datetime(reference_row.get("start_time"), errors="coerce", utc=False)
    video_end_time = pd.to_datetime(reference_row.get("end_time"), errors="coerce", utc=False)
    if pd.isna(video_start_time):
        return _write_empty_event_log(
            config=config,
            report_path=report_path,
            reason="Caretaker reference video row is missing a valid start_time.",
        )

    event_start_time = video_start_time + pd.to_timedelta(config.caretaker_entry_offset_sec, unit="s")
    event_end_time = video_start_time + pd.to_timedelta(config.caretaker_exit_offset_sec, unit="s")
    timezone_name = str(reference_row.get("timezone", ""))
    warnings_text = str(reference_row.get("warnings", "") or "").strip()

    event_row = {
        "event_id": config.caretaker_event_id,
        "event_type": config.caretaker_event_type,
        "system_id": reference_row.get("system_id", "free_range"),
        "room_id": reference_row.get("room_id", "room_1"),
        "session_id": config.caretaker_reference_session_id,
        "source_video_path": reference_row.get("file_path", ""),
        "source_media_id": reference_row.get("media_id", ""),
        "source_video_start_time": video_start_time.isoformat(),
        "entry_offset_sec": config.caretaker_entry_offset_sec,
        "exit_offset_sec": config.caretaker_exit_offset_sec,
        "event_start_time": event_start_time.isoformat(),
        "event_end_time": event_end_time.isoformat(),
        "label_quality": "manual_offset_from_reference_video",
        "notes": "Caretaker enters GX330044.MP4 at 00:03:03 and leaves at 00:05:00.",
    }

    event_log_df = pd.DataFrame([event_row], columns=EVENT_LOG_COLUMNS)
    config.generated_event_log_csv.parent.mkdir(parents=True, exist_ok=True)
    event_log_df.to_csv(config.generated_event_log_csv, index=False)

    lines = [
        "# Caretaker Event Timestamp Report",
        "",
        f"- Reference video path: `{reference_row.get('file_path', '')}`",
        f"- Video start time: `{video_start_time.isoformat()}`",
        f"- Video end time: `{video_end_time.isoformat() if pd.notna(video_end_time) else 'n/a'}`",
        f"- Entry offset: `{config.caretaker_entry_offset_sec}` seconds",
        f"- Exit offset: `{config.caretaker_exit_offset_sec}` seconds",
        f"- Computed absolute entry time: `{event_start_time.isoformat()}`",
        f"- Computed absolute exit time: `{event_end_time.isoformat()}`",
        f"- Timezone: `{timezone_name or 'n/a'}`",
        f"- Timestamp source field used by preprocessing: `CreationDate` (via current timestamp priority order)",
        f"- Manifest warnings for the reference video: `{warnings_text or 'none'}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return event_log_df


def _write_empty_event_log(
    config: BiomarkerTwinConfig,
    report_path: Path,
    reason: str,
) -> pd.DataFrame:
    event_log_df = pd.DataFrame(columns=EVENT_LOG_COLUMNS)
    config.generated_event_log_csv.parent.mkdir(parents=True, exist_ok=True)
    event_log_df.to_csv(config.generated_event_log_csv, index=False)

    lines = [
        "# Caretaker Event Timestamp Report",
        "",
        "No caretaker event log row was generated for this run.",
        "",
        f"- Configured reference video path: `{config.caretaker_reference_video_path.as_posix()}`",
        f"- Reason: `{reason}`",
        "",
        "The biomarker/HMM pipeline will continue, but all windows will remain `outside_event_window`.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return event_log_df


def label_windows_relative_to_events(
    window_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    config: BiomarkerTwinConfig,
) -> pd.DataFrame:
    labelled_df = window_df.copy()
    if labelled_df.empty:
        return labelled_df

    rename_map = {}
    for column in ("event_id", "event_type", "event_phase", "candidate_event_source"):
        if column in labelled_df.columns:
            rename_map[column] = f"resilience_{column}"
    if rename_map:
        labelled_df = labelled_df.rename(columns=rename_map)

    labelled_df["window_mid_time_dt"] = labelled_df["start_time_dt"] + (
        (labelled_df["end_time_dt"] - labelled_df["start_time_dt"]) / 2
    )
    labelled_df["event_id"] = pd.NA
    labelled_df["event_type"] = pd.NA
    labelled_df["event_phase"] = "outside_event_window"
    labelled_df["seconds_from_event_start"] = np.nan
    labelled_df["seconds_from_event_end"] = np.nan
    labelled_df["event_start_time"] = pd.NA
    labelled_df["event_end_time"] = pd.NA
    labelled_df["event_overlap_seconds"] = 0.0
    labelled_df["overlaps_event"] = False
    labelled_df["_event_priority"] = 0

    if event_log_df.empty:
        return labelled_df.drop(columns=["_event_priority"], errors="ignore")

    working_events = event_log_df.copy()
    if "event_start_time" in working_events.columns:
        working_events["event_start_time_dt"] = pd.to_datetime(working_events["event_start_time"], errors="coerce", utc=False)
    elif "start_time" in working_events.columns:
        working_events["event_start_time_dt"] = pd.to_datetime(working_events["start_time"], errors="coerce", utc=False)
    else:
        working_events["event_start_time_dt"] = pd.NaT

    if "event_end_time" in working_events.columns:
        working_events["event_end_time_dt"] = pd.to_datetime(working_events["event_end_time"], errors="coerce", utc=False)
    elif "end_time" in working_events.columns:
        working_events["event_end_time_dt"] = pd.to_datetime(working_events["end_time"], errors="coerce", utc=False)
    else:
        working_events["event_end_time_dt"] = pd.NaT

    for _, event_row in working_events.iterrows():
        event_start = event_row.get("event_start_time_dt")
        event_end = event_row.get("event_end_time_dt")
        if pd.isna(event_start) or pd.isna(event_end):
            continue

        room_id = str(event_row.get("room_id", "") or "")
        room_mask = labelled_df["room_id"].astype(str) == room_id if room_id else pd.Series([True] * len(labelled_df), index=labelled_df.index)
        baseline_start = event_start - pd.to_timedelta(config.baseline_minutes_before_event, unit="m")
        recovery_end = event_end + pd.to_timedelta(config.recovery_minutes_after_event, unit="m")

        overlap_seconds = labelled_df.apply(
            lambda row: _compute_overlap_seconds(
                row["start_time_dt"],
                row["end_time_dt"],
                event_start,
                event_end,
            ),
            axis=1,
        )

        during_mask = room_mask & (overlap_seconds >= config.event_overlap_min_seconds)
        baseline_mask = (
            room_mask
            & (labelled_df["end_time_dt"] <= event_start)
            & (labelled_df["end_time_dt"] > baseline_start)
        )
        recovery_mask = (
            room_mask
            & (labelled_df["start_time_dt"] >= event_end)
            & (labelled_df["start_time_dt"] < recovery_end)
        )

        _apply_event_assignment(
            labelled_df,
            during_mask,
            event_row,
            event_start,
            event_end,
            overlap_seconds,
            phase_name="during_entry",
            priority=3,
        )
        _apply_event_assignment(
            labelled_df,
            baseline_mask,
            event_row,
            event_start,
            event_end,
            overlap_seconds,
            phase_name="pre_entry_baseline",
            priority=2,
        )
        _apply_event_assignment(
            labelled_df,
            recovery_mask,
            event_row,
            event_start,
            event_end,
            overlap_seconds,
            phase_name="post_entry_recovery",
            priority=1,
        )

    labelled_df["overlaps_event"] = labelled_df["overlaps_event"].astype(bool)
    return labelled_df.drop(columns=["_event_priority"], errors="ignore")


def build_event_validation_result(
    labelled_window_df: pd.DataFrame,
    state_sequence_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    config: BiomarkerTwinConfig,
) -> EventValidationResult:
    merge_columns = [
        column
        for column in (
            "window_id",
            "state_id",
            "state_label",
            "state_probability_max",
            "welfare_risk_score",
            "risk_level",
            "sustained_risk_flag",
        )
        if column in state_sequence_df.columns
    ]
    merged_df = labelled_window_df.merge(
        state_sequence_df[merge_columns].drop_duplicates(subset=["window_id"]),
        on="window_id",
        how="left",
    )

    merged_df = merged_df.sort_values(
        [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in merged_df.columns],
        kind="stable",
    ).reset_index(drop=True)

    validation_df = merged_df[
        (merged_df["event_id"].astype(str) == config.caretaker_event_id)
        & (merged_df["event_phase"].isin(PHASE_ORDER))
    ].copy()

    summary_rows: list[dict] = []
    for phase in PHASE_ORDER:
        phase_df = validation_df[validation_df["event_phase"] == phase].copy()
        dominant_label = "n/a"
        dominant_fraction = np.nan
        if not phase_df.empty and "state_label" in phase_df.columns:
            state_counts = phase_df["state_label"].fillna("unlabeled_state").value_counts(normalize=True)
            dominant_label = str(state_counts.index[0])
            dominant_fraction = float(state_counts.iloc[0])

        summary_rows.append(
            {
                "event_id": config.caretaker_event_id,
                "phase": phase,
                "n_windows": int(len(phase_df)),
                "activity_mean_mean": _safe_mean(phase_df, "activity_mean"),
                "activity_mean_max": _safe_max(phase_df, "activity_mean"),
                "mobility_index_mean": _safe_mean(phase_df, "mobility_index"),
                "mobility_index_max": _safe_max(phase_df, "mobility_index"),
                "spatial_freedom_index_mean": _safe_mean(phase_df, "spatial_freedom_index"),
                "occupancy_imbalance_index_mean": _safe_mean(phase_df, "occupancy_imbalance_index"),
                "risk_score_mean": _safe_mean(phase_df, "welfare_risk_score"),
                "risk_score_max": _safe_max(phase_df, "welfare_risk_score"),
                "dominant_state_label": dominant_label,
                "dominant_state_fraction": dominant_fraction,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    processed_start_time = _safe_text_min(merged_df, "start_time")
    processed_end_time = _safe_text_max(merged_df, "start_time")
    event_covered = not validation_df.empty
    labelled_window_count = int(validation_df["window_id"].nunique()) if not validation_df.empty else 0
    report_markdown = _build_event_validation_report(summary_df, validation_df, event_log_df, config)
    return EventValidationResult(
        labelled_state_df=merged_df,
        summary_df=summary_df,
        report_markdown=report_markdown,
        event_covered=event_covered,
        labelled_window_count=labelled_window_count,
        processed_start_time=processed_start_time,
        processed_end_time=processed_end_time,
    )


def generate_event_validation_plots(
    validation_result: EventValidationResult,
    event_log_df: pd.DataFrame,
    config: BiomarkerTwinConfig,
) -> None:
    merged_df = validation_result.labelled_state_df.copy()
    target_event_df = merged_df[
        (merged_df["event_id"].astype(str) == config.caretaker_event_id)
        & (merged_df["event_phase"].isin(PHASE_ORDER))
    ].copy()
    target_event_df["minutes_from_event_start"] = pd.to_numeric(
        target_event_df["seconds_from_event_start"], errors="coerce"
    ) / 60.0

    event_row = event_log_df[event_log_df["event_id"].astype(str) == config.caretaker_event_id]
    event_end_minutes = (config.caretaker_exit_offset_sec - config.caretaker_entry_offset_sec) / 60.0
    if not event_row.empty:
        event_start_dt = pd.to_datetime(event_row.iloc[0]["event_start_time"], errors="coerce", utc=False)
    else:
        event_start_dt = pd.NaT

    _plot_event_centered_biomarkers(target_event_df, config.plots_dir / "event_centered_biomarker_timeline.png", event_end_minutes)
    _plot_event_centered_states(target_event_df, config.plots_dir / "event_centered_hmm_state_timeline.png", event_end_minutes)
    _plot_full_state_risk_timeline(merged_df, event_start_dt, config.plots_dir / "aug16_17_hmm_state_risk_timeline.png")
    _plot_event_phase_metric_comparison(validation_result.summary_df, config.plots_dir / "event_phase_metric_comparison.png")
    _plot_state_distribution_by_event_phase(target_event_df, config.plots_dir / "state_distribution_by_event_phase.png")


def _apply_event_assignment(
    labelled_df: pd.DataFrame,
    candidate_mask: pd.Series,
    event_row: pd.Series,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    overlap_seconds: pd.Series,
    phase_name: str,
    priority: int,
) -> None:
    update_mask = candidate_mask & (labelled_df["_event_priority"] < priority)
    if not bool(update_mask.any()):
        return

    seconds_from_start = (
        labelled_df.loc[update_mask, "window_mid_time_dt"] - event_start
    ).dt.total_seconds()
    seconds_from_end = (
        labelled_df.loc[update_mask, "window_mid_time_dt"] - event_end
    ).dt.total_seconds()

    labelled_df.loc[update_mask, "event_id"] = event_row.get("event_id")
    labelled_df.loc[update_mask, "event_type"] = event_row.get("event_type")
    labelled_df.loc[update_mask, "event_phase"] = phase_name
    labelled_df.loc[update_mask, "seconds_from_event_start"] = seconds_from_start
    labelled_df.loc[update_mask, "seconds_from_event_end"] = seconds_from_end
    labelled_df.loc[update_mask, "event_start_time"] = event_start.isoformat()
    labelled_df.loc[update_mask, "event_end_time"] = event_end.isoformat()
    labelled_df.loc[update_mask, "event_overlap_seconds"] = overlap_seconds.loc[update_mask]
    labelled_df.loc[update_mask, "overlaps_event"] = phase_name == "during_entry"
    labelled_df.loc[update_mask, "_event_priority"] = priority


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


def _build_event_validation_report(
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    config: BiomarkerTwinConfig,
) -> str:
    baseline_mean_mobility = _phase_metric(summary_df, "pre_entry_baseline", "mobility_index_mean")
    during_mean_mobility = _phase_metric(summary_df, "during_entry", "mobility_index_mean")
    recovery_mean_mobility = _phase_metric(summary_df, "post_entry_recovery", "mobility_index_mean")
    baseline_mean_risk = _phase_metric(summary_df, "pre_entry_baseline", "risk_score_mean")
    during_mean_risk = _phase_metric(summary_df, "during_entry", "risk_score_mean")
    recovery_mean_risk = _phase_metric(summary_df, "post_entry_recovery", "risk_score_mean")

    mobility_increased = _compare_gt(during_mean_mobility, baseline_mean_mobility)
    mobility_recovered = _compare_lt(recovery_mean_mobility, during_mean_mobility)
    risk_increased = _compare_gt(during_mean_risk, baseline_mean_risk)
    risk_recovered = _compare_lt(recovery_mean_risk, during_mean_risk)

    peak_mobility_row = _peak_row(validation_df, "mobility_index")
    peak_risk_row = _peak_row(validation_df, "welfare_risk_score")
    dominant_state_text = _state_transition_text(validation_df)
    response_biomarker = _strongest_response_metric(summary_df)

    recovery_time_text = "n/a"
    post_df = validation_df[validation_df["event_phase"] == "post_entry_recovery"].copy()
    if not post_df.empty and pd.notna(baseline_mean_mobility):
        recovery_candidates = post_df[pd.to_numeric(post_df["mobility_index"], errors="coerce") <= baseline_mean_mobility]
        if not recovery_candidates.empty:
            recovery_time_text = f"{float(recovery_candidates.iloc[0]['seconds_from_event_end']) / 60.0:.2f} minutes after event end"

    interpretation = "No clear event-aligned change was detected."
    if mobility_increased or risk_increased:
        interpretation = "The video-derived biomarker system detected a change around the labelled caretaker-entry event."

    lines = [
        "# Event Validation Report",
        "",
        f"Known event: `{config.caretaker_event_id}`",
        "",
        "This is a validation against one labelled management event, not biological welfare ground truth.",
        "",
        "## Phase Summary",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        f"- {interpretation}",
        f"- Strongest responding biomarker by mean phase shift: `{response_biomarker}`",
        f"- Mobility increased during entry vs baseline: `{_yes_no(mobility_increased)}`",
        f"- Mobility decreased during recovery vs during-entry: `{_yes_no(mobility_recovered)}`",
        f"- Risk score increased during entry vs baseline: `{_yes_no(risk_increased)}`",
        f"- Risk score decreased during recovery vs during-entry: `{_yes_no(risk_recovered)}`",
        f"- Peak mobility response time relative to event start: `{_peak_time_text(peak_mobility_row)}`",
        f"- Peak risk response time relative to event start: `{_peak_time_text(peak_risk_row)}`",
        f"- Recovery time after event end: `{recovery_time_text}`",
        "",
        "## HMM Behavioural State Context",
        "",
        f"- Dominant state transition sequence around event: `{dominant_state_text}`",
        f"- Most frequent state during entry: `{_phase_state(summary_df, 'during_entry')}`",
        f"- Most frequent state during recovery: `{_phase_state(summary_df, 'post_entry_recovery')}`",
        "",
        "## Caveats",
        "",
        "- Prototype video-derived biomarker and latent-state response only.",
        "- Not a validated welfare diagnosis.",
        "- Event timing is anchored to one manually offset reference video.",
    ]

    if not event_log_df.empty:
        lines.extend(
            [
                "",
                "## Event Log Row",
                "",
                "```text",
                event_log_df[event_log_df["event_id"].astype(str) == config.caretaker_event_id].to_string(index=False),
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def _plot_event_centered_biomarkers(validation_df: pd.DataFrame, output_path: Path, event_end_minutes: float) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    metrics = [
        ("activity_mean", "Activity Mean"),
        ("mobility_index", "Mobility Index"),
        ("spatial_freedom_index", "Spatial Freedom Index"),
        ("occupancy_imbalance_index", "Occupancy Imbalance Index"),
        ("welfare_risk_score", "Risk Score"),
    ]
    for axis, (column, title) in zip(axes, metrics):
        axis.plot(validation_df["minutes_from_event_start"], pd.to_numeric(validation_df[column], errors="coerce"), marker="o")
        _annotate_event_phase_bands(axis)
        axis.axvline(0.0, color="#c94f3d", linestyle="--", linewidth=1.2)
        axis.axvline(event_end_minutes, color="#28536b", linestyle="--", linewidth=1.2)
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].set_title("Event-Centered Biomarker Timeline")
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_event_centered_states(validation_df: pd.DataFrame, output_path: Path, event_end_minutes: float) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    state_axis, risk_axis = axes
    ordered_df = validation_df.sort_values("minutes_from_event_start", kind="stable")
    state_axis.step(ordered_df["minutes_from_event_start"], ordered_df["state_id"], where="post", color="#28536b")
    state_axis.scatter(ordered_df["minutes_from_event_start"], ordered_df["state_id"], color="#28536b", s=24)
    _set_state_ticks(state_axis, ordered_df)
    _annotate_event_phase_bands(state_axis)
    state_axis.axvline(0.0, color="#c94f3d", linestyle="--", linewidth=1.2)
    state_axis.axvline(event_end_minutes, color="#28536b", linestyle="--", linewidth=1.2)
    state_axis.set_ylabel("HMM State")
    state_axis.set_title("Event-Centered HMM State Timeline")
    state_axis.grid(alpha=0.25)

    risk_axis.plot(ordered_df["minutes_from_event_start"], pd.to_numeric(ordered_df["welfare_risk_score"], errors="coerce"), marker="o", color="#c94f3d")
    _annotate_event_phase_bands(risk_axis)
    risk_axis.axvline(0.0, color="#c94f3d", linestyle="--", linewidth=1.2)
    risk_axis.axvline(event_end_minutes, color="#28536b", linestyle="--", linewidth=1.2)
    risk_axis.set_ylabel("Risk Score")
    risk_axis.set_xlabel("Minutes Relative To Caretaker Entry Start")
    risk_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_full_state_risk_timeline(merged_df: pd.DataFrame, event_start_dt: pd.Timestamp, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ordered_df = merged_df.sort_values("start_time_dt", kind="stable")
    state_axis, risk_axis = axes
    state_axis.step(ordered_df["start_time_dt"], ordered_df["state_id"], where="post", color="#28536b")
    _set_state_ticks(state_axis, ordered_df)
    if pd.notna(event_start_dt):
        state_axis.axvline(event_start_dt, color="#c94f3d", linestyle="--", linewidth=1.2)
        risk_axis.axvline(event_start_dt, color="#c94f3d", linestyle="--", linewidth=1.2)
    state_axis.set_ylabel("HMM State")
    state_axis.set_title("Full Aug 16-17 HMM State And Risk Timeline")
    state_axis.grid(alpha=0.25)
    risk_axis.plot(ordered_df["start_time_dt"], pd.to_numeric(ordered_df["welfare_risk_score"], errors="coerce"), color="#c94f3d")
    risk_axis.set_ylabel("Risk Score")
    risk_axis.set_xlabel("Window Start Time")
    risk_axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_event_phase_metric_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("activity_mean_mean", "Activity"),
        ("mobility_index_mean", "Mobility"),
        ("risk_score_mean", "Risk"),
        ("spatial_freedom_index_mean", "Spatial Freedom"),
        ("occupancy_imbalance_index_mean", "Occupancy Imbalance"),
    ]
    plot_df = summary_df.set_index("phase").reindex(PHASE_ORDER)
    x_positions = np.arange(len(PHASE_ORDER))
    width = 0.14

    figure, axis = plt.subplots(figsize=(12, 6))
    for metric_index, (column, label) in enumerate(metrics):
        axis.bar(
            x_positions + ((metric_index - (len(metrics) - 1) / 2) * width),
            pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0),
            width=width,
            label=label,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(PHASE_ORDER, rotation=20, ha="right")
    axis.set_ylabel("Mean Value")
    axis.set_title("Event Phase Metric Comparison")
    axis.legend(loc="upper left", ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_state_distribution_by_event_phase(validation_df: pd.DataFrame, output_path: Path) -> None:
    if validation_df.empty or "state_label" not in validation_df.columns:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, "No event-labelled HMM states available.", ha="center", va="center", fontsize=11)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    distribution_df = (
        validation_df.groupby(["event_phase", "state_label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(PHASE_ORDER, fill_value=0)
    )
    normalized_df = distribution_df.div(distribution_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    figure, axis = plt.subplots(figsize=(11, 6))
    normalized_df.plot(kind="bar", stacked=True, ax=axis, colormap="tab20")
    axis.set_ylabel("Fraction Of Windows")
    axis.set_title("State Distribution By Event Phase")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _annotate_event_phase_bands(axis) -> None:
    axis.axvspan(-10.0, 0.0, color="#76b7b2", alpha=0.10)
    axis.axvspan(0.0, 1.95, color="#c94f3d", alpha=0.08)
    axis.axvspan(1.95, 21.95, color="#f1c453", alpha=0.08)


def _set_state_ticks(axis, dataframe: pd.DataFrame) -> None:
    label_map = dataframe[["state_id", "state_label"]].dropna().drop_duplicates().sort_values("state_id", kind="stable")
    if label_map.empty:
        return
    axis.set_yticks(label_map["state_id"].tolist())
    axis.set_yticklabels(label_map["state_label"].tolist())


def _safe_mean(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe.empty or column not in dataframe.columns:
        return np.nan
    series = pd.to_numeric(dataframe[column], errors="coerce")
    return float(series.mean()) if series.notna().any() else np.nan


def _safe_max(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe.empty or column not in dataframe.columns:
        return np.nan
    series = pd.to_numeric(dataframe[column], errors="coerce")
    return float(series.max()) if series.notna().any() else np.nan


def _safe_text_min(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    series = dataframe[column].dropna().astype(str)
    return str(series.min()) if not series.empty else ""


def _safe_text_max(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    series = dataframe[column].dropna().astype(str)
    return str(series.max()) if not series.empty else ""


def _phase_metric(summary_df: pd.DataFrame, phase: str, column: str) -> float:
    phase_df = summary_df[summary_df["phase"] == phase]
    if phase_df.empty or column not in phase_df.columns:
        return np.nan
    return float(phase_df.iloc[0][column]) if pd.notna(phase_df.iloc[0][column]) else np.nan


def _compare_gt(left: float, right: float) -> bool:
    return pd.notna(left) and pd.notna(right) and float(left) > float(right)


def _compare_lt(left: float, right: float) -> bool:
    return pd.notna(left) and pd.notna(right) and float(left) < float(right)


def _peak_row(dataframe: pd.DataFrame, column: str) -> pd.Series | None:
    if dataframe.empty or column not in dataframe.columns:
        return None
    series = pd.to_numeric(dataframe[column], errors="coerce")
    if not series.notna().any():
        return None
    return dataframe.loc[int(series.idxmax())]


def _peak_time_text(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    seconds_value = pd.to_numeric(row.get("seconds_from_event_start"), errors="coerce")
    if pd.isna(seconds_value):
        return "n/a"
    return f"{float(seconds_value) / 60.0:.2f} minutes"


def _state_transition_text(validation_df: pd.DataFrame) -> str:
    if validation_df.empty or "state_label" not in validation_df.columns:
        return "n/a"
    ordered = validation_df.sort_values("start_time_dt", kind="stable")
    transitions: list[str] = []
    previous_label = None
    for _, row in ordered.iterrows():
        label = str(row.get("state_label", "unlabeled_state"))
        if label == previous_label:
            continue
        minute_value = pd.to_numeric(row.get("seconds_from_event_start"), errors="coerce")
        if pd.notna(minute_value):
            transitions.append(f"{float(minute_value) / 60.0:+.2f} min: {label}")
        else:
            transitions.append(label)
        previous_label = label
    return " -> ".join(transitions) if transitions else "n/a"


def _phase_state(summary_df: pd.DataFrame, phase: str) -> str:
    phase_df = summary_df[summary_df["phase"] == phase]
    if phase_df.empty:
        return "n/a"
    return str(phase_df.iloc[0].get("dominant_state_label", "n/a"))


def _strongest_response_metric(summary_df: pd.DataFrame) -> str:
    baseline_df = summary_df[summary_df["phase"] == "pre_entry_baseline"]
    during_df = summary_df[summary_df["phase"] == "during_entry"]
    if baseline_df.empty or during_df.empty:
        return "n/a"
    metrics = {
        "activity_mean": abs(float(during_df.iloc[0]["activity_mean_mean"]) - float(baseline_df.iloc[0]["activity_mean_mean"])),
        "mobility_index": abs(float(during_df.iloc[0]["mobility_index_mean"]) - float(baseline_df.iloc[0]["mobility_index_mean"])),
        "spatial_freedom_index": abs(float(during_df.iloc[0]["spatial_freedom_index_mean"]) - float(baseline_df.iloc[0]["spatial_freedom_index_mean"])),
        "occupancy_imbalance_index": abs(float(during_df.iloc[0]["occupancy_imbalance_index_mean"]) - float(baseline_df.iloc[0]["occupancy_imbalance_index_mean"])),
        "risk_score": abs(float(during_df.iloc[0]["risk_score_mean"]) - float(baseline_df.iloc[0]["risk_score_mean"])),
    }
    return max(metrics, key=metrics.get)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
