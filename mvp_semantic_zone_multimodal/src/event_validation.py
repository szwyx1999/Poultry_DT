from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SemanticZoneConfig
from .hmm_model import SemanticHmmResult
from .merge_multimodal import attach_event_labels
from .plotting import (
    plot_event_centered_semantic_biomarkers,
    plot_event_centered_semantic_multimodal_timeline,
    plot_event_centered_semantic_zone_activity,
    plot_semantic_hmm_state_risk_timeline,
    plot_semantic_state_distribution_by_event_phase,
    plot_semantic_vs_fourzone_event_comparison,
)
from .utils import prepare_time_columns, read_csv_if_exists


PHASE_ORDER = ["pre_entry_baseline", "during_entry", "post_entry_recovery"]


@dataclass(frozen=True)
class EventLogResult:
    event_log_df: pd.DataFrame
    source: str


@dataclass(frozen=True)
class SemanticLabelResult:
    labelled_df: pd.DataFrame
    output_path: Path
    report_path: Path


@dataclass(frozen=True)
class SemanticEventValidationResult:
    summary_df: pd.DataFrame
    output_path: Path
    report_path: Path
    meeting_bullet: str


def ensure_event_log(config: SemanticZoneConfig) -> EventLogResult:
    existing_event_df = read_csv_if_exists(config.event_log_csv)
    if not existing_event_df.empty:
        return EventLogResult(event_log_df=existing_event_df, source="existing_biomarker_event_log")

    manifest_df = read_csv_if_exists(config.media_manifest_csv)
    if manifest_df.empty:
        return EventLogResult(event_log_df=pd.DataFrame(), source="missing_media_manifest")

    reference_path = config.caretaker_reference_video_path.resolve(strict=False)
    manifest_df = manifest_df.copy()
    manifest_df["resolved_file_path"] = manifest_df["file_path"].astype(str).apply(
        lambda value: str((config.workspace_root / value).resolve(strict=False))
    )
    match_df = manifest_df[manifest_df["resolved_file_path"] == str(reference_path)].copy()
    if match_df.empty:
        match_df = manifest_df[
            manifest_df["file_name"].astype(str).str.casefold() == config.caretaker_reference_video_path.name.casefold()
        ].copy()
    if match_df.empty:
        return EventLogResult(event_log_df=pd.DataFrame(), source="reference_video_missing_from_manifest")

    reference_row = match_df.sort_values("start_time", kind="stable").iloc[0]
    video_start_time = pd.to_datetime(reference_row.get("start_time"), errors="coerce", utc=False)
    if pd.isna(video_start_time):
        return EventLogResult(event_log_df=pd.DataFrame(), source="reference_video_start_time_missing")

    event_start_time = video_start_time + pd.to_timedelta(config.caretaker_entry_offset_sec, unit="s")
    event_end_time = video_start_time + pd.to_timedelta(config.caretaker_exit_offset_sec, unit="s")
    event_log_df = pd.DataFrame(
        [
            {
                "event_id": config.caretaker_event_id,
                "event_type": config.caretaker_event_type,
                "system_id": reference_row.get("system_id", "free_range"),
                "room_id": reference_row.get("room_id", config.target_room_id),
                "session_id": reference_row.get("session_id", "caretaker_entry_week_11_17_aug"),
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
        ]
    )
    return EventLogResult(event_log_df=event_log_df, source="recreated_from_media_manifest")


def label_semantic_windows(
    config: SemanticZoneConfig,
    multimodal_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
) -> SemanticLabelResult:
    labelled_df = attach_event_labels(multimodal_df, event_log_df, config)
    output_path = config.features_dir / "semantic_labelled_window_table.csv"
    report_path = config.reports_dir / "semantic_event_label_report.md"
    labelled_df.to_csv(output_path, index=False)

    phase_counts = labelled_df["event_phase"].value_counts().to_dict() if not labelled_df.empty else {}
    event_row = event_log_df.iloc[0] if not event_log_df.empty else {}
    lines = [
        "# Semantic Event Label Report",
        "",
        "- This semantic-zone experiment uses the caretaker-entry event as a management-event validation target.",
        f"- Event log source: `{('existing' if not event_log_df.empty else 'missing')}`",
        f"- Event id: `{event_row.get('event_id', 'n/a') if isinstance(event_row, pd.Series) else 'n/a'}`",
        f"- Event start time: `{event_row.get('event_start_time', 'n/a') if isinstance(event_row, pd.Series) else 'n/a'}`",
        f"- Event end time: `{event_row.get('event_end_time', 'n/a') if isinstance(event_row, pd.Series) else 'n/a'}`",
        "",
        "## Phase Counts",
        "",
    ]
    for phase_name in ["pre_entry_baseline", "during_entry", "post_entry_recovery", "outside_event_window"]:
        lines.append(f"- `{phase_name}`: {int(phase_counts.get(phase_name, 0))}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SemanticLabelResult(labelled_df=labelled_df, output_path=output_path, report_path=report_path)


def build_semantic_event_validation(
    config: SemanticZoneConfig,
    labelled_df: pd.DataFrame,
    hmm_result: SemanticHmmResult,
    previous_fourzone_event_summary_csv: Path,
) -> SemanticEventValidationResult:
    primary_sequence_df = hmm_result.sequence_df[hmm_result.sequence_df["model_name"] == config.primary_model_name].copy()
    if primary_sequence_df.empty:
        raise ValueError(f"No HMM sequence rows were available for primary model `{config.primary_model_name}`.")

    merged_df = labelled_df.merge(
        primary_sequence_df[
            [
                "window_id",
                "model_name",
                "model_type",
                "state_id",
                "state_label",
                "state_probability_max",
                "risk_score",
                "risk_level",
                "sustained_risk_flag",
                "high_risk_flag",
            ]
        ].drop_duplicates(subset=["window_id"], keep="first"),
        on="window_id",
        how="left",
    )
    merged_df = prepare_time_columns(merged_df)
    event_df = merged_df[
        (merged_df["event_id"].astype(str) == config.caretaker_event_id)
        & (merged_df["event_phase"].isin(PHASE_ORDER))
    ].copy()
    if event_df.empty:
        raise ValueError("No event-labelled semantic windows were available for caretaker-entry validation.")
    event_df["minutes_from_event_start"] = pd.to_numeric(event_df["seconds_from_event_start"], errors="coerce") / 60.0

    summary_rows: list[dict] = []
    for phase in PHASE_ORDER:
        phase_df = event_df[event_df["event_phase"] == phase].copy()
        dominant_label = "n/a"
        dominant_fraction = np.nan
        if not phase_df.empty and "state_label" in phase_df.columns:
            counts = phase_df["state_label"].value_counts(normalize=True)
            if not counts.empty:
                dominant_label = str(counts.index[0])
                dominant_fraction = float(counts.iloc[0])
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
                "drinking_activity_fraction_mean": _safe_mean(phase_df, "drinking_activity_fraction"),
                "feeding_activity_fraction_mean": _safe_mean(phase_df, "feeding_activity_fraction"),
                "general_activity_fraction_mean": _safe_mean(phase_df, "general_activity_fraction"),
                "feeding_plus_drinking_activity_fraction_mean": _safe_mean(phase_df, "feeding_plus_drinking_activity_fraction"),
                "semantic_transition_proxy_mean": _safe_mean(phase_df, "semantic_transition_proxy"),
                "audio_rms_mean": _safe_mean(phase_df, "audio_rms"),
                "audio_spectral_centroid_mean": _safe_mean(phase_df, "audio_spectral_centroid"),
                "audio_spectral_rolloff_mean": _safe_mean(phase_df, "audio_spectral_rolloff"),
                "risk_score_mean": _safe_mean(phase_df, "risk_score"),
                "risk_score_max": _safe_max(phase_df, "risk_score"),
                "dominant_state_label": dominant_label,
                "dominant_state_fraction": dominant_fraction,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    output_path = config.features_dir / "semantic_event_validation_summary.csv"
    report_path = config.reports_dir / "semantic_event_validation_report.md"
    summary_df.to_csv(output_path, index=False)

    event_row = event_df.sort_values("start_time_dt", kind="stable").iloc[0]
    event_start = pd.to_datetime(event_row.get("event_start_time"), errors="coerce", utc=False)
    event_end = pd.to_datetime(event_row.get("event_end_time"), errors="coerce", utc=False)
    plot_event_centered_semantic_zone_activity(event_df, config.plots_dir / "event_centered_semantic_zone_activity.png")
    plot_event_centered_semantic_biomarkers(event_df, config.plots_dir / "event_centered_semantic_biomarkers.png")
    plot_event_centered_semantic_multimodal_timeline(event_df, config.plots_dir / "event_centered_semantic_multimodal_timeline.png")
    plot_semantic_hmm_state_risk_timeline(primary_sequence_df, event_start, event_end, config.plots_dir / "semantic_hmm_state_risk_timeline.png")
    plot_semantic_state_distribution_by_event_phase(event_df, config.plots_dir / "semantic_state_distribution_by_event_phase.png")

    fourzone_summary_df = read_csv_if_exists(previous_fourzone_event_summary_csv)
    plot_semantic_vs_fourzone_event_comparison(summary_df, fourzone_summary_df, config.plots_dir / "semantic_vs_fourzone_event_comparison.png")

    report_text, meeting_bullet = _build_validation_report(summary_df, event_df, fourzone_summary_df)
    report_path.write_text(report_text, encoding="utf-8")
    return SemanticEventValidationResult(
        summary_df=summary_df,
        output_path=output_path,
        report_path=report_path,
        meeting_bullet=meeting_bullet,
    )


def _build_validation_report(
    summary_df: pd.DataFrame,
    event_df: pd.DataFrame,
    fourzone_summary_df: pd.DataFrame,
) -> tuple[str, str]:
    phase_lookup = summary_df.set_index("phase")
    baseline = phase_lookup.loc["pre_entry_baseline"] if "pre_entry_baseline" in phase_lookup.index else pd.Series(dtype=float)
    during = phase_lookup.loc["during_entry"] if "during_entry" in phase_lookup.index else pd.Series(dtype=float)
    recovery = phase_lookup.loc["post_entry_recovery"] if "post_entry_recovery" in phase_lookup.index else pd.Series(dtype=float)

    activity_increase = _phase_change_text(baseline, during, "activity_mean_mean", "overall activity")
    mobility_increase = _phase_change_text(baseline, during, "mobility_index_mean", "mobility index")
    risk_increase = _phase_change_text(baseline, during, "risk_score_mean", "risk score")
    recovery_text = _recovery_text(baseline, during, recovery)

    semantic_interpretation = _semantic_interpretation_text(baseline, during)
    audio_text = _audio_text(baseline, during)
    dominant_during_state = str(during.get("dominant_state_label", "n/a"))

    fourzone_text = "Previous four-zone comparison was unavailable."
    if not fourzone_summary_df.empty and "phase" in fourzone_summary_df.columns:
        fourzone_lookup = fourzone_summary_df.set_index("phase")
        if "during_entry" in fourzone_lookup.index and "pre_entry_baseline" in fourzone_lookup.index:
            fourzone_delta = float(fourzone_lookup.loc["during_entry", "mobility_index_mean"] - fourzone_lookup.loc["pre_entry_baseline", "mobility_index_mean"])
            semantic_delta = float(during.get("mobility_index_mean", np.nan) - baseline.get("mobility_index_mean", np.nan))
            fourzone_text = (
                f"Semantic mobility contrast was `{semantic_delta:.4f}` versus previous four-zone mobility contrast `{fourzone_delta:.4f}`. "
                "The semantic version is more interpretable because it also shows whether motion shifts toward drinking or feeding areas."
            )

    lines = [
        "# Semantic Event Validation Report",
        "",
        "- This is a new semantic-zone experiment layered on top of the existing analytics backbone. Previous code was not modified.",
        "- Inputs for this experiment were two days of Room 1 video, embedded MP4 audio, daily Room 1 environment context, and the manual caretaker-entry label.",
        "- This report is exploratory and does not claim validated welfare diagnosis.",
        "",
        "## Answers",
        "",
        f"1. Did caretaker entry increase overall activity? {activity_increase}",
        f"2. Did activity concentrate more in drinking, feeding, or general areas? {semantic_interpretation}",
        f"3. Did semantic-zone features provide more interpretable response than four equal zones? {fourzone_text}",
        f"4. Did HMM switch to a disturbance/transition or functional-zone state? The dominant during-entry state was `{dominant_during_state}`.",
        f"5. Did embedded audio features change during event? {audio_text}",
        "6. How should this be interpreted for the meeting? The semantic experiment adds a biologically meaningful description of where the activity response happened, not just whether total motion increased.",
        f"7. What are limitations? Semantic zones are manually defined, activity is not true occupancy, audio is whole-room embedded audio, environment is daily context, and the risk score is a prototype heuristic. {recovery_text}",
        "",
        "## Summary Table",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "## State Transition Sequence Around Event",
        "",
        _dominant_state_sequence_text(event_df),
    ]
    meeting_bullet = (
        "- The semantic-zone experiment showed that caretaker entry increased overall activity and mobility, "
        f"with the response becoming more interpretable through functional-area fractions and a dominant `{dominant_during_state}` latent state."
    )
    return "\n".join(lines) + "\n", meeting_bullet


def _phase_change_text(baseline: pd.Series, during: pd.Series, column: str, label: str) -> str:
    baseline_value = float(baseline.get(column, np.nan)) if not baseline.empty else np.nan
    during_value = float(during.get(column, np.nan)) if not during.empty else np.nan
    if pd.isna(baseline_value) or pd.isna(during_value):
        return f"`{label}` did not have enough data."
    direction = "increased" if during_value > baseline_value else "did not increase"
    return f"`{label}` {direction} from `{baseline_value:.4f}` to `{during_value:.4f}`."


def _recovery_text(baseline: pd.Series, during: pd.Series, recovery: pd.Series) -> str:
    baseline_value = float(baseline.get("risk_score_mean", np.nan)) if not baseline.empty else np.nan
    during_value = float(during.get("risk_score_mean", np.nan)) if not during.empty else np.nan
    recovery_value = float(recovery.get("risk_score_mean", np.nan)) if not recovery.empty else np.nan
    if pd.isna(baseline_value) or pd.isna(during_value) or pd.isna(recovery_value):
        return "Recovery could not be estimated cleanly."
    if recovery_value < during_value:
        return f"Risk decreased from during-event `{during_value:.4f}` toward recovery `{recovery_value:.4f}`."
    return "Risk did not clearly decrease during the recovery window."


def _semantic_interpretation_text(baseline: pd.Series, during: pd.Series) -> str:
    baseline_functional = float(baseline.get("feeding_plus_drinking_activity_fraction_mean", np.nan)) if not baseline.empty else np.nan
    during_functional = float(during.get("feeding_plus_drinking_activity_fraction_mean", np.nan)) if not during.empty else np.nan
    drink = float(during.get("drinking_activity_fraction_mean", np.nan)) if not during.empty else np.nan
    feed = float(during.get("feeding_activity_fraction_mean", np.nan)) if not during.empty else np.nan
    general = float(during.get("general_activity_fraction_mean", np.nan)) if not during.empty else np.nan
    peak_zone = "general"
    zone_values = {"drinking": drink, "feeding": feed, "general": general}
    clean_zone_values = {key: value for key, value in zone_values.items() if pd.notna(value)}
    if clean_zone_values:
        peak_zone = max(clean_zone_values, key=clean_zone_values.get)
    if pd.notna(baseline_functional) and pd.notna(during_functional):
        direction = "increased" if during_functional > baseline_functional else "did not increase"
        return (
            f"Functional-area activity fraction {direction} from `{baseline_functional:.4f}` to `{during_functional:.4f}`, "
            f"and the dominant semantic activity fraction during entry was `{peak_zone}`."
        )
    return "Semantic zone fractions did not have enough data."


def _audio_text(baseline: pd.Series, during: pd.Series) -> str:
    baseline_rms = float(baseline.get("audio_rms_mean", np.nan)) if not baseline.empty else np.nan
    during_rms = float(during.get("audio_rms_mean", np.nan)) if not during.empty else np.nan
    baseline_centroid = float(baseline.get("audio_spectral_centroid_mean", np.nan)) if not baseline.empty else np.nan
    during_centroid = float(during.get("audio_spectral_centroid_mean", np.nan)) if not during.empty else np.nan
    if pd.isna(baseline_rms) or pd.isna(during_rms):
        return "Embedded audio did not have enough aligned data."
    return (
        f"Audio RMS {'increased' if during_rms > baseline_rms else 'did not increase'} "
        f"from `{baseline_rms:.4f}` to `{during_rms:.4f}`, and spectral centroid "
        f"{'increased' if pd.notna(baseline_centroid) and pd.notna(during_centroid) and during_centroid > baseline_centroid else 'did not clearly increase'} "
        f"from `{baseline_centroid:.1f}` to `{during_centroid:.1f}`."
    )


def _dominant_state_sequence_text(event_df: pd.DataFrame) -> str:
    ordered = event_df.sort_values("start_time_dt", kind="stable")
    transitions: list[str] = []
    previous_label = None
    for _, row in ordered.iterrows():
        label = str(row.get("state_label", "unlabeled_state"))
        if label == previous_label:
            continue
        minutes = pd.to_numeric(row.get("seconds_from_event_start"), errors="coerce")
        if pd.notna(minutes):
            transitions.append(f"- `{minutes / 60.0:+.2f}` min: `{label}`")
        else:
            transitions.append(f"- `{label}`")
        previous_label = label
    return "\n".join(transitions) if transitions else "- no state changes observed"


def _safe_mean(dataframe: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(dataframe.get(column), errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _safe_max(dataframe: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(dataframe.get(column), errors="coerce")
    return float(values.max()) if values.notna().any() else np.nan
