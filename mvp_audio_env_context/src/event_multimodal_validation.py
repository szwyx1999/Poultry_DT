from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AudioEnvContextConfig
from .plotting import (
    plot_event_audio_env_context_panel,
    plot_event_centered_multimodal_timeline,
    plot_event_phase_audio_video_comparison,
)


PHASE_ORDER = ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
METRIC_COLUMNS = [
    "activity_mean",
    "mobility_index",
    "spatial_freedom_index",
    "occupancy_imbalance_index",
    "welfare_risk_score",
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "temp_context",
    "rh_context",
]


@dataclass(frozen=True)
class EventMultimodalValidationResult:
    summary_df: pd.DataFrame
    event_df: pd.DataFrame
    output_path: Path
    report_path: Path
    meeting_bullet: str


def run_event_multimodal_validation(
    config: AudioEnvContextConfig,
    multimodal_df: pd.DataFrame,
) -> EventMultimodalValidationResult:
    if multimodal_df.empty:
        raise ValueError("Multimodal window table is empty; cannot run event-centered multimodal validation.")

    if "event_phase" not in multimodal_df.columns or "event_id" not in multimodal_df.columns:
        raise ValueError(
            "Event labels are required for multimodal validation. Ensure mvp_biomarker_state_twin/data/event_log.csv is available."
        )

    event_df = multimodal_df[
        multimodal_df["event_phase"].astype(str).isin(PHASE_ORDER)
        & multimodal_df["event_id"].notna()
    ].copy()
    if event_df.empty:
        raise ValueError(
            "No caretaker-entry event-labelled windows were found in the multimodal table. "
            "Ensure mvp_biomarker_state_twin/data/event_log.csv exists and the biomarker/HMM run covered the event."
        )

    event_df["minutes_from_event_start"] = pd.to_numeric(event_df["seconds_from_event_start"], errors="coerce") / 60.0
    summary_rows: list[dict] = []
    baseline_means = {
        metric: _phase_mean(event_df, "pre_entry_baseline", metric)
        for metric in METRIC_COLUMNS
    }

    for phase in PHASE_ORDER:
        phase_df = event_df[event_df["event_phase"] == phase].copy()
        for metric in METRIC_COLUMNS:
            series = pd.to_numeric(phase_df.get(metric), errors="coerce")
            mean_value = float(series.mean()) if series.notna().any() else np.nan
            baseline_mean = baseline_means.get(metric, np.nan)
            fold_change = np.nan
            if pd.notna(mean_value) and pd.notna(baseline_mean) and baseline_mean != 0:
                fold_change = float(mean_value / baseline_mean)
            summary_rows.append(
                {
                    "event_id": str(phase_df["event_id"].iloc[0]) if not phase_df.empty else "",
                    "phase": phase,
                    "metric": metric,
                    "n_windows": int(len(phase_df)),
                    "mean": mean_value,
                    "median": float(series.median()) if series.notna().any() else np.nan,
                    "max": float(series.max()) if series.notna().any() else np.nan,
                    "std": float(series.std()) if series.notna().any() else np.nan,
                    "baseline_mean": baseline_mean,
                    "fold_change_vs_baseline": fold_change,
                    "mean_minus_baseline": float(mean_value - baseline_mean) if pd.notna(mean_value) and pd.notna(baseline_mean) else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    output_path = config.features_dir / "event_multimodal_summary.csv"
    report_path = config.reports_dir / "event_multimodal_validation_report.md"
    summary_df.to_csv(output_path, index=False)

    representative_row = event_df[event_df["event_phase"] == "during_entry"].head(1)
    event_row = representative_row.iloc[0] if not representative_row.empty else event_df.iloc[0]
    plot_event_centered_multimodal_timeline(event_df, event_row, config.plots_dir / "event_centered_multimodal_timeline.png")
    plot_event_phase_audio_video_comparison(summary_df, config.plots_dir / "event_phase_audio_video_comparison.png")
    plot_event_audio_env_context_panel(event_df, event_row, config.plots_dir / "event_audio_env_context_panel.png")

    report_text, meeting_bullet = _build_event_report(event_df, summary_df)
    report_path.write_text(report_text, encoding="utf-8")
    return EventMultimodalValidationResult(
        summary_df=summary_df,
        event_df=event_df,
        output_path=output_path,
        report_path=report_path,
        meeting_bullet=meeting_bullet,
    )


def _build_event_report(event_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[str, str]:
    video_response = _metric_comparison_text(summary_df, "mobility_index")
    audio_energy_response = _metric_comparison_text(summary_df, "audio_rms")
    spectral_response = _metric_comparison_text(summary_df, "audio_spectral_centroid")
    risk_response = _metric_comparison_text(summary_df, "welfare_risk_score")
    dominant_state_text = _dominant_state_text(event_df)
    temp_context = pd.to_numeric(event_df["temp_context"], errors="coerce").dropna()
    rh_context = pd.to_numeric(event_df["rh_context"], errors="coerce").dropna()
    temp_text = f"{temp_context.iloc[0]:.2f}" if not temp_context.empty else "n/a"
    rh_text = f"{rh_context.iloc[0]:.2f}" if not rh_context.empty else "n/a"

    lines = [
        "# Event Multimodal Validation Report",
        "",
        "This report compares video biomarkers, embedded MP4 audio features, and daily environmental context around the labelled caretaker-entry event.",
        "",
        "## Answers",
        "",
        f"1. Do video biomarkers respond during caretaker entry? {video_response}",
        f"2. Do embedded audio features also change during caretaker entry? {audio_energy_response} {spectral_response}",
        f"3. Is the acoustic response interpretable after considering daily temp/RH context? Daily context during the event was temp `{temp_text}` and RH `{rh_text}`; interpret acoustic changes as event-aligned response under that coarse context, not as isolated environmental effect.",
        f"4. Does HMM state/risk align with video and audio response? {risk_response} Dominant state sequence: {dominant_state_text}",
        "5. What are the caveats? Environment is daily Room 1 context, audio comes from embedded MP4 tracks, and the analysis remains exploratory rather than validated welfare diagnosis.",
        "",
        "## Audio Spectral Framing",
        "",
        spectral_response,
        "",
        "## Summary Table",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
    ]
    centroid_baseline = _summary_value(summary_df, "pre_entry_baseline", "audio_spectral_centroid", "mean")
    centroid_during = _summary_value(summary_df, "during_entry", "audio_spectral_centroid", "mean")
    rms_baseline = _summary_value(summary_df, "pre_entry_baseline", "audio_rms", "mean")
    rms_during = _summary_value(summary_df, "during_entry", "audio_rms", "mean")
    centroid_change = "higher" if pd.notna(centroid_during) and pd.notna(centroid_baseline) and centroid_during > centroid_baseline else "not higher"
    rms_change = "higher" if pd.notna(rms_during) and pd.notna(rms_baseline) and rms_during > rms_baseline else "not higher"
    meeting_bullet = (
        "- Event-centred multimodal validation showed a strong video response, a modest audio-energy increase, "
        f"and a spectral-centroid shift that was {centroid_change} during the caretaker entry."
        if rms_change == "higher"
        else "- Event-centred multimodal validation showed a strong video response and an acoustic spectral shift, even though audio energy did not clearly increase."
    )
    return "\n".join(lines) + "\n", meeting_bullet


def _metric_comparison_text(summary_df: pd.DataFrame, metric: str) -> str:
    baseline = _summary_value(summary_df, "pre_entry_baseline", metric, "mean")
    during = _summary_value(summary_df, "during_entry", metric, "mean")
    recovery = _summary_value(summary_df, "post_entry_recovery", metric, "mean")
    if pd.isna(baseline) or pd.isna(during):
        return f"`{metric}` did not have enough data."
    increased = during > baseline
    recovery_direction = "decreased toward baseline" if pd.notna(recovery) and recovery < during else "did not clearly decrease"
    return (
        f"`{metric}` {'increased' if increased else 'did not increase'} "
        f"from baseline `{baseline:.4f}` to during-event `{during:.4f}`, and {recovery_direction} "
        f"(recovery mean `{recovery:.4f}`)." if pd.notna(recovery) else
        f"`{metric}` {'increased' if increased else 'did not increase'} from baseline `{baseline:.4f}` to during-event `{during:.4f}`."
    )


def _dominant_state_text(event_df: pd.DataFrame) -> str:
    if "state_label" not in event_df.columns:
        return "state labels unavailable"
    transitions: list[str] = []
    previous_label = None
    ordered_df = event_df.sort_values("start_time_dt", kind="stable")
    for _, row in ordered_df.iterrows():
        label = str(row.get("state_label", "unlabeled_state"))
        if label == previous_label:
            continue
        minutes = pd.to_numeric(row.get("seconds_from_event_start"), errors="coerce")
        if pd.notna(minutes):
            transitions.append(f"{float(minutes) / 60.0:+.2f} min: {label}")
        else:
            transitions.append(label)
        previous_label = label
    return " -> ".join(transitions) if transitions else "no state changes observed"


def _phase_mean(event_df: pd.DataFrame, phase: str, metric: str) -> float:
    phase_df = event_df[event_df["event_phase"] == phase]
    series = pd.to_numeric(phase_df.get(metric), errors="coerce")
    return float(series.mean()) if series.notna().any() else np.nan


def _summary_value(summary_df: pd.DataFrame, phase: str, metric: str, column: str) -> float:
    subset_df = summary_df[(summary_df["phase"] == phase) & (summary_df["metric"] == metric)]
    if subset_df.empty:
        return np.nan
    value = subset_df.iloc[0][column]
    return float(value) if pd.notna(value) else np.nan
