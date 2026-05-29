from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_audio_env_correlation_heatmap(correlation_df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    if correlation_df.empty:
        _write_empty_plot(axis, "No audio-environment correlations available.")
    else:
        pivot = correlation_df.pivot(index="audio_feature", columns="env_feature", values="spearman_rho")
        image = axis.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index.tolist())
        axis.set_title("Audio-Environment Spearman Correlation Heatmap")
        figure.colorbar(image, ax=axis, shrink=0.8, label="Spearman rho")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_audio_features_over_time(multimodal_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plot_df = _sorted_time_df(multimodal_df)
    metric_specs = [
        ("audio_rms", "Audio RMS"),
        ("audio_short_time_energy", "Audio Short-Time Energy"),
        ("audio_spectral_centroid", "Spectral Centroid"),
    ]
    for axis, (column, title) in zip(axes, metric_specs):
        if plot_df.empty or column not in plot_df.columns:
            _write_empty_plot(axis, f"No {title.lower()} values available.")
            continue
        axis.plot(plot_df["start_time_dt"], pd.to_numeric(plot_df[column], errors="coerce"), linewidth=0.9, alpha=0.85)
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].set_title("Embedded Audio Features Over Time")
    axes[-1].set_xlabel("Window Start Time")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_env_context_over_time(env_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if env_df.empty:
        for axis in axes:
            _write_empty_plot(axis, "No environment context rows available.")
    else:
        plot_df = env_df.copy()
        plot_df["date_dt"] = pd.to_datetime(plot_df["date"], errors="coerce")
        axes[0].plot(plot_df["date_dt"], plot_df["temp_daily_mean"], marker="o", label="Temp Daily Mean")
        axes[0].plot(plot_df["date_dt"], plot_df["temp_daily_range"], marker="s", label="Temp Daily Range")
        axes[0].set_ylabel("Temperature")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.25)

        axes[1].plot(plot_df["date_dt"], plot_df["rh_daily_mean"], marker="o", color="#2a6f97", label="RH Daily Mean")
        axes[1].set_ylabel("Relative Humidity")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.25)
    axes[0].set_title("Room 1 Daily Environment Context")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_centered_multimodal_timeline(event_df: pd.DataFrame, event_row: pd.Series | None, output_path: Path) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    if event_df.empty:
        for axis in axes:
            _write_empty_plot(axis, "No event-centred multimodal windows available.")
    else:
        ordered_df = event_df.sort_values("minutes_from_event_start", kind="stable")
        specs = [
            ("activity_mean", "Activity Mean", "#4c956c"),
            ("mobility_index", "Mobility Index", "#2a6f97"),
            ("audio_rms", "Audio RMS", "#c94f3d"),
            ("audio_spectral_centroid", "Spectral Centroid", "#8a5a44"),
            ("welfare_risk_score", "Risk Score", "#8e5572"),
        ]
        exit_minutes = _event_exit_minutes(event_row)
        for axis, (column, label, color) in zip(axes, specs):
            axis.plot(
                ordered_df["minutes_from_event_start"],
                pd.to_numeric(ordered_df[column], errors="coerce"),
                marker="o",
                linewidth=1.0,
                markersize=3.0,
                color=color,
            )
            _shade_event_phases(axis, ordered_df)
            axis.axvline(0.0, linestyle="--", color="#c94f3d", linewidth=1.1)
            axis.axvline(exit_minutes, linestyle="--", color="#28536b", linewidth=1.1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[0].set_title("Event-Centered Multimodal Timeline")
        if {"temp_context", "rh_context"} <= set(ordered_df.columns):
            temp_value = pd.to_numeric(ordered_df["temp_context"], errors="coerce").dropna()
            rh_value = pd.to_numeric(ordered_df["rh_context"], errors="coerce").dropna()
            context_text = (
                f"Temp context: {temp_value.iloc[0]:.2f}" if not temp_value.empty else "Temp context: n/a"
            ) + " | " + (
                f"RH context: {rh_value.iloc[0]:.2f}" if not rh_value.empty else "RH context: n/a"
            )
            axes[0].text(0.99, 1.02, context_text, transform=axes[0].transAxes, ha="right", fontsize=9)
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_phase_audio_video_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    if summary_df.empty:
        for axis in axes:
            _write_empty_plot(axis, "No event phase summary available.")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    phase_order = ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
    video_metrics = ["activity_mean", "mobility_index", "welfare_risk_score"]
    audio_metrics = ["audio_rms", "audio_short_time_energy", "audio_spectral_centroid"]
    for axis, metrics, title in zip(
        axes,
        (video_metrics, audio_metrics),
        ("Video-Derived Metrics By Event Phase", "Embedded Audio Metrics By Event Phase"),
    ):
        plot_df = summary_df[summary_df["metric"].isin(metrics)].copy()
        if plot_df.empty:
            _write_empty_plot(axis, f"No metrics available for {title.lower()}.")
            continue
        pivot = plot_df.pivot(index="phase", columns="metric", values="mean").reindex(phase_order)
        x_positions = np.arange(len(pivot.index))
        width = 0.18
        for metric_index, metric in enumerate(pivot.columns):
            axis.bar(
                x_positions + ((metric_index - (len(pivot.columns) - 1) / 2) * width),
                pd.to_numeric(pivot[metric], errors="coerce").fillna(0.0),
                width=width,
                label=metric,
            )
        axis.set_ylabel("Mean")
        axis.set_title(title)
        axis.legend(loc="upper right")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(np.arange(3))
    axes[-1].set_xticklabels(phase_order, rotation=20, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_audio_env_context_panel(event_df: pd.DataFrame, event_row: pd.Series | None, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    if event_df.empty:
        for axis in axes:
            _write_empty_plot(axis, "No event-labelled multimodal windows available.")
    else:
        ordered_df = event_df.sort_values("minutes_from_event_start", kind="stable")
        exit_minutes = _event_exit_minutes(event_row)
        panels = [
            ("audio_zero_crossing_rate", "Zero Crossing Rate", "#2a6f97"),
            ("audio_spectral_bandwidth", "Spectral Bandwidth", "#b56576"),
            ("audio_spectral_rolloff", "Spectral Rolloff", "#6d597a"),
        ]
        for axis, (column, label, color) in zip(axes, panels):
            axis.plot(ordered_df["minutes_from_event_start"], pd.to_numeric(ordered_df[column], errors="coerce"), color=color, marker="o", linewidth=1.0, markersize=3.0)
            _shade_event_phases(axis, ordered_df)
            axis.axvline(0.0, linestyle="--", color="#c94f3d", linewidth=1.1)
            axis.axvline(exit_minutes, linestyle="--", color="#28536b", linewidth=1.1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        if {"temp_context", "rh_context"} <= set(ordered_df.columns):
            temp_text = ordered_df["temp_context"].dropna().astype(float)
            rh_text = ordered_df["rh_context"].dropna().astype(float)
            axes[0].text(
                0.02,
                1.02,
                f"Daily temp context: {temp_text.iloc[0]:.2f}" if not temp_text.empty else "Daily temp context: n/a",
                transform=axes[0].transAxes,
                fontsize=9,
            )
            axes[1].text(
                0.02,
                1.02,
                f"Daily RH context: {rh_text.iloc[0]:.2f}" if not rh_text.empty else "Daily RH context: n/a",
                transform=axes[1].transAxes,
                fontsize=9,
            )
    axes[0].set_title("Event Audio + Environment Context Panel")
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_hmm_ablation_event_alignment(sequence_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if sequence_df.empty:
        for axis in axes:
            _write_empty_plot(axis, "No HMM ablation state sequence available.")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    model_order = sequence_df["model_name"].dropna().unique().tolist()
    for axis, model_name in zip(axes, model_order[:2]):
        model_df = sequence_df[sequence_df["model_name"] == model_name].sort_values("minutes_from_event_start", kind="stable")
        axis.step(model_df["minutes_from_event_start"], model_df["state_id"], where="post", linewidth=1.2)
        axis.scatter(model_df["minutes_from_event_start"], model_df["state_id"], s=10)
        axis.axvline(0.0, linestyle="--", color="#c94f3d", linewidth=1.1)
        exit_minutes = _event_exit_minutes(model_df.iloc[0] if not model_df.empty else None)
        axis.axvline(exit_minutes, linestyle="--", color="#28536b", linewidth=1.1)
        axis.set_ylabel(model_name)
        axis.grid(alpha=0.25)
    axes[0].set_title("HMM Ablation Event Alignment")
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_video_only_vs_multimodal_risk(sequence_df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    if sequence_df.empty:
        _write_empty_plot(axis, "No HMM ablation risk sequence available.")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    for model_name, model_df in sequence_df.groupby("model_name", sort=False):
        ordered_df = model_df.sort_values("minutes_from_event_start", kind="stable")
        axis.plot(
            ordered_df["minutes_from_event_start"],
            pd.to_numeric(ordered_df["risk_score"], errors="coerce"),
            marker="o",
            linewidth=1.0,
            markersize=3.0,
            label=model_name,
        )
    axis.axvline(0.0, linestyle="--", color="#c94f3d", linewidth=1.1)
    axis.set_title("Video-Only Vs Multimodal Risk Around Event")
    axis.set_xlabel("Minutes Relative To Caretaker Entry Start")
    axis.set_ylabel("Heuristic Risk Score")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _write_empty_plot(axis, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=11)
    axis.axis("off")


def _sorted_time_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    working_df = dataframe.copy()
    working_df["start_time_dt"] = pd.to_datetime(working_df["start_time"], errors="coerce", utc=False)
    return working_df.sort_values("start_time_dt", kind="stable").reset_index(drop=True)


def _shade_event_phases(axis, event_df: pd.DataFrame) -> None:
    baseline_df = event_df[event_df["event_phase"] == "pre_entry_baseline"]
    during_df = event_df[event_df["event_phase"] == "during_entry"]
    recovery_df = event_df[event_df["event_phase"] == "post_entry_recovery"]
    if not baseline_df.empty:
        axis.axvspan(baseline_df["minutes_from_event_start"].min(), baseline_df["minutes_from_event_start"].max(), color="#76b7b2", alpha=0.08)
    if not during_df.empty:
        axis.axvspan(during_df["minutes_from_event_start"].min(), during_df["minutes_from_event_start"].max(), color="#c94f3d", alpha=0.08)
    if not recovery_df.empty:
        axis.axvspan(recovery_df["minutes_from_event_start"].min(), recovery_df["minutes_from_event_start"].max(), color="#f1c453", alpha=0.08)


def _event_exit_minutes(event_row: pd.Series | None) -> float:
    if event_row is None:
        return 0.0
    start_value = pd.to_datetime(event_row.get("event_start_time"), errors="coerce", utc=False)
    end_value = pd.to_datetime(event_row.get("event_end_time"), errors="coerce", utc=False)
    if pd.isna(start_value) or pd.isna(end_value):
        end_seconds = pd.to_numeric(event_row.get("seconds_from_event_end"), errors="coerce")
        start_seconds = pd.to_numeric(event_row.get("seconds_from_event_start"), errors="coerce")
        if pd.notna(end_seconds) and pd.notna(start_seconds):
            return float(start_seconds - end_seconds) / 60.0
        return 0.0
    return float((end_value - start_value).total_seconds()) / 60.0
