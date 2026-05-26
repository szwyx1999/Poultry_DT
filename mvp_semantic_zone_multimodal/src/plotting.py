from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_ORDER = ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
DISCLAIMER = "Prototype semantic-zone multimodal analysis; not validated welfare diagnosis."


def plot_semantic_zone_activity_timeseries(feature_df: pd.DataFrame, output_path: Path) -> None:
    if feature_df.empty:
        return
    working_df = _prepare_time(feature_df)
    figure, axis = plt.subplots(figsize=(14, 5))
    for zone_id, group_df in working_df.groupby("zone_id", sort=False):
        ordered = group_df.sort_values("start_time_dt", kind="stable")
        axis.plot(ordered["start_time_dt"], ordered["activity_mean"], label=zone_id, linewidth=1.4)
    axis.set_title("Semantic Zone Activity Over Time")
    axis.set_ylabel("Activity Mean")
    axis.set_xlabel("Window Start Time")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_semantic_zone_activity_heatmap(feature_df: pd.DataFrame, output_path: Path) -> None:
    if feature_df.empty:
        return
    working_df = _prepare_time(feature_df)
    pivot_df = (
        working_df.pivot_table(
            index="zone_id",
            columns="start_time_dt",
            values="activity_mean",
            aggfunc="mean",
        )
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(14, 4.5))
    image = axis.imshow(pivot_df.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_yticks(np.arange(len(pivot_df.index)))
    axis.set_yticklabels(pivot_df.index.tolist())
    tick_step = max(1, len(pivot_df.columns) // 10)
    axis.set_xticks(np.arange(len(pivot_df.columns))[::tick_step])
    axis.set_xticklabels([column.strftime("%m-%d %H:%M") for column in pivot_df.columns[::tick_step]], rotation=45, ha="right")
    axis.set_title("Semantic Zone Activity Heatmap")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_semantic_biomarker_timeseries(biomarker_df: pd.DataFrame, output_path: Path) -> None:
    if biomarker_df.empty:
        return
    working_df = _prepare_time(biomarker_df)
    metrics = [
        ("mobility_index", "Mobility Index"),
        ("spatial_freedom_index", "Spatial Freedom Index"),
        ("occupancy_imbalance_index", "Occupancy Imbalance Index"),
        ("feeding_plus_drinking_activity_fraction", "Functional Area Activity Fraction"),
    ]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), sharex=True)
    for axis, (column, label) in zip(axes, metrics):
        axis.plot(working_df["start_time_dt"], working_df[column], linewidth=1.3, color="#28536b")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Semantic Biomarker Timeseries")
    axes[-1].set_xlabel("Window Start Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_centered_semantic_zone_activity(event_df: pd.DataFrame, output_path: Path) -> None:
    if event_df.empty:
        return
    ordered = event_df.sort_values("minutes_from_event_start", kind="stable")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(ordered["minutes_from_event_start"], ordered["drinking_activity_fraction"], label="Drinking Fraction", color="#d62728")
    axis.plot(ordered["minutes_from_event_start"], ordered["feeding_activity_fraction"], label="Feeding Fraction", color="#2ca02c")
    axis.plot(ordered["minutes_from_event_start"], ordered["general_activity_fraction"], label="General Fraction", color="#1f77b4")
    _decorate_event_axis(axis, ordered, title="Event-Centered Semantic Zone Activity")
    axis.set_ylabel("Activity Fraction")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_centered_semantic_biomarkers(event_df: pd.DataFrame, output_path: Path) -> None:
    if event_df.empty:
        return
    ordered = event_df.sort_values("minutes_from_event_start", kind="stable")
    metrics = [
        ("mobility_index", "Mobility Index", "#28536b"),
        ("spatial_freedom_index", "Spatial Freedom Index", "#5f8f3e"),
        ("occupancy_imbalance_index", "Occupancy Imbalance Index", "#c97c1f"),
        ("semantic_transition_proxy", "Transition Proxy", "#c94f3d"),
    ]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 10), sharex=True)
    for axis, (column, label, color) in zip(axes, metrics):
        axis.plot(ordered["minutes_from_event_start"], ordered[column], color=color, linewidth=1.6)
        _decorate_event_axis(axis, ordered, title=None)
        axis.set_ylabel(label)
    axes[0].set_title("Event-Centered Semantic Biomarkers")
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_event_centered_semantic_multimodal_timeline(event_df: pd.DataFrame, output_path: Path) -> None:
    if event_df.empty:
        return
    ordered = event_df.sort_values("minutes_from_event_start", kind="stable")
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(ordered["minutes_from_event_start"], ordered["activity_mean"], label="Activity Mean", color="#28536b")
    axes[0].plot(ordered["minutes_from_event_start"], ordered["mobility_index"], label="Mobility Index", color="#c94f3d")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Video")

    axes[1].plot(ordered["minutes_from_event_start"], ordered["audio_rms"], label="Audio RMS", color="#7a1f5c")
    axes[1].plot(ordered["minutes_from_event_start"], ordered["audio_spectral_centroid"] / 3000.0, label="Spectral Centroid / 3000", color="#1f77b4")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Audio")

    axes[2].plot(ordered["minutes_from_event_start"], ordered["risk_score"], label="Risk Score", color="#c97c1f")
    axes[2].axhline(0.33, color="#c97c1f", linestyle="--", alpha=0.7)
    axes[2].axhline(0.66, color="#c94f3d", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Risk")

    axes[3].plot(ordered["minutes_from_event_start"], ordered["temp_context"], label="Temp Context", color="#ff7f0e")
    axes[3].plot(ordered["minutes_from_event_start"], ordered["rh_context"], label="RH Context", color="#17becf")
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("Env")

    for axis in axes:
        _decorate_event_axis(axis, ordered, title=None)
        axis.grid(alpha=0.25)
    axes[0].set_title("Event-Centered Semantic Multimodal Timeline")
    axes[-1].set_xlabel("Minutes Relative To Caretaker Entry Start")
    figure.text(0.5, 0.01, DISCLAIMER, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_semantic_hmm_state_risk_timeline(
    sequence_df: pd.DataFrame,
    event_start: pd.Timestamp | None,
    event_end: pd.Timestamp | None,
    output_path: Path,
) -> None:
    if sequence_df.empty:
        return
    ordered = _prepare_time(sequence_df).sort_values("start_time_dt", kind="stable")
    figure, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)

    axes[0].step(ordered["start_time_dt"], ordered["state_id"], where="post", color="#28536b", linewidth=1.8)
    axes[0].scatter(ordered["start_time_dt"], ordered["state_id"], c=np.where(ordered["risk_level"].eq("high"), "#c94f3d", "#28536b"), s=18)
    _set_state_ticks(axes[0], ordered)
    axes[0].set_ylabel("State")
    axes[0].grid(alpha=0.25)
    axes[0].set_title(f"Semantic HMM State And Risk Timeline | n_windows={len(ordered)}")

    axes[1].plot(ordered["start_time_dt"], ordered["risk_score"], color="#c94f3d", linewidth=1.4)
    axes[1].fill_between(ordered["start_time_dt"], ordered["risk_score"], color="#c94f3d", alpha=0.18)
    axes[1].axhline(0.33, color="#c97c1f", linestyle="--", alpha=0.8)
    axes[1].axhline(0.66, color="#c94f3d", linestyle="--", alpha=0.8)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Risk Score")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("Window Start Time")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    if event_start is not None and pd.notna(event_start):
        for axis in axes:
            axis.axvline(event_start, color="#111111", linestyle="--", linewidth=1.2)
    if event_end is not None and pd.notna(event_end):
        for axis in axes:
            axis.axvline(event_end, color="#666666", linestyle=":", linewidth=1.2)

    figure.text(0.5, 0.01, DISCLAIMER, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_semantic_state_distribution_by_event_phase(event_df: pd.DataFrame, output_path: Path) -> None:
    if event_df.empty or "state_label" not in event_df.columns:
        return
    counts = (
        event_df[event_df["event_phase"].isin(PHASE_ORDER)]
        .groupby(["event_phase", "state_label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(PHASE_ORDER)
    )
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    figure, axis = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(proportions))
    for state_label in proportions.columns:
        values = proportions[state_label].to_numpy(dtype=float)
        axis.bar(proportions.index, values, bottom=bottom, label=state_label)
        bottom += values
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("State Fraction")
    axis.set_title("Semantic State Distribution By Event Phase")
    axis.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_semantic_vs_fourzone_event_comparison(
    semantic_summary_df: pd.DataFrame,
    fourzone_summary_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    if semantic_summary_df.empty:
        return
    phases = PHASE_ORDER
    semantic_lookup = semantic_summary_df.set_index("phase")
    fourzone_lookup = fourzone_summary_df.set_index("phase") if fourzone_summary_df is not None and not fourzone_summary_df.empty and "phase" in fourzone_summary_df.columns else None

    figure, axes = plt.subplots(2, 1, figsize=(11, 9))
    x_positions = np.arange(len(phases))
    width = 0.18

    semantic_mobility = [float(semantic_lookup.loc[phase, "mobility_index_mean"]) if phase in semantic_lookup.index else np.nan for phase in phases]
    semantic_risk = [float(semantic_lookup.loc[phase, "risk_score_mean"]) if phase in semantic_lookup.index else np.nan for phase in phases]
    fourzone_mobility = [float(fourzone_lookup.loc[phase, "mobility_index_mean"]) if fourzone_lookup is not None and phase in fourzone_lookup.index else np.nan for phase in phases]
    fourzone_risk = [float(fourzone_lookup.loc[phase, "risk_score_mean"]) if fourzone_lookup is not None and phase in fourzone_lookup.index else np.nan for phase in phases]

    axes[0].bar(x_positions - 1.5 * width, semantic_mobility, width=width, label="Semantic Mobility", color="#28536b")
    axes[0].bar(x_positions - 0.5 * width, semantic_risk, width=width, label="Semantic Risk", color="#c94f3d")
    axes[0].bar(x_positions + 0.5 * width, fourzone_mobility, width=width, label="Four-Zone Mobility", color="#5f8f3e")
    axes[0].bar(x_positions + 1.5 * width, fourzone_risk, width=width, label="Four-Zone Risk", color="#c97c1f")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(phases, rotation=15, ha="right")
    axes[0].set_ylabel("Mean Response")
    axes[0].set_title("Semantic Vs Previous Four-Zone Event Comparison")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    drink = [float(semantic_lookup.loc[phase, "drinking_activity_fraction_mean"]) if phase in semantic_lookup.index else np.nan for phase in phases]
    feed = [float(semantic_lookup.loc[phase, "feeding_activity_fraction_mean"]) if phase in semantic_lookup.index else np.nan for phase in phases]
    general = [float(semantic_lookup.loc[phase, "general_activity_fraction_mean"]) if phase in semantic_lookup.index else np.nan for phase in phases]
    axes[1].bar(x_positions - width, drink, width=width, label="Drinking Fraction", color="#d62728")
    axes[1].bar(x_positions, feed, width=width, label="Feeding Fraction", color="#2ca02c")
    axes[1].bar(x_positions + width, general, width=width, label="General Fraction", color="#1f77b4")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(phases, rotation=15, ha="right")
    axes[1].set_ylabel("Mean Activity Fraction")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _decorate_event_axis(axis, event_df: pd.DataFrame, title: str | None) -> None:
    exit_minutes = _event_exit_minutes(event_df)
    axis.axvspan(event_df["minutes_from_event_start"].min(), 0.0, color="#76b7b2", alpha=0.12)
    axis.axvspan(0.0, exit_minutes, color="#c94f3d", alpha=0.12)
    axis.axvspan(exit_minutes, event_df["minutes_from_event_start"].max(), color="#f1c453", alpha=0.10)
    axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.1)
    axis.axvline(exit_minutes, color="#666666", linestyle=":", linewidth=1.1)
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.25)


def _event_exit_minutes(event_df: pd.DataFrame) -> float:
    if "event_start_time" in event_df.columns and "event_end_time" in event_df.columns:
        start = pd.to_datetime(event_df["event_start_time"].iloc[0], errors="coerce", utc=False)
        end = pd.to_datetime(event_df["event_end_time"].iloc[0], errors="coerce", utc=False)
        if pd.notna(start) and pd.notna(end):
            return float((end - start).total_seconds() / 60.0)
    return 1.95


def _prepare_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_df = dataframe.copy()
    if "start_time_dt" not in working_df.columns:
        working_df["start_time_dt"] = pd.to_datetime(working_df["start_time"], errors="coerce", utc=False)
    return working_df


def _set_state_ticks(axis, sequence_df: pd.DataFrame) -> None:
    label_df = sequence_df[["state_id", "state_label"]].drop_duplicates().sort_values("state_id", kind="stable")
    axis.set_yticks(label_df["state_id"].tolist())
    axis.set_yticklabels(label_df["state_label"].tolist())

