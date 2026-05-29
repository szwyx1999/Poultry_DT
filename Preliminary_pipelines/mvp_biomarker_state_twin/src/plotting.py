from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BiomarkerTwinConfig


RISK_CAPTION = "Prototype video-derived latent state and heuristic risk score; not validated welfare diagnosis."


def plot_biomarker_timeseries(window_df: pd.DataFrame, output_path: str) -> None:
    metrics = [
        ("mobility_index", "Mobility"),
        ("spatial_freedom_index", "Spatial Freedom"),
        ("occupancy_imbalance_index", "Occupancy Imbalance"),
        ("activity_mean", "Activity Mean"),
    ]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 10), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    timeline_df = _prepare_timeline_df(window_df)
    for axis, (column, title) in zip(axes, metrics):
        for sequence_key, group_df in timeline_df.groupby(_sequence_group_columns(timeline_df), sort=False, dropna=False):
            axis.plot(group_df["start_time_dt"], group_df[column], marker="o", label=_format_sequence_title(sequence_key, _sequence_group_columns(timeline_df)))
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Window Start Time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_biomarker_timeseries_by_room(window_df: pd.DataFrame, output_path: str) -> None:
    timeline_df = _prepare_timeline_df(window_df)
    grouped_sequences = list(timeline_df.groupby(_sequence_group_columns(timeline_df), sort=False, dropna=False))
    figure, axes = plt.subplots(len(grouped_sequences), 1, figsize=(13, max(4, 3.5 * len(grouped_sequences))), sharex=False)
    if len(grouped_sequences) == 1:
        axes = [axes]

    for axis, (sequence_key, group_df) in zip(axes, grouped_sequences):
        ordered_df = group_df.sort_values("start_time_dt", kind="stable")
        axis.plot(ordered_df["start_time_dt"], ordered_df["mobility_index"], label="Mobility", color="#28536b")
        axis.plot(ordered_df["start_time_dt"], ordered_df["spatial_freedom_index"], label="Spatial Freedom", color="#5f8f3e")
        axis.plot(ordered_df["start_time_dt"], ordered_df["occupancy_imbalance_index"], label="Occupancy Imbalance", color="#c97c1f")
        axis.plot(ordered_df["start_time_dt"], ordered_df["normalized_activity"], label="Normalized Activity", color="#c94f3d")
        axis.set_ylim(-0.05, 1.05)
        axis.set_title(_format_sequence_title(sequence_key, _sequence_group_columns(timeline_df)))
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left", ncol=2)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Window Start Time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_resilience_curves(window_df: pd.DataFrame, event_df: pd.DataFrame, output_path: str, config: BiomarkerTwinConfig) -> None:
    if event_df.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, "No events found.\nFallback event detection did not yield usable peaks.", ha="center", va="center", fontsize=11)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    selected_events = event_df.head(config.max_resilience_events)
    figure, axes = plt.subplots(len(selected_events), 1, figsize=(10, max(4, 3 * len(selected_events))), sharex=False)
    if len(selected_events) == 1:
        axes = [axes]

    for axis, (_, event_row) in zip(axes, selected_events.iterrows()):
        room_group = window_df[window_df["room_id"] == event_row["room_id"]].sort_values("start_time_dt", kind="stable")
        event_windows = room_group[room_group["event_id"] == event_row["event_id"]]
        if event_windows.empty:
            event_windows = room_group
        activity_series = pd.to_numeric(event_windows["activity_mean"], errors="coerce").fillna(pd.to_numeric(event_windows["activity_total"], errors="coerce")).fillna(0.0)
        phases = event_windows["event_phase"].fillna("normal").tolist()
        x_positions = np.arange(len(activity_series))
        axis.plot(x_positions, activity_series, marker="o", color="#28536b", label="Activity")
        axis.axhline(float(event_row["baseline_activity"]), color="#c29436", linestyle="--", label="Baseline")
        for position, phase in zip(x_positions, phases):
            if phase == "during":
                axis.axvspan(position - 0.25, position + 0.25, color="#c94f3d", alpha=0.25)
            elif phase == "recovery":
                axis.axvspan(position - 0.25, position + 0.25, color="#f1c453", alpha=0.20)
            elif phase == "baseline":
                axis.axvspan(position - 0.25, position + 0.25, color="#76b7b2", alpha=0.18)
        axis.set_title(f"{event_row['event_id']} ({event_row['candidate_event_source']})")
        axis.set_ylabel("Activity")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Relative Window Index")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_hmm_state_outputs(
    sequence_df: pd.DataFrame,
    plots_dir: Path,
    hmm_by_sequence_dir: Path,
) -> None:
    timeline_df = _prepare_timeline_df(sequence_df)
    grouped_sequences = list(timeline_df.groupby(_sequence_group_columns(timeline_df), sort=False, dropna=False))
    if not grouped_sequences:
        return

    hmm_by_sequence_dir.mkdir(parents=True, exist_ok=True)
    primary_sequence_key, primary_sequence_df = max(grouped_sequences, key=lambda item: len(item[1]))
    _plot_hmm_detailed_sequence(
        primary_sequence_df,
        sequence_key=primary_sequence_key,
        group_columns=_sequence_group_columns(timeline_df),
        output_path=plots_dir / "hmm_state_timeline.png",
    )

    _plot_hmm_overview(
        grouped_sequences=grouped_sequences,
        group_columns=_sequence_group_columns(timeline_df),
        output_path=plots_dir / "hmm_state_timeline_overview.png",
    )
    _plot_hmm_overview(
        grouped_sequences=grouped_sequences,
        group_columns=_sequence_group_columns(timeline_df),
        output_path=plots_dir / "hmm_state_timeline_by_room.png",
    )

    for sequence_key, group_df in grouped_sequences:
        slug = _sequence_slug(sequence_key, _sequence_group_columns(timeline_df))
        _plot_hmm_detailed_sequence(
            group_df,
            sequence_key=sequence_key,
            group_columns=_sequence_group_columns(timeline_df),
            output_path=hmm_by_sequence_dir / f"{slug}.png",
        )


def plot_state_risk_heatmap(sequence_df: pd.DataFrame, output_path: str) -> None:
    timeline_df = _prepare_timeline_df(sequence_df)
    primary_df = _primary_sequence_df(timeline_df)
    state_scale = _scale_series(primary_df["state_id"])
    heatmap_values = np.vstack(
        [
            state_scale.to_numpy(dtype=float),
            pd.to_numeric(primary_df["welfare_risk_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(primary_df["mobility_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
            pd.to_numeric(primary_df["spatial_freedom_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
            pd.to_numeric(primary_df["occupancy_imbalance_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
            pd.to_numeric(primary_df["normalized_activity"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
        ]
    )
    row_labels = [
        "State (scaled)",
        "Risk Score",
        "Mobility",
        "Spatial Freedom",
        "Occupancy Imbalance",
        "Normalized Activity",
    ]

    figure, axis = plt.subplots(figsize=(13, 4.5))
    image = axis.imshow(heatmap_values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    tick_positions = np.arange(len(primary_df))
    axis.set_xticks(tick_positions[:: max(1, len(tick_positions) // 10 or 1)])
    axis.set_xticklabels(primary_df["start_time_dt"].dt.strftime("%H:%M").tolist()[:: max(1, len(tick_positions) // 10 or 1)], rotation=45, ha="right")
    axis.set_title("State and Risk Heatmap")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_state_summary_profile(state_summary_df: pd.DataFrame, output_path: str) -> None:
    if state_summary_df.empty:
        return
    plot_df = state_summary_df.copy()
    plot_df["display_label"] = plot_df["state_label"]
    metric_columns = [
        ("component_activity_mobility", "Activity+Mobility"),
        ("component_imbalance", "Imbalance"),
        ("component_low_spatial_freedom", "Low Spatial Freedom"),
        ("component_resilience_pressure", "Resilience Pressure"),
        ("component_persistence", "Persistence"),
        ("state_risk_weight", "Risk Weight"),
    ]

    figure, axis = plt.subplots(figsize=(13, 6))
    x_positions = np.arange(len(plot_df))
    width = 0.12
    for metric_index, (column, label) in enumerate(metric_columns):
        axis.bar(x_positions + ((metric_index - (len(metric_columns) - 1) / 2) * width), plot_df[column], width=width, label=label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_df["display_label"], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Scaled Value")
    axis.set_title("State Summary Biomarker/Risk Profile")
    axis.legend(loc="upper left", ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_risk_score_distribution(sequence_df: pd.DataFrame, output_path: str) -> None:
    risk_scores = pd.to_numeric(sequence_df["welfare_risk_score"], errors="coerce").fillna(0.0)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(risk_scores, bins=min(12, max(4, len(risk_scores))), color="#28536b", alpha=0.8, edgecolor="white")
    axis.axvline(0.33, color="#c97c1f", linestyle="--", label="Low/Medium")
    axis.axvline(0.66, color="#c94f3d", linestyle="--", label="Medium/High")
    axis.set_xlabel("Risk Score")
    axis.set_ylabel("Window Count")
    axis.set_title("Risk Score Distribution")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_hmm_detailed_sequence(
    sequence_df: pd.DataFrame,
    sequence_key: object,
    group_columns: list[str],
    output_path: Path,
) -> None:
    ordered_df = sequence_df.sort_values("start_time_dt", kind="stable")
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    state_axis, risk_axis = axes

    state_axis.step(ordered_df["start_time_dt"], ordered_df["state_id"], where="post", color="#28536b", linewidth=2.0)
    state_axis.scatter(ordered_df["start_time_dt"], ordered_df["state_id"], color="#28536b", s=24, zorder=3)
    high_risk_df = ordered_df[ordered_df["risk_level"] == "high"]
    if not high_risk_df.empty:
        state_axis.scatter(high_risk_df["start_time_dt"], high_risk_df["state_id"], color="#c94f3d", s=42, marker="o", label="High risk")
        risk_axis.scatter(high_risk_df["start_time_dt"], high_risk_df["welfare_risk_score"], color="#c94f3d", s=42, marker="o", label="High risk")

    sustained_df = ordered_df[ordered_df["sustained_risk_flag"]]
    if not sustained_df.empty:
        risk_axis.scatter(sustained_df["start_time_dt"], sustained_df["welfare_risk_score"], color="#7a1f5c", s=50, marker="^", label="Sustained risk")

    state_axis.set_ylabel("State")
    state_axis.grid(alpha=0.25)
    _set_state_axis_ticks(state_axis, ordered_df)
    state_axis.set_title(f"{_format_sequence_title(sequence_key, group_columns)} | n_windows={len(ordered_df)}")
    if not high_risk_df.empty:
        state_axis.legend(loc="upper left")

    risk_axis.plot(ordered_df["start_time_dt"], ordered_df["welfare_risk_score"], marker="o", color="#c94f3d")
    risk_axis.fill_between(ordered_df["start_time_dt"], ordered_df["welfare_risk_score"], alpha=0.18, color="#c94f3d")
    risk_axis.axhline(0.33, color="#c97c1f", linestyle="--", alpha=0.8)
    risk_axis.axhline(0.66, color="#c94f3d", linestyle="--", alpha=0.8)
    risk_axis.set_ylabel("Risk Score")
    risk_axis.set_ylim(0.0, 1.05)
    risk_axis.grid(alpha=0.25)
    risk_axis.set_xlabel("Window Start Time")
    risk_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if not high_risk_df.empty or not sustained_df.empty:
        risk_axis.legend(loc="upper left")

    figure.text(0.5, 0.01, RISK_CAPTION, ha="center", va="bottom", fontsize=10)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_hmm_overview(
    grouped_sequences: list[tuple[object, pd.DataFrame]],
    group_columns: list[str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(len(grouped_sequences), 1, figsize=(13, max(4, 2.7 * len(grouped_sequences))), sharex=False)
    if len(grouped_sequences) == 1:
        axes = [axes]

    for axis, (sequence_key, sequence_df) in zip(axes, grouped_sequences):
        ordered_df = sequence_df.sort_values("start_time_dt", kind="stable")
        axis.step(ordered_df["start_time_dt"], ordered_df["state_id"], where="post", color="#28536b", linewidth=1.8)
        high_risk_df = ordered_df[ordered_df["risk_level"] == "high"]
        if not high_risk_df.empty:
            axis.scatter(high_risk_df["start_time_dt"], high_risk_df["state_id"], color="#c94f3d", s=28, marker="o")
        axis.set_title(f"{_format_sequence_title(sequence_key, group_columns)} | n_windows={len(ordered_df)}")
        axis.set_ylabel("State")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        _set_state_axis_ticks(axis, ordered_df)
    axes[-1].set_xlabel("Window Start Time")
    figure.text(0.5, 0.01, RISK_CAPTION, ha="center", va="bottom", fontsize=10)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _prepare_timeline_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    timeline_df = dataframe.copy()
    if "start_time_dt" not in timeline_df.columns:
        timeline_df["start_time_dt"] = pd.to_datetime(timeline_df.get("start_time"), errors="coerce", utc=False)
    sort_columns = [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in timeline_df.columns]
    return timeline_df.sort_values(sort_columns, kind="stable", na_position="last")


def _primary_sequence_df(timeline_df: pd.DataFrame) -> pd.DataFrame:
    grouped_sequences = list(timeline_df.groupby(_sequence_group_columns(timeline_df), sort=False, dropna=False))
    if not grouped_sequences:
        return timeline_df
    _, primary_df = max(grouped_sequences, key=lambda item: len(item[1]))
    return primary_df.sort_values("start_time_dt", kind="stable")


def _sequence_group_columns(timeline_df: pd.DataFrame) -> list[str]:
    if "session_id" in timeline_df.columns and timeline_df["session_id"].notna().any():
        return ["room_id", "session_id"]
    return ["room_id"]


def _format_sequence_title(group_key: object, group_columns: list[str]) -> str:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    return " | ".join(f"{column_name}={value}" for column_name, value in zip(group_columns, group_key))


def _sequence_slug(group_key: object, group_columns: list[str]) -> str:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    parts = [f"{column_name}_{str(value).replace(' ', '_').replace(':', '_')}" for column_name, value in zip(group_columns, group_key)]
    return "__".join(parts)


def _set_state_axis_ticks(axis, sequence_df: pd.DataFrame) -> None:
    label_map = (
        sequence_df[["state_id", "state_label"]]
        .drop_duplicates()
        .sort_values("state_id", kind="stable")
    )
    axis.set_yticks(label_map["state_id"].tolist())
    axis.set_yticklabels(label_map["state_label"].tolist())


def _scale_series(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(numeric_series.min())
    maximum = float(numeric_series.max())
    if minimum == maximum:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    return (numeric_series - minimum) / (maximum - minimum)
