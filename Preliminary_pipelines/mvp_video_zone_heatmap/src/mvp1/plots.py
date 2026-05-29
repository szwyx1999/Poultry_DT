from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .config import Mvp1Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_sanity_plots(
    features_df: pd.DataFrame,
    zone_config: dict,
    config: Mvp1Config,
) -> dict[str, Path]:
    over_time_path = config.plots_dir / "zone_activity_over_time.png"
    heatmap_path = config.plots_dir / "zone_activity_heatmap.png"

    ordered_zones = [zone["zone_id"] for zone in sorted(zone_config["zones"], key=lambda item: (item["row"], item["col"]))]
    valid_df = features_df.copy()
    valid_df["activity_mean_numeric"] = pd.to_numeric(valid_df.get("activity_mean"), errors="coerce")
    valid_df = valid_df.dropna(subset=["activity_mean_numeric"])

    _plot_activity_over_time(valid_df, ordered_zones, over_time_path)
    _plot_activity_heatmap(valid_df, ordered_zones, heatmap_path)
    return {
        "zone_activity_over_time": over_time_path,
        "zone_activity_heatmap": heatmap_path,
    }


def _plot_activity_over_time(valid_df: pd.DataFrame, ordered_zones: list[str], output_path: Path) -> None:
    if valid_df.empty:
        _save_placeholder_plot(output_path, "No valid zone activity data available.")
        return

    plot_df = valid_df.copy()
    plot_df["start_time_dt"] = pd.to_datetime(plot_df["start_time"], errors="coerce")
    plot_df = plot_df.sort_values(["start_time_dt", "window_id", "zone_id"], kind="stable")

    figure, axis = plt.subplots(figsize=(10, 5))
    for zone_id in ordered_zones:
        zone_rows = plot_df[plot_df["zone_id"] == zone_id]
        if zone_rows.empty:
            continue
        axis.plot(
            zone_rows["start_time_dt"],
            zone_rows["activity_mean_numeric"],
            marker="o",
            linewidth=1.5,
            label=zone_id,
        )
    axis.set_title("Zone Activity Over Time")
    axis.set_xlabel("Window Start Time")
    axis.set_ylabel("Activity Mean")
    axis.legend(loc="best")
    axis.grid(alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_activity_heatmap(valid_df: pd.DataFrame, ordered_zones: list[str], output_path: Path) -> None:
    if valid_df.empty:
        _save_placeholder_plot(output_path, "No valid zone activity data available.")
        return

    plot_df = valid_df.copy()
    plot_df["start_time_dt"] = pd.to_datetime(plot_df["start_time"], errors="coerce")
    window_order = (
        plot_df.sort_values(["start_time_dt", "window_id"], kind="stable")["window_id"]
        .drop_duplicates()
        .tolist()
    )
    pivot_table = plot_df.pivot_table(
        index="window_id",
        columns="zone_id",
        values="activity_mean_numeric",
        aggfunc="mean",
    ).reindex(index=window_order, columns=ordered_zones)

    if pivot_table.empty:
        _save_placeholder_plot(output_path, "No valid zone activity data available.")
        return

    heatmap_values = pivot_table.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(heatmap_values, aspect="auto", cmap="viridis")
    axis.set_title("Zone Activity Heatmap")
    axis.set_xlabel("Zone")
    axis.set_ylabel("Window")
    axis.set_xticks(np.arange(len(ordered_zones)))
    axis.set_xticklabels(ordered_zones)
    axis.set_yticks(np.arange(len(window_order)))
    axis.set_yticklabels(window_order)
    figure.colorbar(image, ax=axis, label="Activity Mean")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_placeholder_plot(output_path: Path, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.set_axis_off()
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
