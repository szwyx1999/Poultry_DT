from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SemanticZoneConfig
from .plotting import plot_semantic_biomarker_timeseries
from .utils import distribution_from_values, minmax_scale, prepare_time_columns, safe_float


@dataclass(frozen=True)
class SemanticBiomarkerResult:
    biomarker_df: pd.DataFrame
    output_path: Path
    report_path: Path


def compute_semantic_biomarkers(
    config: SemanticZoneConfig,
    semantic_feature_df: pd.DataFrame,
    event_log_path: Path | None = None,
) -> SemanticBiomarkerResult:
    if semantic_feature_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "window_id",
                "media_id",
                "system_id",
                "room_id",
                "session_id",
                "start_time",
                "end_time",
                "mobility_index",
                "spatial_freedom_index",
                "occupancy_imbalance_index",
                "activity_mean",
                "normalized_activity",
                "drinking_activity_fraction",
                "feeding_activity_fraction",
                "general_activity_fraction",
                "drinking_to_feeding_activity_ratio",
                "feeding_plus_drinking_activity_fraction",
                "functional_area_activity_fraction",
                "semantic_transition_proxy",
            ]
        )
        output_path = config.features_dir / "semantic_biomarker_window_table.csv"
        report_path = config.reports_dir / "semantic_biomarker_report.md"
        empty_df.to_csv(output_path, index=False)
        report_path.write_text("# Semantic Biomarker Report\n\nNo semantic feature rows were available.\n", encoding="utf-8")
        return SemanticBiomarkerResult(empty_df, output_path, report_path)

    working_df = prepare_time_columns(semantic_feature_df)
    group_columns = ["window_id", "room_id", "session_id", "start_time", "end_time"]
    optional_columns = ["media_id", "system_id", "duration_seconds", "zone_config_id"]
    rows: list[dict] = []

    grouped = working_df.sort_values(["room_id", "session_id", "start_time_dt", "zone_id"], kind="stable").groupby(group_columns, sort=False, dropna=False)
    for group_key, window_group in grouped:
        first_row = window_group.iloc[0]
        zone_lookup = {
            str(row["zone_id"]): row
            for _, row in window_group.iterrows()
        }
        zone_values = np.array(
            [
                safe_float(zone_lookup.get("drinking_zone", {}).get("activity_mean") if "drinking_zone" in zone_lookup else np.nan, 0.0) or 0.0,
                safe_float(zone_lookup.get("feeding_zone", {}).get("activity_mean") if "feeding_zone" in zone_lookup else np.nan, 0.0) or 0.0,
                safe_float(zone_lookup.get("general_zone", {}).get("activity_mean") if "general_zone" in zone_lookup else np.nan, 0.0) or 0.0,
            ],
            dtype=float,
        )
        probabilities = distribution_from_values(zone_values)
        activity_mean = float(np.nanmean(zone_values))
        drinking_fraction, feeding_fraction, general_fraction = probabilities.tolist()
        rows.append(
            {
                "window_id": str(group_key[0]),
                "room_id": str(group_key[1]),
                "session_id": str(group_key[2]),
                "start_time": str(group_key[3]),
                "end_time": str(group_key[4]),
                "media_id": str(first_row.get("media_id", "")),
                "system_id": str(first_row.get("system_id", "")),
                "duration_seconds": safe_float(first_row.get("duration_seconds"), np.nan),
                "zone_config_id": str(first_row.get("zone_config_id", "")),
                "activity_mean": activity_mean,
                "activity_total": float(np.nansum(zone_values)),
                "drinking_activity_fraction": drinking_fraction,
                "feeding_activity_fraction": feeding_fraction,
                "general_activity_fraction": general_fraction,
                "drinking_to_feeding_activity_ratio": float(drinking_fraction / feeding_fraction) if feeding_fraction > 0 else np.nan,
                "feeding_plus_drinking_activity_fraction": float(drinking_fraction + feeding_fraction),
                "functional_area_activity_fraction": float(drinking_fraction + feeding_fraction),
                "spatial_freedom_index": _spatial_freedom(probabilities),
                "occupancy_imbalance_index": _imbalance(probabilities),
                "warnings": _merge_text(window_group.get("warnings")),
            }
        )

    biomarker_df = pd.DataFrame(rows)
    biomarker_df = prepare_time_columns(biomarker_df)
    biomarker_df = biomarker_df.sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)

    transition_values: list[float] = []
    for _, group_df in biomarker_df.groupby(["room_id", "session_id"], sort=False, dropna=False):
        previous = None
        for _, row in group_df.iterrows():
            current = np.array(
                [
                    row["drinking_activity_fraction"],
                    row["feeding_activity_fraction"],
                    row["general_activity_fraction"],
                ],
                dtype=float,
            )
            transition_values.append(0.0 if previous is None else float(0.5 * np.abs(current - previous).sum()))
            previous = current
    biomarker_df["semantic_transition_proxy"] = transition_values
    biomarker_df["normalized_activity"] = biomarker_df.groupby("room_id", sort=False)["activity_mean"].transform(minmax_scale)
    biomarker_df["mobility_index"] = (
        0.7 * biomarker_df["normalized_activity"].fillna(0.0)
        + 0.3 * biomarker_df["semantic_transition_proxy"].fillna(0.0)
    ).clip(0.0, 1.0)

    output_path = config.features_dir / "semantic_biomarker_window_table.csv"
    report_path = config.reports_dir / "semantic_biomarker_report.md"
    biomarker_df.to_csv(output_path, index=False)
    plot_semantic_biomarker_timeseries(biomarker_df, config.plots_dir / "semantic_biomarker_timeseries.png")
    report_path.write_text(_build_report(biomarker_df, event_log_path), encoding="utf-8")
    return SemanticBiomarkerResult(biomarker_df=biomarker_df, output_path=output_path, report_path=report_path)


def _spatial_freedom(probabilities: np.ndarray) -> float:
    if probabilities.size <= 1:
        return 1.0
    mask = probabilities > 0
    entropy = -np.sum(probabilities[mask] * np.log(probabilities[mask]))
    return float(entropy / np.log(probabilities.size))


def _imbalance(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        return np.nan
    uniform = np.repeat(1.0 / probabilities.size, probabilities.size)
    return float(0.5 * np.abs(probabilities - uniform).sum())


def _merge_text(series: pd.Series | None) -> str:
    if series is None:
        return ""
    values = [str(value).strip() for value in series.dropna().astype(str) if str(value).strip()]
    return ";".join(dict.fromkeys(values))


def _build_report(biomarker_df: pd.DataFrame, event_log_path: Path | None) -> str:
    peak_zone = "n/a"
    if not biomarker_df.empty:
        mean_fractions = biomarker_df[
            [
                "drinking_activity_fraction",
                "feeding_activity_fraction",
                "general_activity_fraction",
            ]
        ].mean()
        peak_zone = str(mean_fractions.idxmax()) if mean_fractions.notna().any() else "n/a"

    lines = [
        "# Semantic Biomarker Report",
        "",
        "- Semantic zones differ from the old four-way layout because they represent drinking, feeding, and general area rather than equal image slices.",
        f"- Number of semantic biomarker windows: {len(biomarker_df)}",
        f"- Time span start: `{biomarker_df['start_time'].min() if not biomarker_df.empty else 'n/a'}`",
        f"- Time span end: `{biomarker_df['start_time'].max() if not biomarker_df.empty else 'n/a'}`",
        f"- Mean dominant semantic zone by activity fraction: `{peak_zone}`",
        "",
        "## Summary Statistics",
        "",
        "```text",
        biomarker_df[
            [
                "activity_mean",
                "mobility_index",
                "spatial_freedom_index",
                "occupancy_imbalance_index",
                "drinking_activity_fraction",
                "feeding_activity_fraction",
                "general_activity_fraction",
                "semantic_transition_proxy",
            ]
        ].describe().to_string() if not biomarker_df.empty else "No biomarker rows available.",
        "```",
        "",
        "## Interpretation Notes",
        "",
        "- Activity fractions describe semantic-zone activity distribution, not true occupancy or bird counts.",
        "- Drinking and feeding zones are small functional areas, so their fraction changes should be interpreted as relative concentration of motion intensity.",
        f"- Event log available for later alignment: `{bool(event_log_path and event_log_path.exists())}`",
    ]
    return "\n".join(lines) + "\n"
