from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import SemanticZoneConfig
from .plotting import plot_semantic_zone_activity_heatmap, plot_semantic_zone_activity_timeseries
from .semantic_zone_builder import SemanticZoneBuildResult, build_zone_masks, polygon_to_mask, scale_polygon
from .utils import combine_warnings, prepare_time_columns, read_csv_if_exists, resolve_video_path, safe_float

try:  # pragma: no cover - depends on local environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


FEATURE_COLUMNS = [
    "window_id",
    "media_id",
    "system_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "video_path",
    "video_start_offset_sec",
    "duration_seconds",
    "zone_config_id",
    "zone_id",
    "semantic_type",
    "zone_area_pixels",
    "zone_area_fraction",
    "activity_mean",
    "activity_std",
    "activity_sum",
    "activity_norm",
    "frame_count_used",
    "quality_status",
    "warnings",
]


@dataclass(frozen=True)
class SemanticZoneFeatureResult:
    selected_windows_df: pd.DataFrame
    feature_df: pd.DataFrame
    output_path: Path
    report_path: Path
    elapsed_seconds: float


def compute_semantic_zone_video_features(
    config: SemanticZoneConfig,
    zone_result: SemanticZoneBuildResult,
) -> SemanticZoneFeatureResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for semantic-zone video feature extraction.")

    start_clock = time.perf_counter()
    selected_windows_df = _load_target_windows(config)
    output_path = config.features_dir / "semantic_zone_video_features.csv"
    report_path = config.reports_dir / "semantic_video_processing_coverage.md"

    all_rows: list[dict] = []
    failed_windows: list[str] = []
    media_groups = list(selected_windows_df.groupby("media_id", sort=False, dropna=False))
    for _, media_df in media_groups:
        media_df = media_df.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
        interval_data = _load_or_compute_media_cache(config, zone_result.zone_config, media_df)
        media_rows = _aggregate_media_windows(media_df, zone_result.zone_config, interval_data)
        all_rows.extend(media_rows)
        failed_windows.extend(
            sorted({str(row["window_id"]) for row in media_rows if str(row["quality_status"]) == "failed"})
        )

    feature_df = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)
    if not feature_df.empty:
        feature_df = prepare_time_columns(feature_df)
        feature_df = feature_df.sort_values(
            ["room_id", "session_id", "start_time_dt", "window_id", "zone_id"],
            kind="stable",
        ).reset_index(drop=True)
        total_by_window = feature_df.groupby("window_id", sort=False)["activity_sum"].transform("sum")
        zone_count_by_window = feature_df.groupby("window_id", sort=False)["zone_id"].transform("count").replace(0, np.nan)
        feature_df["activity_norm"] = np.where(
            pd.to_numeric(total_by_window, errors="coerce").fillna(0.0) > 0,
            pd.to_numeric(feature_df["activity_sum"], errors="coerce").fillna(0.0) / total_by_window,
            1.0 / zone_count_by_window.fillna(1.0),
        )
        feature_df["activity_norm"] = pd.to_numeric(feature_df["activity_norm"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    plot_semantic_zone_activity_timeseries(feature_df, config.plots_dir / "semantic_zone_activity_timeseries.png")
    plot_semantic_zone_activity_heatmap(feature_df, config.plots_dir / "semantic_zone_activity_heatmap.png")

    elapsed_seconds = time.perf_counter() - start_clock
    report_path.write_text(
        _build_coverage_report(
            config=config,
            selected_windows_df=selected_windows_df,
            feature_df=feature_df,
            failed_windows=failed_windows,
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )
    return SemanticZoneFeatureResult(
        selected_windows_df=selected_windows_df,
        feature_df=feature_df,
        output_path=output_path,
        report_path=report_path,
        elapsed_seconds=elapsed_seconds,
    )


def _load_target_windows(config: SemanticZoneConfig) -> pd.DataFrame:
    selected_df = read_csv_if_exists(config.selected_windows_csv)
    if selected_df.empty:
        selected_df = read_csv_if_exists(config.video_window_index_csv)
    if selected_df.empty:
        raise ValueError("No selected windows or indexed windows were available for semantic-zone processing.")

    working_df = selected_df.copy()
    if "resolved_video_path" not in working_df.columns:
        working_df["resolved_video_path"] = working_df["video_path"].astype(str).apply(
            lambda value: str(resolve_video_path(config.workspace_root, value))
        )

    room_mask = working_df["room_id"].astype(str) == config.target_room_id
    folder_mask = working_df["video_path"].astype(str).str.contains(config.target_raw_subdir.replace("\\", "/"), case=False, regex=False, na=False)
    if not bool(folder_mask.any()):
        folder_mask = working_df["resolved_video_path"].astype(str).str.contains("Room 1 (16, 17 Aug)", case=False, regex=False, na=False)
    filtered_df = working_df[room_mask & folder_mask].copy()
    if filtered_df.empty:
        raise ValueError(
            "No windows matched the configured semantic-zone target folder: "
            f"{config.target_raw_subdir}"
        )

    filtered_df = (
        filtered_df.drop_duplicates(subset=["window_id"], keep="first")
        .sort_values(["room_id", "session_id", "start_time", "window_id"], kind="stable")
        .reset_index(drop=True)
    )
    return filtered_df


def _load_or_compute_media_cache(
    config: SemanticZoneConfig,
    zone_config: dict,
    media_df: pd.DataFrame,
) -> dict[str, object]:
    media_id = str(media_df.iloc[0]["media_id"])
    cache_path = config.cache_dir / f"semantic_video_media_{media_id}.joblib"
    if config.video_cache_enabled and cache_path.exists() and not config.force_recompute:
        return joblib.load(cache_path)

    interval_data = _compute_media_interval_metrics(config, zone_config, media_df)
    if config.video_cache_enabled:
        joblib.dump(interval_data, cache_path)
    return interval_data


def _compute_media_interval_metrics(
    config: SemanticZoneConfig,
    zone_config: dict,
    media_df: pd.DataFrame,
) -> dict[str, object]:
    resolved_path = Path(str(media_df.iloc[0]["resolved_video_path"]))
    warnings: list[str] = []
    if not resolved_path.exists():
        return {
            "interval_center_sec": np.array([], dtype=float),
            "zone_metrics": {},
            "warnings": ["missing_video_file"],
        }

    capture = cv2.VideoCapture(str(resolved_path))
    if not capture or not capture.isOpened():
        if capture:
            capture.release()
        return {
            "interval_center_sec": np.array([], dtype=float),
            "zone_metrics": {},
            "warnings": ["opencv_open_failed"],
        }

    max_offset = float(
        (
            pd.to_numeric(media_df["video_start_offset_sec"], errors="coerce").fillna(0.0)
            + pd.to_numeric(media_df["duration_seconds"], errors="coerce").fillna(0.0)
        ).max()
    )
    sample_period = 1.0 / max(config.frame_sample_rate, 1e-6)
    sample_times = np.arange(0.0, max_offset + sample_period * 0.5, sample_period, dtype=float)
    sampled_frames: list[np.ndarray] = []
    sampled_times: list[float] = []
    resize_height: int | None = None
    resized_masks: dict[str, np.ndarray] | None = None

    for sample_time in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(sample_time) * 1000.0)
        success, frame = capture.read()
        if not success or frame is None:
            warnings.append(f"sample_read_failed_at_{sample_time:.2f}s")
            continue
        resized = _resize_frame(frame, config.resize_width)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if resize_height is None:
            resize_height = int(grayscale.shape[0])
            resized_masks = _build_resized_masks(zone_config, config.resize_width, resize_height)
        sampled_frames.append(grayscale)
        sampled_times.append(float(sample_time))

    capture.release()
    if len(sampled_frames) < 2 or resized_masks is None:
        warnings.append("insufficient_sampled_frames")
        return {
            "interval_center_sec": np.array([], dtype=float),
            "zone_metrics": {},
            "warnings": warnings,
        }

    zone_metrics = {
        zone_id: {
            "activity_mean": [],
            "activity_std": [],
            "activity_sum": [],
        }
        for zone_id in ("drinking_zone", "feeding_zone", "general_zone")
    }
    interval_centers: list[float] = []
    for previous_time, previous_frame, current_time, current_frame in zip(
        sampled_times[:-1],
        sampled_frames[:-1],
        sampled_times[1:],
        sampled_frames[1:],
    ):
        diff = np.abs(current_frame - previous_frame)
        interval_centers.append(float((previous_time + current_time) / 2.0))
        for zone_id, mask in resized_masks.items():
            zone_values = diff[mask]
            if zone_values.size == 0:
                zone_metrics[zone_id]["activity_mean"].append(np.nan)
                zone_metrics[zone_id]["activity_std"].append(np.nan)
                zone_metrics[zone_id]["activity_sum"].append(np.nan)
            else:
                zone_metrics[zone_id]["activity_mean"].append(float(zone_values.mean()))
                zone_metrics[zone_id]["activity_std"].append(float(zone_values.std()))
                zone_metrics[zone_id]["activity_sum"].append(float(zone_values.sum()))

    return {
        "interval_center_sec": np.asarray(interval_centers, dtype=float),
        "zone_metrics": {
            zone_id: {metric_name: np.asarray(metric_values, dtype=float) for metric_name, metric_values in metric_dict.items()}
            for zone_id, metric_dict in zone_metrics.items()
        },
        "warnings": warnings,
    }


def _build_resized_masks(zone_config: dict, resize_width: int, resize_height: int) -> dict[str, np.ndarray]:
    source_width = int(zone_config["image_width"])
    source_height = int(zone_config["image_height"])
    occupancy = np.zeros((resize_height, resize_width), dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for zone in zone_config.get("zones", []):
        zone_id = str(zone["zone_id"])
        polygon = zone.get("polygon")
        if polygon:
            scaled_polygon = scale_polygon(
                polygon=polygon,
                source_width=source_width,
                source_height=source_height,
                target_width=resize_width,
                target_height=resize_height,
            )
            mask = polygon_to_mask(scaled_polygon, image_width=resize_width, image_height=resize_height)
            masks[zone_id] = mask
            occupancy |= mask
    masks["general_zone"] = ~occupancy
    return masks


def _aggregate_media_windows(
    media_df: pd.DataFrame,
    zone_config: dict,
    interval_data: dict[str, object],
) -> list[dict]:
    interval_center_sec = np.asarray(interval_data.get("interval_center_sec", np.array([], dtype=float)), dtype=float)
    zone_metrics = interval_data.get("zone_metrics", {})
    interval_warnings = interval_data.get("warnings", [])

    original_masks = build_zone_masks(
        zone_config,
        image_width=int(zone_config["image_width"]),
        image_height=int(zone_config["image_height"]),
    )
    total_pixels = int(zone_config["image_width"]) * int(zone_config["image_height"])
    zone_area_lookup = {
        zone_id: {
            "zone_area_pixels": int(mask.sum()),
            "zone_area_fraction": float(mask.sum() / total_pixels),
        }
        for zone_id, mask in original_masks.items()
    }
    semantic_type_lookup = {
        str(zone["zone_id"]): str(zone.get("semantic_type", "general"))
        for zone in zone_config.get("zones", [])
    }

    rows: list[dict] = []
    for _, window_row in media_df.iterrows():
        window_id = str(window_row["window_id"])
        start_offset = float(pd.to_numeric(window_row.get("video_start_offset_sec"), errors="coerce") or 0.0)
        duration_seconds = float(pd.to_numeric(window_row.get("duration_seconds"), errors="coerce") or 0.0)
        end_offset = start_offset + duration_seconds
        mask = (interval_center_sec >= start_offset) & (interval_center_sec < end_offset)
        frame_count_used = int(mask.sum())
        quality_status = "ok"
        warning_text = combine_warnings(window_row.get("warnings", ""), interval_warnings)
        if frame_count_used == 0:
            quality_status = "failed"
            warning_text = combine_warnings(warning_text, "no_interval_overlap")

        for zone_id in ("drinking_zone", "feeding_zone", "general_zone"):
            zone_metric_dict = zone_metrics.get(zone_id, {})
            if frame_count_used > 0 and zone_metric_dict:
                activity_mean = float(np.nanmean(zone_metric_dict["activity_mean"][mask]))
                activity_std = float(np.nanmean(zone_metric_dict["activity_std"][mask]))
                activity_sum = float(np.nansum(zone_metric_dict["activity_sum"][mask]))
            else:
                activity_mean = np.nan
                activity_std = np.nan
                activity_sum = np.nan

            rows.append(
                {
                    "window_id": window_id,
                    "media_id": str(window_row.get("media_id", "")),
                    "system_id": str(window_row.get("system_id", "")),
                    "room_id": str(window_row.get("room_id", "")),
                    "session_id": str(window_row.get("session_id", "")),
                    "start_time": str(window_row.get("start_time", "")),
                    "end_time": str(window_row.get("end_time", "")),
                    "video_path": str(window_row.get("video_path", "")),
                    "video_start_offset_sec": start_offset,
                    "duration_seconds": duration_seconds,
                    "zone_config_id": str(zone_config.get("zone_config_id", "")),
                    "zone_id": zone_id,
                    "semantic_type": semantic_type_lookup.get(zone_id, zone_id.replace("_zone", "")),
                    "zone_area_pixels": zone_area_lookup[zone_id]["zone_area_pixels"],
                    "zone_area_fraction": zone_area_lookup[zone_id]["zone_area_fraction"],
                    "activity_mean": activity_mean,
                    "activity_std": activity_std,
                    "activity_sum": activity_sum,
                    "activity_norm": np.nan,
                    "frame_count_used": frame_count_used,
                    "quality_status": quality_status,
                    "warnings": warning_text,
                }
            )
    return rows


def _resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = float(resize_width) / float(width)
    resize_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def _build_coverage_report(
    config: SemanticZoneConfig,
    selected_windows_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    failed_windows: list[str],
    elapsed_seconds: float,
) -> str:
    raw_target_dir = config.workspace_root / "first_week_data_cleaning" / "data" / "raw" / Path(config.target_raw_subdir)
    raw_videos_found = sorted(
        [
            path
            for path in raw_target_dir.glob("*.MP4")
            if path.is_file()
        ]
    ) if raw_target_dir.exists() else []
    indexed_videos_used = int(selected_windows_df["media_id"].nunique()) if not selected_windows_df.empty else 0
    processed_windows = int(feature_df["window_id"].nunique()) if not feature_df.empty else 0
    per_zone_summary = (
        feature_df.groupby("zone_id", sort=False)["activity_mean"].agg(["count", "mean", "std", "max"]).round(6)
        if not feature_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# Semantic Video Processing Coverage",
        "",
        "- This is a new semantic-zone experiment. Previous code and previous outputs were not modified.",
        f"- Raw target videos found: {len(raw_videos_found)}",
        f"- Indexed videos used: {indexed_videos_used}",
        f"- Selected windows: {len(selected_windows_df)}",
        f"- Processed windows: {processed_windows}",
        f"- Total semantic feature rows: {len(feature_df)}",
        f"- Unique media files: {feature_df['media_id'].nunique() if not feature_df.empty else 0}",
        f"- Time span start: `{selected_windows_df['start_time'].min() if not selected_windows_df.empty else 'n/a'}`",
        f"- Time span end: `{selected_windows_df['start_time'].max() if not selected_windows_df.empty else 'n/a'}`",
        f"- Failed windows: {len(set(failed_windows))}",
        f"- Runtime estimate: `{elapsed_seconds / 60.0:.2f}` minutes",
        "",
        "## Per-Zone Activity Summary",
        "",
        "```text",
        per_zone_summary.to_string() if not per_zone_summary.empty else "No semantic feature rows available.",
        "```",
        "",
        "## Failed Window IDs",
        "",
    ]
    if failed_windows:
        for window_id in list(dict.fromkeys(failed_windows))[:50]:
            lines.append(f"- `{window_id}`")
        if len(set(failed_windows)) > 50:
            lines.append(f"- plus {len(set(failed_windows)) - 50} more")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
