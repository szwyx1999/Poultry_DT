from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Mvp1Config
from .utils import combine_warnings, optional_text, row_value, safe_float
from .video_reader import WindowReadResult, _safe_cap_get, require_cv2
from .zone_utils import compute_zone_boundaries


FEATURE_COLUMNS = [
    "window_id",
    "media_id",
    "system_id",
    "room_id",
    "session_id",
    "zone_id",
    "start_time",
    "end_time",
    "video_path",
    "video_start_offset_sec",
    "duration_seconds",
    "activity_mean",
    "activity_std",
    "motion_pixel_ratio",
    "frame_count",
    "warnings",
]


@dataclass(slots=True)
class StreamedVideoMetrics:
    interval_end_offsets_sec: np.ndarray
    sample_offsets_sec: np.ndarray
    zone_metrics: dict[str, dict[str, np.ndarray]]
    warnings: list[str]


def extract_video_zone_features(
    selected_windows: pd.DataFrame,
    read_results: dict[str, WindowReadResult],
    zone_config: dict,
    config: Mvp1Config,
    cv2_module=None,
) -> pd.DataFrame:
    if selected_windows.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows: list[dict] = []
    module = cv2_module

    for _, window_row in selected_windows.iterrows():
        window_id = str(row_value(window_row, "window_id", ""))
        result = read_results.get(window_id)
        base_warnings = combine_warnings(row_value(window_row, "warnings", ""), result.warnings if result else [])

        if result is None:
            rows.extend(
                _blank_zone_rows(window_row, zone_config, frame_count=0, warnings=combine_warnings(base_warnings, "read_result_missing"))
            )
            continue

        frame_count = len(result.frames)
        if frame_count < 2:
            rows.extend(
                _blank_zone_rows(
                    window_row,
                    zone_config,
                    frame_count=frame_count,
                    warnings=combine_warnings(base_warnings, "insufficient_frames_for_optical_flow"),
                )
            )
            continue

        if module is None:
            module = require_cv2()

        prepared_frames = [
            _to_grayscale(_resize_frame(frame, config.resize_width, module))
            for frame in result.frames
        ]
        zone_boundaries = compute_zone_boundaries(
            frame_height=prepared_frames[0].shape[0],
            frame_width=prepared_frames[0].shape[1],
            zone_config=zone_config,
            layout_override=config.zone_layout,
        )

        zone_metrics: dict[str, dict[str, list[float]]] = {
            zone["zone_id"]: {"activity_values": [], "motion_ratios": []}
            for zone in zone_boundaries
        }

        for previous_frame, current_frame in zip(prepared_frames, prepared_frames[1:]):
            flow = module.calcOpticalFlowFarneback(
                previous_frame,
                current_frame,
                None,
                config.optical_flow_params["pyr_scale"],
                config.optical_flow_params["levels"],
                config.optical_flow_params["winsize"],
                config.optical_flow_params["iterations"],
                config.optical_flow_params["poly_n"],
                config.optical_flow_params["poly_sigma"],
                config.optical_flow_params["flags"],
            )
            magnitude = np.sqrt((flow[..., 0] ** 2) + (flow[..., 1] ** 2))
            for zone in zone_boundaries:
                zone_slice = magnitude[
                    zone["y_start"] : zone["y_end"],
                    zone["x_start"] : zone["x_end"],
                ]
                zone_metrics[zone["zone_id"]]["activity_values"].append(float(zone_slice.mean()))
                zone_metrics[zone["zone_id"]]["motion_ratios"].append(
                    float((zone_slice > config.motion_threshold).mean())
                )

        for zone in zone_boundaries:
            zone_id = zone["zone_id"]
            activity_values = zone_metrics[zone_id]["activity_values"]
            motion_ratios = zone_metrics[zone_id]["motion_ratios"]
            rows.append(
                _build_feature_row(
                    window_row=window_row,
                    zone_id=zone_id,
                    frame_count=frame_count,
                    warnings=base_warnings,
                    activity_mean=float(np.mean(activity_values)),
                    activity_std=float(np.std(activity_values)),
                    motion_pixel_ratio=float(np.mean(motion_ratios)),
                )
            )

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def extract_video_zone_features_streaming(
    selected_windows: pd.DataFrame,
    zone_config: dict,
    config: Mvp1Config,
    cv2_module=None,
) -> pd.DataFrame:
    if selected_windows.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    module = cv2_module if cv2_module is not None else require_cv2()
    rows: list[dict] = []

    grouped = selected_windows.groupby("media_id", sort=False, dropna=False)
    for _, media_windows in grouped:
        media_windows = media_windows.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
        first_row = media_windows.iloc[0]
        base_path_text = optional_text(first_row.get("resolved_video_path")) or ""
        resolved_video_path = Path(base_path_text) if base_path_text else Path()
        media_warnings = combine_warnings(first_row.get("warnings", ""))

        if not base_path_text or not resolved_video_path.exists():
            for _, window_row in media_windows.iterrows():
                rows.extend(
                    _blank_zone_rows(
                        window_row,
                        zone_config,
                        frame_count=0,
                        warnings=combine_warnings(media_warnings, "missing_video_file"),
                    )
                )
            continue

        streamed_metrics = _stream_video_zone_metrics(
            resolved_video_path=resolved_video_path,
            zone_config=zone_config,
            config=config,
            cv2_module=module,
        )

        if streamed_metrics.interval_end_offsets_sec.size == 0:
            for _, window_row in media_windows.iterrows():
                rows.extend(
                    _blank_zone_rows(
                        window_row,
                        zone_config,
                        frame_count=int(streamed_metrics.sample_offsets_sec.size),
                        warnings=combine_warnings(media_warnings, streamed_metrics.warnings, "insufficient_frames_for_optical_flow"),
                    )
                )
            continue

        for _, window_row in media_windows.iterrows():
            rows.extend(
                _build_window_feature_rows_from_stream(
                    window_row=window_row,
                    streamed_metrics=streamed_metrics,
                    zone_config=zone_config,
                    warnings=combine_warnings(window_row.get("warnings", ""), streamed_metrics.warnings),
                )
            )

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def save_feature_csv(features_df: pd.DataFrame, output_path: Path) -> Path:
    features_df.to_csv(output_path, index=False)
    return output_path


def load_completed_feature_rows(
    output_path: Path,
    selected_windows: pd.DataFrame,
    zone_config: dict,
) -> tuple[pd.DataFrame, set[str]]:
    if not output_path.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS), set()

    existing_df = pd.read_csv(output_path)
    if existing_df.empty or "window_id" not in existing_df.columns:
        return pd.DataFrame(columns=FEATURE_COLUMNS), set()

    selected_window_ids = set(selected_windows["window_id"].astype(str).tolist())
    existing_df = existing_df[existing_df["window_id"].astype(str).isin(selected_window_ids)].copy()
    if existing_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), set()

    expected_zone_count = len(zone_config.get("zones", []))
    completed_counts = existing_df.groupby("window_id", dropna=False)["zone_id"].nunique()
    if expected_zone_count > 0:
        completed_window_ids = {
            str(window_id)
            for window_id, zone_count in completed_counts.items()
            if int(zone_count) >= expected_zone_count
        }
    else:
        completed_window_ids = set(completed_counts.index.astype(str).tolist())

    return existing_df, completed_window_ids


def _blank_zone_rows(
    window_row: pd.Series,
    zone_config: dict,
    frame_count: int,
    warnings: str,
) -> list[dict]:
    blank_rows: list[dict] = []
    ordered_zones = sorted(zone_config["zones"], key=lambda item: (item["row"], item["col"]))
    for zone in ordered_zones:
        blank_rows.append(
            _build_feature_row(
                window_row=window_row,
                zone_id=zone["zone_id"],
                frame_count=frame_count,
                warnings=warnings,
                activity_mean=None,
                activity_std=None,
                motion_pixel_ratio=None,
            )
        )
    return blank_rows


def _build_window_feature_rows_from_stream(
    window_row: pd.Series,
    streamed_metrics: StreamedVideoMetrics,
    zone_config: dict,
    warnings: str,
) -> list[dict]:
    start_offset = safe_float(row_value(window_row, "video_start_offset_sec", 0.0), 0.0) or 0.0
    duration_seconds = safe_float(row_value(window_row, "duration_seconds", 0.0), 0.0) or 0.0
    end_offset = start_offset + duration_seconds

    interval_mask = (
        (streamed_metrics.interval_end_offsets_sec > (start_offset + 1e-9))
        & (streamed_metrics.interval_end_offsets_sec <= (end_offset + 1e-6))
    )
    sample_frame_count = int(
        (
            (streamed_metrics.sample_offsets_sec >= max(0.0, start_offset) - 1e-9)
            & (streamed_metrics.sample_offsets_sec <= end_offset + 1e-6)
        ).sum()
    )

    if interval_mask.sum() < 1:
        return _blank_zone_rows(
            window_row,
            zone_config,
            frame_count=sample_frame_count,
            warnings=combine_warnings(warnings, "insufficient_frames_for_optical_flow"),
        )

    rows: list[dict] = []
    ordered_zones = [zone["zone_id"] for zone in sorted(zone_config["zones"], key=lambda item: (item["row"], item["col"]))]
    for zone_id in ordered_zones:
        zone_data = streamed_metrics.zone_metrics.get(zone_id, {})
        activity_values = np.asarray(zone_data.get("activity_values", np.array([])), dtype=float)[interval_mask]
        motion_ratios = np.asarray(zone_data.get("motion_ratios", np.array([])), dtype=float)[interval_mask]
        if activity_values.size == 0:
            rows.extend(
                _blank_zone_rows(
                    window_row,
                    zone_config,
                    frame_count=sample_frame_count,
                    warnings=combine_warnings(warnings, "empty_zone_activity_values"),
                )
            )
            return rows

        rows.append(
            _build_feature_row(
                window_row=window_row,
                zone_id=zone_id,
                frame_count=sample_frame_count,
                warnings=warnings,
                activity_mean=float(np.mean(activity_values)),
                activity_std=float(np.std(activity_values)),
                motion_pixel_ratio=float(np.mean(motion_ratios)) if motion_ratios.size else None,
            )
        )

    return rows


def _build_feature_row(
    window_row: pd.Series,
    zone_id: str,
    frame_count: int,
    warnings: str,
    activity_mean: float | None,
    activity_std: float | None,
    motion_pixel_ratio: float | None,
) -> dict:
    return {
        "window_id": row_value(window_row, "window_id", ""),
        "media_id": row_value(window_row, "media_id", ""),
        "system_id": row_value(window_row, "system_id", ""),
        "room_id": row_value(window_row, "room_id", ""),
        "session_id": row_value(window_row, "session_id", ""),
        "zone_id": zone_id,
        "start_time": row_value(window_row, "start_time", ""),
        "end_time": row_value(window_row, "end_time", ""),
        "video_path": row_value(window_row, "video_path", ""),
        "video_start_offset_sec": row_value(window_row, "video_start_offset_sec", ""),
        "duration_seconds": row_value(window_row, "duration_seconds", ""),
        "activity_mean": activity_mean,
        "activity_std": activity_std,
        "motion_pixel_ratio": motion_pixel_ratio,
        "frame_count": frame_count,
        "warnings": warnings,
    }


def _stream_video_zone_metrics(
    resolved_video_path: Path,
    zone_config: dict,
    config: Mvp1Config,
    cv2_module,
) -> StreamedVideoMetrics:
    capture = cv2_module.VideoCapture(str(resolved_video_path))
    if not capture or not capture.isOpened():
        if capture:
            capture.release()
        return StreamedVideoMetrics(
            interval_end_offsets_sec=np.array([], dtype=float),
            sample_offsets_sec=np.array([], dtype=float),
            zone_metrics={},
            warnings=["opencv_open_failed"],
        )

    fps = _safe_cap_get(capture, cv2_module.CAP_PROP_FPS)
    total_frame_count = _safe_cap_get(capture, cv2_module.CAP_PROP_FRAME_COUNT)
    total_duration_seconds = None
    if fps and fps > 0 and total_frame_count and total_frame_count > 0:
        total_duration_seconds = total_frame_count / fps
    sample_period = 1.0 / config.frame_sample_rate
    sample_index = 0
    previous_frame_gray = None
    previous_sample_offset = None
    sample_offsets: list[float] = []
    interval_end_offsets: list[float] = []
    zone_boundaries: list[dict] | None = None
    zone_metrics: dict[str, dict[str, list[float]]] = {}
    warnings: list[str] = []

    while True:
        target_sample_sec = sample_index * sample_period
        if total_duration_seconds is not None and target_sample_sec > (total_duration_seconds + 1e-6):
            break

        if not _seek_capture_to_sample(capture, cv2_module, target_sample_sec, fps):
            break
        success, frame = capture.read()
        if not success:
            break

        frame_time_sec = _estimate_sample_time_seconds(
            capture=capture,
            cv2_module=cv2_module,
            target_sample_sec=target_sample_sec,
            fps=fps,
        )
        if previous_sample_offset is not None and frame_time_sec <= previous_sample_offset + 1e-6:
            warnings.append("non_increasing_sample_timestamps")
            break

        resized_frame = _resize_frame(frame, config.resize_width, cv2_module)
        current_frame_gray = _to_grayscale(resized_frame)

        if zone_boundaries is None:
            zone_boundaries = compute_zone_boundaries(
                frame_height=current_frame_gray.shape[0],
                frame_width=current_frame_gray.shape[1],
                zone_config=zone_config,
                layout_override=config.zone_layout,
            )
            zone_metrics = {
                zone["zone_id"]: {"activity_values": [], "motion_ratios": []}
                for zone in zone_boundaries
            }
        sample_offsets.append(frame_time_sec)
        if previous_frame_gray is not None:
            flow = cv2_module.calcOpticalFlowFarneback(
                previous_frame_gray,
                current_frame_gray,
                None,
                config.optical_flow_params["pyr_scale"],
                config.optical_flow_params["levels"],
                config.optical_flow_params["winsize"],
                config.optical_flow_params["iterations"],
                config.optical_flow_params["poly_n"],
                config.optical_flow_params["poly_sigma"],
                config.optical_flow_params["flags"],
            )
            magnitude = np.sqrt((flow[..., 0] ** 2) + (flow[..., 1] ** 2))
            interval_end_offsets.append(frame_time_sec)
            for zone in zone_boundaries:
                zone_slice = magnitude[
                    zone["y_start"] : zone["y_end"],
                    zone["x_start"] : zone["x_end"],
                ]
                zone_metrics[zone["zone_id"]]["activity_values"].append(float(zone_slice.mean()))
                zone_metrics[zone["zone_id"]]["motion_ratios"].append(
                    float((zone_slice > config.motion_threshold).mean())
                )

        previous_frame_gray = current_frame_gray
        previous_sample_offset = frame_time_sec
        sample_index += 1

    capture.release()

    if not sample_offsets:
        warnings.append("no_frames_read")

    return StreamedVideoMetrics(
        interval_end_offsets_sec=np.asarray(interval_end_offsets, dtype=float),
        sample_offsets_sec=np.asarray(sample_offsets, dtype=float),
        zone_metrics={
            zone_id: {
                "activity_values": np.asarray(values["activity_values"], dtype=float),
                "motion_ratios": np.asarray(values["motion_ratios"], dtype=float),
            }
            for zone_id, values in zone_metrics.items()
        },
        warnings=warnings,
    )


def _seek_capture_to_sample(capture, cv2_module, target_sample_sec: float, fps: float | None) -> bool:
    if fps and fps > 0:
        target_frame_index = max(0, int(round(target_sample_sec * fps)))
        if capture.set(cv2_module.CAP_PROP_POS_FRAMES, target_frame_index):
            return True
    return bool(capture.set(cv2_module.CAP_PROP_POS_MSEC, max(0.0, target_sample_sec * 1000.0)))


def _resize_frame(frame: np.ndarray, target_width: int, cv2_module) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2_module.resize(
        frame,
        (target_width, target_height),
        interpolation=getattr(cv2_module, "INTER_AREA", 3),
    )


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        blue = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        red = frame[:, :, 2].astype(np.float32)
        grayscale = (0.114 * blue) + (0.587 * green) + (0.299 * red)
        return grayscale.astype(np.uint8)
    raise ValueError("Unsupported frame shape for grayscale conversion.")


def _estimate_video_time_seconds(capture, cv2_module, frame_index: int, fps: float | None) -> float:
    current_msec = _safe_cap_get(capture, cv2_module.CAP_PROP_POS_MSEC)
    if current_msec is not None and current_msec > 0:
        return max(0.0, current_msec / 1000.0)
    if fps and fps > 0:
        return frame_index / fps
    return 0.0


def _estimate_sample_time_seconds(capture, cv2_module, target_sample_sec: float, fps: float | None) -> float:
    current_msec = _safe_cap_get(capture, cv2_module.CAP_PROP_POS_MSEC)
    if current_msec is not None and current_msec > 0:
        return max(0.0, current_msec / 1000.0)

    if fps and fps > 0:
        current_frame_index = _safe_cap_get(capture, cv2_module.CAP_PROP_POS_FRAMES)
        if current_frame_index is not None and current_frame_index > 0:
            return max(0.0, (current_frame_index - 1.0) / fps)

    return max(0.0, target_sample_sec)
