from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import textwrap

import matplotlib
import numpy as np
import pandas as pd

from .config import Mvp1Config
from .utils import combine_warnings, safe_float

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - exercised indirectly when OpenCV is installed
    import cv2 as _cv2
except ImportError:  # pragma: no cover - this path is test-friendly
    _cv2 = None


@dataclass(slots=True)
class WindowReadResult:
    window_id: str
    resolved_video_path: Path
    frames: list[np.ndarray] = field(default_factory=list)
    frame_offsets_sec: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_fps: float | None = None


def require_cv2():
    if _cv2 is None:
        raise RuntimeError(
            "OpenCV is required for MVP 1 video reading. Install "
            "`opencv-python-headless` from mvp_video_zone_heatmap/requirements.txt."
        )
    return _cv2


def collect_window_reads(
    selected_windows: pd.DataFrame,
    config: Mvp1Config,
    cv2_module=None,
) -> dict[str, WindowReadResult]:
    results: dict[str, WindowReadResult] = {}
    if selected_windows.empty:
        return results

    module = cv2_module if cv2_module is not None else require_cv2()
    for window_row in selected_windows.to_dict(orient="records"):
        result = read_video_window(window_row, config, cv2_module=module)
        save_preview_frame(window_row, result, config)
        results[result.window_id] = result
    return results


def read_video_window(
    window_row: dict,
    config: Mvp1Config,
    cv2_module=None,
) -> WindowReadResult:
    module = cv2_module if cv2_module is not None else require_cv2()

    window_id = str(window_row.get("window_id", "unknown_window"))
    path_text = str(window_row.get("resolved_video_path", "")).strip()
    resolved_video_path = Path(path_text) if path_text else Path()
    result = WindowReadResult(window_id=window_id, resolved_video_path=resolved_video_path)

    if not path_text or not resolved_video_path.exists():
        result.warnings.append("missing_video_file")
        return result

    capture = module.VideoCapture(str(resolved_video_path))
    if not capture or not capture.isOpened():
        result.warnings.append("opencv_open_failed")
        if capture:
            capture.release()
        return result

    start_offset_sec = safe_float(window_row.get("video_start_offset_sec"), 0.0) or 0.0
    duration_seconds = safe_float(window_row.get("duration_seconds"), 0.0) or 0.0
    sample_period = 1.0 / config.frame_sample_rate
    if not capture.set(module.CAP_PROP_POS_MSEC, max(start_offset_sec, 0.0) * 1000.0):
        result.warnings.append("seek_failed")

    result.source_fps = _safe_cap_get(capture, module.CAP_PROP_FPS)
    next_sample_sec = 0.0
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        relative_sec = _estimate_relative_timestamp(
            capture=capture,
            cv2_module=module,
            start_offset_sec=start_offset_sec,
            frame_index=frame_index,
            fps=result.source_fps,
        )
        frame_index += 1

        if relative_sec > duration_seconds + 1e-6:
            break

        if not result.frames or relative_sec + 1e-9 >= next_sample_sec:
            result.frames.append(frame.copy())
            result.frame_offsets_sec.append(min(relative_sec, duration_seconds))
            next_sample_sec = len(result.frames) * sample_period

    capture.release()

    if not result.frames:
        result.warnings.append("no_frames_read")

    return result


def save_preview_frame(window_row: dict, result: WindowReadResult, config: Mvp1Config) -> Path:
    preview_path = config.sample_frames_dir / f"{result.window_id}.png"
    if result.frames:
        preview_frame = _select_preview_frame(result.frames, config.preview_frame_position)
        plt.imsave(preview_path, _frame_to_rgb(preview_frame))
        return preview_path

    warning_text = combine_warnings(window_row.get("warnings", ""), result.warnings) or "no_preview_available"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_axis_off()
    axis.text(0.5, 0.62, result.window_id, ha="center", va="center", fontsize=14, fontweight="bold")
    axis.text(
        0.5,
        0.35,
        textwrap.fill(warning_text, width=40),
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)
    return preview_path


def _select_preview_frame(frames: list[np.ndarray], position: str) -> np.ndarray:
    if position == "first":
        return frames[0]
    if position == "last":
        return frames[-1]
    return frames[len(frames) // 2]


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, :3][:, :, ::-1]
    return frame


def _estimate_relative_timestamp(
    capture,
    cv2_module,
    start_offset_sec: float,
    frame_index: int,
    fps: float | None,
) -> float:
    current_msec = _safe_cap_get(capture, cv2_module.CAP_PROP_POS_MSEC)
    if current_msec is not None and current_msec > 0:
        return max(0.0, current_msec / 1000.0 - start_offset_sec)
    if fps and fps > 0:
        return frame_index / fps
    return 0.0


def _safe_cap_get(capture, property_id) -> float | None:
    try:
        value = capture.get(property_id)
    except Exception:  # pragma: no cover - defensive fallback for OpenCV bindings
        return None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)
