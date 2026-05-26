from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .utils import optional_text, resolve_project_path


@dataclass(slots=True)
class Mvp1Config:
    project_root: Path
    config_path: Path
    project_name: str
    preprocess_root: Path
    video_window_index: Path
    zone_config: Path
    features_dir: Path
    unity_json_dir: Path
    plots_dir: Path
    sample_frames_dir: Path
    reports_dir: Path
    max_windows: int | None
    max_windows_per_session: int | None
    room_id: str | None
    session_id: str | None
    target_raw_subdir: str | None
    selection_strategy: str
    include_all_available: bool
    skip_bad_quality: bool
    require_quality_ok: bool
    frame_sample_rate: float
    resize_width: int
    preview_frame_position: str
    save_preview_frames: bool
    resume_enabled: bool
    force_recompute: bool
    zone_layout: str
    optical_flow_method: str
    optical_flow_params: dict[str, object]
    motion_threshold: float
    normalize_activity: bool


DEFAULT_CONFIG_PATH = "mvp_video_zone_heatmap/configs/mvp1_config.yaml"
ALLOWED_PREVIEW_POSITIONS = {"first", "middle", "last"}
ALLOWED_LAYOUTS = {"2x2", "3x3"}
ALLOWED_SELECTION_STRATEGIES = {"even", "all"}


def load_config(project_root: Path, config_path: Path | str = DEFAULT_CONFIG_PATH) -> Mvp1Config:
    resolved_config_path = resolve_project_path(project_root, config_path)
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved_config_path.as_posix()}"
        )

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project = raw.get("project", {}) or {}
    inputs = raw.get("inputs", {}) or {}
    outputs = raw.get("outputs", {}) or {}
    selection = raw.get("selection", {}) or {}
    video = raw.get("video", {}) or {}
    zones = raw.get("zones", {}) or {}
    optical_flow = raw.get("optical_flow", {}) or {}
    export = raw.get("export", {}) or {}
    runtime = raw.get("runtime", {}) or {}

    raw_max_windows = selection.get("max_windows", 60)
    max_windows = None if raw_max_windows is None else int(raw_max_windows)
    raw_max_windows_per_session = selection.get("max_windows_per_session")
    max_windows_per_session = None if raw_max_windows_per_session in (None, "") else int(raw_max_windows_per_session)

    config = Mvp1Config(
        project_root=project_root,
        config_path=resolved_config_path,
        project_name=str(project.get("name", "poultry_mvp1_video_zone_heatmap")),
        preprocess_root=resolve_project_path(
            project_root, str(inputs.get("preprocess_root", "first_week_data_cleaning"))
        ),
        video_window_index=resolve_project_path(
            project_root,
            str(
                inputs.get(
                    "video_window_index",
                    "first_week_data_cleaning/data/processed/metadata/video_window_index.csv",
                )
            ),
        ),
        zone_config=resolve_project_path(
            project_root,
            str(
                inputs.get(
                    "zone_config",
                    "mvp_video_zone_heatmap/data/metadata/zone_config.json",
                )
            ),
        ),
        features_dir=resolve_project_path(
            project_root,
            str(outputs.get("features_dir", "mvp_video_zone_heatmap/outputs/features")),
        ),
        unity_json_dir=resolve_project_path(
            project_root,
            str(outputs.get("unity_json_dir", "mvp_video_zone_heatmap/outputs/unity_json")),
        ),
        plots_dir=resolve_project_path(
            project_root,
            str(outputs.get("plots_dir", "mvp_video_zone_heatmap/outputs/plots")),
        ),
        sample_frames_dir=resolve_project_path(
            project_root,
            str(
                outputs.get(
                    "sample_frames_dir",
                    "mvp_video_zone_heatmap/outputs/plots/sample_frames",
                )
            ),
        ),
        reports_dir=resolve_project_path(
            project_root,
            str(outputs.get("reports_dir", "mvp_video_zone_heatmap/outputs/reports")),
        ),
        max_windows=max_windows,
        max_windows_per_session=max_windows_per_session,
        room_id=optional_text(selection.get("target_room_id", selection.get("room_id"))),
        session_id=optional_text(selection.get("target_session_id", selection.get("session_id"))),
        target_raw_subdir=optional_text(selection.get("target_raw_subdir")),
        selection_strategy=str(selection.get("selection_strategy", "even")).strip().lower(),
        include_all_available=bool(selection.get("include_all_available", False)),
        skip_bad_quality=bool(selection.get("skip_bad_quality", False)),
        require_quality_ok=bool(selection.get("require_quality_ok", False)),
        frame_sample_rate=float(video.get("frame_sample_rate", 2)),
        resize_width=int(video.get("resize_width", 640)),
        preview_frame_position=str(video.get("preview_frame_position", "middle")).strip().lower(),
        save_preview_frames=bool(video.get("save_preview_frames", True)),
        resume_enabled=bool(runtime.get("resume_enabled", True)),
        force_recompute=bool(runtime.get("force_recompute", False)),
        zone_layout=str(zones.get("layout", "2x2")).strip().lower(),
        optical_flow_method=str(optical_flow.get("method", "farneback")).strip().lower(),
        optical_flow_params={
            "pyr_scale": float(optical_flow.get("pyr_scale", 0.5)),
            "levels": int(optical_flow.get("levels", 3)),
            "winsize": int(optical_flow.get("winsize", 15)),
            "iterations": int(optical_flow.get("iterations", 3)),
            "poly_n": int(optical_flow.get("poly_n", 5)),
            "poly_sigma": float(optical_flow.get("poly_sigma", 1.2)),
            "flags": int(optical_flow.get("flags", 0)),
        },
        motion_threshold=float(optical_flow.get("motion_threshold", 1.0)),
        normalize_activity=bool(export.get("normalize_activity", True)),
    )
    _validate_config(config)
    _create_output_directories(config)
    return config


def _validate_config(config: Mvp1Config) -> None:
    if not config.video_window_index.exists():
        raise FileNotFoundError(
            f"Video window index not found: {config.video_window_index.as_posix()}"
        )
    if not config.zone_config.exists():
        raise FileNotFoundError(
            f"Zone config not found: {config.zone_config.as_posix()}"
        )
    if not config.preprocess_root.exists():
        raise FileNotFoundError(
            f"Preprocess root not found: {config.preprocess_root.as_posix()}"
        )
    if config.max_windows is not None and config.max_windows <= 0:
        raise ValueError("selection.max_windows must be greater than zero when provided.")
    if config.max_windows_per_session is not None and config.max_windows_per_session <= 0:
        raise ValueError("selection.max_windows_per_session must be greater than zero when provided.")
    if config.frame_sample_rate <= 0:
        raise ValueError("video.frame_sample_rate must be greater than zero.")
    if config.resize_width <= 0:
        raise ValueError("video.resize_width must be greater than zero.")
    if config.preview_frame_position not in ALLOWED_PREVIEW_POSITIONS:
        raise ValueError(
            "video.preview_frame_position must be one of first, middle, or last."
        )
    if config.zone_layout not in ALLOWED_LAYOUTS:
        raise ValueError("zones.layout must be one of 2x2 or 3x3.")
    if config.selection_strategy not in ALLOWED_SELECTION_STRATEGIES:
        raise ValueError("selection.selection_strategy must be one of even or all.")
    if config.optical_flow_method != "farneback":
        raise ValueError("Only optical_flow.method=farneback is supported in MVP 1.")
    if config.motion_threshold < 0:
        raise ValueError("optical_flow.motion_threshold must be non-negative.")


def _create_output_directories(config: Mvp1Config) -> None:
    for directory in (
        config.features_dir,
        config.unity_json_dir,
        config.plots_dir,
        config.sample_frames_dir,
        config.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
