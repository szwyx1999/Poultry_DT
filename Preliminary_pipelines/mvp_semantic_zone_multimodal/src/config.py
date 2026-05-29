from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = WORKSPACE_ROOT / "mvp_semantic_zone_multimodal"


@dataclass(frozen=True)
class SemanticZoneConfig:
    workspace_root: Path
    package_root: Path
    reference_image: Path
    annotated_image: Path
    manual_zone_yaml: Path
    video_window_index_csv: Path
    media_manifest_csv: Path
    selected_windows_csv: Path
    previous_fourzone_biomarker_csv: Path
    previous_fourzone_event_summary_csv: Path
    existing_audio_window_features_csv: Path
    event_log_csv: Path
    env_xlsx: Path
    output_dir: Path
    zones_dir: Path
    features_dir: Path
    reports_dir: Path
    plots_dir: Path
    cache_dir: Path
    unity_json_dir: Path
    target_raw_subdir: str
    target_room_id: str
    frame_sample_rate: float
    resize_width: int
    activity_method: str
    video_cache_enabled: bool
    force_recompute: bool
    force_invalid_zones: bool
    ffmpeg_path: str
    audio_sample_rate: int
    audio_frame_seconds: float
    audio_cache_enabled: bool
    audio_force_recompute: bool
    audio_max_report_failures: int
    caretaker_reference_video_path: Path
    caretaker_event_id: str
    caretaker_event_type: str
    caretaker_entry_offset_sec: int
    caretaker_exit_offset_sec: int
    baseline_minutes_before_event: int
    recovery_minutes_after_event: int
    event_overlap_min_seconds: float
    env_am_hour_cutoff: int
    primary_model_name: str
    hmm_n_states: str | int
    max_hmm_states: int
    covariance_type: str
    random_state: int


def load_config(config_path: str | Path | None = None, force: bool = False) -> SemanticZoneConfig:
    resolved_path = Path(config_path) if config_path else PACKAGE_ROOT / "config" / "default.yaml"
    if not resolved_path.is_absolute():
        resolved_path = WORKSPACE_ROOT / resolved_path

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    paths = raw_config.get("paths", {})
    selection = raw_config.get("selection", {})
    video = raw_config.get("video", {})
    audio = raw_config.get("audio", {})
    events = raw_config.get("events", {})
    model = raw_config.get("model", {})

    output_dir = _resolve_path(paths.get("output_dir", "mvp_semantic_zone_multimodal/outputs"))
    return SemanticZoneConfig(
        workspace_root=WORKSPACE_ROOT,
        package_root=PACKAGE_ROOT,
        reference_image=_resolve_path(paths.get("reference_image")),
        annotated_image=_resolve_path(paths.get("annotated_image")),
        manual_zone_yaml=_resolve_path(paths.get("manual_zone_yaml")),
        video_window_index_csv=_resolve_path(paths.get("video_window_index_csv")),
        media_manifest_csv=_resolve_path(paths.get("media_manifest_csv")),
        selected_windows_csv=_resolve_path(paths.get("selected_windows_csv")),
        previous_fourzone_biomarker_csv=_resolve_path(paths.get("previous_fourzone_biomarker_csv")),
        previous_fourzone_event_summary_csv=_resolve_path(paths.get("previous_fourzone_event_summary_csv")),
        existing_audio_window_features_csv=_resolve_path(paths.get("existing_audio_window_features_csv")),
        event_log_csv=_resolve_path(paths.get("event_log_csv")),
        env_xlsx=_resolve_path(paths.get("env_xlsx")),
        output_dir=output_dir,
        zones_dir=output_dir / "zones",
        features_dir=output_dir / "features",
        reports_dir=output_dir / "reports",
        plots_dir=output_dir / "plots",
        cache_dir=output_dir / "cache",
        unity_json_dir=output_dir / "unity_json",
        target_raw_subdir=str(selection.get("target_raw_subdir", "video/Room 1/Room 1 (16, 17 Aug)")),
        target_room_id=str(selection.get("target_room_id", "room_1")),
        frame_sample_rate=float(video.get("frame_sample_rate", 0.5)),
        resize_width=int(video.get("resize_width", 256)),
        activity_method=str(video.get("activity_method", "frame_difference")),
        video_cache_enabled=bool(video.get("cache_enabled", True)),
        force_recompute=bool(video.get("force_recompute", False) or force),
        force_invalid_zones=bool(video.get("force_invalid_zones", False) or force),
        ffmpeg_path=str(audio.get("ffmpeg_path", "ffmpeg")),
        audio_sample_rate=int(audio.get("sample_rate", 22050)),
        audio_frame_seconds=float(audio.get("frame_seconds", 1.0)),
        audio_cache_enabled=bool(audio.get("cache_enabled", True)),
        audio_force_recompute=bool(audio.get("force_recompute", False) or force),
        audio_max_report_failures=int(audio.get("max_report_failures", 25)),
        caretaker_reference_video_path=_resolve_path(events.get("caretaker_reference_video_path")),
        caretaker_event_id=str(events.get("caretaker_event_id", "caretaker_entry_room1_week11_2025_08_17")),
        caretaker_event_type=str(events.get("caretaker_event_type", "caretaker_entry")),
        caretaker_entry_offset_sec=int(events.get("caretaker_entry_offset_sec", 183)),
        caretaker_exit_offset_sec=int(events.get("caretaker_exit_offset_sec", 300)),
        baseline_minutes_before_event=int(events.get("baseline_minutes_before_event", 10)),
        recovery_minutes_after_event=int(events.get("recovery_minutes_after_event", 20)),
        event_overlap_min_seconds=float(events.get("event_overlap_min_seconds", 1.0)),
        env_am_hour_cutoff=int(events.get("env_am_hour_cutoff", 12)),
        primary_model_name=str(model.get("primary_model_name", "semantic_video_only")),
        hmm_n_states=_optional_auto_or_int(model.get("hmm_n_states", "auto")),
        max_hmm_states=int(model.get("max_hmm_states", 4)),
        covariance_type=str(model.get("covariance_type", "diag")),
        random_state=int(model.get("random_state", 42)),
    )


def ensure_output_dirs(config: SemanticZoneConfig) -> None:
    for directory in (
        config.output_dir,
        config.zones_dir,
        config.features_dir,
        config.reports_dir,
        config.plots_dir,
        config.cache_dir,
        config.unity_json_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_path(path_value: str | Path | None) -> Path:
    path_obj = Path(path_value) if path_value is not None else Path()
    if path_obj.is_absolute():
        return path_obj
    return WORKSPACE_ROOT / path_obj


def _optional_auto_or_int(value: object) -> str | int:
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        return int(normalized)
    return int(value)
