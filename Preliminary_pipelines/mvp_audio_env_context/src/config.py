from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = WORKSPACE_ROOT / "mvp_audio_env_context"


@dataclass(frozen=True)
class AudioEnvContextConfig:
    workspace_root: Path
    package_root: Path
    video_window_index_csv: Path
    media_manifest_csv: Path
    selected_windows_csv: Path
    video_zone_features_csv: Path
    biomarker_window_table_csv: Path
    hmm_state_sequence_csv: Path
    event_log_csv: Path
    env_xlsx: Path
    output_dir: Path
    features_dir: Path
    reports_dir: Path
    plots_dir: Path
    ffmpeg_path: str
    audio_sample_rate: int
    audio_frame_seconds: float
    audio_cache_enabled: bool
    audio_force_recompute: bool
    audio_max_report_failures: int
    baseline_minutes_before_event: int
    recovery_minutes_after_event: int
    event_overlap_min_seconds: float
    env_am_hour_cutoff: int
    enable_hmm_ablation: bool
    hmm_n_states: str | int
    max_hmm_states: int
    covariance_type: str
    random_state: int


def load_config(config_path: str | Path | None = None) -> AudioEnvContextConfig:
    resolved_path = Path(config_path) if config_path else PACKAGE_ROOT / "config" / "default.yaml"
    if not resolved_path.is_absolute():
        resolved_path = WORKSPACE_ROOT / resolved_path

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    paths = raw_config.get("paths", {})
    audio = raw_config.get("audio", {})
    events = raw_config.get("events", {})
    analysis = raw_config.get("analysis", {})

    output_dir = _resolve_path(paths.get("output_dir", "mvp_audio_env_context/outputs"))
    return AudioEnvContextConfig(
        workspace_root=WORKSPACE_ROOT,
        package_root=PACKAGE_ROOT,
        video_window_index_csv=_resolve_path(paths.get("video_window_index_csv", "first_week_data_cleaning/data/processed/metadata/video_window_index.csv")),
        media_manifest_csv=_resolve_path(paths.get("media_manifest_csv", "first_week_data_cleaning/data/processed/metadata/media_manifest.csv")),
        selected_windows_csv=_resolve_path(paths.get("selected_windows_csv", "mvp_video_zone_heatmap/outputs/features/selected_windows.csv")),
        video_zone_features_csv=_resolve_path(paths.get("video_zone_features_csv", "mvp_video_zone_heatmap/outputs/features/video_zone_features.csv")),
        biomarker_window_table_csv=_resolve_path(paths.get("biomarker_window_table_csv", "mvp_biomarker_state_twin/outputs/features/biomarker_window_table.csv")),
        hmm_state_sequence_csv=_resolve_path(paths.get("hmm_state_sequence_csv", "mvp_biomarker_state_twin/outputs/features/hmm_state_sequence.csv")),
        event_log_csv=_resolve_path(paths.get("event_log_csv", "mvp_biomarker_state_twin/data/event_log.csv")),
        env_xlsx=_resolve_path(paths.get("env_xlsx", "first_week_data_cleaning/data/raw/env/Combined Room 1.xlsx")),
        output_dir=output_dir,
        features_dir=output_dir / "features",
        reports_dir=output_dir / "reports",
        plots_dir=output_dir / "plots",
        ffmpeg_path=str(audio.get("ffmpeg_path", "ffmpeg")),
        audio_sample_rate=int(audio.get("sample_rate", 22050)),
        audio_frame_seconds=float(audio.get("frame_seconds", 1.0)),
        audio_cache_enabled=bool(audio.get("cache_enabled", True)),
        audio_force_recompute=bool(audio.get("force_recompute", False)),
        audio_max_report_failures=int(audio.get("max_report_failures", 25)),
        baseline_minutes_before_event=int(events.get("baseline_minutes_before_event", 10)),
        recovery_minutes_after_event=int(events.get("recovery_minutes_after_event", 20)),
        event_overlap_min_seconds=float(events.get("event_overlap_min_seconds", 1.0)),
        env_am_hour_cutoff=int(events.get("env_am_hour_cutoff", 12)),
        enable_hmm_ablation=bool(analysis.get("enable_hmm_ablation", True)),
        hmm_n_states=_optional_auto_or_int(analysis.get("hmm_n_states", "auto")),
        max_hmm_states=int(analysis.get("max_hmm_states", 4)),
        covariance_type=str(analysis.get("covariance_type", "diag")),
        random_state=int(analysis.get("random_state", 42)),
    )


def ensure_output_dirs(config: AudioEnvContextConfig) -> None:
    for directory in (config.output_dir, config.features_dir, config.reports_dir, config.plots_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_path(path_value: str | Path) -> Path:
    path_obj = Path(path_value)
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
