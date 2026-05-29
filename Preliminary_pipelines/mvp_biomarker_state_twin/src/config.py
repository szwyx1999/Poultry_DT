from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = WORKSPACE_ROOT / "mvp_biomarker_state_twin"


@dataclass(frozen=True)
class BiomarkerTwinConfig:
    workspace_root: Path
    package_root: Path
    zone_features_csv: Path
    selected_windows_csv: Path
    mvp1_unity_json: Path
    mvp1_config_yaml: Path
    video_window_index_csv: Path
    media_manifest_csv: Path
    zone_config_json: Path
    event_log_candidates: tuple[Path, ...]
    generated_event_log_csv: Path
    output_dir: Path
    features_dir: Path
    plots_dir: Path
    plots_hmm_by_sequence_dir: Path
    reports_dir: Path
    model_dir: Path
    unity_json_dir: Path
    unity_copy_paths: tuple[Path, ...]
    max_windows: int | None
    max_windows_per_room: int | None
    max_windows_per_session: int | None
    min_windows_for_hmm: int
    auto_refresh_mvp1_if_needed: bool
    mvp1_refresh_window_target: int
    target_raw_subdir: str | None
    mobility_activity_weight: float
    mobility_transition_weight: float
    disturbance_peak_zscore: float
    fallback_max_events_per_room: int
    resilience_baseline_windows: int
    resilience_max_recovery_windows: int
    resilience_recovery_threshold_fraction: float
    caretaker_reference_video_path: Path
    caretaker_reference_session_id: str
    caretaker_event_id: str
    caretaker_event_type: str
    caretaker_entry_offset_sec: int
    caretaker_exit_offset_sec: int
    baseline_minutes_before_event: int
    recovery_minutes_after_event: int
    event_overlap_min_seconds: float
    hmm_n_states: str | int
    max_hmm_states: int
    covariance_type: str
    random_state: int
    add_missingness_indicators: bool
    min_room_windows_for_individual_model: int
    sustained_risk_windows: int
    risk_activity_mobility_weight: float
    risk_imbalance_weight: float
    risk_low_spatial_freedom_weight: float
    risk_resilience_pressure_weight: float
    risk_persistence_weight: float
    max_resilience_events: int


def load_config(config_path: str | Path | None = None) -> BiomarkerTwinConfig:
    resolved_path = Path(config_path) if config_path else PACKAGE_ROOT / "config" / "default.yaml"
    if not resolved_path.is_absolute():
        resolved_path = WORKSPACE_ROOT / resolved_path
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    paths = raw_config.get("paths", {})
    biomarker_window_selection = raw_config.get("biomarker_window_selection", {})
    biomarkers = raw_config.get("biomarkers", {})
    events = raw_config.get("events", {})
    model = raw_config.get("model", {})
    risk = raw_config.get("risk", {})
    plots = raw_config.get("plots", {})

    output_dir = _resolve_path(paths.get("package_output_dir", "mvp_biomarker_state_twin/outputs"))
    return BiomarkerTwinConfig(
        workspace_root=WORKSPACE_ROOT,
        package_root=PACKAGE_ROOT,
        zone_features_csv=_resolve_path(paths.get("zone_features_csv", "mvp_video_zone_heatmap/outputs/features/video_zone_features.csv")),
        selected_windows_csv=_resolve_path(paths.get("selected_windows_csv", "mvp_video_zone_heatmap/outputs/features/selected_windows.csv")),
        mvp1_unity_json=_resolve_path(paths.get("mvp1_unity_json", "mvp_video_zone_heatmap/outputs/unity_json/mvp1_zone_activity_timeline.json")),
        mvp1_config_yaml=_resolve_path(paths.get("mvp1_config_yaml", "mvp_video_zone_heatmap/configs/mvp1_config.yaml")),
        video_window_index_csv=_resolve_path(paths.get("video_window_index_csv", "first_week_data_cleaning/data/processed/metadata/video_window_index.csv")),
        media_manifest_csv=_resolve_path(paths.get("media_manifest_csv", "first_week_data_cleaning/data/processed/metadata/media_manifest.csv")),
        zone_config_json=_resolve_path(paths.get("zone_config_json", "mvp_video_zone_heatmap/data/metadata/zone_config.json")),
        event_log_candidates=tuple(
            _resolve_path(path_value)
            for path_value in paths.get(
                "event_log_candidates",
                [
                    "data/metadata/event_log.csv",
                    "mvp_biomarker_state_twin/data/event_log.csv",
                    "first_week_data_cleaning/data/metadata/event_log.csv",
                ],
            )
        ),
        generated_event_log_csv=_resolve_path(paths.get("generated_event_log_csv", "mvp_biomarker_state_twin/data/event_log.csv")),
        output_dir=output_dir,
        features_dir=output_dir / "features",
        plots_dir=output_dir / "plots",
        plots_hmm_by_sequence_dir=output_dir / "plots" / "hmm_by_sequence",
        reports_dir=output_dir / "reports",
        model_dir=output_dir / "model",
        unity_json_dir=output_dir / "unity_json",
        unity_copy_paths=tuple(
            _resolve_path(path_value)
            for path_value in paths.get(
                "unity_copy_paths",
                [
                    "PoultryTwinDemo/demo1/Assets/StreamingAssets/poultry_twin_demo_timeline.json",
                ],
            )
        ),
        max_windows=_optional_int(biomarker_window_selection.get("max_windows")),
        max_windows_per_room=_optional_int(biomarker_window_selection.get("max_windows_per_room")),
        max_windows_per_session=_optional_int(biomarker_window_selection.get("max_windows_per_session")),
        min_windows_for_hmm=int(biomarker_window_selection.get("min_windows_for_hmm", 30)),
        auto_refresh_mvp1_if_needed=bool(biomarker_window_selection.get("auto_refresh_mvp1_if_needed", True)),
        mvp1_refresh_window_target=int(biomarker_window_selection.get("mvp1_refresh_window_target", 60)),
        target_raw_subdir=_optional_text(paths.get("target_raw_subdir")),
        mobility_activity_weight=float(biomarkers.get("mobility_activity_weight", 0.7)),
        mobility_transition_weight=float(biomarkers.get("mobility_transition_weight", 0.3)),
        disturbance_peak_zscore=float(biomarkers.get("disturbance_peak_zscore", 1.0)),
        fallback_max_events_per_room=int(biomarkers.get("fallback_max_events_per_room", 3)),
        resilience_baseline_windows=int(biomarkers.get("resilience_baseline_windows", 2)),
        resilience_max_recovery_windows=int(biomarkers.get("resilience_max_recovery_windows", 6)),
        resilience_recovery_threshold_fraction=float(biomarkers.get("resilience_recovery_threshold_fraction", 0.2)),
        caretaker_reference_video_path=_resolve_path(
            events.get(
                "caretaker_reference_video_path",
                "first_week_data_cleaning/data/raw/Caretaker Entry Video/Room 1/Week 11/17 Aug/GX330044.MP4",
            )
        ),
        caretaker_reference_session_id=str(events.get("caretaker_reference_session_id", "caretaker_entry_week_11_17_aug")),
        caretaker_event_id=str(events.get("caretaker_event_id", "caretaker_entry_room1_week11_2025_08_17")),
        caretaker_event_type=str(events.get("caretaker_event_type", "caretaker_entry")),
        caretaker_entry_offset_sec=int(events.get("caretaker_entry_offset_sec", 183)),
        caretaker_exit_offset_sec=int(events.get("caretaker_exit_offset_sec", 300)),
        baseline_minutes_before_event=int(events.get("baseline_minutes_before_event", 10)),
        recovery_minutes_after_event=int(events.get("recovery_minutes_after_event", 20)),
        event_overlap_min_seconds=float(events.get("event_overlap_min_seconds", 1.0)),
        hmm_n_states=_optional_auto_or_int(model.get("hmm_n_states", "auto")),
        max_hmm_states=int(model.get("max_hmm_states", 4)),
        covariance_type=str(model.get("covariance_type", "diag")),
        random_state=int(model.get("random_state", 42)),
        add_missingness_indicators=bool(model.get("add_missingness_indicators", True)),
        min_room_windows_for_individual_model=int(model.get("min_room_windows_for_individual_model", 8)),
        sustained_risk_windows=int(model.get("sustained_risk_windows", 3)),
        risk_activity_mobility_weight=float(risk.get("activity_mobility_weight", 0.30)),
        risk_imbalance_weight=float(risk.get("imbalance_weight", 0.25)),
        risk_low_spatial_freedom_weight=float(risk.get("low_spatial_freedom_weight", 0.20)),
        risk_resilience_pressure_weight=float(risk.get("resilience_pressure_weight", 0.15)),
        risk_persistence_weight=float(risk.get("persistence_weight", 0.10)),
        max_resilience_events=int(plots.get("max_resilience_events", 4)),
    )


def config_as_dict(config: BiomarkerTwinConfig) -> dict[str, Any]:
    return {
        "workspace_root": str(config.workspace_root),
        "package_root": str(config.package_root),
        "zone_features_csv": str(config.zone_features_csv),
        "selected_windows_csv": str(config.selected_windows_csv),
        "mvp1_unity_json": str(config.mvp1_unity_json),
        "mvp1_config_yaml": str(config.mvp1_config_yaml),
        "video_window_index_csv": str(config.video_window_index_csv),
        "media_manifest_csv": str(config.media_manifest_csv),
        "zone_config_json": str(config.zone_config_json),
        "event_log_candidates": [str(path_value) for path_value in config.event_log_candidates],
        "generated_event_log_csv": str(config.generated_event_log_csv),
        "output_dir": str(config.output_dir),
        "unity_copy_paths": [str(path_value) for path_value in config.unity_copy_paths],
        "max_windows": config.max_windows,
        "max_windows_per_room": config.max_windows_per_room,
        "max_windows_per_session": config.max_windows_per_session,
        "min_windows_for_hmm": config.min_windows_for_hmm,
        "auto_refresh_mvp1_if_needed": config.auto_refresh_mvp1_if_needed,
        "mvp1_refresh_window_target": config.mvp1_refresh_window_target,
        "target_raw_subdir": config.target_raw_subdir,
        "mobility_activity_weight": config.mobility_activity_weight,
        "mobility_transition_weight": config.mobility_transition_weight,
        "disturbance_peak_zscore": config.disturbance_peak_zscore,
        "fallback_max_events_per_room": config.fallback_max_events_per_room,
        "resilience_baseline_windows": config.resilience_baseline_windows,
        "resilience_max_recovery_windows": config.resilience_max_recovery_windows,
        "resilience_recovery_threshold_fraction": config.resilience_recovery_threshold_fraction,
        "caretaker_reference_video_path": str(config.caretaker_reference_video_path),
        "caretaker_reference_session_id": config.caretaker_reference_session_id,
        "caretaker_event_id": config.caretaker_event_id,
        "caretaker_event_type": config.caretaker_event_type,
        "caretaker_entry_offset_sec": config.caretaker_entry_offset_sec,
        "caretaker_exit_offset_sec": config.caretaker_exit_offset_sec,
        "baseline_minutes_before_event": config.baseline_minutes_before_event,
        "recovery_minutes_after_event": config.recovery_minutes_after_event,
        "event_overlap_min_seconds": config.event_overlap_min_seconds,
        "hmm_n_states": config.hmm_n_states,
        "max_hmm_states": config.max_hmm_states,
        "covariance_type": config.covariance_type,
        "random_state": config.random_state,
        "add_missingness_indicators": config.add_missingness_indicators,
        "min_room_windows_for_individual_model": config.min_room_windows_for_individual_model,
        "sustained_risk_windows": config.sustained_risk_windows,
        "risk_activity_mobility_weight": config.risk_activity_mobility_weight,
        "risk_imbalance_weight": config.risk_imbalance_weight,
        "risk_low_spatial_freedom_weight": config.risk_low_spatial_freedom_weight,
        "risk_resilience_pressure_weight": config.risk_resilience_pressure_weight,
        "risk_persistence_weight": config.risk_persistence_weight,
        "max_resilience_events": config.max_resilience_events,
    }


def _resolve_path(path_value: str | Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return WORKSPACE_ROOT / path_obj


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_auto_or_int(value: object) -> str | int:
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        return int(normalized)
    return int(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
