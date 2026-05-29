from __future__ import annotations

from pathlib import Path

import pandas as pd

from mvp_biomarker_state_twin.src.config import BiomarkerTwinConfig
from mvp_biomarker_state_twin.src.pipeline import run_pipeline


def test_pipeline_runs_on_synthetic_inputs(tmp_path: Path) -> None:
    workspace_root = tmp_path
    package_root = workspace_root / "mvp_biomarker_state_twin"
    outputs_root = package_root / "outputs"
    inputs_root = workspace_root / "synthetic_inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)

    zone_features_path = inputs_root / "video_zone_features.csv"
    selected_windows_path = inputs_root / "selected_windows.csv"
    window_index_path = inputs_root / "video_window_index.csv"
    media_manifest_path = inputs_root / "media_manifest.csv"
    zone_config_path = inputs_root / "zone_config.json"

    zone_features_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "zone_id": "zone_A",
                "start_time": "2025-08-16T09:00:00-03:00",
                "end_time": "2025-08-16T09:00:30-03:00",
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 0.0,
                "duration_seconds": 30.0,
                "activity_mean": 0.10,
                "activity_std": 0.01,
                "motion_pixel_ratio": 0.05,
                "frame_count": 60,
                "warnings": "",
            },
            {
                "window_id": "w1",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "zone_id": "zone_B",
                "start_time": "2025-08-16T09:00:00-03:00",
                "end_time": "2025-08-16T09:00:30-03:00",
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 0.0,
                "duration_seconds": 30.0,
                "activity_mean": 0.20,
                "activity_std": 0.01,
                "motion_pixel_ratio": 0.06,
                "frame_count": 60,
                "warnings": "",
            },
            {
                "window_id": "w2",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "zone_id": "zone_A",
                "start_time": "2025-08-16T09:05:00-03:00",
                "end_time": "2025-08-16T09:05:30-03:00",
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 300.0,
                "duration_seconds": 30.0,
                "activity_mean": 0.30,
                "activity_std": 0.02,
                "motion_pixel_ratio": 0.10,
                "frame_count": 60,
                "warnings": "",
            },
            {
                "window_id": "w2",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "zone_id": "zone_B",
                "start_time": "2025-08-16T09:05:00-03:00",
                "end_time": "2025-08-16T09:05:30-03:00",
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 300.0,
                "duration_seconds": 30.0,
                "activity_mean": 0.80,
                "activity_std": 0.03,
                "motion_pixel_ratio": 0.18,
                "frame_count": 60,
                "warnings": "",
            },
        ]
    )
    zone_features_df.to_csv(zone_features_path, index=False)

    selected_windows_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "start_time": "2025-08-16T09:00:00-03:00",
                "end_time": "2025-08-16T09:00:30-03:00",
                "duration_seconds": 30.0,
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 0.0,
                "has_audio": False,
                "zone_config_id": "default_zone",
                "quality_status": "ok",
                "warnings": "",
                "selection_rank": 1,
            },
            {
                "window_id": "w2",
                "media_id": "m1",
                "system_id": "sys1",
                "room_id": "room_1",
                "session_id": "session_1",
                "start_time": "2025-08-16T09:05:00-03:00",
                "end_time": "2025-08-16T09:05:30-03:00",
                "duration_seconds": 30.0,
                "video_path": "synthetic.mp4",
                "video_start_offset_sec": 300.0,
                "has_audio": False,
                "zone_config_id": "default_zone",
                "quality_status": "ok",
                "warnings": "",
                "selection_rank": 2,
            },
        ]
    )
    selected_windows_df.to_csv(selected_windows_path, index=False)
    selected_windows_df.to_csv(window_index_path, index=False)
    pd.DataFrame([{"media_id": "m1", "zone_config_id": "default_zone"}]).to_csv(media_manifest_path, index=False)
    zone_config_path.write_text(
        '{"room_id":"room_1","zones":[{"zone_id":"zone_A","row":0,"col":0},{"zone_id":"zone_B","row":0,"col":1}]}',
        encoding="utf-8",
    )

    config = BiomarkerTwinConfig(
        workspace_root=workspace_root,
        package_root=package_root,
        zone_features_csv=zone_features_path,
        selected_windows_csv=selected_windows_path,
        mvp1_unity_json=inputs_root / "mvp1.json",
        mvp1_config_yaml=inputs_root / "mvp1_config.yaml",
        video_window_index_csv=window_index_path,
        media_manifest_csv=media_manifest_path,
        zone_config_json=zone_config_path,
        event_log_candidates=tuple(),
        generated_event_log_csv=package_root / "data" / "event_log.csv",
        output_dir=outputs_root,
        features_dir=outputs_root / "features",
        plots_dir=outputs_root / "plots",
        plots_hmm_by_sequence_dir=outputs_root / "plots" / "hmm_by_sequence",
        reports_dir=outputs_root / "reports",
        model_dir=outputs_root / "model",
        unity_json_dir=outputs_root / "unity_json",
        unity_copy_paths=(outputs_root / "unity_json" / "copied.json",),
        max_windows=None,
        max_windows_per_room=None,
        max_windows_per_session=None,
        min_windows_for_hmm=30,
        auto_refresh_mvp1_if_needed=False,
        mvp1_refresh_window_target=60,
        target_raw_subdir=None,
        mobility_activity_weight=0.7,
        mobility_transition_weight=0.3,
        disturbance_peak_zscore=0.5,
        fallback_max_events_per_room=2,
        resilience_baseline_windows=1,
        resilience_max_recovery_windows=2,
        resilience_recovery_threshold_fraction=0.2,
        caretaker_reference_video_path=inputs_root / "reference.mp4",
        caretaker_reference_session_id="caretaker_entry_week_11_17_aug",
        caretaker_event_id="caretaker_entry_room1_week11_2025_08_17",
        caretaker_event_type="caretaker_entry",
        caretaker_entry_offset_sec=183,
        caretaker_exit_offset_sec=300,
        baseline_minutes_before_event=10,
        recovery_minutes_after_event=20,
        event_overlap_min_seconds=1.0,
        hmm_n_states=2,
        max_hmm_states=4,
        covariance_type="diag",
        random_state=42,
        add_missingness_indicators=True,
        min_room_windows_for_individual_model=4,
        sustained_risk_windows=2,
        risk_activity_mobility_weight=0.30,
        risk_imbalance_weight=0.25,
        risk_low_spatial_freedom_weight=0.20,
        risk_resilience_pressure_weight=0.15,
        risk_persistence_weight=0.10,
        max_resilience_events=2,
    )

    outputs = run_pipeline(config)
    assert Path(outputs["canonical_zone_feature_table"]).exists()
    assert Path(outputs["biomarker_window_table"]).exists()
    assert Path(outputs["hmm_state_sequence"]).exists()
    assert Path(outputs["data_coverage_report"]).exists()
    assert Path(outputs["unity_json"]).exists()
