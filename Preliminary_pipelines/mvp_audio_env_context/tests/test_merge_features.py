from __future__ import annotations

import pandas as pd

from mvp_audio_env_context.src.config import AudioEnvContextConfig
from mvp_audio_env_context.src.merge_features import _attach_environment_context, _attach_event_labels


def _test_config() -> AudioEnvContextConfig:
    from pathlib import Path

    root = Path(".")
    return AudioEnvContextConfig(
        workspace_root=root,
        package_root=root / "mvp_audio_env_context",
        video_window_index_csv=root / "a.csv",
        media_manifest_csv=root / "b.csv",
        selected_windows_csv=root / "c.csv",
        video_zone_features_csv=root / "d.csv",
        biomarker_window_table_csv=root / "e.csv",
        hmm_state_sequence_csv=root / "f.csv",
        event_log_csv=root / "g.csv",
        env_xlsx=root / "h.xlsx",
        output_dir=root / "outputs",
        features_dir=root / "outputs" / "features",
        reports_dir=root / "outputs" / "reports",
        plots_dir=root / "outputs" / "plots",
        ffmpeg_path="ffmpeg",
        audio_sample_rate=22050,
        audio_frame_seconds=1.0,
        audio_cache_enabled=True,
        audio_force_recompute=False,
        audio_max_report_failures=10,
        baseline_minutes_before_event=10,
        recovery_minutes_after_event=20,
        event_overlap_min_seconds=1.0,
        env_am_hour_cutoff=12,
        enable_hmm_ablation=False,
        hmm_n_states="auto",
        max_hmm_states=4,
        covariance_type="diag",
        random_state=42,
    )


def test_attach_environment_context_selects_am_and_pm() -> None:
    merged_df = pd.DataFrame(
        [
            {"start_time": "2025-08-17T08:00:00-03:00", "end_time": "2025-08-17T08:00:30-03:00"},
            {"start_time": "2025-08-17T15:00:00-03:00", "end_time": "2025-08-17T15:00:30-03:00"},
        ]
    )
    merged_df["start_time_dt"] = pd.to_datetime(merged_df["start_time"], utc=False)
    merged_df["end_time_dt"] = pd.to_datetime(merged_df["end_time"], utc=False)
    env_df = pd.DataFrame(
        [
            {
                "date": "2025-08-17",
                "temp_am_mean": 25.0,
                "temp_pm": 27.0,
                "temp_daily_mean": 26.0,
                "rh_am": 40.0,
                "rh_pm": 50.0,
                "rh_daily_mean": 45.0,
                "temp_daily_range": 2.0,
                "env_quality_flag": "ok",
            }
        ]
    )

    result_df = _attach_environment_context(merged_df, env_df, _test_config())
    assert float(result_df.loc[0, "temp_context"]) == 25.0
    assert float(result_df.loc[1, "temp_context"]) == 27.0
    assert float(result_df.loc[0, "rh_context"]) == 40.0
    assert float(result_df.loc[1, "rh_context"]) == 50.0


def test_attach_event_labels_marks_overlap() -> None:
    merged_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "room_id": "room_1",
                "start_time": "2025-08-17T08:48:50-03:00",
                "end_time": "2025-08-17T08:49:20-03:00",
            }
        ]
    )
    merged_df["start_time_dt"] = pd.to_datetime(merged_df["start_time"], utc=False)
    merged_df["end_time_dt"] = pd.to_datetime(merged_df["end_time"], utc=False)
    event_log_df = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "event_type": "caretaker_entry",
                "room_id": "room_1",
                "event_start_time": "2025-08-17T08:49:00-03:00",
                "event_end_time": "2025-08-17T08:50:57-03:00",
            }
        ]
    )

    result_df = _attach_event_labels(merged_df, event_log_df, _test_config())
    assert result_df.loc[0, "event_phase"] == "during_entry"
    assert bool(result_df.loc[0, "overlaps_event"]) is True
