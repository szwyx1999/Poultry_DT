from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from mvp_audio_env_context.src.config import AudioEnvContextConfig
from mvp_audio_env_context.src.env_loader import load_environment_daily


def test_load_environment_daily_normalizes_columns() -> None:
    tmp_path = Path(__file__).resolve().parent / "_tmp_env_loader"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    workbook_path = tmp_path / "Combined Room 1.xlsx"
    pd.DataFrame(
        [
            {
                "Date": "2025-08-16",
                "Temp_AM_Min": 24.0,
                "Temp_AM_Max": 28.0,
                "Temp_PM": 27.0,
                "RH_AM": 40.0,
                "RH_PM": 50.0,
            }
        ]
    ).to_excel(workbook_path, index=False)

    output_root = tmp_path / "outputs"
    config = AudioEnvContextConfig(
        workspace_root=tmp_path,
        package_root=tmp_path / "mvp_audio_env_context",
        video_window_index_csv=tmp_path / "video_window_index.csv",
        media_manifest_csv=tmp_path / "media_manifest.csv",
        selected_windows_csv=tmp_path / "selected_windows.csv",
        video_zone_features_csv=tmp_path / "video_zone_features.csv",
        biomarker_window_table_csv=tmp_path / "biomarker_window_table.csv",
        hmm_state_sequence_csv=tmp_path / "hmm_state_sequence.csv",
        event_log_csv=tmp_path / "event_log.csv",
        env_xlsx=workbook_path,
        output_dir=output_root,
        features_dir=output_root / "features",
        reports_dir=output_root / "reports",
        plots_dir=output_root / "plots",
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
    config.features_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result = load_environment_daily(config)
    row = result.env_df.iloc[0]
    assert row["temp_am_mean"] == 26.0
    assert row["temp_daily_mean"] == 26.5
    assert row["rh_daily_mean"] == 45.0
    assert row["temp_daily_range"] == 4.0
    assert row["env_quality_flag"] == "ok"
