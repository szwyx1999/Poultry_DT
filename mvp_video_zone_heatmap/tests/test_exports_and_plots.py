from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from mvp_video_zone_heatmap.src.mvp1.config import Mvp1Config
from mvp_video_zone_heatmap.src.mvp1.export_unity import export_unity_timeline
from mvp_video_zone_heatmap.src.mvp1.plots import generate_sanity_plots


class ExportAndPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp_root = Path.cwd() / ".tmp_tests_mvp1"
        self.workspace_temp_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.workspace_temp_root / uuid.uuid4().hex
        (self.project_root / "features").mkdir(parents=True)
        (self.project_root / "unity_json").mkdir(parents=True)
        (self.project_root / "plots/sample_frames").mkdir(parents=True)

        self.config = Mvp1Config(
            project_root=self.project_root,
            config_path=self.project_root / "config.yaml",
            project_name="poultry_mvp1_video_zone_heatmap",
            preprocess_root=self.project_root / "first_week_data_cleaning",
            video_window_index=self.project_root / "first_week_data_cleaning/data/processed/metadata/video_window_index.csv",
            zone_config=self.project_root / "zone_config.json",
            features_dir=self.project_root / "features",
            unity_json_dir=self.project_root / "unity_json",
            plots_dir=self.project_root / "plots",
            sample_frames_dir=self.project_root / "plots/sample_frames",
            max_windows=10,
            room_id=None,
            session_id=None,
            require_quality_ok=False,
            frame_sample_rate=2.0,
            resize_width=640,
            preview_frame_position="middle",
            zone_layout="2x2",
            optical_flow_method="farneback",
            optical_flow_params={
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            },
            motion_threshold=1.0,
            normalize_activity=True,
        )
        self.zone_config = {
            "room_id": "room1",
            "layout": "2x2",
            "zones": [
                {"zone_id": "zone_A", "row": 0, "col": 0},
                {"zone_id": "zone_B", "row": 0, "col": 1},
                {"zone_id": "zone_C", "row": 1, "col": 0},
                {"zone_id": "zone_D", "row": 1, "col": 1},
            ],
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)
        shutil.rmtree(self.workspace_temp_root, ignore_errors=True)

    def test_export_unity_timeline_normalizes_activity_and_preserves_nulls(self) -> None:
        features_df = pd.DataFrame(
            [
                {
                    "window_id": "window_1",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "zone_id": "zone_A",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "video_path": "path.mp4",
                    "video_start_offset_sec": 0.0,
                    "duration_seconds": 30.0,
                    "activity_mean": 0.0,
                    "activity_std": 0.0,
                    "motion_pixel_ratio": 0.0,
                    "frame_count": 5,
                    "warnings": "",
                },
                {
                    "window_id": "window_1",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "zone_id": "zone_B",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "video_path": "path.mp4",
                    "video_start_offset_sec": 0.0,
                    "duration_seconds": 30.0,
                    "activity_mean": None,
                    "activity_std": None,
                    "motion_pixel_ratio": None,
                    "frame_count": 0,
                    "warnings": "missing_video_file",
                },
                {
                    "window_id": "window_1",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "zone_id": "zone_C",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "video_path": "path.mp4",
                    "video_start_offset_sec": 0.0,
                    "duration_seconds": 30.0,
                    "activity_mean": 2.0,
                    "activity_std": 0.5,
                    "motion_pixel_ratio": 0.5,
                    "frame_count": 5,
                    "warnings": "",
                },
                {
                    "window_id": "window_1",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "zone_id": "zone_D",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "video_path": "path.mp4",
                    "video_start_offset_sec": 0.0,
                    "duration_seconds": 30.0,
                    "activity_mean": 4.0,
                    "activity_std": 1.0,
                    "motion_pixel_ratio": 1.0,
                    "frame_count": 5,
                    "warnings": "",
                },
            ]
        )

        output_path = export_unity_timeline(features_df, self.zone_config, self.config)

        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        zones = payload["timeline"][0]["zones"]
        self.assertEqual(zones[0]["activity"], 0.0)
        self.assertIsNone(zones[1]["activity"])
        self.assertEqual(zones[2]["activity"], 0.5)
        self.assertEqual(zones[3]["activity"], 1.0)

    def test_generate_plots_writes_placeholder_when_no_valid_activity_exists(self) -> None:
        features_df = pd.DataFrame(
            [
                {
                    "window_id": "window_1",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "zone_id": "zone_A",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "video_path": "path.mp4",
                    "video_start_offset_sec": 0.0,
                    "duration_seconds": 30.0,
                    "activity_mean": None,
                    "activity_std": None,
                    "motion_pixel_ratio": None,
                    "frame_count": 0,
                    "warnings": "missing_video_file",
                }
            ]
        )

        plot_paths = generate_sanity_plots(features_df, self.zone_config, self.config)

        self.assertTrue(plot_paths["zone_activity_over_time"].exists())
        self.assertTrue(plot_paths["zone_activity_heatmap"].exists())


if __name__ == "__main__":
    unittest.main()
