from __future__ import annotations

import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

from mvp_video_zone_heatmap.src.mvp1.config import load_config


CONFIG_TEXT = """project:
  name: poultry_mvp1_video_zone_heatmap
inputs:
  preprocess_root: first_week_data_cleaning
  video_window_index: first_week_data_cleaning/data/processed/metadata/video_window_index.csv
  zone_config: mvp_video_zone_heatmap/data/metadata/zone_config.json
outputs:
  features_dir: mvp_video_zone_heatmap/outputs/features
  unity_json_dir: mvp_video_zone_heatmap/outputs/unity_json
  plots_dir: mvp_video_zone_heatmap/outputs/plots
  sample_frames_dir: mvp_video_zone_heatmap/outputs/plots/sample_frames
selection:
  max_windows: 3
  room_id: null
  session_id: null
  require_quality_ok: false
video:
  frame_sample_rate: 2
  resize_width: 640
  preview_frame_position: middle
zones:
  layout: 2x2
optical_flow:
  method: farneback
  pyr_scale: 0.5
  levels: 3
  winsize: 15
  iterations: 3
  poly_n: 5
  poly_sigma: 1.2
  flags: 0
  motion_threshold: 1.0
export:
  normalize_activity: true
"""


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp_root = Path.cwd() / ".tmp_tests_mvp1"
        self.workspace_temp_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.workspace_temp_root / uuid.uuid4().hex
        (self.project_root / "first_week_data_cleaning/data/processed/metadata").mkdir(parents=True)
        (self.project_root / "mvp_video_zone_heatmap/configs").mkdir(parents=True)
        (self.project_root / "mvp_video_zone_heatmap/data/metadata").mkdir(parents=True)

        index_path = self.project_root / "first_week_data_cleaning/data/processed/metadata/video_window_index.csv"
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "window_id",
                    "media_id",
                    "system_id",
                    "room_id",
                    "session_id",
                    "start_time",
                    "end_time",
                    "duration_seconds",
                    "video_path",
                    "video_start_offset_sec",
                    "has_audio",
                    "zone_config_id",
                    "quality_status",
                    "warnings",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "window_id": "win_1",
                    "media_id": "media_1",
                    "system_id": "system_a",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "start_time": "2025-08-16T09:00:03-03:00",
                    "end_time": "2025-08-16T09:00:33-03:00",
                    "duration_seconds": "30.0",
                    "video_path": "data/raw/video/example.mp4",
                    "video_start_offset_sec": "0.0",
                    "has_audio": "True",
                    "zone_config_id": "default_zone",
                    "quality_status": "warning",
                    "warnings": "unknown_system",
                }
            )

        (self.project_root / "mvp_video_zone_heatmap/configs/mvp1_config.yaml").write_text(
            CONFIG_TEXT,
            encoding="utf-8",
        )
        (self.project_root / "mvp_video_zone_heatmap/data/metadata/zone_config.json").write_text(
            json.dumps(
                {
                    "room_id": "room1",
                    "layout": "2x2",
                    "zones": [
                        {"zone_id": "zone_A", "row": 0, "col": 0},
                        {"zone_id": "zone_B", "row": 0, "col": 1},
                        {"zone_id": "zone_C", "row": 1, "col": 0},
                        {"zone_id": "zone_D", "row": 1, "col": 1},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)
        shutil.rmtree(self.workspace_temp_root, ignore_errors=True)

    def test_load_config_resolves_inputs_and_creates_outputs(self) -> None:
        config = load_config(
            project_root=self.project_root,
            config_path="mvp_video_zone_heatmap/configs/mvp1_config.yaml",
        )

        self.assertTrue(config.video_window_index.exists())
        self.assertTrue(config.zone_config.exists())
        self.assertTrue(config.features_dir.exists())
        self.assertTrue(config.unity_json_dir.exists())
        self.assertTrue(config.plots_dir.exists())
        self.assertTrue(config.sample_frames_dir.exists())
        self.assertEqual(config.preprocess_root, self.project_root / "first_week_data_cleaning")


if __name__ == "__main__":
    unittest.main()
