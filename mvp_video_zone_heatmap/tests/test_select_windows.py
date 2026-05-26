from __future__ import annotations

import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

from mvp_video_zone_heatmap.src.mvp1.config import load_config
from mvp_video_zone_heatmap.src.mvp1.select_windows import select_video_windows


CONFIG_TEMPLATE = """project:
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
  max_windows: {max_windows}
  room_id: {room_id}
  session_id: {session_id}
  require_quality_ok: {require_quality_ok}
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


class SelectWindowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp_root = Path.cwd() / ".tmp_tests_mvp1"
        self.workspace_temp_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.workspace_temp_root / uuid.uuid4().hex
        (self.project_root / "first_week_data_cleaning/data/processed/metadata").mkdir(parents=True)
        (self.project_root / "first_week_data_cleaning/data/raw/video").mkdir(parents=True)
        (self.project_root / "mvp_video_zone_heatmap/configs").mkdir(parents=True)
        (self.project_root / "mvp_video_zone_heatmap/data/metadata").mkdir(parents=True)

        self.index_path = self.project_root / "first_week_data_cleaning/data/processed/metadata/video_window_index.csv"
        with self.index_path.open("w", encoding="utf-8", newline="") as handle:
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
            for index in range(10):
                relative_video_path = f"data/raw/video/video_{index:02d}.mp4"
                (self.project_root / "first_week_data_cleaning" / relative_video_path).write_bytes(b"video")
                writer.writerow(
                    {
                        "window_id": f"window_{index + 1:02d}",
                        "media_id": f"media_{index + 1:02d}",
                        "system_id": "unknown_system",
                        "room_id": "room_1",
                        "session_id": "session_1",
                        "start_time": f"2025-08-16T09:{index:02d}:00-03:00",
                        "end_time": f"2025-08-16T09:{index:02d}:30-03:00",
                        "duration_seconds": "30.0",
                        "video_path": relative_video_path,
                        "video_start_offset_sec": str(index * 10.0),
                        "has_audio": "True",
                        "zone_config_id": "default_zone",
                        "quality_status": "warning",
                        "warnings": "unknown_system",
                    }
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

    def test_select_windows_evenly_spreads_across_timeline(self) -> None:
        config_path = self.project_root / "mvp_video_zone_heatmap/configs/mvp1_config.yaml"
        config_path.write_text(
            CONFIG_TEMPLATE.format(
                max_windows=3,
                room_id="null",
                session_id="null",
                require_quality_ok="false",
            ),
            encoding="utf-8",
        )
        config = load_config(self.project_root, config_path)

        selected_df = select_video_windows(config)

        self.assertEqual(selected_df["window_id"].tolist(), ["window_01", "window_06", "window_10"])
        self.assertEqual(selected_df["selection_rank"].tolist(), [1, 2, 3])
        self.assertTrue(selected_df["resolved_video_path"].iloc[0].endswith("video_00.mp4"))

    def test_require_quality_ok_can_yield_zero_selected_rows(self) -> None:
        config_path = self.project_root / "mvp_video_zone_heatmap/configs/mvp1_config.yaml"
        config_path.write_text(
            CONFIG_TEMPLATE.format(
                max_windows=3,
                room_id="null",
                session_id="null",
                require_quality_ok="true",
            ),
            encoding="utf-8",
        )
        config = load_config(self.project_root, config_path)

        selected_df = select_video_windows(config)

        self.assertTrue(selected_df.empty)
        self.assertTrue((config.features_dir / "selected_windows.csv").exists())


if __name__ == "__main__":
    unittest.main()
