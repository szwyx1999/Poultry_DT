from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from mvp_video_zone_heatmap.src.mvp1.config import Mvp1Config
from mvp_video_zone_heatmap.src.mvp1.video_reader import WindowReadResult, save_preview_frame


class VideoReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp_root = Path.cwd() / ".tmp_tests_mvp1"
        self.workspace_temp_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.workspace_temp_root / uuid.uuid4().hex
        (self.project_root / "plots/sample_frames").mkdir(parents=True)

        self.config = Mvp1Config(
            project_root=self.project_root,
            config_path=self.project_root / "config.yaml",
            project_name="poultry_mvp1_video_zone_heatmap",
            preprocess_root=self.project_root / "first_week_data_cleaning",
            video_window_index=self.project_root / "video_window_index.csv",
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

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)
        shutil.rmtree(self.workspace_temp_root, ignore_errors=True)

    def test_save_preview_frame_writes_image_for_real_frame_and_placeholder(self) -> None:
        frame_result = WindowReadResult(
            window_id="window_with_frame",
            resolved_video_path=Path("video.mp4"),
            frames=[np.zeros((32, 32, 3), dtype=np.uint8)],
            frame_offsets_sec=[0.0],
            warnings=[],
        )
        placeholder_result = WindowReadResult(
            window_id="window_without_frame",
            resolved_video_path=Path("missing.mp4"),
            frames=[],
            frame_offsets_sec=[],
            warnings=["missing_video_file"],
        )

        preview_path = save_preview_frame({"window_id": "window_with_frame", "warnings": ""}, frame_result, self.config)
        placeholder_path = save_preview_frame(
            {"window_id": "window_without_frame", "warnings": "missing_video_file"},
            placeholder_result,
            self.config,
        )

        self.assertTrue(preview_path.exists())
        self.assertTrue(placeholder_path.exists())


if __name__ == "__main__":
    unittest.main()
