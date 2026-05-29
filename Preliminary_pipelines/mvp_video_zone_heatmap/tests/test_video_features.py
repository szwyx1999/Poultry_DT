from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mvp_video_zone_heatmap.src.mvp1.config import Mvp1Config
from mvp_video_zone_heatmap.src.mvp1.video_features import extract_video_zone_features
from mvp_video_zone_heatmap.src.mvp1.video_reader import WindowReadResult


class FakeCv2:
    INTER_AREA = 3

    @staticmethod
    def resize(frame, dimensions, interpolation=None):
        target_width, target_height = dimensions
        if frame.shape[1] == target_width and frame.shape[0] == target_height:
            return frame
        y_indices = np.linspace(0, frame.shape[0] - 1, num=target_height).astype(int)
        x_indices = np.linspace(0, frame.shape[1] - 1, num=target_width).astype(int)
        return frame[y_indices][:, x_indices]

    @staticmethod
    def calcOpticalFlowFarneback(previous, current, _unused, *args):
        difference = np.abs(current.astype(np.float32) - previous.astype(np.float32)) / 255.0
        flow = np.zeros((current.shape[0], current.shape[1], 2), dtype=np.float32)
        flow[..., 0] = difference
        return flow


class VideoFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Mvp1Config(
            project_root=Path("."),
            config_path=Path("mvp1_config.yaml"),
            project_name="poultry_mvp1_video_zone_heatmap",
            preprocess_root=Path("first_week_data_cleaning"),
            video_window_index=Path("video_window_index.csv"),
            zone_config=Path("zone_config.json"),
            features_dir=Path("features"),
            unity_json_dir=Path("unity_json"),
            plots_dir=Path("plots"),
            sample_frames_dir=Path("plots/sample_frames"),
            max_windows=10,
            room_id=None,
            session_id=None,
            require_quality_ok=False,
            frame_sample_rate=2.0,
            resize_width=100,
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
            motion_threshold=0.2,
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

    def test_extract_video_zone_features_handles_valid_and_failed_windows(self) -> None:
        frame_1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_2 = frame_1.copy()
        frame_2[50:100, 50:100, :] = 255
        frame_3 = frame_2.copy()
        frame_3[50:100, 50:100, :] = 128

        selected_windows = pd.DataFrame(
            [
                {
                    "window_id": "window_valid",
                    "media_id": "media_1",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "start_time": "2025-08-16T09:00:00-03:00",
                    "end_time": "2025-08-16T09:00:30-03:00",
                    "duration_seconds": 30.0,
                    "video_path": "data/raw/video_1.mp4",
                    "video_start_offset_sec": 0.0,
                    "warnings": "",
                },
                {
                    "window_id": "window_failed",
                    "media_id": "media_2",
                    "system_id": "system_1",
                    "room_id": "room_1",
                    "session_id": "session_1",
                    "start_time": "2025-08-16T09:00:30-03:00",
                    "end_time": "2025-08-16T09:01:00-03:00",
                    "duration_seconds": 30.0,
                    "video_path": "data/raw/video_2.mp4",
                    "video_start_offset_sec": 30.0,
                    "warnings": "missing_video_file",
                },
            ]
        )
        read_results = {
            "window_valid": WindowReadResult(
                window_id="window_valid",
                resolved_video_path=Path("video_1.mp4"),
                frames=[frame_1, frame_2, frame_3],
                frame_offsets_sec=[0.0, 0.5, 1.0],
                warnings=[],
                source_fps=2.0,
            ),
            "window_failed": WindowReadResult(
                window_id="window_failed",
                resolved_video_path=Path("video_2.mp4"),
                frames=[],
                frame_offsets_sec=[],
                warnings=["missing_video_file"],
                source_fps=None,
            ),
        }

        features_df = extract_video_zone_features(
            selected_windows=selected_windows,
            read_results=read_results,
            zone_config=self.zone_config,
            config=self.config,
            cv2_module=FakeCv2(),
        )

        self.assertEqual(len(features_df), 8)
        valid_rows = features_df[features_df["window_id"] == "window_valid"]
        failed_rows = features_df[features_df["window_id"] == "window_failed"]
        zone_d_activity = valid_rows.loc[valid_rows["zone_id"] == "zone_D", "activity_mean"].iloc[0]
        zone_a_activity = valid_rows.loc[valid_rows["zone_id"] == "zone_A", "activity_mean"].iloc[0]

        self.assertGreater(zone_d_activity, zone_a_activity)
        self.assertTrue(failed_rows["activity_mean"].isna().all())
        self.assertTrue(
            failed_rows["warnings"].str.contains("insufficient_frames_for_optical_flow").all()
        )


if __name__ == "__main__":
    unittest.main()
