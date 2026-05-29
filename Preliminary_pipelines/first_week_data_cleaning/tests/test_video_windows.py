from __future__ import annotations

import unittest

from src.preprocess.build_video_windows import build_video_windows


class VideoWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_rows = [
            {
                "media_id": "video_000001",
                "system_id": "free_range",
                "room_id": "room1",
                "session_id": "session_2025_09_01",
                "zone_config_id": "zone_alpha",
                "file_path": "data/raw/free_range/room1/session_2025_09_01/video/GX010027.MP4",
                "start_time": "2025-09-01T10:00:00-03:00",
                "duration_seconds": 60.0,
                "has_audio": True,
                "quality_status": "ok",
                "warnings": "",
            }
        ]

    def test_window_generation_uses_single_video_timeline(self) -> None:
        windows = build_video_windows(
            self.manifest_rows,
            window_seconds=30.0,
            stride_seconds=10.0,
        )
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0]["video_start_offset_sec"], 0.0)
        self.assertEqual(windows[-1]["video_start_offset_sec"], 30.0)
        self.assertEqual(windows[-1]["end_time"], "2025-09-01T10:01:00-03:00")

    def test_zone_config_and_audio_flags_are_propagated(self) -> None:
        windows = build_video_windows(
            self.manifest_rows,
            window_seconds=30.0,
            stride_seconds=10.0,
        )
        self.assertEqual(windows[0]["zone_config_id"], "zone_alpha")
        self.assertEqual(windows[0]["has_audio"], True)

    def test_rows_without_time_or_duration_produce_no_windows(self) -> None:
        windows = build_video_windows(
            [
                {
                    **self.manifest_rows[0],
                    "start_time": "",
                    "duration_seconds": "",
                }
            ],
            window_seconds=30.0,
            stride_seconds=10.0,
        )
        self.assertEqual(windows, [])


if __name__ == "__main__":
    unittest.main()
