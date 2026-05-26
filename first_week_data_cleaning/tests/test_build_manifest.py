from __future__ import annotations

import unittest

from src.preprocess.build_manifest import (
    build_manifest_row,
    derive_path_context,
    normalize_token,
)
from src.preprocess.config import PreprocessingConfig


class BuildManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PreprocessingConfig()
        self.file_info = {
            "media_id": "video_000001",
            "media_type": "video",
            "file_path": "data/raw/free_range/room1/session_2025_09_01/video/GX010027.MP4",
            "relative_raw_path": "free_range/room1/session_2025_09_01/video/GX010027.MP4",
            "file_name": "GX010027.MP4",
            "file_extension": ".mp4",
            "file_size_bytes": 12345,
        }

    def test_derive_path_context_uses_matching_layout(self) -> None:
        context = derive_path_context(
            self.file_info["relative_raw_path"],
            self.config.path_layouts,
            default_modality="video",
        )
        self.assertEqual(context["system_id"], "free_range")
        self.assertEqual(context["room_id"], "room1")
        self.assertEqual(context["session_id"], "session_2025_09_01")
        self.assertEqual(context["modality"], "video")

    def test_derive_path_context_supports_missing_system_level(self) -> None:
        context = derive_path_context(
            "room1/session_2025_09_01/video/GX010027.MP4",
            self.config.path_layouts,
            default_modality="video",
        )
        self.assertIsNone(context["system_id"])
        self.assertEqual(context["room_id"], "room1")
        self.assertEqual(context["session_id"], "session_2025_09_01")
        self.assertEqual(context["modality"], "video")

    def test_derive_path_context_supports_modality_first_layout(self) -> None:
        context = derive_path_context(
            "video/room1/session_2025_09_01/GX010027.MP4",
            self.config.path_layouts,
            default_modality="video",
        )
        self.assertIsNone(context["system_id"])
        self.assertEqual(context["room_id"], "room1")
        self.assertEqual(context["session_id"], "session_2025_09_01")
        self.assertEqual(context["modality"], "video")

    def test_mapping_csv_overrides_path_values(self) -> None:
        row = build_manifest_row(
            file_info=self.file_info,
            exif_metadata={
                "CreateDate": "2025:09:01 12:00:00",
                "Duration": "58",
            },
            ffprobe_metadata={
                "format": {"duration": "60"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30000/1001",
                        "codec_name": "h264",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                    },
                ],
            },
            config=self.config,
            mapping_rows=[
                {
                    "pattern": r"free_range/room1",
                    "system_id": "Free Range A",
                    "room_id": "Room 7",
                    "session_id": "Morning",
                    "modality": "video",
                    "zone_config_id": "zone_alpha",
                }
            ],
            exif_warnings=[],
            ffprobe_warnings=[],
        )
        self.assertEqual(row["system_id"], "free_range_a")
        self.assertEqual(row["room_id"], "room_7")
        self.assertEqual(row["session_id"], "morning")
        self.assertEqual(row["zone_config_id"], "zone_alpha")
        self.assertEqual(row["duration_seconds"], 60.0)
        self.assertEqual(row["has_audio"], True)
        self.assertIn("mapping_override", row["warnings"].split(";"))

    def test_missing_system_defaults_to_unknown(self) -> None:
        row = build_manifest_row(
            file_info={
                **self.file_info,
                "file_path": "data/raw/room1/session_2025_09_01/video/GX010027.MP4",
                "relative_raw_path": "room1/session_2025_09_01/video/GX010027.MP4",
            },
            exif_metadata={
                "CreateDate": "2025:09:01 12:00:00",
                "Duration": "120",
            },
            ffprobe_metadata={
                "format": {"duration": "120"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1280,
                        "height": 720,
                        "avg_frame_rate": "25/1",
                        "codec_name": "h265",
                    }
                ],
            },
            config=self.config,
            mapping_rows=[],
            exif_warnings=[],
            ffprobe_warnings=[],
        )
        self.assertEqual(row["system_id"], "unknown_system")
        self.assertEqual(row["has_audio"], False)
        warnings = row["warnings"].split(";")
        self.assertIn("unknown_system", warnings)
        self.assertIn("no_embedded_audio_stream", warnings)
        self.assertEqual(row["quality_status"], "warning")

    def test_normalize_token(self) -> None:
        self.assertEqual(normalize_token("Room 1"), "room_1")


if __name__ == "__main__":
    unittest.main()
