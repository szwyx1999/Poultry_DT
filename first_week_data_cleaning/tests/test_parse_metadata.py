from __future__ import annotations

import unittest

from src.preprocess.parse_metadata import (
    extract_ffprobe_stream_metadata,
    parse_best_timestamp,
    parse_duration_value,
    parse_exif_datetime,
)


class ParseMetadataTests(unittest.TestCase):
    def test_parse_colon_timestamp_localizes_timezone(self) -> None:
        parsed = parse_exif_datetime("2026:05:12 15:21:00", "America/Halifax")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-05-12T15:21:00-03:00")

    def test_parse_offset_timestamp_converts_timezone(self) -> None:
        parsed = parse_exif_datetime("2026:05:12 18:21:00+00:00", "America/Halifax")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-05-12T15:21:00-03:00")

    def test_best_timestamp_uses_priority_order(self) -> None:
        parsed, source, warning = parse_best_timestamp(
            {
                "MediaCreateDate": "2026:05:12 15:21:00",
                "CreateDate": "2026:05:12 15:20:00",
            },
            ["CreateDate", "MediaCreateDate"],
            "America/Halifax",
        )
        self.assertIsNone(warning)
        self.assertEqual(source, "CreateDate")
        self.assertEqual(parsed.isoformat(), "2026-05-12T15:20:00-03:00")

    def test_parse_duration_supports_seconds(self) -> None:
        self.assertEqual(parse_duration_value("600.5 s"), 600.5)

    def test_parse_duration_supports_hh_mm_ss(self) -> None:
        self.assertEqual(parse_duration_value("0:10:00"), 600.0)

    def test_parse_duration_supports_mm_ss(self) -> None:
        self.assertEqual(parse_duration_value("10:30"), 630.0)

    def test_extract_ffprobe_stream_metadata_parses_video_and_audio(self) -> None:
        metadata, warnings = extract_ffprobe_stream_metadata(
            {
                "format": {"duration": "60.25"},
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
            }
        )
        self.assertEqual(metadata["width"], 1920)
        self.assertEqual(metadata["height"], 1080)
        self.assertAlmostEqual(metadata["frame_rate"], 29.97002997, places=6)
        self.assertEqual(metadata["video_codec"], "h264")
        self.assertEqual(metadata["duration_seconds"], 60.25)
        self.assertEqual(metadata["has_audio"], True)
        self.assertEqual(metadata["audio_codec"], "aac")
        self.assertEqual(metadata["audio_sample_rate"], 48000)
        self.assertEqual(metadata["audio_channels"], 2)
        self.assertEqual(warnings, [])

    def test_extract_ffprobe_stream_metadata_handles_missing_audio(self) -> None:
        metadata, warnings = extract_ffprobe_stream_metadata(
            {
                "format": {"duration": "60"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1280,
                        "height": 720,
                        "avg_frame_rate": "25/1",
                        "codec_name": "hevc",
                    }
                ],
            }
        )
        self.assertEqual(metadata["has_audio"], False)
        self.assertIn("no_embedded_audio_stream", warnings)


if __name__ == "__main__":
    unittest.main()
