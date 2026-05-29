from __future__ import annotations

import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.preprocess.main import run_pipeline


CONFIG_TEXT = """raw_root: data/raw
timezone: America/Halifax
timestamp_priority:
  - CreateDate
  - MediaCreateDate
  - TrackCreateDate
  - DateTimeOriginal
  - FileModifyDate
window_seconds: 30
stride_seconds: 10
path_layouts:
  - [system_id, room_id, session_id, modality]
  - [room_id, session_id, modality]
  - [modality, room_id, session_id]
mapping_csv: data/metadata/path_mapping.csv
default_zone_config_id: default_zone
mapping_precedence: csv_overrides_path
"""


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp_root = Path.cwd() / ".tmp_tests"
        self.workspace_temp_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.workspace_temp_root / uuid.uuid4().hex
        self.project_root.mkdir(parents=True, exist_ok=False)
        (self.project_root / "data/raw/free_range/room1/session_2025_09_01/video").mkdir(
            parents=True
        )
        (self.project_root / "data/raw/room2/session_2025_09_02/video").mkdir(parents=True)
        (self.project_root / "data/metadata").mkdir(parents=True)
        (self.project_root / "data/raw/free_range/room1/session_2025_09_01/video/video1.mp4").write_bytes(b"video1")
        (self.project_root / "data/raw/room2/session_2025_09_02/video/video2.MP4").write_bytes(b"video2")
        (self.project_root / "data/metadata/preprocessing_config.yaml").write_text(
            CONFIG_TEXT,
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)
        shutil.rmtree(self.workspace_temp_root, ignore_errors=True)

    def test_pipeline_generates_required_outputs(self) -> None:
        def fake_extract_exif(*args, **kwargs):
            file_name = Path(kwargs["file_path"]).name
            sidecar_path = kwargs["sidecar_path"]
            sidecar_path.write_text(json.dumps([{"FileName": file_name}]), encoding="utf-8")
            return (
                {
                    "CreateDate": "2025:09:01 10:00:00" if file_name == "video1.mp4" else "2025:09:02 11:00:00",
                    "Duration": "60",
                },
                [],
            )

        def fake_extract_ffprobe(*args, **kwargs):
            file_name = Path(kwargs["file_path"]).name
            sidecar_path = kwargs["sidecar_path"]
            payload = {
                "format": {"duration": "60"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30000/1001",
                        "codec_name": "h264",
                    }
                ],
            }
            if file_name == "video1.mp4":
                payload["streams"].append(
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                    }
                )
            sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload, []

        with mock.patch("src.preprocess.main.require_exiftool", return_value="exiftool"), mock.patch(
            "src.preprocess.main.require_ffprobe", return_value="ffprobe"
        ), mock.patch(
            "src.preprocess.main.extract_exif_metadata",
            side_effect=fake_extract_exif,
        ), mock.patch(
            "src.preprocess.main.extract_ffprobe_metadata",
            side_effect=fake_extract_ffprobe,
        ):
            result = run_pipeline(
                project_root=self.project_root,
                config_path=self.project_root / "data/metadata/preprocessing_config.yaml",
            )

        self.assertTrue((self.project_root / "data/processed/metadata/media_manifest.csv").exists())
        self.assertTrue((self.project_root / "data/processed/metadata/video_window_index.csv").exists())
        self.assertTrue((self.project_root / "data/processed/metadata/preprocessing_report.md").exists())
        self.assertTrue((self.project_root / "data/processed/exif/video_000001_exif.json").exists())
        self.assertTrue((self.project_root / "data/processed/ffprobe/video_000001_ffprobe.json").exists())
        self.assertEqual(len(result["manifest_rows"]), 2)
        self.assertEqual(len(result["video_windows"]), 8)

    def test_missing_ffprobe_raises_clear_error(self) -> None:
        with mock.patch(
            "src.preprocess.main.require_exiftool",
            return_value="exiftool",
        ), mock.patch(
            "src.preprocess.main.require_ffprobe",
            side_effect=RuntimeError("ffprobe was not found on PATH."),
        ):
            with self.assertRaisesRegex(RuntimeError, "ffprobe was not found on PATH"):
                run_pipeline(
                    project_root=self.project_root,
                    config_path=self.project_root / "data/metadata/preprocessing_config.yaml",
                )

    def test_missing_exiftool_raises_clear_error(self) -> None:
        with mock.patch(
            "src.preprocess.main.require_exiftool",
            side_effect=RuntimeError("exiftool was not found on PATH."),
        ):
            with self.assertRaisesRegex(RuntimeError, "exiftool was not found on PATH"):
                run_pipeline(
                    project_root=self.project_root,
                    config_path=self.project_root / "data/metadata/preprocessing_config.yaml",
                )

    def test_bad_file_metadata_does_not_stop_manifest(self) -> None:
        def fake_extract_exif(*args, **kwargs):
            file_name = Path(kwargs["file_path"]).name
            kwargs["sidecar_path"].write_text("{}", encoding="utf-8")
            if file_name == "video1.mp4":
                return {}, ["exif_extract_failed"]
            return ({"CreateDate": "2025:09:02 11:00:00", "Duration": "60"}, [])

        def fake_extract_ffprobe(*args, **kwargs):
            file_name = Path(kwargs["file_path"]).name
            kwargs["sidecar_path"].write_text("{}", encoding="utf-8")
            if file_name == "video1.mp4":
                return {}, ["ffprobe_extract_failed"]
            return (
                {
                    "format": {"duration": "60"},
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
                [],
            )

        with mock.patch("src.preprocess.main.require_exiftool", return_value="exiftool"), mock.patch(
            "src.preprocess.main.require_ffprobe", return_value="ffprobe"
        ), mock.patch(
            "src.preprocess.main.extract_exif_metadata",
            side_effect=fake_extract_exif,
        ), mock.patch(
            "src.preprocess.main.extract_ffprobe_metadata",
            side_effect=fake_extract_ffprobe,
        ):
            run_pipeline(
                project_root=self.project_root,
                config_path=self.project_root / "data/metadata/preprocessing_config.yaml",
            )

        manifest_path = self.project_root / "data/processed/metadata/media_manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(1 for row in rows if row["quality_status"] == "error"), 1)
        self.assertEqual(sum(1 for row in rows if row["quality_status"] != "error"), 1)


if __name__ == "__main__":
    unittest.main()
