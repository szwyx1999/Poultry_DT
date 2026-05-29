from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .build_manifest import write_csv


WINDOW_COLUMNS = [
    "window_id",
    "media_id",
    "system_id",
    "room_id",
    "session_id",
    "source_type",
    "start_time",
    "end_time",
    "duration_seconds",
    "video_path",
    "video_start_offset_sec",
    "has_audio",
    "zone_config_id",
    "quality_status",
    "warnings",
]


def build_video_windows(
    manifest_rows: list[dict],
    window_seconds: float,
    stride_seconds: float,
) -> list[dict]:
    windows: list[dict] = []
    window_delta = timedelta(seconds=window_seconds)
    stride_delta = timedelta(seconds=stride_seconds)
    tolerance = timedelta(milliseconds=1)

    for row in manifest_rows:
        start_time_text = row.get("start_time", "")
        duration_value = row.get("duration_seconds", "")
        if not start_time_text or duration_value in ("", None):
            continue

        start_time = datetime.fromisoformat(start_time_text)
        duration_seconds_value = float(duration_value)
        media_end = start_time + timedelta(seconds=duration_seconds_value)

        cursor = start_time
        window_index = 1
        while cursor + window_delta <= media_end + tolerance:
            window_end = cursor + window_delta
            windows.append(
                {
                    "window_id": f"{row['media_id']}_w{window_index:04d}",
                    "media_id": row["media_id"],
                    "system_id": row["system_id"],
                    "room_id": row["room_id"],
                    "session_id": row["session_id"],
                    "source_type": row.get("source_type", ""),
                    "start_time": cursor.isoformat(),
                    "end_time": window_end.isoformat(),
                    "duration_seconds": round(window_seconds, 3),
                    "video_path": row["file_path"],
                    "video_start_offset_sec": round(
                        (cursor - start_time).total_seconds(),
                        3,
                    ),
                    "has_audio": row["has_audio"],
                    "zone_config_id": row["zone_config_id"],
                    "quality_status": row["quality_status"],
                    "warnings": row["warnings"],
                }
            )
            window_index += 1
            cursor += stride_delta

    return windows


def write_video_window_index_csv(path: Path, rows: list[dict]) -> None:
    write_csv(path, WINDOW_COLUMNS, rows)
