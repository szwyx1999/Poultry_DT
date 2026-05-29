from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Iterable
from zoneinfo import ZoneInfo


DURATION_KEYS = ["Duration", "MediaDuration", "TrackDuration"]


def parse_best_timestamp(
    metadata: dict,
    timestamp_priority: Iterable[str],
    timezone_name: str,
) -> tuple[datetime | None, str | None, str | None]:
    saw_invalid_value = False
    invalid_source: str | None = None
    for key in timestamp_priority:
        raw_value = metadata.get(key)
        if raw_value in (None, ""):
            continue
        parsed = parse_exif_datetime(raw_value, timezone_name)
        if parsed is not None:
            return parsed, key, None
        saw_invalid_value = True
        invalid_source = key
    if saw_invalid_value:
        return None, invalid_source, "invalid_timestamp"
    return None, None, "missing_timestamp"


def parse_exif_datetime(raw_value: object, timezone_name: str) -> datetime | None:
    if raw_value in (None, ""):
        return None

    timezone = ZoneInfo(timezone_name)
    value = str(raw_value).strip()
    value = value.replace(" UTC", "Z")

    colon_pattern = re.compile(
        r"^(?P<year>\d{4}):(?P<month>\d{2}):(?P<day>\d{2})[ T]"
        r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
        r"(?P<fraction>\.\d+)?(?P<tz>Z|[+-]\d{2}:\d{2}|[+-]\d{4})?$"
    )
    match = colon_pattern.match(value)
    if match:
        tz_part = _normalize_tz(match.group("tz") or "")
        iso_value = (
            f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
            f"T{match.group('hour')}:{match.group('minute')}:{match.group('second')}"
            f"{match.group('fraction') or ''}{tz_part}"
        )
        try:
            parsed = datetime.fromisoformat(iso_value)
        except ValueError:
            return None
        return _apply_timezone(parsed, timezone)

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _apply_timezone(parsed, timezone)


def parse_duration_seconds(metadata: dict) -> tuple[float | None, str | None]:
    saw_invalid_value = False
    for key in DURATION_KEYS:
        raw_value = metadata.get(key)
        if raw_value in (None, ""):
            continue
        parsed = parse_duration_value(raw_value)
        if parsed is not None:
            return parsed, None
        saw_invalid_value = True
    if saw_invalid_value:
        return None, "invalid_duration"
    return None, "missing_duration"


def parse_duration_value(raw_value: object) -> float | None:
    if raw_value in (None, ""):
        return None

    if isinstance(raw_value, (int, float)):
        if math.isfinite(float(raw_value)):
            return float(raw_value)
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    colon_pattern = re.compile(
        r"^(?P<hours>\d+):(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)$"
    )
    minute_pattern = re.compile(
        r"^(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)$"
    )
    match = colon_pattern.match(value)
    if match:
        return (
            int(match.group("hours")) * 3600
            + int(match.group("minutes")) * 60
            + float(match.group("seconds"))
        )

    match = minute_pattern.match(value)
    if match:
        return int(match.group("minutes")) * 60 + float(match.group("seconds"))

    numeric_match = re.search(r"-?\d+(?:\.\d+)?", value)
    if numeric_match:
        return float(numeric_match.group(0))

    return None


def compute_end_time(
    start_time: datetime | None,
    duration_seconds: float | None,
) -> datetime | None:
    if start_time is None or duration_seconds is None:
        return None
    return start_time + timedelta(seconds=duration_seconds)


def extract_ffprobe_stream_metadata(ffprobe_metadata: dict) -> tuple[dict, list[str]]:
    streams = ffprobe_metadata.get("streams", [])
    format_data = ffprobe_metadata.get("format", {})

    video_stream = first_matching_stream(streams, "video")
    audio_stream = first_matching_stream(streams, "audio")
    warnings: list[str] = []

    ffprobe_duration = first_present(
        [
            format_data.get("duration"),
            video_stream.get("duration") if video_stream else None,
            audio_stream.get("duration") if audio_stream else None,
        ]
    )
    duration_seconds = parse_duration_value(ffprobe_duration)
    if ffprobe_duration not in (None, "") and duration_seconds is None:
        warnings.append("invalid_ffprobe_duration")

    result = {
        "width": parse_int(video_stream.get("width")) if video_stream else None,
        "height": parse_int(video_stream.get("height")) if video_stream else None,
        "frame_rate": parse_frame_rate(
            first_present(
                [
                    video_stream.get("avg_frame_rate") if video_stream else None,
                    video_stream.get("r_frame_rate") if video_stream else None,
                ]
            )
        ),
        "video_codec": normalize_text(
            first_present(
                [
                    video_stream.get("codec_name") if video_stream else None,
                    video_stream.get("codec_long_name") if video_stream else None,
                ]
            )
        ),
        "duration_seconds": duration_seconds,
        "has_audio": audio_stream is not None,
        "audio_codec": normalize_text(
            first_present(
                [
                    audio_stream.get("codec_name") if audio_stream else None,
                    audio_stream.get("codec_long_name") if audio_stream else None,
                ]
            )
        ),
        "audio_sample_rate": parse_int(
            audio_stream.get("sample_rate") if audio_stream else None
        ),
        "audio_channels": parse_int(
            audio_stream.get("channels") if audio_stream else None
        ),
    }

    if video_stream is None:
        warnings.append("missing_video_stream")

    if audio_stream is None:
        warnings.append("no_embedded_audio_stream")

    return result, warnings


def first_matching_stream(streams: object, codec_type: str) -> dict | None:
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type:
            return stream
    return None


def first_present(values: Iterable[object]) -> object | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def normalize_text(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip()


def parse_int(value: object | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def parse_float(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def parse_frame_rate(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        try:
            fraction = Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
        if fraction.denominator == 0:
            return None
        return float(fraction)

    return parse_float(text)


def _normalize_tz(tz_value: str) -> str:
    if not tz_value:
        return ""
    if tz_value == "Z":
        return "+00:00"
    if re.match(r"^[+-]\d{4}$", tz_value):
        return f"{tz_value[:3]}:{tz_value[3:]}"
    return tz_value


def _apply_timezone(parsed: datetime, timezone: ZoneInfo) -> datetime:
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)
