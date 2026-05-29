from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

from .config import PreprocessingConfig
from .parse_metadata import (
    compute_end_time,
    extract_ffprobe_stream_metadata,
    parse_best_timestamp,
    parse_duration_seconds,
)


MANIFEST_COLUMNS = [
    "media_id",
    "media_type",
    "system_id",
    "room_id",
    "session_id",
    "modality",
    "source_type",
    "zone_config_id",
    "file_path",
    "file_name",
    "file_extension",
    "source_rel_dir",
    "file_size_bytes",
    "start_time",
    "end_time",
    "duration_seconds",
    "timezone",
    "width",
    "height",
    "frame_rate",
    "video_codec",
    "has_audio",
    "audio_codec",
    "audio_sample_rate",
    "audio_channels",
    "quality_status",
    "warnings",
]

CONTEXT_FIELDS = ("system_id", "room_id", "session_id", "modality", "source_type")


def build_manifest_row(
    file_info: dict,
    exif_metadata: dict,
    ffprobe_metadata: dict,
    config: PreprocessingConfig,
    mapping_rows: list[dict],
    exif_warnings: list[str] | None = None,
    ffprobe_warnings: list[str] | None = None,
) -> dict:
    warnings = set(exif_warnings or [])
    warnings.update(ffprobe_warnings or [])

    context = derive_path_context(
        file_info["relative_raw_path"],
        config.path_layouts,
        default_modality=file_info["media_type"],
    )
    warnings.update(context.pop("warnings"))

    mapping_match, extra_matches = find_first_mapping_match(
        file_info["file_path"],
        mapping_rows,
    )
    if mapping_match:
        for field_name in (*CONTEXT_FIELDS, "zone_config_id"):
            context[field_name], changed = override_value(
                context.get(field_name),
                mapping_match.get(field_name),
            )
            if changed:
                warnings.add("mapping_override")
    if extra_matches:
        warnings.add("multiple_mapping_matches")

    system_id = context.get("system_id") or "unknown_system"
    room_id = context.get("room_id") or "unknown_room"
    session_id = context.get("session_id") or "unknown_session"
    modality = context.get("modality") or file_info["media_type"]
    source_type = context.get("source_type") or ""
    zone_config_id = context.get("zone_config_id") or config.default_zone_config_id

    if system_id == "unknown_system":
        warnings.add("unknown_system")
    if room_id == "unknown_room":
        warnings.add("unknown_room")
    if session_id == "unknown_session":
        warnings.add("unknown_session")

    start_time, _, timestamp_warning = parse_best_timestamp(
        exif_metadata,
        config.timestamp_priority,
        config.timezone,
    )
    if timestamp_warning:
        warnings.add(timestamp_warning)

    ffprobe_fields, stream_warnings = extract_ffprobe_stream_metadata(ffprobe_metadata)
    warnings.update(normalize_duration_warnings(stream_warnings))

    exif_duration_seconds, exif_duration_warning = parse_duration_seconds(exif_metadata)
    duration_seconds, duration_warning = choose_duration_seconds(
        ffprobe_duration=ffprobe_fields["duration_seconds"],
        ffprobe_warnings=stream_warnings,
        exif_duration=exif_duration_seconds,
        exif_duration_warning=exif_duration_warning,
    )
    if duration_warning:
        warnings.add(duration_warning)

    end_time = compute_end_time(start_time, duration_seconds)
    quality_status = determine_quality_status(warnings)

    return {
        "media_id": file_info["media_id"],
        "media_type": file_info["media_type"],
        "system_id": system_id,
        "room_id": room_id,
        "session_id": session_id,
        "modality": modality,
        "source_type": source_type,
        "zone_config_id": zone_config_id,
        "file_path": file_info["file_path"],
        "file_name": file_info["file_name"],
        "file_extension": file_info["file_extension"],
        "source_rel_dir": context["source_rel_dir"],
        "file_size_bytes": file_info["file_size_bytes"],
        "start_time": start_time.isoformat() if start_time else "",
        "end_time": end_time.isoformat() if end_time else "",
        "duration_seconds": round(duration_seconds, 3) if duration_seconds is not None else "",
        "timezone": config.timezone,
        "width": ffprobe_fields["width"] or "",
        "height": ffprobe_fields["height"] or "",
        "frame_rate": round(ffprobe_fields["frame_rate"], 3)
        if ffprobe_fields["frame_rate"] is not None
        else "",
        "video_codec": ffprobe_fields["video_codec"] or "",
        "has_audio": ffprobe_fields["has_audio"],
        "audio_codec": ffprobe_fields["audio_codec"] or "",
        "audio_sample_rate": ffprobe_fields["audio_sample_rate"] or "",
        "audio_channels": ffprobe_fields["audio_channels"] or "",
        "quality_status": quality_status,
        "warnings": serialize_warnings(warnings),
    }


def write_manifest_csv(path: Path, rows: Iterable[dict]) -> None:
    write_csv(path, MANIFEST_COLUMNS, rows)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_mapping_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def derive_path_context(
    relative_raw_path: str,
    path_layouts: list[list[str]],
    default_modality: str,
) -> dict:
    path = Path(relative_raw_path)
    parent_parts = list(path.parent.parts) if path.parent != Path(".") else []
    warnings: set[str] = set()

    context = {
        "system_id": None,
        "room_id": None,
        "session_id": None,
        "modality": None,
        "source_type": None,
        "zone_config_id": None,
        "source_rel_dir": "/".join(parent_parts),
        "warnings": warnings,
    }

    modality_token = normalize_token(default_modality) or default_modality
    matched_layout = False
    best_match: tuple[int, dict[str, str | None]] | None = None
    for layout in path_layouts:
        if len(layout) != len(parent_parts):
            continue
        candidate: dict[str, str | None] = {}
        score = 0
        for field_name, part_value in zip(layout, parent_parts, strict=True):
            normalized_value = normalize_token(part_value)
            candidate[field_name] = normalized_value
            if field_name == "modality" and normalized_value == modality_token:
                score += 10
        if best_match is None or score > best_match[0]:
            best_match = (score, candidate)

    if best_match is not None:
        matched_layout = True
        _, matched_candidate = best_match
        for field_name in CONTEXT_FIELDS:
            if field_name in matched_candidate:
                context[field_name] = matched_candidate[field_name]

    if not matched_layout and parent_parts:
        warnings.add("path_context_unmatched")

    if context["modality"] is None:
        context["modality"] = modality_token

    return context


def find_first_mapping_match(
    file_path: str,
    mapping_rows: list[dict],
) -> tuple[dict | None, list[dict]]:
    matches = [
        row
        for row in mapping_rows
        if row.get("pattern") and re.search(row["pattern"], file_path, flags=re.IGNORECASE)
    ]
    if not matches:
        return None, []
    return matches[0], matches[1:]


def override_value(
    current_value: str | None,
    new_value: str | None,
) -> tuple[str | None, bool]:
    if not new_value:
        return current_value, False
    normalized = normalize_token(new_value)
    if current_value == normalized:
        return current_value, False
    return normalized, True


def choose_duration_seconds(
    ffprobe_duration: float | None,
    ffprobe_warnings: list[str],
    exif_duration: float | None,
    exif_duration_warning: str | None,
) -> tuple[float | None, str | None]:
    if ffprobe_duration is not None:
        return ffprobe_duration, None
    if exif_duration is not None:
        return exif_duration, None
    if "invalid_ffprobe_duration" in ffprobe_warnings or exif_duration_warning == "invalid_duration":
        return None, "invalid_duration"
    return None, "missing_duration"


def normalize_duration_warnings(warnings: list[str]) -> set[str]:
    normalized: set[str] = set()
    for warning in warnings:
        if warning == "invalid_ffprobe_duration":
            continue
        normalized.add(warning)
    return normalized


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().casefold()).strip("_")
    return normalized or None


def determine_quality_status(warnings: set[str]) -> str:
    critical_warnings = {
        "exif_extract_failed",
        "ffprobe_extract_failed",
        "missing_timestamp",
        "invalid_timestamp",
        "missing_duration",
        "invalid_duration",
        "missing_video_stream",
    }
    if warnings & critical_warnings:
        return "error"
    if warnings:
        return "warning"
    return "ok"


def serialize_warnings(warnings: set[str] | list[str]) -> str:
    if not warnings:
        return ""
    return ";".join(sorted(set(warnings)))
