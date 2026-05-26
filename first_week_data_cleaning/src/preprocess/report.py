from __future__ import annotations

from collections import Counter
from pathlib import Path


def build_preprocessing_report(
    manifest_rows: list[dict],
    video_windows: list[dict],
) -> str:
    warning_counts = Counter()
    for row in manifest_rows:
        warning_counts.update(split_warnings(row.get("warnings", "")))

    valid_timestamps = sum(1 for row in manifest_rows if row.get("start_time"))
    valid_duration = sum(1 for row in manifest_rows if row.get("duration_seconds") not in ("", None))
    with_audio = sum(1 for row in manifest_rows if to_bool(row.get("has_audio")))
    without_audio = len(manifest_rows) - with_audio

    warning_lines = [
        f"- `{warning}`: {count}"
        for warning, count in sorted(warning_counts.items())
    ]
    if not warning_lines:
        warning_lines = ["- None"]

    lines = [
        "# Preprocessing Report",
        "",
        "## Summary",
        f"- Number of MP4 files found: {len(manifest_rows)}",
        f"- Number of files with valid timestamps: {valid_timestamps}",
        f"- Number of files missing timestamps: {len(manifest_rows) - valid_timestamps}",
        f"- Number of files with valid duration: {valid_duration}",
        f"- Number of files missing duration: {len(manifest_rows) - valid_duration}",
        f"- Number of files with embedded audio: {with_audio}",
        f"- Number of files without embedded audio: {without_audio}",
        f"- Number of video windows generated: {len(video_windows)}",
        "",
        "## Warnings",
        *warning_lines,
        "",
        "## Suggested Next Step",
        build_suggested_next_step(video_windows),
        "",
    ]
    return "\n".join(lines)


def write_preprocessing_report(path: Path, report_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


def split_warnings(serialized_warnings: str) -> list[str]:
    if not serialized_warnings:
        return []
    return [item for item in serialized_warnings.split(";") if item]


def build_suggested_next_step(video_windows: list[dict]) -> str:
    if video_windows:
        return (
            "- Use `data/processed/metadata/video_window_index.csv` as the handoff table for MVP "
            "feature extraction. The later stage should open `video_path`, seek to "
            "`video_start_offset_sec`, process `duration_seconds`, and optionally read embedded "
            "audio from the same MP4."
        )

    return (
        "- No video windows were generated. Inspect `media_manifest.csv`, confirm timestamps and "
        "durations were parsed correctly, and adjust `preprocessing_config.yaml` if the raw folder "
        "layout needs different path parsing rules."
    )


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() == "true"
