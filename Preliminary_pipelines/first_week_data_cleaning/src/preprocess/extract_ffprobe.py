from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def find_ffprobe_executable() -> str | None:
    for candidate in ("ffprobe", "ffprobe.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def require_ffprobe() -> str:
    ffprobe_path = find_ffprobe_executable()
    if not ffprobe_path:
        raise RuntimeError(
            "ffprobe was not found on PATH. Install FFmpeg and make sure the "
            "`ffprobe` command is available before running preprocessing."
        )
    return ffprobe_path


def extract_ffprobe_metadata(
    ffprobe_path: str,
    file_path: Path,
    sidecar_path: Path,
) -> tuple[dict, list[str]]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(file_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        payload = {
            "error": str(exc),
            "file_path": str(file_path),
        }
        _write_json(sidecar_path, payload)
        return {}, ["ffprobe_extract_failed"]

    if completed.returncode != 0:
        payload = {
            "error": completed.stderr.strip() or "ffprobe failed",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "file_path": str(file_path),
        }
        _write_json(sidecar_path, payload)
        return {}, ["ffprobe_extract_failed"]

    try:
        parsed = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {
            "error": "invalid_json_from_ffprobe",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "file_path": str(file_path),
        }
        _write_json(sidecar_path, payload)
        return {}, ["ffprobe_extract_failed"]

    _write_json(sidecar_path, parsed)
    return parsed, []


def prune_stale_ffprobe_sidecars(ffprobe_dir: Path, active_media_ids: set[str]) -> list[Path]:
    removed_paths: list[Path] = []
    if not ffprobe_dir.exists():
        return removed_paths

    for sidecar_path in ffprobe_dir.glob("*_ffprobe.json"):
        media_id = sidecar_path.name[: -len("_ffprobe.json")]
        if media_id in active_media_ids:
            continue
        sidecar_path.unlink()
        removed_paths.append(sidecar_path)
    return removed_paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
