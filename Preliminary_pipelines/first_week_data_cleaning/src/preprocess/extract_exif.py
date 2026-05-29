from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def find_exiftool_executable() -> str | None:
    for candidate in ("exiftool", "exiftool.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def require_exiftool() -> str:
    exiftool_path = find_exiftool_executable()
    if not exiftool_path:
        raise RuntimeError(
            "exiftool was not found on PATH. Install ExifTool and make sure the "
            "`exiftool` command is available before running preprocessing."
        )
    return exiftool_path


def extract_exif_metadata(
    exiftool_path: str,
    file_path: Path,
    sidecar_path: Path,
) -> tuple[dict, list[str]]:
    try:
        completed = subprocess.run(
            [exiftool_path, "-json", str(file_path)],
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
        return {}, ["exif_extract_failed"]

    if completed.returncode != 0:
        payload = {
            "error": completed.stderr.strip() or "exiftool failed",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "file_path": str(file_path),
        }
        _write_json(sidecar_path, payload)
        return {}, ["exif_extract_failed"]

    try:
        parsed = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        payload = {
            "error": "invalid_json_from_exiftool",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "file_path": str(file_path),
        }
        _write_json(sidecar_path, payload)
        return {}, ["exif_extract_failed"]

    _write_json(sidecar_path, parsed)
    metadata = parsed[0] if isinstance(parsed, list) and parsed else {}
    return metadata, []


def prune_stale_exif_sidecars(exif_dir: Path, active_media_ids: set[str]) -> list[Path]:
    removed_paths: list[Path] = []
    if not exif_dir.exists():
        return removed_paths

    for sidecar_path in exif_dir.glob("*_exif.json"):
        media_id = sidecar_path.name[: -len("_exif.json")]
        if media_id in active_media_ids:
            continue
        sidecar_path.unlink()
        removed_paths.append(sidecar_path)
    return removed_paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
