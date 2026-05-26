from __future__ import annotations

from pathlib import Path


VIDEO_EXTENSIONS = {".mp4"}


def scan_media_files(project_root: Path, raw_root: Path) -> list[dict]:
    if not raw_root.exists():
        return []

    candidate_paths = sorted(
        (
            file_path
            for file_path in raw_root.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(project_root).as_posix().casefold(),
    )

    media_files: list[dict] = []
    for index, file_path in enumerate(candidate_paths, start=1):
        relative_repo_path = file_path.relative_to(project_root).as_posix()
        relative_raw_path = file_path.relative_to(raw_root).as_posix()
        media_files.append(
            {
                "media_id": f"video_{index:06d}",
                "media_type": "video",
                "absolute_path": file_path,
                "file_path": relative_repo_path,
                "relative_raw_path": relative_raw_path,
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": file_path.stat().st_size,
            }
        )

    return media_files
