from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .build_manifest import build_manifest_row, load_mapping_csv, write_manifest_csv
from .build_video_windows import build_video_windows, write_video_window_index_csv
from .config import load_config
from .extract_exif import (
    extract_exif_metadata,
    prune_stale_exif_sidecars,
    require_exiftool,
)
from .extract_ffprobe import (
    extract_ffprobe_metadata,
    prune_stale_ffprobe_sidecars,
    require_ffprobe,
)
from .report import build_preprocessing_report, write_preprocessing_report
from .scan_files import scan_media_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw poultry welfare MP4 metadata into manifest and window tables."
    )
    parser.add_argument(
        "--config",
        default="data/metadata/preprocessing_config.yaml",
        help="Path to preprocessing YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    config_path = project_root / args.config
    try:
        run_pipeline(project_root=project_root, config_path=config_path)
    except Exception as exc:  # pragma: no cover - CLI guard
        logger = logging.getLogger("preprocess")
        if logger.handlers:
            logger.error(str(exc))
        else:
            print(str(exc))
        return 1
    return 0


def run_pipeline(project_root: Path, config_path: Path) -> dict:
    output_dirs = prepare_output_directories(project_root)
    logger = configure_logging(output_dirs["logs"])
    logger.info("Loading configuration from %s", display_path(project_root, config_path))
    config = load_config(config_path)
    remove_legacy_outputs(output_dirs["metadata"], logger)

    raw_root = resolve_project_path(project_root, config.raw_root)
    mapping_csv_path = resolve_project_path(project_root, config.mapping_csv)

    logger.info("Scanning MP4 files under %s", display_path(project_root, raw_root))
    media_files = scan_media_files(project_root=project_root, raw_root=raw_root)
    logger.info("Found %s MP4 file(s).", len(media_files))

    exiftool_path = require_exiftool()
    ffprobe_path = require_ffprobe()
    mapping_rows = load_mapping_csv(mapping_csv_path)

    manifest_rows: list[dict] = []
    active_media_ids: set[str] = set()
    total_files = len(media_files)
    for index, file_info in enumerate(media_files, start=1):
        logger.info("[%s/%s] Reading metadata for %s", index, total_files, file_info["file_name"])

        exif_sidecar_path = output_dirs["exif"] / f"{file_info['media_id']}_exif.json"
        ffprobe_sidecar_path = output_dirs["ffprobe"] / f"{file_info['media_id']}_ffprobe.json"

        exif_metadata, exif_warnings = extract_exif_metadata(
            exiftool_path=exiftool_path,
            file_path=file_info["absolute_path"],
            sidecar_path=exif_sidecar_path,
        )
        ffprobe_metadata, ffprobe_warnings = extract_ffprobe_metadata(
            ffprobe_path=ffprobe_path,
            file_path=file_info["absolute_path"],
            sidecar_path=ffprobe_sidecar_path,
        )

        active_media_ids.add(file_info["media_id"])
        manifest_rows.append(
            build_manifest_row(
                file_info=file_info,
                exif_metadata=exif_metadata,
                ffprobe_metadata=ffprobe_metadata,
                config=config,
                mapping_rows=mapping_rows,
                exif_warnings=exif_warnings,
                ffprobe_warnings=ffprobe_warnings,
            )
        )

    removed_exif = prune_stale_exif_sidecars(output_dirs["exif"], active_media_ids)
    removed_ffprobe = prune_stale_ffprobe_sidecars(output_dirs["ffprobe"], active_media_ids)
    if removed_exif:
        logger.info("Pruned %s stale EXIF sidecar(s).", len(removed_exif))
    if removed_ffprobe:
        logger.info("Pruned %s stale FFprobe sidecar(s).", len(removed_ffprobe))

    logger.info("Writing media manifest...")
    manifest_path = output_dirs["metadata"] / "media_manifest.csv"
    write_manifest_csv(manifest_path, manifest_rows)

    logger.info("Building video window index...")
    video_windows = build_video_windows(
        manifest_rows=manifest_rows,
        window_seconds=config.window_seconds,
        stride_seconds=config.stride_seconds,
    )
    video_window_index_path = output_dirs["metadata"] / "video_window_index.csv"
    write_video_window_index_csv(video_window_index_path, video_windows)

    logger.info("Writing preprocessing report...")
    report_text = build_preprocessing_report(manifest_rows, video_windows)
    report_path = output_dirs["metadata"] / "preprocessing_report.md"
    write_preprocessing_report(report_path, report_text)

    logger.info("Preprocessing complete.")
    return {
        "manifest_path": manifest_path,
        "video_window_index_path": video_window_index_path,
        "report_path": report_path,
        "manifest_rows": manifest_rows,
        "video_windows": video_windows,
    }


def prepare_output_directories(project_root: Path) -> dict[str, Path]:
    metadata_dir = project_root / "data/processed/metadata"
    exif_dir = project_root / "data/processed/exif"
    ffprobe_dir = project_root / "data/processed/ffprobe"
    logs_dir = project_root / "outputs/logs"
    for directory in (metadata_dir, exif_dir, ffprobe_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "metadata": metadata_dir,
        "exif": exif_dir,
        "ffprobe": ffprobe_dir,
        "logs": logs_dir,
    }


def remove_legacy_outputs(metadata_dir: Path, logger: logging.Logger) -> None:
    for legacy_name in ("av_pairs.csv", "mvp_windows.csv"):
        legacy_path = metadata_dir / legacy_name
        if not legacy_path.exists():
            continue
        legacy_path.unlink()
        logger.info("Removed legacy output %s", legacy_path.name)


def configure_logging(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(logs_dir / "preprocess.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def resolve_project_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def display_path(project_root: Path, target_path: Path) -> str:
    try:
        return target_path.relative_to(project_root).as_posix()
    except ValueError:
        return target_path.as_posix()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
