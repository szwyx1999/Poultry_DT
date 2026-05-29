from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_TIMESTAMP_PRIORITY = [
    "CreationDate",
    "CreateDate",
    "MediaCreateDate",
    "TrackCreateDate",
    "DateTimeOriginal",
    "FileModifyDate",
]

DEFAULT_PATH_LAYOUTS = [
    ["system_id", "room_id", "session_id", "modality"],
    ["room_id", "session_id", "modality"],
    ["modality", "room_id", "session_id"],
]

ALLOWED_PATH_FIELDS = {"system_id", "room_id", "session_id", "modality", "source_type"}


@dataclass(slots=True)
class PreprocessingConfig:
    raw_root: str = "data/raw"
    timezone: str = "America/Halifax"
    timestamp_priority: list[str] = field(default_factory=lambda: list(DEFAULT_TIMESTAMP_PRIORITY))
    window_seconds: float = 30.0
    stride_seconds: float = 10.0
    path_layouts: list[list[str]] = field(
        default_factory=lambda: [list(layout) for layout in DEFAULT_PATH_LAYOUTS]
    )
    mapping_csv: str = "data/metadata/path_mapping.csv"
    default_zone_config_id: str = "default"
    mapping_precedence: str = "csv_overrides_path"


def load_config(config_path: Path) -> PreprocessingConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.as_posix()}. "
            "Create data/metadata/preprocessing_config.yaml or pass --config."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    config = PreprocessingConfig(
        raw_root=str(raw_data.get("raw_root", "data/raw")),
        timezone=str(raw_data.get("timezone", "America/Halifax")),
        timestamp_priority=list(
            raw_data.get("timestamp_priority", DEFAULT_TIMESTAMP_PRIORITY)
        ),
        window_seconds=float(raw_data.get("window_seconds", 30.0)),
        stride_seconds=float(raw_data.get("stride_seconds", 10.0)),
        path_layouts=[
            list(layout)
            for layout in raw_data.get("path_layouts", DEFAULT_PATH_LAYOUTS)
        ],
        mapping_csv=str(raw_data.get("mapping_csv", "data/metadata/path_mapping.csv")),
        default_zone_config_id=str(raw_data.get("default_zone_config_id", "default")),
        mapping_precedence=str(
            raw_data.get("mapping_precedence", "csv_overrides_path")
        ),
    )
    _validate_config(config, config_path)
    return config


def _validate_config(config: PreprocessingConfig, config_path: Path) -> None:
    if not config.raw_root.strip():
        raise ValueError(f"{config_path.as_posix()} must define a non-empty raw_root.")

    if not config.timestamp_priority:
        raise ValueError(
            f"{config_path.as_posix()} must define at least one timestamp field."
        )

    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero.")

    if config.stride_seconds <= 0:
        raise ValueError("stride_seconds must be greater than zero.")

    if not config.path_layouts:
        raise ValueError("path_layouts must include at least one layout.")

    for index, layout in enumerate(config.path_layouts, start=1):
        if not layout:
            raise ValueError(f"path_layouts entry {index} must not be empty.")
        if len(layout) != len(set(layout)):
            raise ValueError(f"path_layouts entry {index} contains duplicate fields.")
        unknown_fields = [field_name for field_name in layout if field_name not in ALLOWED_PATH_FIELDS]
        if unknown_fields:
            raise ValueError(
                f"path_layouts entry {index} contains unsupported fields: {', '.join(unknown_fields)}."
            )

    if config.mapping_precedence != "csv_overrides_path":
        raise ValueError(
            "Only mapping_precedence='csv_overrides_path' is supported in v1."
        )
