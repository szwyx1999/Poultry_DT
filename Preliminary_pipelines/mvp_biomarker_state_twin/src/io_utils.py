from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import BiomarkerTwinConfig


DATETIME_COLUMNS = ("start_time", "end_time")


def ensure_output_dirs(config: BiomarkerTwinConfig) -> None:
    for path_value in (
        config.output_dir,
        config.features_dir,
        config.plots_dir,
        config.plots_hmm_by_sequence_dir,
        config.reports_dir,
        config.model_dir,
        config.unity_json_dir,
    ):
        path_value.mkdir(parents=True, exist_ok=True)


def read_csv(path_value: Path) -> pd.DataFrame:
    if not path_value.exists():
        return pd.DataFrame()
    return pd.read_csv(path_value)


def read_csv_with_datetimes(path_value: Path) -> pd.DataFrame:
    dataframe = read_csv(path_value)
    return add_datetime_columns(dataframe)


def add_datetime_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    augmented = dataframe.copy()
    for column in DATETIME_COLUMNS:
        if column in augmented.columns:
            augmented[f"{column}_dt"] = pd.to_datetime(augmented[column], errors="coerce", utc=False)
    return augmented


def load_json(path_value: Path) -> dict:
    with path_value.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(path_value: Path, content: str) -> Path:
    path_value.parent.mkdir(parents=True, exist_ok=True)
    with path_value.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return path_value


def find_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path_value in paths:
        if path_value.exists():
            return path_value
    return None
