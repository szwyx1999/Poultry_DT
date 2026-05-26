from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Mvp1Config
from .utils import display_path, optional_text


SELECTED_WINDOWS_FILENAME = "selected_windows.csv"


def select_video_windows(config: Mvp1Config) -> pd.DataFrame:
    windows_df = pd.read_csv(config.video_window_index)
    loaded_count = len(windows_df)

    filtered_df = windows_df.copy()
    room_filter_applied = None
    session_filter_applied = None
    target_subdir_filter_applied = None
    quality_filter_applied = False

    if config.room_id and "room_id" in filtered_df.columns:
        room_filter_applied = config.room_id
        filtered_df = filtered_df[filtered_df["room_id"].astype(str) == config.room_id]
    if config.session_id and "session_id" in filtered_df.columns:
        session_filter_applied = config.session_id
        filtered_df = filtered_df[filtered_df["session_id"].astype(str) == config.session_id]
    if config.target_raw_subdir and "video_path" in filtered_df.columns:
        target_subdir_filter_applied = config.target_raw_subdir
        normalized_target = _normalize_subdir_text(config.target_raw_subdir)
        filtered_df = filtered_df[
            filtered_df["video_path"].astype(str).apply(
                lambda value: normalized_target in _normalize_subdir_text(value)
            )
        ]
    if config.skip_bad_quality and "quality_status" in filtered_df.columns:
        quality_filter_applied = True
        filtered_df = filtered_df[
            filtered_df["quality_status"].astype(str).str.lower() != "error"
        ]
    if config.require_quality_ok and "quality_status" in filtered_df.columns:
        quality_filter_applied = True
        filtered_df = filtered_df[
            filtered_df["quality_status"].astype(str).str.lower() == "ok"
        ]

    sort_columns = [column for column in ("start_time", "window_id") if column in filtered_df.columns]
    if sort_columns:
        filtered_df = filtered_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    else:
        filtered_df = filtered_df.reset_index(drop=True)

    if config.max_windows_per_session is not None and not filtered_df.empty:
        session_group_columns = [column for column in ("room_id", "session_id", "media_id") if column in filtered_df.columns]
        filtered_df = (
            filtered_df.groupby(session_group_columns, dropna=False, sort=False, group_keys=False)
            .head(config.max_windows_per_session)
            .reset_index(drop=True)
        )

    selected_df = _select_rows(filtered_df, config)
    selected_df["resolved_video_path"] = selected_df.get("video_path", pd.Series(dtype=object)).apply(
        lambda value: _resolve_video_path(config.preprocess_root, value)
    )
    selected_df["selection_rank"] = np.arange(1, len(selected_df) + 1, dtype=int)

    output_path = config.features_dir / SELECTED_WINDOWS_FILENAME
    selected_df.to_csv(output_path, index=False)

    missing_path_count = 0
    if "resolved_video_path" in selected_df.columns:
        missing_path_count = sum(
            1
            for path_text in selected_df["resolved_video_path"].tolist()
            if not path_text or not Path(path_text).exists()
        )

    print(f"Loaded windows: {loaded_count}")
    print(f"Selected windows: {len(selected_df)}")
    print(f"Room filter: {room_filter_applied or 'none'}")
    print(f"Session filter: {session_filter_applied or 'none'}")
    print(f"Target raw subdir filter: {target_subdir_filter_applied or 'none'}")
    print(f"Require quality ok: {quality_filter_applied}")
    print(f"Selection strategy: {config.selection_strategy}")
    print(f"Missing resolved video paths: {missing_path_count}")
    print(f"Selected windows saved to: {display_path(config.project_root, output_path)}")
    return selected_df


def _select_rows(dataframe: pd.DataFrame, config: Mvp1Config) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    if config.include_all_available or config.selection_strategy == "all":
        return dataframe.reset_index(drop=True)

    if config.max_windows is None:
        return dataframe.reset_index(drop=True)

    if len(dataframe) <= config.max_windows:
        return dataframe.reset_index(drop=True)

    raw_positions = np.linspace(0, len(dataframe) - 1, num=config.max_windows)
    indices = np.floor(raw_positions + 0.5).astype(int)
    return dataframe.iloc[indices].reset_index(drop=True)


def _normalize_subdir_text(value: object) -> str:
    return str(value).replace("\\", "/").strip().casefold()


def _resolve_video_path(preprocess_root: Path, value: object) -> str:
    path_text = optional_text(value)
    if not path_text:
        return ""
    candidate = Path(path_text)
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False))
    return str((preprocess_root / candidate).resolve(strict=False))
