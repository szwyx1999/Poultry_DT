from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import yaml

from .config import BiomarkerTwinConfig


@dataclass(frozen=True)
class SourceCoverageSummary:
    video_window_index_rows: int
    video_window_index_unique_windows: int
    video_window_index_unique_rooms: int
    video_window_index_unique_sessions: int
    raw_target_video_count: int
    indexed_target_video_count: int
    selected_window_rows: int
    selected_window_unique_windows: int
    selected_window_unique_media: int
    processed_start_time_min: str
    processed_start_time_max: str
    zone_feature_rows: int
    zone_feature_unique_windows: int
    zone_feature_unique_rooms: int
    zone_feature_unique_sessions: int
    failed_target_videos: tuple[str, ...]
    all_target_videos_processed: bool
    refresh_performed: bool
    refresh_reason: str


def ensure_sufficient_mvp1_features(config: BiomarkerTwinConfig) -> SourceCoverageSummary:
    before_summary = _summarize_source_tables(config)
    refresh_performed = False
    refresh_reason = "Existing MVP1 feature table already satisfied the biomarker/HMM minimum window count."

    if (
        config.auto_refresh_mvp1_if_needed
        and before_summary.zone_feature_unique_windows < config.min_windows_for_hmm
        and before_summary.video_window_index_unique_windows > before_summary.zone_feature_unique_windows
    ):
        _rerun_mvp1_with_refresh_target(config)
        refresh_performed = True
        refresh_reason = (
            "MVP1 feature table had too few windows for the HMM minimum, so MVP1 was rerun "
            f"with selection.max_windows={config.mvp1_refresh_window_target}."
        )

    after_summary = _summarize_source_tables(config)
    return SourceCoverageSummary(
        video_window_index_rows=after_summary.video_window_index_rows,
        video_window_index_unique_windows=after_summary.video_window_index_unique_windows,
        video_window_index_unique_rooms=after_summary.video_window_index_unique_rooms,
        video_window_index_unique_sessions=after_summary.video_window_index_unique_sessions,
        raw_target_video_count=after_summary.raw_target_video_count,
        indexed_target_video_count=after_summary.indexed_target_video_count,
        selected_window_rows=after_summary.selected_window_rows,
        selected_window_unique_windows=after_summary.selected_window_unique_windows,
        selected_window_unique_media=after_summary.selected_window_unique_media,
        processed_start_time_min=after_summary.processed_start_time_min,
        processed_start_time_max=after_summary.processed_start_time_max,
        zone_feature_rows=after_summary.zone_feature_rows,
        zone_feature_unique_windows=after_summary.zone_feature_unique_windows,
        zone_feature_unique_rooms=after_summary.zone_feature_unique_rooms,
        zone_feature_unique_sessions=after_summary.zone_feature_unique_sessions,
        failed_target_videos=after_summary.failed_target_videos,
        all_target_videos_processed=after_summary.all_target_videos_processed,
        refresh_performed=refresh_performed,
        refresh_reason=refresh_reason,
    )


def apply_window_selection(window_df: pd.DataFrame, config: BiomarkerTwinConfig) -> pd.DataFrame:
    if window_df.empty:
        return window_df.copy()

    filtered_df = window_df.sort_values(
        [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in window_df.columns],
        kind="stable",
    ).reset_index(drop=True)

    if config.max_windows_per_session is not None:
        session_group_columns = [column for column in ("room_id", "session_id") if column in filtered_df.columns]
        filtered_df = (
            filtered_df.groupby(session_group_columns, dropna=False, sort=False, group_keys=False)
            .head(config.max_windows_per_session)
            .reset_index(drop=True)
        )

    if config.max_windows_per_room is not None:
        filtered_df = (
            filtered_df.groupby("room_id", dropna=False, sort=False, group_keys=False)
            .head(config.max_windows_per_room)
            .reset_index(drop=True)
        )

    if config.max_windows is not None:
        filtered_df = filtered_df.head(config.max_windows).reset_index(drop=True)

    return filtered_df


def write_data_coverage_report(
    source_summary: SourceCoverageSummary,
    canonical_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    state_summary_df: pd.DataFrame,
    configured_state_setting: str | int,
    effective_states: int,
    occupied_states: int,
    max_windows_setting: int | None,
    max_windows_per_room_setting: int | None,
    max_windows_per_session_setting: int | None,
    min_windows_for_hmm_setting: int,
    output_path: Path,
) -> None:
    unique_rooms = int(biomarker_df["room_id"].nunique()) if not biomarker_df.empty else 0
    unique_sessions = int(biomarker_df["session_id"].nunique()) if "session_id" in biomarker_df.columns and not biomarker_df.empty else 0
    unique_windows = int(biomarker_df["window_id"].nunique()) if not biomarker_df.empty else 0
    lines = [
        "# Data Coverage Report",
        "",
        "This report documents how much video-derived MVP1 data was available and how much was actually used for biomarker and latent-state modelling.",
        "",
        "## Source Tables",
        "",
        f"- `video_window_index.csv` rows: {source_summary.video_window_index_rows}",
        f"- `video_window_index.csv` unique windows: {source_summary.video_window_index_unique_windows}",
        f"- `video_window_index.csv` unique rooms: {source_summary.video_window_index_unique_rooms}",
        f"- `video_window_index.csv` unique sessions: {source_summary.video_window_index_unique_sessions}",
        f"- Target raw videos found: {source_summary.raw_target_video_count}",
        f"- Target videos indexed in `media_manifest.csv`: {source_summary.indexed_target_video_count}",
        f"- `selected_windows.csv` rows: {source_summary.selected_window_rows}",
        f"- `selected_windows.csv` unique windows: {source_summary.selected_window_unique_windows}",
        f"- `selected_windows.csv` unique media files: {source_summary.selected_window_unique_media}",
        f"- `video_zone_features.csv` rows: {source_summary.zone_feature_rows}",
        f"- `video_zone_features.csv` unique windows: {source_summary.zone_feature_unique_windows}",
        f"- `video_zone_features.csv` unique rooms: {source_summary.zone_feature_unique_rooms}",
        f"- `video_zone_features.csv` unique sessions: {source_summary.zone_feature_unique_sessions}",
        f"- Full processed time span start: {source_summary.processed_start_time_min or 'n/a'}",
        f"- Full processed time span end: {source_summary.processed_start_time_max or 'n/a'}",
        f"- All target-folder videos processed: {'yes' if source_summary.all_target_videos_processed else 'no'}",
        f"- MVP1 refresh performed during this run: {'yes' if source_summary.refresh_performed else 'no'}",
        f"- MVP1 refresh note: {source_summary.refresh_reason}",
        "",
        "## Biomarker/HMM Input",
        "",
        f"- `canonical_zone_feature_table.csv` rows: {len(canonical_df)}",
        f"- `biomarker_window_table.csv` rows used for modelling: {len(biomarker_df)}",
        f"- Unique rooms used: {unique_rooms}",
        f"- Unique sessions used: {unique_sessions}",
        f"- Unique windows used: {unique_windows}",
        f"- Configured HMM state setting: {configured_state_setting if configured_state_setting != '' else 'auto'}",
        f"- Fitted HMM components used: {effective_states}",
        f"- Occupied latent states observed in `hmm_state_sequence.csv`: {occupied_states}",
        "",
        "## Window Selection Settings",
        "",
        f"- `max_windows`: {'null' if max_windows_setting is None else max_windows_setting}",
        f"- `max_windows_per_room`: {'null' if max_windows_per_room_setting is None else max_windows_per_room_setting}",
        f"- `max_windows_per_session`: {'null' if max_windows_per_session_setting is None else max_windows_per_session_setting}",
        f"- `min_windows_for_hmm`: {min_windows_for_hmm_setting}",
        "",
        "## State Summary Coverage",
        "",
        f"- Occupied state rows represented in `hmm_state_summary.csv`: {occupied_states}",
        f"- State rows in `hmm_state_summary.csv`: {len(state_summary_df)}",
        "",
        "Video-derived MVP1 features only were used at this stage. Audio, environmental, and thermal signals were not integrated into this run.",
    ]
    if source_summary.failed_target_videos:
        lines.extend(
            [
                "",
                "## Missing Or Failed Videos",
                "",
            ]
        )
        for video_name in source_summary.failed_target_videos:
            lines.append(f"- `{video_name}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_source_tables(config: BiomarkerTwinConfig) -> SourceCoverageSummary:
    video_window_index_df = _read_csv(config.video_window_index_csv)
    media_manifest_df = _read_csv(config.media_manifest_csv)
    selected_windows_df = _read_csv(config.selected_windows_csv)
    zone_features_df = _read_csv(config.zone_features_csv)
    target_manifest_df = _filter_target_manifest(media_manifest_df, config)
    selected_window_unique_media = _safe_nunique(selected_windows_df, "media_id")
    processed_start_time_min = _safe_min_text(selected_windows_df, "start_time")
    processed_start_time_max = _safe_max_text(selected_windows_df, "start_time")
    if not target_manifest_df.empty and "quality_status" in target_manifest_df.columns:
        failed_target_videos = tuple(
            target_manifest_df[
                target_manifest_df["quality_status"].astype(str).str.lower() == "error"
            ].get("file_name", pd.Series(dtype=object)).astype(str).tolist()
        )
    else:
        failed_target_videos = tuple()
    return SourceCoverageSummary(
        video_window_index_rows=len(video_window_index_df),
        video_window_index_unique_windows=_safe_nunique(video_window_index_df, "window_id"),
        video_window_index_unique_rooms=_safe_nunique(video_window_index_df, "room_id"),
        video_window_index_unique_sessions=_safe_nunique(video_window_index_df, "session_id"),
        raw_target_video_count=_count_target_raw_videos(config),
        indexed_target_video_count=len(target_manifest_df),
        selected_window_rows=len(selected_windows_df),
        selected_window_unique_windows=_safe_nunique(selected_windows_df, "window_id"),
        selected_window_unique_media=selected_window_unique_media,
        processed_start_time_min=processed_start_time_min,
        processed_start_time_max=processed_start_time_max,
        zone_feature_rows=len(zone_features_df),
        zone_feature_unique_windows=_safe_nunique(zone_features_df, "window_id"),
        zone_feature_unique_rooms=_safe_nunique(zone_features_df, "room_id"),
        zone_feature_unique_sessions=_safe_nunique(zone_features_df, "session_id"),
        failed_target_videos=failed_target_videos,
        all_target_videos_processed=_count_target_raw_videos(config) == len(target_manifest_df),
        refresh_performed=False,
        refresh_reason="",
    )


def _rerun_mvp1_with_refresh_target(config: BiomarkerTwinConfig) -> None:
    from mvp_video_zone_heatmap.src.mvp1.main import run_pipeline as run_mvp1_pipeline

    with config.mvp1_config_yaml.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    raw_config.setdefault("selection", {})
    raw_config["selection"]["max_windows"] = int(config.mvp1_refresh_window_target)

    with NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
        temp_config_path = Path(handle.name)

    try:
        run_mvp1_pipeline(project_root=config.workspace_root, config_path=temp_config_path)
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()


def _read_csv(path_value: Path) -> pd.DataFrame:
    if not path_value.exists():
        return pd.DataFrame()
    return pd.read_csv(path_value)


def _safe_nunique(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].nunique())


def _safe_min_text(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    series = dataframe[column].dropna().astype(str)
    return str(series.min()) if not series.empty else ""


def _safe_max_text(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    series = dataframe[column].dropna().astype(str)
    return str(series.max()) if not series.empty else ""


def _filter_target_manifest(media_manifest_df: pd.DataFrame, config: BiomarkerTwinConfig) -> pd.DataFrame:
    if media_manifest_df.empty or not config.target_raw_subdir or "file_path" not in media_manifest_df.columns:
        return media_manifest_df.copy()
    normalized_target = str(config.target_raw_subdir).replace("\\", "/").strip().casefold()
    return media_manifest_df[
        media_manifest_df["file_path"].astype(str).apply(
            lambda value: normalized_target in str(value).replace("\\", "/").strip().casefold()
        )
    ].copy()


def _count_target_raw_videos(config: BiomarkerTwinConfig) -> int:
    if not config.target_raw_subdir:
        return 0
    target_dir = config.workspace_root / "first_week_data_cleaning" / "data" / "raw" / Path(config.target_raw_subdir)
    if not target_dir.exists():
        return 0
    return len(
        {
            path.resolve(strict=False)
            for pattern in ("*.MP4", "*.mp4")
            for path in target_dir.rglob(pattern)
        }
    )
