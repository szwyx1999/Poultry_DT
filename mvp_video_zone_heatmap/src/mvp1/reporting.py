from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Mvp1Config


def write_processing_coverage_report(
    config: Mvp1Config,
    selected_windows: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Path:
    report_path = config.reports_dir / "aug16_17_processing_coverage.md"
    manifest_path = config.preprocess_root / "data/processed/metadata/media_manifest.csv"
    manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    target_filter = _normalize_path_text(config.target_raw_subdir or "")
    if target_filter and not manifest_df.empty and "file_path" in manifest_df.columns:
        manifest_df = manifest_df[
            manifest_df["file_path"].astype(str).apply(lambda value: target_filter in _normalize_path_text(value))
        ].copy()

    raw_video_count = _count_target_raw_videos(config)
    indexed_video_count = len(manifest_df)
    selected_window_count = int(selected_windows["window_id"].nunique()) if not selected_windows.empty and "window_id" in selected_windows.columns else 0
    feature_window_count = int(features_df["window_id"].nunique()) if not features_df.empty and "window_id" in features_df.columns else 0
    unique_sessions = int(selected_windows["session_id"].nunique()) if not selected_windows.empty and "session_id" in selected_windows.columns else 0
    unique_media_files = int(selected_windows["media_id"].nunique()) if not selected_windows.empty and "media_id" in selected_windows.columns else 0
    start_time_min = _safe_min_text(selected_windows, "start_time")
    start_time_max = _safe_max_text(selected_windows, "start_time")

    failed_video_rows = manifest_df[
        manifest_df.get("quality_status", pd.Series(dtype=object)).astype(str).str.lower() == "error"
    ].copy() if not manifest_df.empty else pd.DataFrame()

    feature_warning_windows = pd.DataFrame()
    if not features_df.empty and "warnings" in features_df.columns:
        feature_warning_windows = (
            features_df[features_df["warnings"].astype(str).str.strip() != ""]
            .groupby(["window_id", "media_id", "video_path"], dropna=False, sort=False)["warnings"]
            .first()
            .reset_index()
        )

    event_log_path = config.project_root / "mvp_biomarker_state_twin" / "data" / "event_log.csv"
    caretaker_event_time = None
    event_covered = "unknown"
    if event_log_path.exists():
        event_log_df = pd.read_csv(event_log_path)
        if not event_log_df.empty and "event_start_time" in event_log_df.columns:
            target_events = event_log_df[event_log_df["event_id"].astype(str) == "caretaker_entry_room1_week11_2025_08_17"]
            if not target_events.empty:
                caretaker_event_time = str(target_events.iloc[0]["event_start_time"])
                start_dt = pd.to_datetime(start_time_min, errors="coerce", utc=False)
                end_dt = pd.to_datetime(start_time_max, errors="coerce", utc=False)
                event_dt = pd.to_datetime(caretaker_event_time, errors="coerce", utc=False)
                if pd.notna(start_dt) and pd.notna(end_dt) and pd.notna(event_dt):
                    event_covered = "yes" if start_dt <= event_dt <= end_dt else "no"

    lines = [
        "# Aug 16-17 Processing Coverage",
        "",
        "This report summarizes the full Room 1 Aug 16-17 MVP1 processing run.",
        "",
        f"- Target raw subdir: `{config.target_raw_subdir or 'none'}`",
        f"- Selection strategy: `{config.selection_strategy}`",
        f"- Include all available: {'yes' if config.include_all_available else 'no'}",
        f"- Skip bad quality: {'yes' if config.skip_bad_quality else 'no'}",
        f"- Raw videos found in target folder: {raw_video_count}",
        f"- Videos indexed in `media_manifest.csv`: {indexed_video_count}",
        f"- Selected windows: {selected_window_count}",
        f"- Extracted feature windows: {feature_window_count}",
        f"- Start time min: {start_time_min or 'n/a'}",
        f"- Start time max: {start_time_max or 'n/a'}",
        f"- Unique sessions: {unique_sessions}",
        f"- Unique media files: {unique_media_files}",
        f"- Failed indexed videos: {len(failed_video_rows)}",
        f"- Windows with non-empty warnings in `video_zone_features.csv`: {len(feature_warning_windows)}",
        f"- Caretaker event start time: {caretaker_event_time or 'n/a'}",
        f"- Caretaker event falls inside processed window range: {event_covered}",
        f"- All raw videos under target folder were indexed: {'yes' if raw_video_count == indexed_video_count else 'no'}",
    ]

    if not failed_video_rows.empty:
        display_columns = [column for column in ("file_name", "start_time", "quality_status", "warnings") if column in failed_video_rows.columns]
        lines.extend(
            [
                "",
                "## Failed Indexed Videos",
                "",
                "```text",
                failed_video_rows[display_columns].to_string(index=False),
                "```",
            ]
        )

    if not feature_warning_windows.empty:
        lines.extend(
            [
                "",
                "## Warning Windows",
                "",
                "```text",
                feature_warning_windows.head(40).to_string(index=False),
                "```",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _count_target_raw_videos(config: Mvp1Config) -> int:
    if not config.target_raw_subdir:
        return 0
    target_dir = config.preprocess_root / "data/raw" / Path(config.target_raw_subdir)
    if not target_dir.exists():
        return 0
    return len(
        {
            path.resolve(strict=False)
            for pattern in ("*.MP4", "*.mp4")
            for path in target_dir.rglob(pattern)
        }
    )


def _normalize_path_text(value: object) -> str:
    return str(value).replace("\\", "/").strip().casefold()


def _safe_min_text(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    return str(dataframe[column].dropna().astype(str).min()) if not dataframe[column].dropna().empty else ""


def _safe_max_text(dataframe: pd.DataFrame, column: str) -> str:
    if dataframe.empty or column not in dataframe.columns:
        return ""
    return str(dataframe[column].dropna().astype(str).max()) if not dataframe[column].dropna().empty else ""
