from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config
from .export_unity import export_unity_timeline
from .plots import generate_sanity_plots
from .reporting import write_processing_coverage_report
from .select_windows import select_video_windows
from .video_features import (
    extract_video_zone_features,
    extract_video_zone_features_streaming,
    load_completed_feature_rows,
    save_feature_csv,
)
from .video_reader import collect_window_reads
from .zone_utils import load_zone_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zone-level activity features from preprocessed poultry video windows."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the MVP 1 YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all selected window features instead of resuming from the existing CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    try:
        run_pipeline(project_root=project_root, config_path=args.config, force=args.force)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc))
        return 1
    return 0


def run_pipeline(project_root: Path, config_path: str | Path = DEFAULT_CONFIG_PATH, force: bool = False) -> dict[str, Path]:
    config = load_config(project_root=project_root, config_path=config_path)
    if force:
        config.force_recompute = True
    zone_config = load_zone_config(config.zone_config)

    print("Selecting video windows...")
    selected_windows = select_video_windows(config)

    features_path = config.features_dir / "video_zone_features.csv"
    features_df = _extract_features(selected_windows, zone_config, config, features_path)

    print("Exporting Unity-readable JSON...")
    unity_json_path = export_unity_timeline(features_df, zone_config, config)

    print("Generating sanity-check plots...")
    plot_paths = generate_sanity_plots(features_df, zone_config, config)
    coverage_report_path = write_processing_coverage_report(config, selected_windows, features_df)

    print("MVP 1 pipeline complete.")
    print(f"Features CSV: {features_path.as_posix()}")
    print(f"Unity JSON: {unity_json_path.as_posix()}")
    print(f"Plots: {plot_paths['zone_activity_over_time'].as_posix()}, {plot_paths['zone_activity_heatmap'].as_posix()}")
    print(f"Coverage report: {coverage_report_path.as_posix()}")

    return {
        "features_path": features_path,
        "unity_json_path": unity_json_path,
        "zone_activity_over_time_path": plot_paths["zone_activity_over_time"],
        "zone_activity_heatmap_path": plot_paths["zone_activity_heatmap"],
        "coverage_report_path": coverage_report_path,
    }


def _extract_features(
    selected_windows: pd.DataFrame,
    zone_config: dict,
    config,
    features_path: Path,
) -> pd.DataFrame:
    if selected_windows.empty:
        empty_df = pd.DataFrame()
        save_feature_csv(empty_df, features_path)
        return empty_df

    use_streaming_mode = (
        config.include_all_available
        or config.selection_strategy == "all"
        or len(selected_windows) > 250
        or not config.save_preview_frames
        or config.resume_enabled
    )

    if not use_streaming_mode:
        print("Reading MP4 windows and saving preview frames...")
        read_results = collect_window_reads(selected_windows, config)
        print("Extracting zone-level activity features...")
        features_df = extract_video_zone_features(
            selected_windows=selected_windows,
            read_results=read_results,
            zone_config=zone_config,
            config=config,
        )
        features_df = _sort_feature_rows(features_df)
        save_feature_csv(features_df, features_path)
        return features_df

    cached_df = pd.DataFrame()
    completed_window_ids: set[str] = set()
    if config.resume_enabled and not config.force_recompute:
        cached_df, completed_window_ids = load_completed_feature_rows(features_path, selected_windows, zone_config)
        cached_df = cached_df[cached_df["window_id"].astype(str).isin(completed_window_ids)].copy() if not cached_df.empty else cached_df
        if completed_window_ids:
            print(f"Resuming from existing feature cache: {len(completed_window_ids)} completed window(s) found.")

    pending_windows = selected_windows[
        ~selected_windows["window_id"].astype(str).isin(completed_window_ids)
    ].copy()

    feature_chunks: list[pd.DataFrame] = [cached_df] if not cached_df.empty else []
    pending_media_groups = list(pending_windows.groupby("media_id", sort=False, dropna=False))
    if not pending_media_groups:
        print("All selected windows already have cached feature rows.")
        if not feature_chunks:
            features_df = pd.DataFrame()
        else:
            features_df = _sort_feature_rows(pd.concat(feature_chunks, ignore_index=True))
        save_feature_csv(features_df, features_path)
        return features_df

    print(f"Extracting zone-level activity features in streaming mode for {len(pending_media_groups)} media file(s)...")
    for media_index, (_, media_windows) in enumerate(pending_media_groups, start=1):
        media_windows = media_windows.reset_index(drop=True)
        media_id = str(media_windows.iloc[0]["media_id"])
        print(f"[{media_index}/{len(pending_media_groups)}] Processing {media_id} ({len(media_windows)} window(s))")
        media_features_df = extract_video_zone_features_streaming(
            selected_windows=media_windows,
            zone_config=zone_config,
            config=config,
        )
        feature_chunks.append(media_features_df)
        partial_df = _sort_feature_rows(pd.concat(feature_chunks, ignore_index=True))
        save_feature_csv(partial_df, features_path)

    features_df = _sort_feature_rows(pd.concat(feature_chunks, ignore_index=True))
    save_feature_csv(features_df, features_path)
    return features_df


def _sort_feature_rows(features_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty:
        return features_df
    sort_columns = [column for column in ("start_time", "window_id", "zone_id") if column in features_df.columns]
    return features_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
