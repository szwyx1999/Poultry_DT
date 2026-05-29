# MVP 1 Video Zone Heatmap

`mvp_video_zone_heatmap/` is the first modeling layer built on top of the week-one preprocessing outputs. Its job is to prove that real indexed MP4 windows can be converted into zone-level activity features.

## What MVP 1 is

- A video zone heatmap prototype.
- A bridge from cleaned video windows to simple activity features.
- An early building block for the larger poultry welfare digital twin.

## Dependency on `first_week_data_cleaning`

MVP 1 consumes the cleaned video window index produced by `first_week_data_cleaning/`.

- Input handoff table: `first_week_data_cleaning/data/processed/metadata/video_window_index.csv`
- Optional debug context: `first_week_data_cleaning/data/processed/metadata/media_manifest.csv`
- Raw MP4 paths are taken from the window index and resolved relative to `first_week_data_cleaning/`

## How to run

Install dependencies:

```bash
python -m pip install -r mvp_video_zone_heatmap/requirements.txt
```

Run the pipeline from the workspace root:

```bash
python -m mvp_video_zone_heatmap.src.mvp1.main
```

Run with a custom config:

```bash
python -m mvp_video_zone_heatmap.src.mvp1.main --config path/to/custom_config.yaml
```

## Outputs

- `mvp_video_zone_heatmap/outputs/features/selected_windows.csv`
- `mvp_video_zone_heatmap/outputs/features/video_zone_features.csv`
- `mvp_video_zone_heatmap/outputs/unity_json/mvp1_zone_activity_timeline.json`
- `mvp_video_zone_heatmap/outputs/plots/zone_activity_over_time.png`
- `mvp_video_zone_heatmap/outputs/plots/zone_activity_heatmap.png`
- `mvp_video_zone_heatmap/outputs/plots/sample_frames/*.png`

## Results Description

- `selected_windows.csv`: confirms which windows were chosen from the cleaned index.
- Preview frames: quick visual check that the selected MP4 windows opened correctly.
- `video_zone_features.csv`: one row per window and image-space zone.
- Plots: confirms activity is not all zero or all missing.
- `mvp1_zone_activity_timeline.json`: structured export for later Unity playback.

