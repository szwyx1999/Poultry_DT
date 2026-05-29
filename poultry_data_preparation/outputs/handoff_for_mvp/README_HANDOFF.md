# Handoff For MVP

This handoff folder contains processed metadata and feature tables for downstream MVPs.

## Files

- `media_manifest.csv`: raw-video media-level manifest with technical metadata and timestamps
- `video_window_index.csv`: fixed video windows and primary handoff for window-based processing
- `semantic_zone_video_features.csv`: one row per room/media/window/semantic zone activity feature
- `semantic_biomarker_window_table.csv`: window-level semantic activity distribution features
- `audio_window_features.csv`: embedded MP4 audio features aggregated to the same windows
- `env_daily.csv`: room-level daily environment context
- `processed_multimodal_window_table.csv`: merged handoff table for downstream analytics
- `semantic_zone_configs.json`: semantic zone geometry definitions per room

## Notes

- Videos are not copied into this folder.
- Path columns may be absolute or relative depending on `paths_in_outputs` in the config.
- Downstream MVPs should read `video_window_index.csv` and `processed_multimodal_window_table.csv` as their main window-level handoff tables.
