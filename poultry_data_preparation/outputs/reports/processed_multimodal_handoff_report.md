# Processed Multimodal Handoff Report

- Total windows: 83548
- Windows per room: {'room_1': 83548}
- Time span per room: {'room_1': {'min': '2025-08-10T11:25:46', 'max': '2026-05-25T18:32:27.775652-03:00'}}
- Percent with video features: 100.00% 
- Percent with audio features: 100.00% 
- Percent with environment match: 100.00% 

## Missingness Summary

- `window_id`: 0
- `media_id`: 0
- `room_id`: 0
- `session_id`: 0
- `start_time`: 0
- `end_time`: 0
- `local_date`: 1
- `video_path`: 0
- `activity_mean`: 0
- `normalized_activity`: 0
- `mobility_index`: 0
- `spatial_freedom_index`: 0
- `occupancy_imbalance_index`: 0
- `drinking_activity_fraction`: 0
- `feeding_activity_fraction`: 0
- `general_activity_fraction`: 0
- `semantic_transition_proxy`: 0
- `audio_available`: 0
- `audio_rms`: 1
- `audio_short_time_energy`: 1
- `audio_zero_crossing_rate`: 1
- `audio_spectral_centroid`: 1
- `audio_spectral_bandwidth`: 1
- `audio_spectral_rolloff`: 1
- `temp_context`: 1
- `rh_context`: 1
- `temp_daily_mean`: 1
- `rh_daily_mean`: 1
- `env_quality_flag`: 1
- `quality_status`: 0
- `warnings`: 0

## Recommended Next Commands

- Downstream semantic-zone analytics can read `outputs/features/processed_multimodal_window_table.csv`.
- Window-based pipelines can also read `outputs/metadata/video_window_index.csv` and `outputs/features/semantic_biomarker_window_table.csv`.
