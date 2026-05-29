# MVP Audio + Environment Context

This package extends the existing video-only poultry digital twin analytics backbone with:

- embedded MP4 audio features extracted from the same indexed video windows
- daily Room 1 environmental context from `Combined Room 1.xlsx`
- multimodal event validation around the labelled caretaker-entry event
- optional HMM ablation comparing video-only vs video+audio+environment feature sets

What this MVP does:

- reads the already-processed metadata, MVP1 zone activity features, and biomarker/HMM outputs
- extracts embedded audio from MP4 files only
- merges audio, daily environment context, and video-derived biomarker/HMM outputs at the window level
- produces exploratory reports and plots for multimodal interpretation

## Inputs

Required upstream inputs:

- `first_week_data_cleaning/data/processed/metadata/video_window_index.csv`
- `first_week_data_cleaning/data/processed/metadata/media_manifest.csv`
- `mvp_video_zone_heatmap/outputs/features/selected_windows.csv`
- `mvp_video_zone_heatmap/outputs/features/video_zone_features.csv`
- `mvp_biomarker_state_twin/outputs/features/biomarker_window_table.csv`
- `mvp_biomarker_state_twin/outputs/features/hmm_state_sequence.csv`
- `first_week_data_cleaning/data/raw/env/Combined Room 1.xlsx`

Optional:

- `mvp_biomarker_state_twin/data/event_log.csv`

## Run

From the workspace root:

```bash
python -m mvp_audio_env_context.src.main --config mvp_audio_env_context/config/default.yaml
```

## Main Outputs

Feature tables:

- `mvp_audio_env_context/outputs/features/env_room1_daily.csv`
- `mvp_audio_env_context/outputs/features/audio_window_features.csv`
- `mvp_audio_env_context/outputs/features/multimodal_window_table.csv`
- `mvp_audio_env_context/outputs/features/audio_env_correlation_table.csv`
- `mvp_audio_env_context/outputs/features/event_multimodal_summary.csv`

Reports:

- `mvp_audio_env_context/outputs/reports/env_data_report.md`
- `mvp_audio_env_context/outputs/reports/audio_extraction_report.md`
- `mvp_audio_env_context/outputs/reports/multimodal_merge_report.md`
- `mvp_audio_env_context/outputs/reports/audio_env_correlation_report.md`
- `mvp_audio_env_context/outputs/reports/event_multimodal_validation_report.md`
- `mvp_audio_env_context/outputs/reports/progress_summary_for_meeting.md`

Optional ablation outputs:

- `mvp_audio_env_context/outputs/features/hmm_ablation_state_sequences.csv`
- `mvp_audio_env_context/outputs/features/hmm_ablation_metrics.csv`
- `mvp_audio_env_context/outputs/reports/multimodal_hmm_ablation_report.md`

