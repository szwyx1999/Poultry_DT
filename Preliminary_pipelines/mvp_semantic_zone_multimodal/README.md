# MVP Semantic Zone Multimodal

This package is a new analytics experiment that replaces the previous four equal image-space zones with manually defined semantic zones for Room 1 Aug 16-17 videos:

- drinking area
- feeding area
- general area


## What This Experiment Does

- reads the existing Room 1 Aug 16-17 indexed windows
- detects semantic zones from the annotated reference frame and falls back to a manual YAML config if automatic box detection is not clean enough
- computes semantic-zone activity for all available windows
- extracts embedded MP4 audio only
- merges daily Room 1 environmental context
- fits two semantic-zone HMM variants:
  - `semantic_video_only`
  - `semantic_multimodal`
- validates against the labelled caretaker-entry event


## Inputs

- `first_week_data_cleaning/data/metadata/semantic_zone_refs/room1_aug16_17_reference.png`
- `first_week_data_cleaning/data/metadata/semantic_zone_refs/room1_aug16_17_reference_with_notes.png`
- `first_week_data_cleaning/data/processed/metadata/video_window_index.csv`
- `first_week_data_cleaning/data/processed/metadata/media_manifest.csv`
- `first_week_data_cleaning/data/raw/video/Room 1/Room 1 (16, 17 Aug)`
- `first_week_data_cleaning/data/raw/env/Combined Room 1.xlsx`
- `mvp_biomarker_state_twin/data/event_log.csv` if available

## Run

From the workspace root:

```bash
python -m mvp_semantic_zone_multimodal.src.main --config mvp_semantic_zone_multimodal/config/default.yaml
```

The default config targets all currently available Room 1 Aug 16-17 windows from:

- `first_week_data_cleaning/data/raw/video/Room 1/Room 1 (16, 17 Aug)`

## Core Outputs

- `outputs/zones/semantic_zone_config_room1_aug16_17.json`
- `outputs/zones/semantic_zone_overlay.png`
- `outputs/features/semantic_zone_video_features.csv`
- `outputs/features/semantic_biomarker_window_table.csv`
- `outputs/features/audio_window_features.csv`
- `outputs/features/env_room1_daily.csv`
- `outputs/features/semantic_multimodal_window_table.csv`
- `outputs/features/semantic_labelled_window_table.csv`
- `outputs/features/semantic_hmm_state_sequence.csv`
- `outputs/features/semantic_hmm_state_summary.csv`
- `outputs/features/semantic_event_validation_summary.csv`
- `outputs/reports/semantic_zone_progress_summary_for_meeting.md`