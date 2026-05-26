# Data Coverage Report

This report documents how much video-derived MVP1 data was available and how much was actually used for biomarker and latent-state modelling.

## Source Tables

- `video_window_index.csv` rows: 12128
- `video_window_index.csv` unique windows: 12128
- `video_window_index.csv` unique rooms: 1
- `video_window_index.csv` unique sessions: 2
- Target raw videos found: 39
- Target videos indexed in `media_manifest.csv`: 39
- `selected_windows.csv` rows: 11530
- `selected_windows.csv` unique windows: 11530
- `selected_windows.csv` unique media files: 39
- `video_zone_features.csv` rows: 46120
- `video_zone_features.csv` unique windows: 11530
- `video_zone_features.csv` unique rooms: 1
- `video_zone_features.csv` unique sessions: 1
- Full processed time span start: 2025-08-16T06:00:03-03:00
- Full processed time span end: 2025-08-17T14:14:53-03:00
- All target-folder videos processed: yes
- MVP1 refresh performed during this run: no
- MVP1 refresh note: Existing MVP1 feature table already satisfied the biomarker/HMM minimum window count.

## Biomarker/HMM Input

- `canonical_zone_feature_table.csv` rows: 46120
- `biomarker_window_table.csv` rows used for modelling: 11530
- Unique rooms used: 1
- Unique sessions used: 1
- Unique windows used: 11530
- Configured HMM state setting: auto
- Fitted HMM components used: 4
- Occupied latent states observed in `hmm_state_sequence.csv`: 3

## Window Selection Settings

- `max_windows`: null
- `max_windows_per_room`: null
- `max_windows_per_session`: null
- `min_windows_for_hmm`: 30

## State Summary Coverage

- Occupied state rows represented in `hmm_state_summary.csv`: 3
- State rows in `hmm_state_summary.csv`: 3

Video-derived MVP1 features only were used at this stage. Audio, environmental, and thermal signals were not integrated into this run.
