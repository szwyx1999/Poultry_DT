# Input Folder Structure

This file documents the raw input layout expected by `poultry_data_preparation` and shows the current sample subset that exists in this workspace.

## Generic Expected Layout

```text
data/
  raw/
    video/
      Room 1/
        Room 1 (16, 17 Aug)/
          GX010044.MP4
          GX020044.MP4
          ...
    env/
      Combined Room 1.xlsx
      Combined Room 2.xlsx
    Caretaker Entry Video/
      Room 1/
        Week 11/
          17 Aug/
            GX320044.MP4
            GX330044.MP4
  metadata/
    semantic_zone_refs/
      room1_aug16_17_reference.png
      room1_aug16_17_reference_with_notes.png
```

## Current Workspace Sample Subset

The current raw sample subset lives under:

```text
first_week_data_cleaning/
  data/
    raw/
      video/
        Room 1/
          Room 1 (10, 11, 12, 13 Aug)/
            GX010042.MP4
            GX010043.MP4
            GX020043.MP4
          Room 1 (16, 17 Aug)/
            GX010044.MP4
            GX020044.MP4
            ...
      env/
        Combined Room 1.xlsx
      Caretaker Entry Video/
        Room 1/
          Week 11/
            17 Aug/
              GX320044.MP4
              GX330044.MP4
    metadata/
      semantic_zone_refs/
        room1_aug16_17_reference.png
        room1_aug16_17_reference_with_notes.png
```

## Required vs Optional Inputs

Required for the main preprocessing workflow:

- raw MP4 video files
- semantic-zone reference image files
- room-level environment workbook files

Optional or not part of the current main preprocessing output path:

- caretaker-entry reference videos
- any future external behavior-detection CSVs
- any external audio files

## Timestamp Notes

No separate timestamp CSV is required.

The preprocessing code resolves video timestamps from:

1. `exiftool` MP4 metadata when available
2. `ffprobe` creation-time metadata when available
3. file modification time as a fallback

Those resolved timestamps appear later in:

- `outputs/metadata/media_manifest.csv`
- `outputs/metadata/video_window_index.csv`
- `outputs/handoff_for_mvp/media_manifest.csv`
- `outputs/handoff_for_mvp/video_window_index.csv`

## Prepared Files Passed To The Current Gaussian HMM Workflow

The current `mvp_prepared_data_state_model` main workflow reads these prepared outputs:

- `outputs/handoff_for_mvp/processed_multimodal_window_table.csv`
- `outputs/handoff_for_mvp/video_window_index.csv`
- `outputs/handoff_for_mvp/media_manifest.csv`

It does not read raw videos directly.
