# Poultry Data Preparation

## Purpose

This package converts raw MP4 video, embedded MP4 audio, semantic zone reference images, and Room-level environment spreadsheets into standardized processed CSV or parquet tables for downstream analytics.

## Required Input Folder Structure

```text
data/
  raw/
    video/
      Room 1/
        Room 1 (16, 17 Aug)/
          *.MP4
      Room 2/
        ...
    env/
      Combined Room 1.xlsx
      Combined Room 2.xlsx
  metadata/
    semantic_zone_refs/
      room1_aug16_17_reference.png
      room1_aug16_17_reference_with_notes.png
      room2_..._reference.png
      room2_..._reference_with_notes.png
```

## How Room IDs Are Inferred

Room IDs are parsed from folder names and filenames. Examples:

- `Room 1` -> `room_1`
- `room1_aug16_17_reference.png` -> `room_1`
- `Combined Room 2.xlsx` -> `room_2`

If a room cannot be parsed, the pipeline assigns `unknown_room`, emits a warning, and includes it in reports.

## Run A Small Test

From the workspace root:

```bash
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage all --max-windows 100
```

Small test with both media and window caps:

```bash
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage all --max-media 2 --max-windows 100
```

Dry run:

```bash
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage all --dry-run
```

## Run Full External-Drive Processing

Example:

```bash
python -m poultry_data_preparation.src.main --config E:/PoultryProject/config/full_run.yaml --stage all
```

The config can live with the external project and resolve all paths relative to its `project_root`.

## Stages

Supported stages:

- `index`
- `zones`
- `video_features`
- `audio_features`
- `env`
- `merge`
- `all`

Examples:

```bash
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage index
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage zones
python -m poultry_data_preparation.src.main --config poultry_data_preparation/config/default.yaml --stage merge --max-windows 100
```

## Resume And Safety

- Video and audio stages cache per-media computations under `outputs/cache/`.
- Reruns reuse cached media-level features unless `force_recompute=true` is set in the config.
- Dry-run mode scans inputs and writes reports, but does not perform heavy video or audio extraction.
- Videos are never copied into the output or handoff folders.

## Outputs

Primary outputs:

- `outputs/metadata/media_manifest.csv`
- `outputs/metadata/video_window_index.csv`
- `outputs/zones/semantic_zone_configs.json`
- `outputs/features/semantic_zone_video_features.csv`
- `outputs/features/semantic_biomarker_window_table.csv`
- `outputs/features/audio_window_features.csv`
- `outputs/features/env_daily.csv`
- `outputs/features/processed_multimodal_window_table.csv`

Handoff package:

- `outputs/handoff_for_mvp/README_HANDOFF.md`
- copies of the main processed tables