
## File structure

```text
data/
  raw/                                   # Root directory for raw MP4 data
    free_range/room1/session_2025_09_01/video/GX010027.mp4
    room1/session_2025_09_01/video/GX010027.mp4
    video/Room 1/Room 1 (16, 17 Aug)/GX010044.MP4

  metadata/
    preprocessing_config.yaml            # Preprocessing configuration
    path_mapping.csv                     # Optional path parsing override table

  processed/
    metadata/
      media_manifest.csv                 # One row per MP4
      video_window_index.csv             # Main entry table for the downstream MVP
      preprocessing_report.md            # Human-readable summary report
    exif/
      <media_id>_exif.json               # Raw exiftool result for each MP4
    ffprobe/
      <media_id>_ffprobe.json            # Raw ffprobe result for each MP4

outputs/
  logs/
    preprocess.log                       # Run log

src/
  preprocess/

tests/
```

## Where the input data is located

The preprocessing stage only depends on the following inputs:

- Raw videos: `data/raw/**/*.mp4`
- Raw videos: `data/raw/**/*.MP4`
- Configuration file: `data/metadata/preprocessing_config.yaml`
- Optional override table: `data/metadata/path_mapping.csv`


## Raw directory formats

By default, the following path layouts relative to `data/raw/` are supported:

- `system_id/room_id/session_id/modality`

For example:

```text
data/raw/free_range/room1/session_2025_09_01/video/GX010027.mp4
```

will be parsed as:

- `system_id=free_range`
- `room_id=room1`
- `session_id=session_2025_09_01`
- `modality=video`

If a field is missing from the path, it will be written as:

- `unknown_system`
- `unknown_room`
- `unknown_session`

If manual overrides are needed for path parsing results, use:

- `data/metadata/path_mapping.csv`

Example:

```csv
pattern,system_id,room_id,session_id,modality,zone_config_id
free_range/room1,free_range,room1,session_2025_09_01,video,zone_alpha
```


## How to run

Run from the project root directory:

```bash
python -m src.preprocess.main
```

To specify another configuration file:

```bash
python -m src.preprocess.main --config path/to/custom_config.yaml
```

## Where the preprocessing outputs are located

### 1 Main outputs

1. `data/processed/metadata/media_manifest.csv`
2. `data/processed/metadata/video_window_index.csv`
3. `data/processed/metadata/preprocessing_report.md`
4. `data/processed/exif/<media_id>_exif.json`
5. `data/processed/ffprobe/<media_id>_ffprobe.json`
6. `outputs/logs/preprocess.log`

### 2 Purpose of each file

`data/processed/metadata/media_manifest.csv`

- One row per MP4
- Used for data quality checks, tracing, and debugging
- Includes path context, timestamps, duration, resolution, frame rate, codec, whether embedded audio exists, warnings, and more
- If a downstream window has an issue, this file can be looked up using `media_id`

`data/processed/metadata/video_window_index.csv`

- Main entry point for the downstream MVP feature extraction stage
- Each row represents one video segment that needs to be processed
- Downstream code should prioritize reading this file instead of rescanning `data/raw/`

`data/processed/metadata/preprocessing_report.md`

- Human-readable summary report
- Used to quickly check data volume, missing items, and warning types
- Not used as the main input for downstream programs

`data/processed/exif/<media_id>_exif.json`

- Raw `exiftool` output
- Used to trace timestamp sources and troubleshoot EXIF anomalies

`data/processed/ffprobe/<media_id>_ffprobe.json`

- Raw `ffprobe` output
- Used to trace low-level information for video streams / audio streams / codec / duration

`outputs/logs/preprocess.log`

- Log for the entire run
- Used to troubleshoot which files errored or generated warnings during processing

