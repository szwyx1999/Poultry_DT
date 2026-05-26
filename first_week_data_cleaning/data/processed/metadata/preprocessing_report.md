# Preprocessing Report

## Summary
- Number of MP4 files found: 41
- Number of files with valid timestamps: 41
- Number of files missing timestamps: 0
- Number of files with valid duration: 41
- Number of files missing duration: 0
- Number of files with embedded audio: 41
- Number of files without embedded audio: 0
- Number of video windows generated: 12128

## Warnings
- `mapping_override`: 41

## Suggested Next Step
- Use `data/processed/metadata/video_window_index.csv` as the handoff table for MVP feature extraction. The later stage should open `video_path`, seek to `video_start_offset_sec`, process `duration_seconds`, and optionally read embedded audio from the same MP4.
