# Media Index Report

- Dry run: `False`
- ffprobe available: `True`
- exiftool available: `True`
- Number of videos found: 310
- Files missing timestamps and using file mtime fallback: 1
- Files missing audio: 1

## Videos Per Room

- `room_1`: 310 videos (start `2025-08-10T11:25:46`, end `2026-05-25T18:32:27.775652-03:00`)

## Codec Summary

```text
video_codec audio_codec  count
       hevc         aac    309
        NaN         NaN      1
```

## Warning Summary

- `GX410044.MP4`: `ffprobe_failed:[mov,mp4,m4a,3gp,3g2,mj2 @ 0000022336160240] moov atom not found
F:\temp\data\raw\video\Room 1\Room 1 (13, 14, 15 Aug)\GX410044.MP4: Invalid data found when processing input;timestamp_source=file_mtime_fallback`
