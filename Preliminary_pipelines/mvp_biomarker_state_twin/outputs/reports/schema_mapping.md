# Schema Mapping

This report documents how the MVP1 feature exports were normalized into the canonical zone feature table.

## Identifier Mapping

| Canonical field | Source column |
| --- | --- |
| `window_id` | `window_id` |
| `media_id` | `media_id` |
| `system_id` | `system_id` |
| `room_id` | `room_id` |
| `session_id` | `session_id` |
| `zone_id` | `zone_id` |
| `start_time` | `start_time` |
| `end_time` | `end_time` |
| `zone_config_id` | `zone_config_id` |

## Activity And Metadata Mapping

| Canonical field | Source column |
| --- | --- |
| `activity_mean` | `activity_mean` |
| `activity_std` | `activity_std` |
| `motion_pixel_ratio` | `motion_pixel_ratio` |
| `frame_count` | `frame_count` |
| `video_path` | `video_path` |
| `video_start_offset_sec` | `video_start_offset_sec` |
| `duration_seconds` | `duration_seconds` |
| `has_audio` | `has_audio` |
| `quality_status` | `quality_status` |
| `selection_rank` | `selection_rank` |
| `warnings` | `warnings` |

## Canonical Table Summary

- Rows: 46120
- Unique windows: 11530
- Unique rooms: 1
- Unique zones: 4
- Activity proxy priority: `activity_mean` -> `motion_pixel_ratio` -> `activity_std`

## Notes

The canonical table preserves one row per `window_id` and `zone_id`.

Observed activity proxy source counts: {'activity_mean': 46120}.

When `zone_config_id` was absent from `video_zone_features.csv`, it was backfilled from the selected-window and preprocessing metadata tables.
