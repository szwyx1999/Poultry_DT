# HMM Sequence Diagnostics

This report validates that the latent-state sequence is window-level and temporally well-defined before plotting.

## Summary

- Total rows in `canonical_zone_feature_table.csv`: 46120
- Total rows in `biomarker_window_table.csv`: 11530
- Total rows in `hmm_state_sequence.csv`: 11530
- Model type: gaussian_hmm
- Requested HMM state setting: auto
- Fitted HMM components used: 4
- Occupied latent states observed: 3
- Number of unique rooms: 1
- Number of unique sessions: 1
- Number of unique windows: 11530
- Number of event-labelled windows: 192
- Caretaker event covered by processed windows: yes
- Number of unique room/window pairs: 11530
- Duplicate `room_id` + `window_id` row count: 0
- Duplicate `room_id` + `session_id` + `start_time` row count: 0
- Any time window with multiple assigned states: no
- Multiple-state `room_id` + `session_id` + `start_time` group count: 0
- Valid temporal sequence for plotting: yes

## Sequence Breakdown

```text
room_id       session_id  row_count  unique_windows            start_time_min            start_time_max
 room_1 room_1_16_17_aug      11530           11530 2025-08-16T06:00:03-03:00 2025-08-17T14:14:53-03:00
```
