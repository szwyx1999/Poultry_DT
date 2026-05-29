# Event Validation Report

Known event: `caretaker_entry_room1_week11_2025_08_17`

This is a validation against one labelled management event, not biological welfare ground truth.

## Phase Summary

```text
                               event_id               phase  n_windows  activity_mean_mean  activity_mean_max  mobility_index_mean  mobility_index_max  spatial_freedom_index_mean  occupancy_imbalance_index_mean  risk_score_mean  risk_score_max      dominant_state_label  dominant_state_fraction
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline         58            0.292306           0.510139             0.191593            0.342997                    0.956132                        0.143457         0.223632        0.223632  distributed_low_activity                 1.000000
caretaker_entry_room1_week11_2025_08_17        during_entry         14            0.759885           1.025385             0.491312            0.660612                    0.871986                        0.251116         0.414972        0.429689 disturbance_or_transition                 0.928571
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery        120            0.243906           0.569966             0.163854            0.349436                    0.949343                        0.157059         0.223632        0.223642  distributed_low_activity                 1.000000
```

## Interpretation

- The video-derived biomarker system detected a change around the labelled caretaker-entry event.
- Strongest responding biomarker by mean phase shift: `activity_mean`
- Mobility increased during entry vs baseline: `yes`
- Mobility decreased during recovery vs during-entry: `yes`
- Risk score increased during entry vs baseline: `yes`
- Risk score decreased during recovery vs during-entry: `yes`
- Peak mobility response time relative to event start: `1.70 minutes`
- Peak risk response time relative to event start: `1.37 minutes`
- Recovery time after event end: `3.58 minutes after event end`

## HMM Behavioural State Context

- Dominant state transition sequence around event: `-10.15 min: distributed_low_activity -> +0.03 min: disturbance_or_transition -> +2.20 min: distributed_low_activity`
- Most frequent state during entry: `disturbance_or_transition`
- Most frequent state during recovery: `distributed_low_activity`

## Caveats

- Prototype video-derived biomarker and latent-state response only.
- Not a validated welfare diagnosis.
- Event timing is anchored to one manually offset reference video.

## Event Log Row

```text
                               event_id      event_type  system_id room_id                     session_id                                                 source_video_path source_media_id   source_video_start_time  entry_offset_sec  exit_offset_sec          event_start_time            event_end_time                      label_quality                                                             notes
caretaker_entry_room1_week11_2025_08_17 caretaker_entry free_range  room_1 caretaker_entry_week_11_17_aug data/raw/Caretaker Entry Video/Room 1/Week 11/17 Aug/GX330044.MP4    video_000002 2025-08-17T08:45:57-03:00               183              300 2025-08-17T08:49:00-03:00 2025-08-17T08:50:57-03:00 manual_offset_from_reference_video Caretaker enters GX330044.MP4 at 00:03:03 and leaves at 00:05:00.
```
