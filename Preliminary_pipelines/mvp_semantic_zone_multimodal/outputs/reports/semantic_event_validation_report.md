# Semantic Event Validation Report

- This is a new semantic-zone experiment layered on top of the existing analytics backbone. Previous code was not modified.
- Inputs for this experiment were two days of Room 1 video, embedded MP4 audio, daily Room 1 environment context, and the manual caretaker-entry label.
- This report is exploratory and does not claim validated welfare diagnosis.

## Answers

1. Did caretaker entry increase overall activity? `overall activity` increased from `0.0075` to `0.0198`.
2. Did activity concentrate more in drinking, feeding, or general areas? Functional-area activity fraction increased from `0.7886` to `0.8347`, and the dominant semantic activity fraction during entry was `feeding`.
3. Did semantic-zone features provide more interpretable response than four equal zones? Semantic mobility contrast was `0.2920` versus previous four-zone mobility contrast `0.2997`. The semantic version is more interpretable because it also shows whether motion shifts toward drinking or feeding areas.
4. Did HMM switch to a disturbance/transition or functional-zone state? The dominant during-entry state was `feeder_concentrated_activity`.
5. Did embedded audio features change during event? Audio RMS increased from `0.0646` to `0.0655`, and spectral centroid increased from `1475.4` to `1745.0`.
6. How should this be interpreted for the meeting? The semantic experiment adds a biologically meaningful description of where the activity response happened, not just whether total motion increased.
7. What are limitations? Semantic zones are manually defined, activity is not true occupancy, audio is whole-room embedded audio, environment is daily context, and the risk score is a prototype heuristic. Risk decreased from during-event `0.6469` toward recovery `0.5232`.

## Summary Table

```text
                               event_id               phase  n_windows  activity_mean_mean  activity_mean_max  mobility_index_mean  mobility_index_max  spatial_freedom_index_mean  occupancy_imbalance_index_mean  drinking_activity_fraction_mean  feeding_activity_fraction_mean  general_activity_fraction_mean  feeding_plus_drinking_activity_fraction_mean  semantic_transition_proxy_mean  audio_rms_mean  audio_spectral_centroid_mean  audio_spectral_rolloff_mean  risk_score_mean  risk_score_max         dominant_state_label  dominant_state_fraction
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline         58            0.007465           0.010143             0.185059            0.246221                    0.941238                        0.155014                         0.321919                        0.466706                        0.211375                                      0.788625                        0.043960        0.064553                   1475.440167                  3033.264371         0.568173        0.647371 feeder_concentrated_activity                 0.586207
caretaker_entry_room1_week11_2025_08_17        during_entry         14            0.019842           0.028266             0.477058            0.668330                    0.904057                        0.199356                         0.322520                        0.512170                        0.165310                                      0.834690                        0.052753        0.065509                   1744.977033                  3974.516654         0.646850        0.647371 feeder_concentrated_activity                 1.000000
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery        120            0.006623           0.012770             0.163247            0.307731                    0.917495                        0.188894                         0.267542                        0.517640                        0.214817                                      0.785183                        0.036895        0.065138                   1394.090206                  2759.624172         0.523154        0.647371     distributed_low_activity                 0.641667
```

## State Transition Sequence Around Event

- `-10.15` min: `distributed_low_activity`
- `-9.32` min: `feeder_concentrated_activity`
- `-8.82` min: `distributed_low_activity`
- `-8.32` min: `feeder_concentrated_activity`
- `-2.80` min: `distributed_low_activity`
- `-0.13` min: `feeder_concentrated_activity`
- `+3.03` min: `distributed_low_activity`
- `+3.53` min: `feeder_concentrated_activity`
- `+6.70` min: `distributed_low_activity`
- `+8.20` min: `feeder_concentrated_activity`
- `+8.70` min: `distributed_low_activity`
- `+9.37` min: `feeder_concentrated_activity`
- `+10.37` min: `distributed_low_activity`
- `+12.20` min: `feeder_concentrated_activity`
- `+12.70` min: `distributed_low_activity`
- `+14.70` min: `feeder_concentrated_activity`
- `+14.87` min: `distributed_low_activity`
- `+16.37` min: `feeder_concentrated_activity`
- `+17.37` min: `distributed_low_activity`
