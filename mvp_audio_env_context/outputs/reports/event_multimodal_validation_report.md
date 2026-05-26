# Event Multimodal Validation Report

This report compares video biomarkers, embedded MP4 audio features, and daily environmental context around the labelled caretaker-entry event.

## Answers

1. Do video biomarkers respond during caretaker entry? `mobility_index` increased from baseline `0.1916` to during-event `0.4913`, and decreased toward baseline (recovery mean `0.1639`).
2. Do embedded audio features also change during caretaker entry? `audio_rms` increased from baseline `0.0646` to during-event `0.0655`, and decreased toward baseline (recovery mean `0.0651`). `audio_spectral_centroid` increased from baseline `1475.4402` to during-event `1744.9770`, and decreased toward baseline (recovery mean `1394.0902`).
3. Is the acoustic response interpretable after considering daily temp/RH context? Daily context during the event was temp `23.70` and RH `52.00`; interpret acoustic changes as event-aligned response under that coarse context, not as isolated environmental effect.
4. Does HMM state/risk align with video and audio response? `welfare_risk_score` increased from baseline `0.2236` to during-event `0.4150`, and decreased toward baseline (recovery mean `0.2236`). Dominant state sequence: -10.15 min: distributed_low_activity -> +0.03 min: disturbance_or_transition -> +2.20 min: distributed_low_activity
5. What are the caveats? Environment is daily Room 1 context, audio comes from embedded MP4 tracks, and the analysis remains exploratory rather than validated welfare diagnosis.

## Audio Spectral Framing

`audio_spectral_centroid` increased from baseline `1475.4402` to during-event `1744.9770`, and decreased toward baseline (recovery mean `1394.0902`).

## Summary Table

```text
                               event_id               phase                    metric  n_windows        mean      median         max          std  baseline_mean  fold_change_vs_baseline  mean_minus_baseline
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline             activity_mean         58    0.292306    0.291604    0.510139 7.270774e-02       0.292306                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline            mobility_index         58    0.191593    0.192233    0.342997 4.351259e-02       0.191593                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline     spatial_freedom_index         58    0.956132    0.960411    0.999442 2.408031e-02       0.956132                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline occupancy_imbalance_index         58    0.143457    0.144402    0.249808 4.995426e-02       0.143457                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline        welfare_risk_score         58    0.223632    0.223632    0.223632 7.350346e-09       0.223632                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline                 audio_rms         58    0.064553    0.064528    0.066380 7.974316e-04       0.064553                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline   audio_short_time_energy         58    0.004189    0.004183    0.004429 1.033307e-04       0.004189                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline  audio_zero_crossing_rate         58    0.043745    0.043361    0.048302 2.512988e-03       0.043745                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline   audio_spectral_centroid         58 1475.440167 1484.678528 1692.231201 1.187699e+02    1475.440167                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline  audio_spectral_bandwidth         58 1986.502056 1991.379883 2212.223877 1.318981e+02    1986.502056                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline    audio_spectral_rolloff         58 3033.264371 3020.033325 3802.100098 3.682682e+02    3033.264371                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline              temp_context         58   23.700000   23.700000   23.700000 0.000000e+00      23.700000                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17  pre_entry_baseline                rh_context         58   52.000000   52.000000   52.000000 0.000000e+00      52.000000                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17        during_entry             activity_mean         14    0.759885    0.758125    1.025385 1.587759e-01       0.292306                 2.599618         4.675786e-01
caretaker_entry_room1_week11_2025_08_17        during_entry            mobility_index         14    0.491312    0.495551    0.660612 8.723560e-02       0.191593                 2.564348         2.997183e-01
caretaker_entry_room1_week11_2025_08_17        during_entry     spatial_freedom_index         14    0.871986    0.900064    0.966649 8.786911e-02       0.956132                 0.911993        -8.414610e-02
caretaker_entry_room1_week11_2025_08_17        during_entry occupancy_imbalance_index         14    0.251116    0.237300    0.416418 9.507425e-02       0.143457                 1.750461         1.076591e-01
caretaker_entry_room1_week11_2025_08_17        during_entry        welfare_risk_score         14    0.414972    0.429689    0.429689 5.506868e-02       0.223632                 1.855601         1.913397e-01
caretaker_entry_room1_week11_2025_08_17        during_entry                 audio_rms         14    0.065509    0.064882    0.067921 1.305974e-03       0.064553                 1.014810         9.560449e-04
caretaker_entry_room1_week11_2025_08_17        during_entry   audio_short_time_energy         14    0.004372    0.004265    0.004713 2.278735e-04       0.004189                 1.043600         1.826503e-04
caretaker_entry_room1_week11_2025_08_17        during_entry  audio_zero_crossing_rate         14    0.052997    0.052749    0.061490 4.410056e-03       0.043745                 1.211497         9.251934e-03
caretaker_entry_room1_week11_2025_08_17        during_entry   audio_spectral_centroid         14 1744.977033 1751.568970 1974.141357 1.601736e+02    1475.440167                 1.182682         2.695369e+02
caretaker_entry_room1_week11_2025_08_17        during_entry  audio_spectral_bandwidth         14 2229.739842 2260.027832 2427.327393 1.518181e+02    1986.502056                 1.122445         2.432378e+02
caretaker_entry_room1_week11_2025_08_17        during_entry    audio_spectral_rolloff         14 3974.516654 4004.833374 4738.466797 5.032018e+02    3033.264371                 1.310310         9.412523e+02
caretaker_entry_room1_week11_2025_08_17        during_entry              temp_context         14   23.700000   23.700000   23.700000 3.686825e-15      23.700000                 1.000000        -3.552714e-15
caretaker_entry_room1_week11_2025_08_17        during_entry                rh_context         14   52.000000   52.000000   52.000000 0.000000e+00      52.000000                 1.000000         0.000000e+00
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery             activity_mean        120    0.243906    0.214967    0.569966 9.998524e-02       0.292306                 0.834418        -4.840066e-02
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery            mobility_index        120    0.163854    0.149837    0.349436 5.881433e-02       0.191593                 0.855220        -2.773881e-02
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery     spatial_freedom_index        120    0.949343    0.955229    0.994739 3.464808e-02       0.956132                 0.992900        -6.788682e-03
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery occupancy_imbalance_index        120    0.157059    0.153626    0.319314 5.981520e-02       0.143457                 1.094816         1.360207e-02
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery        welfare_risk_score        120    0.223632    0.223632    0.223642 9.382715e-07       0.223632                 1.000000         9.087581e-08
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery                 audio_rms        120    0.065138    0.065107    0.067951 8.725560e-04       0.064553                 1.009068         5.853870e-04
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery   audio_short_time_energy        120    0.004274    0.004264    0.004646 1.131999e-04       0.004189                 1.020188         8.457127e-05
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery  audio_zero_crossing_rate        120    0.042478    0.041335    0.058379 4.095622e-03       0.043745                 0.971046        -1.266614e-03
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery   audio_spectral_centroid        120 1394.090206 1357.170227 1987.483765 1.704907e+02    1475.440167                 0.944864        -8.134996e+01
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery  audio_spectral_bandwidth        120 1885.965604 1859.236633 2435.690918 1.830765e+02    1986.502056                 0.949390        -1.005365e+02
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery    audio_spectral_rolloff        120 2759.624172 2617.033325 4775.566895 4.838143e+02    3033.264371                 0.909787        -2.736402e+02
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery              temp_context        120   23.700000   23.700000   23.700000 7.135220e-15      23.700000                 1.000000        -7.105427e-15
caretaker_entry_room1_week11_2025_08_17 post_entry_recovery                rh_context        120   52.000000   52.000000   52.000000 0.000000e+00      52.000000                 1.000000         0.000000e+00
```
