# Multimodal HMM Ablation Report

This ablation compares a video-only latent-state model against a video+audio+environment feature set.

- Shared state count used: 4
- Interpretation: Video-only HMM retained stronger event contrast than the multimodal variant.

## Model Notes

- `video_only` used 4 states on 11530 windows and produced mean posterior confidence `0.982`.
- `video_audio_env` used 4 states on 11530 windows and produced mean posterior confidence `0.998`.

## Metrics

```text
     model_name  n_states                              metric_name     value
     video_only         4                 event_detection_contrast  0.104543
     video_only         4              state_purity_by_event_phase  0.724138
     video_only         4                   time_to_detect_minutes       NaN
     video_only         4 false_positive_rate_outside_event_window  0.000000
     video_only         4                log_likelihood_per_sample  1.923060
     video_only         4                posterior_confidence_mean  0.981769
     video_only         4        posterior_confidence_during_event  1.000000
     video_only         4                       risk_mean_baseline  0.637770
     video_only         4                   risk_mean_during_event  0.742313
     video_only         4                       risk_mean_recovery  0.478616
video_audio_env         4                 event_detection_contrast  0.041451
video_audio_env         4              state_purity_by_event_phase  0.891516
video_audio_env         4                   time_to_detect_minutes  0.033333
video_audio_env         4 false_positive_rate_outside_event_window  0.305698
video_audio_env         4                log_likelihood_per_sample -3.430893
video_audio_env         4                posterior_confidence_mean  0.997567
video_audio_env         4        posterior_confidence_during_event  0.999585
video_audio_env         4                       risk_mean_baseline  0.692666
video_audio_env         4                   risk_mean_during_event  0.734117
video_audio_env         4                       risk_mean_recovery  0.693778
```

## Caveat

- Environment is daily/coarse, so it may add contextual structure without improving short event detection.
- If multimodal does not improve event contrast, that should be reported honestly rather than forced into a welfare claim.
