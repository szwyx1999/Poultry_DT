# Audio-Environment Correlation Report

This analysis treats environment as daily Room 1 context. It should be interpreted as contextual association, not causal proof.

## Resolution Note

- Correlations were computed on date-level aggregated audio means, not raw window-level repetition.
- Number of overlapping local dates with audio + environment context: 2
- This avoids overstating significance from repeated daily context values across many windows.

## Strongest Humidity Associations

No humidity associations were available at the current date-level overlap.

## Strongest Temperature Associations

No temperature associations were available at the current date-level overlap.

## RH And Spectral Feature Framing

No humidity-spectral associations were available.

## Table

```text
     env_feature            audio_feature  n_samples  pearson_r  pearson_p  spearman_rho  spearman_p  pearson_fdr_bh  spearman_fdr_bh
    temp_context                audio_rms          2        NaN        NaN           NaN         NaN             NaN              NaN
    temp_context  audio_short_time_energy          2        NaN        NaN           NaN         NaN             NaN              NaN
    temp_context audio_zero_crossing_rate          2        NaN        NaN           NaN         NaN             NaN              NaN
    temp_context  audio_spectral_centroid          2        NaN        NaN           NaN         NaN             NaN              NaN
    temp_context audio_spectral_bandwidth          2        NaN        NaN           NaN         NaN             NaN              NaN
    temp_context   audio_spectral_rolloff          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context                audio_rms          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context  audio_short_time_energy          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context audio_zero_crossing_rate          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context  audio_spectral_centroid          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context audio_spectral_bandwidth          2        NaN        NaN           NaN         NaN             NaN              NaN
      rh_context   audio_spectral_rolloff          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean                audio_rms          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean  audio_short_time_energy          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean audio_zero_crossing_rate          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean  audio_spectral_centroid          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean audio_spectral_bandwidth          2        NaN        NaN           NaN         NaN             NaN              NaN
 temp_daily_mean   audio_spectral_rolloff          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean                audio_rms          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean  audio_short_time_energy          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean audio_zero_crossing_rate          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean  audio_spectral_centroid          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean audio_spectral_bandwidth          2        NaN        NaN           NaN         NaN             NaN              NaN
   rh_daily_mean   audio_spectral_rolloff          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range                audio_rms          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range  audio_short_time_energy          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range audio_zero_crossing_rate          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range  audio_spectral_centroid          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range audio_spectral_bandwidth          2        NaN        NaN           NaN         NaN             NaN              NaN
temp_daily_range   audio_spectral_rolloff          2        NaN        NaN           NaN         NaN             NaN              NaN
```

## Caveat

- Environment is daily/coarse, so these correlations provide contextual framing only.
- In line with the project framing, humidity may modulate acoustic features and should be modelled as context rather than treated as standalone welfare proof.
