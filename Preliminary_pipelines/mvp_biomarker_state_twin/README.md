# MVP Biomarker State Twin

This MVP builds the next analytics layer on top of `mvp_video_zone_heatmap/`. It turns MVP1 zone-level activity features into candidate digital biomarkers, a canonical feature table for latent behavioural state modelling, a Gaussian-HMM-or-fallback state model, and a Unity-demo-ready JSON timeline.

## Workflow:
- Reads the existing MVP1 outputs instead of reprocessing raw video.
- Normalizes the MVP1 zone feature table into a canonical per-window/per-zone table.
- Computes four candidate biomarkers:
  - Mobility Index
  - Spatial Freedom Index
  - Occupancy Imbalance Index
  - Resilience Curve metrics
- Fits a Gaussian HMM when `hmmlearn` is available.
- Falls back to a Gaussian Mixture plus transition smoothing if the HMM is unavailable or fails.
- Exports a Unity-ready timeline JSON and copies it into the Unity project `StreamingAssets` folders.


## Dependency On MVP1

The pipeline reads:

- `mvp_video_zone_heatmap/outputs/features/video_zone_features.csv`
- `mvp_video_zone_heatmap/outputs/features/selected_windows.csv`
- `mvp_video_zone_heatmap/outputs/unity_json/mvp1_zone_activity_timeline.json`

It can also backfill metadata from:

- `first_week_data_cleaning/data/processed/metadata/video_window_index.csv`
- `first_week_data_cleaning/data/processed/metadata/media_manifest.csv`

## Run

From the workspace root:

```bash
python -m mvp_biomarker_state_twin.src.main
```

Or with an explicit config:

```bash
python -m mvp_biomarker_state_twin.src.main --config mvp_biomarker_state_twin/config/default.yaml
```

## Outputs

Key outputs land under `mvp_biomarker_state_twin/outputs/`:

- `features/canonical_zone_feature_table.csv`
- `features/biomarker_window_table.csv`
- `features/resilience_event_table.csv`
- `features/hmm_state_sequence.csv`
- `features/hmm_state_summary.csv`
- `model/hmm_model.joblib`
- `reports/data_coverage_report.md`
- `reports/schema_mapping.md`
- `reports/hmm_sequence_diagnostics.md`
- `reports/risk_score_explanation.md`
- `plots/biomarker_timeseries.png`
- `plots/biomarker_timeseries_by_room.png`
- `plots/resilience_curves.png`
- `plots/hmm_state_timeline.png`
- `plots/hmm_state_timeline_overview.png`
- `plots/state_risk_heatmap.png`
- `plots/state_summary_radar_or_bar.png`
- `plots/risk_score_distribution.png`
- `unity_json/poultry_twin_demo_timeline.json`

The Unity JSON is also copied to:

- `PoultryTwinDemo/demo1/Assets/StreamingAssets/poultry_twin_demo_timeline.json`


## State Labels

State labels are derived from the state-level biomarker means in `hmm_state_summary.csv`, not hard-coded from `state_id`.

The heuristics currently distinguish patterns such as:

- `stable_low_activity`
- `localized_low_activity_concentration`
- `normal_distributed_activity`
- `disturbance_or_transition`
- `concentrated_high_activity`
- `sustained_deviation`

Additional fallback labels such as `distributed_low_activity`, `localized_activity_shift`, or `mixed_activity_pattern` can appear when the state profile does not cleanly match one of the core patterns.


