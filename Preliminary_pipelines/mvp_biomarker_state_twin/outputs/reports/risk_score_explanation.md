# Risk Score Explanation

The current welfare risk score is a prototype heuristic derived from the latent-state summary. It is intended for exploratory digital-twin demonstration work only.

## Warning

Prototype risk score, not validated welfare diagnosis.

## Formula

State-level risk weight is computed as a weighted combination of:

- high activity plus high mobility
- high occupancy imbalance
- low spatial freedom
- resilience pressure
- persistence of the state across consecutive windows

For each state:

`state_risk_weight = (w_activity_mobility * component_activity_mobility)`
` + (w_imbalance * component_imbalance)`
` + (w_low_spatial_freedom * component_low_spatial_freedom)`
` + (w_resilience_pressure * component_resilience_pressure)`
` + (w_persistence * component_persistence)`

Window-level risk score is then the posterior expectation over the active latent state probabilities.

## Feature Weights

- `w_activity_mobility`: 0.30
- `w_imbalance`: 0.25
- `w_low_spatial_freedom`: 0.20
- `w_resilience_pressure`: 0.15
- `w_persistence`: 0.10

## Model Context

- Model type: `gaussian_hmm`
- HMM state setting: `auto`
- Fitted HMM components used: 4
- Occupied latent states observed in the sequence: 3
- Feature columns used by the state model: mobility_index, spatial_freedom_index, occupancy_imbalance_index, normalized_activity, resilience_pressure_index, mobility_index__missing, spatial_freedom_index__missing, occupancy_imbalance_index__missing, normalized_activity__missing, resilience_pressure_index__missing

## State Summary Table

Only occupied states appear in the table below. If the model was fit with more components than were actually used by the current run, unoccupied components are not listed here.

```text
 state_id                          state_label  window_count  window_fraction  normalized_activity_mean  mobility_index_mean  spatial_freedom_index_mean  occupancy_imbalance_index_mean  resilience_pressure_index_mean  state_run_length_mean  component_activity_mobility  component_imbalance  component_low_spatial_freedom  component_resilience_pressure  component_persistence  state_risk_weight
        1 localized_low_activity_concentration          5382         0.466782                  0.000009             0.010826                    0.731132                        0.392218                             0.0            4677.378298                     0.000005             0.392218                       0.268868                            0.5               1.000000           0.326830
        2             distributed_low_activity          6092         0.528361                  0.186281             0.148990                    0.949274                        0.153714                             0.0            1207.794485                     0.248016             0.153714                       0.050726                            0.5               0.256534           0.223632
        3            disturbance_or_transition            56         0.004857                  0.601107             0.456875                    0.814121                        0.309390                             0.0              10.607143                     0.800553             0.309390                       0.185879                            0.5               0.000000           0.429689
```
