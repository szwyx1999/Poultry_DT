# Semantic Biomarker Report

- Semantic zones differ from the old four-way layout because they represent drinking, feeding, and general area rather than equal image slices.
- Number of semantic biomarker windows: 11530
- Time span start: `2025-08-16T06:00:03-03:00`
- Time span end: `2025-08-17T14:14:53-03:00`
- Mean dominant semantic zone by activity fraction: `general_activity_fraction`

## Summary Statistics

```text
       activity_mean  mobility_index  spatial_freedom_index  occupancy_imbalance_index  drinking_activity_fraction  feeding_activity_fraction  general_activity_fraction  semantic_transition_proxy
count   11530.000000    11530.000000           11530.000000               11530.000000                11530.000000               11530.000000               11530.000000               11530.000000
mean        0.003870        0.098226               0.666246                   0.361564                    0.186532                   0.300088                   0.513380                   0.034644
std         0.003948        0.093023               0.272638                   0.175164                    0.103002                   0.257140                   0.336986                   0.027818
min         0.000114        0.000561              -0.000000                   0.019918                    0.000000                   0.000000                   0.086600                   0.000000
25%         0.000198        0.010172               0.411967                   0.198116                    0.100031                   0.029358                   0.194560                   0.015282
50%         0.004313        0.106174               0.803939                   0.317961                    0.177928                   0.417140                   0.278100                   0.028266
75%         0.006673        0.165057               0.913804                   0.536343                    0.262660                   0.540688                   0.869676                   0.046721
max         0.030054        0.706730               0.999141                   0.666667                    0.561203                   0.775990                   1.000000                   0.590973
```

## Interpretation Notes

- Activity fractions describe semantic-zone activity distribution, not true occupancy or bird counts.
- Drinking and feeding zones are small functional areas, so their fraction changes should be interpreted as relative concentration of motion intensity.
- Event log available for later alignment: `True`
