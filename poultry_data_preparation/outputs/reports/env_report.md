# Environment Report

Daily environment data are treated as contextual covariates, not minute-level causal signals.

## Rows Per Room

- `room_1`: 149 rows (start `2025-06-05`, end `2025-10-31`)

## Missing Values

- `room_id`: 0
- `date`: 0
- `temp_am_min`: 1
- `temp_am_max`: 1
- `temp_pm`: 0
- `rh_am`: 1
- `rh_pm`: 0
- `temp_am_mean`: 1
- `temp_daily_mean`: 0
- `rh_daily_mean`: 0
- `temp_daily_range`: 1
- `env_quality_flag`: 0
- `source_file`: 0

## Summary Statistics

```text
       temp_am_min  temp_am_max     temp_pm  temp_am_mean  temp_daily_mean  temp_daily_range       rh_am       rh_pm  rh_daily_mean
count   148.000000   148.000000  149.000000    148.000000       149.000000        148.000000  148.000000  149.000000     149.000000
mean     24.214189    25.200000   24.715436     24.707095        24.731376          0.985811   48.128378   49.348993      48.741611
std       2.264185     2.168262    2.188537      2.194334         2.207650          0.628789   12.105702   13.370463      12.223925
min      21.300000    21.800000   23.500000     21.600000        22.650000         -0.300000   26.000000   26.000000      26.000000
25%      23.300000    24.100000   23.700000     23.750000        23.725000          0.700000   41.000000   41.000000      41.500000
50%      23.400000    24.300000   23.800000     23.850000        23.825000          0.850000   46.000000   45.000000      46.500000
75%      23.600000    24.825000   24.000000     24.012500        24.025000          1.000000   54.500000   58.000000      54.500000
max      31.400000    33.300000   31.700000     32.000000        31.800000          3.400000   89.000000   84.000000      83.500000
```
