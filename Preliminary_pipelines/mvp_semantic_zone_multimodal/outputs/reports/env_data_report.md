# Environment Data Report

- Room 1 environment data are used as daily contextual covariates, not minute-level causal signals.
- Number of rows: 149
- Date range start: `2025-06-05`
- Date range end: `2025-10-31`

## Missing Values

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

## Summary Statistics

```text
                  count       mean        std    min     25%     50%      75%   max
temp_am_min       148.0  24.214189   2.264185  21.30  23.300  23.400  23.6000  31.4
temp_am_max       148.0  25.200000   2.168262  21.80  24.100  24.300  24.8250  33.3
temp_pm           149.0  24.715436   2.188537  23.50  23.700  23.800  24.0000  31.7
temp_am_mean      148.0  24.707095   2.194334  21.60  23.750  23.850  24.0125  32.0
temp_daily_mean   149.0  24.731376   2.207650  22.65  23.725  23.825  24.0250  31.8
temp_daily_range  148.0   0.985811   0.628789  -0.30   0.700   0.850   1.0000   3.4
rh_am             148.0  48.128378  12.105702  26.00  41.000  46.000  54.5000  89.0
rh_pm             149.0  49.348993  13.370463  26.00  41.000  45.000  58.0000  84.0
rh_daily_mean     149.0  48.741611  12.223925  26.00  41.500  46.500  54.5000  83.5
```

## Parsing Warnings

- none
