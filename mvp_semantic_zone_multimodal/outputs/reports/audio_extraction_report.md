# Audio Extraction Report

- Audio comes from embedded MP4 audio only. No external WAV files were used.
- Number of windows attempted: 11530
- Number with audio available: 11530
- Number failed or unavailable: 0
- Reused precomputed embedded-audio rows from earlier pipeline: 0

## Feature Summary Statistics

```text
                            count         mean         std          min          25%          50%          75%          max
audio_rms                 11530.0     0.055350    0.012112     0.036139     0.043729     0.054596     0.068357     0.078017
audio_short_time_energy   11530.0     0.003231    0.001373     0.001309     0.001917     0.003008     0.004704     0.006197
audio_zero_crossing_rate  11530.0     0.045051    0.007988     0.031164     0.039148     0.044272     0.049271     0.108217
audio_spectral_centroid   11530.0  1316.377232  225.622845  1015.177917  1124.633820  1252.699219  1466.950195  2327.271240
audio_spectral_bandwidth  11530.0  1741.220508  265.436241  1373.836304  1493.798462  1699.852966  1944.460358  2636.034180
audio_spectral_rolloff    11530.0  2590.336692  578.156825  1893.266724  2200.274963  2374.950073  2859.333374  5764.933105
```

## Failed Window IDs

- none

## Quality Caveat

- Embedded room audio is whole-room audio and should not be interpreted as isolated chicken vocalization.
