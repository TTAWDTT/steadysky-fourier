# Phase 3 Loss-Curriculum Results

Phase 3 tested whether moving the low-to-high Fourier curriculum from the data distribution into the loss would avoid the Phase 1/2 problem of treating low-pass fields as raw physical states.

Both arms trained from scratch for the full 150 epochs with raw inputs and raw targets.

## Runs

| Arm | Run number | Epoch 150 train loss | Epoch 150 validation loss | 120-month rollout |
|---|---|---:|---:|---|
| Fourier loss curriculum | `phase3_freq_loss_edim384` | 0.002245 | 0.234777 | completed |
| Fourier loss + tendency anomaly | `phase3_freq_anom_edim384` | 0.002762 | 0.311504 | completed |

## Selected Long-Rollout Metrics

Selected tos metrics:

| Lead | Raw | Pure Fourier | Mixed | Residual | Freq loss | Freq+anom | Persistence | Climatology |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12m RMSE | 0.9499 | 0.7417 | 0.9786 | 0.9348 | 1.1040 | 1.1687 | 0.6554 | 0.6962 |
| 60m RMSE | 1.4055 | 0.9081 | 1.4361 | 1.2652 | 1.4577 | 1.4119 | 0.7188 | 0.7585 |
| 120m RMSE | 1.3446 | 1.2703 | 1.5073 | 1.4973 | 1.5767 | 1.9562 | 0.7442 | 0.8319 |
| 12m ACC | 0.1491 | 0.1311 | 0.1629 | 0.1963 | 0.1313 | 0.1070 | 0.2914 | n/a |
| 60m ACC | 0.0783 | -0.0092 | 0.0758 | 0.1187 | 0.0767 | 0.0621 | 0.2049 | n/a |
| 120m ACC | 0.1161 | -0.1146 | 0.1276 | 0.1619 | -0.1460 | -0.1622 | 0.2428 | n/a |

Selected Nino3.4 metrics:

| Lead | Raw | Pure Fourier | Mixed | Residual | Freq loss | Freq+anom | Persistence | Climatology |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12m RMSE | 0.7118 | 0.6031 | 0.8493 | 0.6199 | 1.0735 | 1.1145 | 0.7641 | 0.7310 |
| 60m RMSE | 1.1860 | 0.8412 | 1.4287 | 1.0464 | 2.0461 | 1.9546 | 0.8266 | 0.9129 |
| 120m RMSE | 1.9285 | 1.2118 | 2.1575 | 1.8393 | 2.0529 | 1.7766 | 1.1928 | 1.3040 |
| 12m amplitude | 0.4083 | 0.3459 | 0.3479 | 0.3531 | 0.7262 | 0.8098 | 0.9982 | n/a |
| 60m amplitude | 0.4366 | 0.5054 | 0.8183 | 0.3419 | 1.6343 | 1.5363 | 0.7992 | n/a |
| 120m amplitude | 0.7769 | 0.5286 | 1.0041 | 0.6951 | 0.8927 | 0.6199 | 0.5595 | n/a |
| 120m cumulative corr. | 0.1320 | 0.1200 | 0.1635 | 0.1804 | 0.0822 | 0.0930 | -0.0052 | n/a |

## Interpretation

Phase 3 is a useful negative result.

The loss-curriculum arms restore more Nino3.4 anomaly amplitude than the pure Fourier curriculum in some horizons, which means the added frequency/tendency supervision does push against amplitude collapse. But this comes with worse RMSE and worse long-lead pattern correlation. At 120 months, both Phase 3 arms have negative tos ACC and worse tos RMSE than the raw baseline.

The current best candidate among the tested variants remains the Phase 2 residual arm for spatial structure, while pure Fourier remains strongest among model arms for some long-horizon RMSE values. Neither solves the persistence/climatology challenge.

The next methodological step should not simply increase Fourier-loss weight. A better direction is likely a more conservative residual-style curriculum, potentially combined later with short rollout training, because the present loss-only curriculum over-emphasizes spectral amplitude without preserving phase skill.
