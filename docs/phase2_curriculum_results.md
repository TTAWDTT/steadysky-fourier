# Phase 2 Curriculum Results

Phase 2 evaluates two curriculum variants after the initial raw-vs-Fourier comparison suggested a possible smoothing side effect.

The controlled rule remains unchanged: same SFNO/FourCastNet2 architecture, Makani training path, loss, optimizer, normalization, split, batch size, epoch budget, and post-training 120-month rollout. Only the training data injection schedule differs.

## Runs

| Arm | Run number | Epochs | Status |
|---|---|---:|---|
| Raw baseline | `phase1_raw_edim384` | 150 | completed in Phase 1 |
| Pure Fourier | `phase1_fourier_edim384` | 150 | completed in Phase 1 |
| Mixed | `phase2_mixed_edim384` | 150 | completed |
| Residual | `phase2_residual_edim384` | 150 | completed |

## Phase 2 Training Losses

| Arm | Final train loss | Final validation loss | Best validation loss |
|---|---:|---:|---:|
| Mixed | 0.001649 | 0.130286 | 0.129990 |
| Residual | 0.001469 | 0.131078 | 0.130835 |

Training-time validation is not the research endpoint. It is retained as a health check. The formal comparison uses post-training long rollouts.

## 120-Month Rollout Metrics

Selected tos metrics at 12, 60, and 120 months:

| Lead | Method | tos RMSE | tos ACC |
|---:|---|---:|---:|
| 12 | Raw | 0.9499 | 0.1491 |
| 12 | Pure Fourier | 0.7417 | 0.1311 |
| 12 | Mixed | 0.9786 | 0.1629 |
| 12 | Residual | 0.9348 | 0.1963 |
| 12 | Persistence | 0.6554 | 0.2914 |
| 12 | Climatology | 0.6962 | n/a |
| 60 | Raw | 1.4055 | 0.0783 |
| 60 | Pure Fourier | 0.9081 | -0.0092 |
| 60 | Mixed | 1.4361 | 0.0758 |
| 60 | Residual | 1.2652 | 0.1187 |
| 60 | Persistence | 0.7188 | 0.2049 |
| 60 | Climatology | 0.7585 | n/a |
| 120 | Raw | 1.3446 | 0.1161 |
| 120 | Pure Fourier | 1.2703 | -0.1146 |
| 120 | Mixed | 1.5073 | 0.1276 |
| 120 | Residual | 1.4973 | 0.1619 |
| 120 | Persistence | 0.7442 | 0.2428 |
| 120 | Climatology | 0.8319 | n/a |

Selected Nino3.4 diagnostics:

| Lead | Method | Nino3.4 RMSE | Nino3.4 amplitude ratio |
|---:|---|---:|---:|
| 12 | Raw | 0.7118 | 0.4083 |
| 12 | Pure Fourier | 0.6031 | 0.3459 |
| 12 | Mixed | 0.8493 | 0.3479 |
| 12 | Residual | 0.6199 | 0.3531 |
| 12 | Persistence | 0.7641 | 0.9982 |
| 12 | Climatology | 0.7310 | n/a |
| 60 | Raw | 1.1860 | 0.4366 |
| 60 | Pure Fourier | 0.8412 | 0.5054 |
| 60 | Mixed | 1.4287 | 0.8183 |
| 60 | Residual | 1.0464 | 0.3419 |
| 60 | Persistence | 0.8266 | 0.7992 |
| 60 | Climatology | 0.9129 | n/a |
| 120 | Raw | 1.9285 | 0.7769 |
| 120 | Pure Fourier | 1.2118 | 0.5286 |
| 120 | Mixed | 2.1575 | 1.0041 |
| 120 | Residual | 1.8393 | 0.6951 |
| 120 | Persistence | 1.1928 | 0.5595 |
| 120 | Climatology | 1.3040 | n/a |

## Interpretation

The Phase 2 variants help separate stability from useful forecast skill.

- Pure Fourier still gives lower long-lead RMSE in some fields, but it also suppresses anomaly amplitude and can lose pattern correlation.
- Mixed restores Nino3.4 amplitude at long lead, but its RMSE worsens, suggesting amplitude preservation alone is not enough.
- Residual is the more promising Phase 2 arm for spatial structure: it improves tos ACC over raw and pure Fourier at the selected long leads, but it does not beat persistence or climatology in tos RMSE.
- Persistence and climatology are strong sanity baselines on this four-variable monthly ocean-like split; beating raw is not sufficient for a strong long-range skill claim.

The current conclusion is therefore conservative: these curricula change long-rollout behavior and expose a useful stability-smoothing tradeoff, but they do not yet demonstrate robust long-range predictive skill.
