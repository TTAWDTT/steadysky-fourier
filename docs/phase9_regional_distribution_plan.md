# Phase 9 Regional Distribution Rollout Plan

Date: 2026-07-17

## Main-Line Decision

Phase 6B is now treated as the main research line:

```text
residual-soft curriculum
+ scheduled rollout exposure
+ batch-level feature-MMD distribution matching
```

The Phase 9 question is not whether Fourier curriculum alone works. The
question is whether Phase 6B's global distribution matching can be made more
structural without repeating Phase 8's overly blunt scalar invariant failure.

## Baseline

| Item | Value |
|---|---|
| Baseline arm | `distribution_rollout` |
| Run number | `phase6_distribution_rollout_edim384` |
| Model | SFNO `sfno_walker_1deg_edim384_layers8` |
| Parameters | 147,776,272 |
| Training schedule | 10, 15, 20, 25, 35, 45 epochs |
| Multistep schedule | 1, 1, 1, 3, 6, 12 |
| Batch schedule | 16, 16, 16, 8, 4, 2 |
| Extra loss | `feature_mmd` in stages 4-6 |
| MMD weights | 0.020, 0.035, 0.050 |
| Formal evaluation | 120-month autoregressive rollout |
| Best known `tos` result | RMSE 0.7375, ACC 9.4922, CRPS 0.4818 |

## New Arms

| Arm | Run number | Mechanism | Purpose |
|---|---|---|---|
| Phase 9A | `phase9_regional_distribution_rollout_edim384` | Replace `feature_mmd` with `regional_feature_mmd` | Test whether basin/region-scale structure improves long-rollout stability |
| Phase 9B | `phase9_coupled_regional_distribution_rollout_edim384` | Regional MMD plus cross-channel covariance features | Test whether explicit variable-coupling statistics help preserve the attractor |

Both arms keep the Phase 6B data schedule, rollout schedule, batch sizes, base
Makani L2 field loss, and MMD weights. This makes the comparison a clean
ablation of the distribution feature map.

## Loss

For rollout stages 4-6:

```text
L = L_field + lambda_mmd L_regional_feature_mmd
```

where `lambda_mmd` is:

| Stage | Weight |
|---|---:|
| 4 | 0.020 |
| 5 | 0.035 |
| 6 | 0.050 |

`regional_feature_mmd` includes the Phase 6B features:

- field mean
- log variance
- coarse low-pass mean
- RBF MMD with bandwidth 1.0

and adds:

| Feature | Phase 9A | Phase 9B |
|---|---:|---:|
| 4 x 4 regional means | yes | yes |
| cross-channel covariance | no | yes |

## Expected Read

Phase 9A is the conservative improvement. It should help if Phase 6B's global
features were too coarse and missed regional anomaly organization.

Phase 9B is riskier. It can help if `tauu`, `tauv`, `tos`, and `zos` coupling is
central to the long-run attractor, but it may overconstrain rollout states if
the covariance features are too high-dimensional or noisy at small batch sizes.

## Success Criterion

The primary criterion remains formal 120-month Makani evaluation:

| Metric | Better direction |
|---|---|
| `tos` RMSE at 120 months | lower than 0.7375 |
| `tos` CRPS at 120 months | lower than 0.4818 |
| `tos` ACC at 120 months | higher than 9.4922 |

Secondary diagnostics should inspect Nino3.4, lead maps, error maps, spectra,
and drift curves if either arm matches or beats Phase 6B.
