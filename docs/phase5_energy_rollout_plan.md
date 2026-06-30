# Phase 5 Energy-Preserving Rollout Plan

Phase 5 follows the Phase 2-4 pattern:

- Phase 2 residual training gave the best model-side spatial-structure signal,
  but did not lower long-lead RMSE enough.
- Phase 3 Fourier-band losses preserved more anomaly amplitude in some cases,
  but damaged RMSE and long-lead pattern correlation.
- Phase 4 residual+rollout training sharply lowered long-lead RMSE, but mostly
  by damping anomalies toward a smoother attractor.

The next experiment keeps the part that worked in Phase 4, then adds a small
constraint against anomaly collapse.

## Fixed Setup

Both Phase 5 arms keep the same fixed experimental surface as Phase 4:

| Item | Value |
|---|---|
| Architecture | `sfno_walker_1deg_edim384_layers8` |
| Parameters | 147,776,272 |
| Variables | `tauu`, `tauv`, `tos`, `zos` |
| Data schedule | Phase 4 soft residual schedule |
| Rollout schedule | stages 1-3: 1 step, stage 4: 3 steps, stage 5: 6 steps, stage 6: 12 steps |
| Epochs | `10,15,20,25,35,45` = 150 total |
| Evaluation | post-training 120-month rollout |

## New Arms

| Arm | Run number | Added rollout-stage regularizer |
|---|---|---|
| Energy rollout | `phase5_energy_rollout_edim384` | match low+mid spatial anomaly spectral energy |
| Spectrum rollout | `phase5_spectrum_rollout_edim384` | match low+mid+weak-high spatial anomaly spectral energy |

The regularizer is only active in stages 4-6, where autoregressive rollout
training begins. The L2 field loss remains the dominant loss. The regularizer
weights are deliberately small:

| Stage | Multistep count | Regularizer weight |
|---:|---:|---:|
| 4 | 3 | 0.04 |
| 5 | 6 | 0.06 |
| 6 | 12 | 0.08 |

## Rationale

The Phase 4 residual+rollout arm learned a stable attractor, but its 120-month
tos amplitude ratio was only 0.351. This suggests that rollout training taught
the model that a damped forecast is a low-risk way to reduce accumulated MSE.

Phase 5 does not ask the model to match Fourier phase directly. It only asks
the forecast to retain target-like anomaly energy in broad spectral bands while
field L2 still controls pointwise accuracy. This is intended to preserve the
stabilizing effect of rollout training while making collapse toward
climatology more expensive.

## Success Criteria

The Phase 5 arms should be judged against Phase 4 residual+rollout, raw,
persistence, and climatology.

The desired result is not just lower RMSE. A successful arm should:

- keep long-lead tos RMSE near or below Phase 4 residual+rollout,
- improve 120-month tos ACC relative to Phase 4 residual+rollout,
- raise tos and Nino3.4 amplitude ratios toward persistence without exploding
  them above 1,
- avoid the Phase 3 failure mode where amplitude is restored but RMSE and ACC
  degrade.

If RMSE worsens while amplitude recovers, Phase 5 should be treated as another
negative result: the constraint would be too strong or insufficiently phase
aware.
