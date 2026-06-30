# Phase 6 Attractor And Probabilistic Plan

Phase 6 is the follow-up plan after Phase 5. It is based on the two larger
ideas from the long-rollout synthesis:

1. invariant-statistics / attractor matching,
2. probabilistic or stochastic residual modeling.

These ideas should not be merged into one ambiguous experiment. They test
different claims.

## Phase 6A: Invariant-Statistics Rollout

This is the primary Phase 6 arm because it keeps the architecture fixed.

| Item | Value |
|---|---|
| Run name | `phase6_attractor_rollout_edim384` |
| Architecture | `sfno_walker_1deg_edim384_layers8` |
| Parameters | 147,776,272 |
| Data schedule | Phase 4/5 soft residual curriculum |
| Rollout schedule | stages 1-3: 1 step, stage 4: 3 steps, stage 5: 6 steps, stage 6: 12 steps |
| Epoch budget | 150 |
| Main change | add rollout-window attractor-statistics loss in stages 4-6 |

### Objective

The loss should remain dominated by field L2. The new term should compare
rollout-window statistics rather than pointwise long-lead phase:

```text
L = L_field
  + lambda_energy L_log_spectral_energy
  + lambda_stats L_attractor_stats
```

`L_attractor_stats` should include:

- channel-wise anomaly mean and variance,
- low/mid spectral energy,
- Nino3.4-like regional tos mean and variance,
- cross-variable correlation for `tauu`, `tauv`, `tos`, and `zos`.

The key is that these statistics are computed over the batch and rollout
window. This targets the long-run distribution without forcing exact chaotic
phase matching.

### Prediction

Compared with Phase 4 residual+rollout and Phase 5:

- tos 120-month RMSE should stay near the stable Phase 4/5 region,
- tos and Nino3.4 amplitude should rise toward persistence,
- tos ACC should improve if the model preserves coherent low modes,
- high-k spectral energy should not explode.

### Failure Mode

If the loss is too strong, it will behave like Phase 3: amplitude returns but
phase skill and RMSE degrade. If it is too weak, it will behave like Phase 4:
stable but damped.

## Phase 6B: Probabilistic Residual

This is exploratory because it changes the modeling claim. It is no longer
only a data-injection or loss-scheduling intervention.

| Item | Value |
|---|---|
| Run name | `phase6_prob_residual_edim384` |
| Architecture | SFNO mean model plus stochastic residual mechanism |
| Status | exploratory, not a direct ablation of the original claim |
| Goal | restore conditional variance without sacrificing rollout stability |

### Minimal Version

The least invasive version should avoid rebuilding Makani from scratch:

1. Train or reuse a deterministic stable mean model.
2. Estimate residual statistics from validation rollouts:
   `r_t = x_{t+1} - F_theta(x_t)`.
3. Add calibrated stochastic residuals during rollout evaluation:
   low/mid spectral noise with seasonal and channel-dependent variance.

This creates a probabilistic post-processing baseline before committing to a
new neural residual head.

### Stronger Version

If the minimal version is promising, train a learned residual generator:

```text
x_{t+1} = F_theta(x_t) + G_phi(x_t, z_t)
```

where `F_theta` is the deterministic mean model and `G_phi` restores conditional
variance. Evaluation should use ensemble metrics such as CRPS, spread-skill,
spectral fidelity, and ensemble Nino3.4 distribution.

### Prediction

This direction may not improve deterministic RMSE, because a stochastic sample
is not the conditional mean. Its success criteria are different:

- better CRPS,
- better spread-skill,
- better anomaly amplitude and spectra,
- stable ensemble climatology,
- plausible Nino3.4 distribution.

## Recommended Execution Order

1. Finish Phase 5 and evaluate it.
2. If Phase 5 improves amplitude without badly hurting RMSE, run Phase 6A as a
   stronger attractor-statistics version.
3. If Phase 5 still collapses amplitude, Phase 6A becomes the next primary
   attempt.
4. Run Phase 6B only as an explicitly exploratory branch, because it changes the
   research question from deterministic forecasting to probabilistic climate
   emulation.

## Optimization Target

The Phase 6A deterministic target is:

| Metric | Target |
|---|---|
| 120m tos RMSE | no worse than Phase 4 residual+rollout by more than 10% |
| 120m tos ACC | above Phase 4 residual+rollout |
| 120m tos amplitude ratio | at least 0.65, ideally 0.8-1.1 |
| 120m Nino3.4 amplitude ratio | at least 0.65 without RMSE blow-up |
| low/mid spectral energy | closer to truth than Phase 4 |
| high-k energy | no unstable explosion |

The Phase 6B probabilistic target is ensemble-based and should not be judged by
single-sample RMSE alone.

## Clean Claim Separation

Phase 6A supports the main SteadySky claim:

> Long-rollout stability can be improved by combining rollout training with
> invariant-statistics constraints while keeping architecture fixed.

Phase 6B supports a broader claim:

> Deterministic long-rollout models tend toward conditional means; preserving
> long-run climate variability may require probabilistic residual dynamics.

