# Phase 8 Invariant Rollout Plan

Phase 7 rules out two tempting continuations of Phase 6B:

- simply extending train-time rollout exposure can become numerically unstable,
- adding a short-lead field anchor can improve local pressure while degrading
  long-horizon attractor behavior.

Phase 8 therefore targets invariant structure directly. The experimental
question is:

> Can we keep Phase 6B's distribution-level benefit while penalizing the specific
> long-run attractor failures: basin drift, seasonal-cycle loss, and spectral
> shape collapse?

The architecture, optimizer, split, normalization, 150-epoch budget, and
120-month post-training rollout evaluation remain fixed. Phase 8 should be a
Phase 6B follow-up, not a Phase 7 follow-up: keep the stable 12-step rollout
envelope and change only one invariant mechanism per arm.

## Baseline To Preserve

Phase 6B is the current reference point:

| Arm | 120-month tos RMSE | 120-month tos ACC | 120-month tos CRPS | Read |
|---|---:|---:|---:|---|
| Phase 4 residual+rollout | 0.7416 | 9.3691 | 0.4900 | stable but damped |
| Phase 6B distribution rollout | **0.7375** | **9.4922** | **0.4818** | best current arm |
| Phase 7A long distribution best | 0.8628 | 9.0205 | 0.6067 | longer rollout did not help |
| Phase 7B anchored distribution | 0.8103 | 8.8920 | 0.5375 | short-lead anchor did not help |

Phase 8 succeeds only if it keeps the Phase 6B formal metric advantage while
improving at least one long-run structure diagnostic.

## Arms

| Arm | Run number | New mechanism | Why this is different from Phase 7 |
|---|---|---|---|
| Phase 8A drift-aware distribution rollout | `phase8_drift_distribution_rollout_edim384` | Phase 6B feature-MMD plus a weak spatial-mean drift penalty in rollout stages | Penalizes slow attractor displacement instead of short-lead pointwise error |
| Phase 8B spectral-shape distribution rollout | `phase8_spectral_distribution_rollout_edim384` | Phase 6B feature-MMD plus a weak low/mid/high spectral-shape penalty in rollout stages | Preserves broad energy geometry without forcing exact Fourier phase |

## Loss Definitions

Both arms keep the base Makani field loss:

```text
L = L_field + lambda_mmd L_feature_mmd + lambda_inv L_invariant
```

`L_field` is the same squared L2 loss used by the earlier Makani runs.
`L_feature_mmd` is the Phase 6B batch-level feature MMD over coarse field
statistics. `L_invariant` is the only Phase 8-specific term.

### Phase 8A: spatial-mean drift loss

For each example and channel, compute a weighted spatial mean over the valid
field:

```text
mu(pred) = mean_space(pred)
mu(targ) = mean_space(targ)
L_drift = (mu(pred) - mu(targ))^2
```

The intent is not to make every grid point match. It is to make slow basin-scale
or global displacement expensive, especially the cold/negative `tos` drift seen
after long rollout.

Implementation note: use Makani's loss weights when available, so masked/ocean
regions and channel weighting remain consistent with the rest of the training
path.

### Phase 8B: spectral-shape loss

Compute spatial FFT power in broad radial bands:

```text
E_low, E_mid, E_high
P_band = E_band / (E_low + E_mid + E_high + eps)
L_shape = sum_band (log P_band(pred) - log P_band(targ))^2
```

This differs from Phase 3 and Phase 5:

- it compares broad normalized shape, not Fourier coefficients,
- it does not prescribe phase,
- it does not force absolute energy,
- it is deliberately weak and only active in rollout stages.

The goal is to prevent the closed-loop attractor from degenerating into a
smooth spectral shape while avoiding the Phase 3 failure mode.

## Training Schedule

Use the successful Phase 6B stability envelope rather than Phase 7A's 24-step
extension or Phase 7's earlier stage-3 distribution pressure.

| Stage | Data | Epoch endpoint | Multistep count | MMD weight | Invariant weight |
|---:|---|---:|---:|---:|---:|
| 1 | residual soft lp004 | 10 | 1 | 0 | 0 |
| 2 | residual soft lp008 | 25 | 1 | 0 | 0 |
| 3 | residual soft lp016 | 45 | 1 | 0 | 0 |
| 4 | residual soft lp032 | 70 | 3 | 0.020 | 0.005 |
| 5 | residual soft lp064 | 105 | 6 | 0.035 | 0.010 |
| 6 | raw | 150 | 12 | 0.050 | 0.015 |

This is intentionally conservative. Phase 7A suggests that stronger closed-loop
pressure can destabilize training, and Phase 7B suggests that adding local
trajectory pressure can hurt the long-run attractor. Phase 8 should therefore
change the invariant term, not the rollout envelope.

No early stop should be used for ordinary validation degradation. Stop only for
non-finite loss, hard runtime failure, or disk/GPU safety.

## Launch Order

Run Phase 8A first if only one GPU is available.

| Priority | Arm | Reason |
|---:|---|---|
| 1 | Phase 8A drift-aware distribution | Lowest conceptual risk; attacks observed drift without using Fourier supervision |
| 2 | Phase 8B spectral-shape distribution | More speculative; useful only if kept weak enough to avoid Phase 3-style spectral overconstraint |

If two GPUs are safely available, launch both arms together. If GPU1 remains
occupied, run 8A first and queue 8B after 8A reaches stage 4 or after 8A
finishes, depending on disk pressure.

## Evaluation

Formal evaluation remains the same post-training 120-month Makani rollout:

- `tos` RMSE, ACC, ACC AUC, CRPS at 120 months,
- same metrics for `tauu`, `tauv`, and `zos`,
- saved forecasts for diagnostic plots.

Supporting diagnostics compare Phase 8 against:

- Phase 4 residual+rollout,
- Phase 6B distribution rollout,
- Phase 7B anchored distribution,
- persistence and climatology references in the diagnostic path.

Primary success criterion:

> Beat or match Phase 6B formal `tos` RMSE/CRPS while improving at least one
> long-horizon structure diagnostic: Nino3.4 cumulative correlation, tos drift,
> or tos spectral-shape ratio.

Secondary reads:

| Diagnostic | Desired movement | Why |
|---|---|---|
| `tos` drift curve | closer to zero than Phase 6B | direct check for attractor displacement |
| Nino3.4 cumulative correlation | above Phase 6B and Phase 7B | avoids amplitude-only gains |
| Nino3.4 amplitude ratio | not lower than Phase 6B | prevents new collapse |
| low/mid/high spectral-shape ratio | closer to truth without high-k explosion | distinguishes shape preservation from noise |
| non-finite checks | no NaN in train/valid loss or gradients | Phase 7A guardrail |

## Implementation Notes

Phase 8 should reuse the existing Makani training path and custom-loss registry.
New losses must return Makani-compatible per-example, per-channel values and
remain disabled in stages 1-3.

Required code changes:

1. Add `SpatialMeanDriftLoss` to `scripts/install_makani_phase3_losses.py`.
2. Add `SpectralShapeLoss` to `scripts/install_makani_phase3_losses.py`.
3. Register both losses in Makani's loss registry install helper.
4. Add `drift_distribution_rollout` and `spectral_distribution_rollout` arms to
   `scripts/run_phase1_training_schedule.sh`.
5. Add both arms to `scripts/run_phase1_long_rollout_eval.sh`.
6. Extend the comparison plotting helper to include Phase 8 arms after
   evaluation.

Do not change:

- SFNO architecture,
- total epoch budget,
- train/valid/test split,
- normalization,
- optimizer family,
- Phase 6B feature-MMD definition,
- post-training 120-month evaluation protocol.

The important guardrail is conceptual: Phase 8 losses must not become another
short-lead pointwise anchor. They should compare low-order properties of the
closed-loop distribution so that the model is nudged toward the right climate
without being forced to match an exact long-lead phase.

## Decision

Phase 8 is locked as:

1. `phase8_drift_distribution_rollout_edim384`
2. `phase8_spectral_distribution_rollout_edim384`

Both are Phase 6B-plus-one-mechanism ablations. The first tests whether
long-run drift is the missing invariant; the second tests whether broad
spectral shape is the missing invariant. Neither should be interpreted as a new
architecture or as a replacement for the original data-injection research
question.
