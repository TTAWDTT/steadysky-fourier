# Phase 8 Invariant Rollout Plan

Phase 7 rules out two tempting continuations of Phase 6B:

- simply extending train-time rollout exposure can become numerically unstable,
- adding a short-lead field anchor can improve local pressure while degrading
  long-horizon attractor behavior.

Phase 8 therefore should target invariant structure directly. The experimental
question is:

> Can we keep Phase 6B's distribution-level benefit while penalizing the specific
> long-run attractor failures: basin drift, seasonal-cycle loss, and spectral
> shape collapse?

The architecture, optimizer, split, normalization, 150-epoch budget, and
120-month post-training rollout evaluation remain fixed.

## Arms

| Arm | Run number | New mechanism | Why this is different from Phase 7 |
|---|---|---|---|
| Phase 8A drift-aware distribution rollout | `phase8_drift_distribution_rollout_edim384` | Phase 6B feature-MMD plus a weak global/basin mean-drift penalty in rollout stages | Penalizes slow attractor displacement instead of short-lead pointwise error |
| Phase 8B spectral-shape distribution rollout | `phase8_spectral_distribution_rollout_edim384` | Phase 6B feature-MMD plus a weak low/mid-band spectral-shape penalty in rollout stages | Preserves broad energy geometry without forcing exact Fourier phase |

## Training Schedule

Use the successful Phase 6B/Phase 7B stability envelope rather than Phase 7A's
24-step extension.

| Stage | Data | Epoch endpoint | Multistep count | Added invariant weight |
|---:|---|---:|---:|---:|
| 1 | residual soft lp004 | 10 | 1 | 0 |
| 2 | residual soft lp008 | 25 | 1 | 0 |
| 3 | residual soft lp016 | 45 | 3 | small |
| 4 | residual soft lp032 | 70 | 6 | moderate |
| 5 | residual soft lp064 | 105 | 12 | moderate |
| 6 | raw | 150 | 12 | moderate |

No early stop should be used for ordinary validation degradation. Stop only for
non-finite loss, hard runtime failure, or disk/GPU safety.

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

## Implementation Notes

Phase 8 should reuse the existing Makani training path and custom-loss registry.
If a new loss is added, it should return Makani-compatible per-example,
per-channel values and remain disabled in stages 1-2.

The important guardrail is conceptual: Phase 8 losses must not become another
short-lead pointwise anchor. They should compare low-order properties of the
closed-loop distribution so that the model is nudged toward the right climate
without being forced to match an exact long-lead phase.
