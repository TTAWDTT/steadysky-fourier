# Phase 6B Distribution Rollout Method

Date: 2026-07-17

## Position in the Research

Phase 6B is the current best-performing method and should be treated as the new
main line of the project.

The original idea was Fourier layerwise data injection: teach the model slow
low-frequency structure first, then progressively add higher-frequency
components. That idea produced useful stabilization, but the experiments showed
an important limitation: low-frequency-first training can also make the model
too smooth and damped.

Phase 6B shifts the core claim:

> Long-rollout stability is a closed-loop distribution problem. A model should
> not only minimize one-step field error; its own autoregressive states should
> remain close to the real system's state distribution.

## Method Summary

Phase 6B combines three components:

```text
residual-soft data curriculum
+ scheduled rollout exposure
+ batch-level feature-MMD distribution matching
```

It does not change the model architecture. The improvement is purely in the
training/data-injection objective.

## Fixed Model and Data Setup

| Item | Value |
|---|---|
| Architecture | Makani SFNO `sfno_walker_1deg_edim384_layers8` |
| Trainable parameters | 147,776,272 |
| Variables | `tauu`, `tauv`, `tos`, `zos` |
| Data family | four-variable Walker/ocean 1-degree dataset |
| Training length | 150 epochs |
| Formal evaluation | 120-month autoregressive rollout |
| Main metrics | Makani `tos` RMSE, ACC, CRPS |

## Training Schedule

Phase 6B keeps the residual-soft curriculum introduced before Phase 6. The data
starts from easier low-pass/residual-soft targets, then returns to raw data.
The key difference from earlier phases is that rollout exposure grows in the
later stages.

| Stage | Training data | Epochs | Multistep count | Batch size |
|---|---|---:|---:|---:|
| 1 | `train_residual_soft_lp004_l020` | 10 | 1 | 16 |
| 2 | `train_residual_soft_lp008_l030` | 15 | 1 | 16 |
| 3 | `train_residual_soft_lp016_l045` | 20 | 1 | 16 |
| 4 | `train_residual_soft_lp032_l060` | 25 | 3 | 8 |
| 5 | `train_residual_soft_lp064_l080` | 35 | 6 | 4 |
| 6 | `train_raw` | 45 | 12 | 2 |

The stage epoch counts are cumulative in the Makani config, but the actual
incremental budget is:

```text
10 + 15 + 20 + 25 + 35 + 45 = 150 epochs
```

## Loss Function

For stages 1-3, Phase 6B uses the normal Makani field loss:

```text
L = L_field
```

For stages 4-6, Phase 6B adds a weak feature-MMD distribution loss:

```text
L = L_field + lambda_mmd L_feature_mmd
```

| Stage | `lambda_mmd` |
|---|---:|
| 4 | 0.020 |
| 5 | 0.035 |
| 6 | 0.050 |

`L_field` is the standard Makani L2 field loss with constant channel weights and
temperature-difference normalization.

## What Feature-MMD Measures

The MMD loss compares a batch of predicted rollout states with a batch of target
states. It does not force every predicted sample to exactly match its paired
future. Instead, it asks whether the predicted batch and target batch look like
they were drawn from similar coarse state distributions.

The feature vector contains:

| Feature | Role |
|---|---|
| Field mean | Controls large-scale state location |
| Log variance | Penalizes variance collapse and over-smoothing |
| Low-pass mean | Preserves coarse spatial structure |
| RBF MMD kernel | Matches batch distributions rather than individual trajectories |

The implementation normalizes feature dimensions inside the batch before
computing the RBF MMD. This keeps one feature family from dominating simply due
to scale.

## Why It Works Better Than Pure Fourier Curriculum

Pure Fourier curriculum teaches low-frequency structure early. That helps
stability, but it can also teach a low-energy attractor: the model remains calm,
but anomalies become too smooth and phase skill remains weak.

Phase 6B attacks the later failure mode directly:

1. **Rollout exposure** makes training see model-generated states, not only
   teacher-forced one-step states.
2. **Distribution matching** makes collapsed or unrealistic rollout batches
   expensive.
3. **Batch-level matching** avoids over-penalizing long-lead phase shifts, which
   are expected in long chaotic or climate-like rollouts.

This is the important conceptual move:

> The target is not exact long-lead trajectory matching. The target is keeping
> the model-induced closed-loop distribution near the real data distribution.

## Formal Result

Phase 6B is currently the best formal 120-month result.

| Arm | 120-month `tos` RMSE | 120-month `tos` ACC | 120-month `tos` CRPS |
|---|---:|---:|---:|
| Raw baseline | 1.2134 | 6.4593 | 0.8532 |
| Phase 1 Fourier curriculum | 1.0546 | 8.3921 | 0.6513 |
| Phase 4 residual+rollout | 0.7416 | 9.3691 | 0.4900 |
| **Phase 6B distribution rollout** | **0.7375** | **9.4922** | **0.4818** |

Relative to raw baseline:

| Metric | Change |
|---|---:|
| `tos` RMSE | -39.2% |
| `tos` CRPS | -43.5% |
| `tos` ACC | +47.0% relative improvement |

The gain over Phase 4 is smaller but important: Phase 4 already found strong
stability through rollout exposure, while Phase 6B improves the distributional
realism of that stable rollout.

## What Phase 6B Does Not Solve Yet

Phase 6B should not be oversold. It is the best current mechanism, not a final
solution.

Known limitations:

| Limitation | Meaning |
|---|---|
| Global features are coarse | Mean/variance/low-pass features may miss regional structure |
| Long-lead phase skill remains uncertain | Distribution matching does not guarantee ENSO/Nino3.4 phase correctness |
| Single architecture so far | Evidence is strongest for this SFNO setup only |
| Single formal data family | Not yet a full ERA5-style benchmark |
| No multi-seed uncertainty yet | We do not know run-to-run variance |

## Why Phase 9 Starts From Phase 6B

Phase 9 keeps the Phase 6B training envelope and changes only the feature map
inside MMD.

| Phase | Change from Phase 6B |
|---|---|
| Phase 9A | Add coarse regional means to MMD features |
| Phase 9B | Add regional means plus cross-channel covariance features |

This is a clean continuation because it tests whether the best current idea,
closed-loop distribution matching, becomes stronger when the distribution
features better represent regional structure and variable coupling.

## Short Name

A good working name for the Phase 6B method is:

```text
Closed-loop Attractor Distribution Matching
```

or, more concretely:

```text
Rollout Feature-MMD Training
```
