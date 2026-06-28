# Phase 4 Residual Rollout Plan

Phase 4 follows the Phase 3 negative result. The best signal so far is not the Fourier-loss curriculum, but the Phase 2 residual curriculum: it gave the strongest long-lead spatial-structure metrics among model variants.

Phase 4 therefore keeps the residual idea and tests two changes:

1. use a softer residual schedule so early stages are not too far from raw data;
2. add short autoregressive rollout training in the second arm to attack phase drift and error accumulation.

## Arms

| GPU | Arm | Run number | Data schedule | Training rollout |
|---:|---|---|---|---|
| 0 | Soft residual | `phase4_residual_soft_edim384` | low-pass plus 20/30/45/60/80/100% raw residual | 1-step throughout |
| 1 | Soft residual + rollout | `phase4_residual_rollout_edim384` | same soft residual data | 1, 1, 1, 3, 6, 12 steps |

Both arms keep the SFNO/FourCastNet2 architecture and 150-epoch stage boundaries.

## Data Schedule

Each transformed training field uses the same continuous blend formula as Phase 2:

```text
stage = lowpass + raw_weight * (raw - lowpass)
```

| Stage | Data | Raw residual weight | Epochs |
|---:|---|---:|---:|
| 1 | `train_residual_soft_lp004_l020` | 0.20 | 10 |
| 2 | `train_residual_soft_lp008_l030` | 0.30 | 15 |
| 3 | `train_residual_soft_lp016_l045` | 0.45 | 20 |
| 4 | `train_residual_soft_lp032_l060` | 0.60 | 25 |
| 5 | `train_residual_soft_lp064_l080` | 0.80 | 35 |
| 6 | `train_raw` | 1.00 | 45 |

This schedule is more conservative than Phase 2 residual, whose early weights were 0.05, 0.15, and 0.30.

## Rollout Training Budget

The rollout arm uses longer backprop-through-time windows only after the first three stages:

| Stage | Multistep count | Batch size |
|---:|---:|---:|
| 1 | 1 | 16 |
| 2 | 1 | 16 |
| 3 | 1 | 16 |
| 4 | 3 | 8 |
| 5 | 6 | 4 |
| 6 | 12 | 2 |

The smaller late-stage batch sizes are a memory constraint for autoregressive training, not a reduced sample experiment. The launcher also enables Makani multistep activation checkpointing when `multistep_count > 1`.

## Evaluation

Both Phase 4 arms use the same 120-month post-training rollout suite as earlier phases.

Primary success criteria:

- improve 60m/120m tos and zos ACC relative to Phase 2 residual;
- improve Nino3.4 cumulative correlation beyond the Phase 2 residual result;
- avoid the Phase 3 failure mode where amplitude recovers but phase skill collapses;
- avoid large RMSE degradation relative to raw.

The persistence and climatology baselines remain mandatory sanity checks.
