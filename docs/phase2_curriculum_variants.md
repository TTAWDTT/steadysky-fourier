# Phase 2 Curriculum Variants

Phase 1 suggested that pure low-to-high Fourier curriculum can improve some long-rollout RMSE values, but may also act as a smoothing regularizer. Phase 2 tests two data-injection variants that keep the architecture, optimizer, loss, split, batch size, and 150-epoch budget unchanged.

## Arms

| Arm | Run number | Training stages | Purpose |
|---|---|---|---|
| Mixed Fourier | `phase2_mixed_edim384` | low-pass fields blended with raw residual throughout stages | Keep raw signal present from the beginning |
| Residual Fourier | `phase2_residual_edim384` | low-pass fields plus an increasing residual weight | Test whether explicitly restoring residual signal reduces smoothing |

Each transformed training field is generated as:

```text
stage = lowpass + raw_weight * (raw - lowpass)
```

Validation and test data remain raw.

## Stage Definitions

| Stage | Mixed data | Raw residual weight | Residual data | Residual weight | Epochs |
|---:|---|---:|---|---:|---:|
| 1 | `train_mixed_lp004_r020` | 0.20 | `train_residual_lp004_l005` | 0.05 | 10 |
| 2 | `train_mixed_lp008_r030` | 0.30 | `train_residual_lp008_l015` | 0.15 | 15 |
| 3 | `train_mixed_lp016_r040` | 0.40 | `train_residual_lp016_l030` | 0.30 | 20 |
| 4 | `train_mixed_lp032_r050` | 0.50 | `train_residual_lp032_l050` | 0.50 | 25 |
| 5 | `train_mixed_lp064_r065` | 0.65 | `train_residual_lp064_l075` | 0.75 | 35 |
| 6 | `train_raw` | 1.00 | `train_raw` | 1.00 | 45 |

The mixed arm is the conservative variant: it keeps substantial raw content in every stage. The residual arm is more diagnostic: it starts closer to low-pass training and restores residual signal more deliberately.

## Interpretation

The key question is not only whether either variant beats the raw baseline in long-horizon RMSE. The stricter question is whether it improves long-rollout metrics without collapsing anomaly amplitude, variance, pattern ACC, or spectral energy. The same skill-vs-smoothing diagnostics from Phase 1 should be run after training.
