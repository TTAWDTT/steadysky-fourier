# Phase 6 Attractor And Distribution Rollout Plan

Phase 6 is a two-arm follow-up to Phase 5.

Phase 5 asks whether preserving spectral energy during rollout training can
undo Phase 4's anomaly damping. Phase 6 asks two slightly broader questions:

1. Can rollout-stage attractor-statistics matching preserve anomaly variance
   and cross-variable structure without prescribing exact phase?
2. Can batch-level distribution matching preserve the rollout attractor without
   forcing sample-wise phase alignment?

These are different mechanisms. They are kept as separate arms so each remains
interpretable.

## Arms

| Arm | Run number | New mechanism |
|---|---|---|
| Phase 6A Attractor rollout | `phase6_attractor_rollout_edim384` | per-sample spatial mean/variance/covariance matching |
| Phase 6B Distribution rollout | `phase6_distribution_rollout_edim384` | batch-level feature MMD distribution matching |

Everything else follows the Phase 4/5 residual+rollout setup:

| Item | Value |
|---|---|
| Architecture | `sfno_walker_1deg_edim384_layers8` |
| Parameters | 147,776,272 |
| Data schedule | soft residual stages, then raw |
| Epochs | `10,15,20,25,35,45` = 150 |
| Rollout schedule | stages 1-3: 1 step, stage 4: 3 steps, stage 5: 6 steps, stage 6: 12 steps |
| Batch schedule | 16, 16, 16, 8, 4, 2 |

## Losses

Stages 1-3 use the normal field L2 loss. Stages 4-6 add a weak
regularizer.

### Phase 6A

```text
L = L_field
  + lambda_attr L_attractor_stats
```

where `L_attractor_stats` compares:

- spatial mean,
- log spatial variance,
- adjacent-channel spatial covariance.

The weights are intentionally small:

| Stage | Multistep count | Attractor weight |
|---:|---:|---:|
| 4 | 3 | 0.03 |
| 5 | 6 | 0.05 |
| 6 | 12 | 0.07 |

### Phase 6B

```text
L = L_field
  + lambda_mmd L_feature_mmd
```

`L_feature_mmd` computes coarse features for every sample, then compares the
prediction and target batch distributions with an RBF-kernel MMD. Features
include:

- channel-wise spatial mean,
- channel-wise log spatial variance,
- coarse low-pass pooled field values.

| Stage | Multistep count | MMD weight |
|---:|---:|---:|
| 4 | 3 | 0.02 |
| 5 | 6 | 0.035 |
| 6 | 12 | 0.05 |

## Why This Differs From Phase 5

Phase 5 preserves spectral energy. Phase 6A preserves per-sample field-level
attractor statistics. Phase 6B preserves a batch-level distribution of coarse
features. Both are less tied to exact Fourier bands and more directly aimed at
the collapse we saw in Phase 4:

- low anomaly variance,
- weak Nino3.4 amplitude,
- low-energy smooth fields.

Unlike Phase 3 Fourier coefficient loss, this does not ask the model to match
spectral phase. It only makes the low-energy attractor expensive.

## Success Criteria

Compare against raw, Phase 4 residual+rollout, Phase 5, persistence, and
climatology.

Desired outcome:

- 120-month tos RMSE remains near Phase 4 residual+rollout,
- 120-month tos ACC improves over Phase 4 residual+rollout,
- tos amplitude ratio rises above Phase 4's 0.351,
- Nino3.4 amplitude rises without the Phase 3 RMSE/ACC failure,
- spectral energy does not explode at high wavenumber.

If RMSE worsens sharply while amplitude improves, this is another Phase 3-like
negative result. If RMSE stays low but amplitude remains collapsed, the
regularizer is too weak or too local. If 6B helps more than 6A, the lesson is
that invariant-measure style distribution matching is better than per-sample
statistic matching for this problem.
