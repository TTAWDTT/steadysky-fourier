# Phase 6 Attractor-Statistics Rollout Plan

Phase 6 is a single-mechanism follow-up to Phase 5.

Phase 5 asks whether preserving spectral energy during rollout training can
undo Phase 4's anomaly damping. Phase 6 asks a slightly broader question:

> Can a rollout-stage attractor-statistics loss preserve anomaly variance and
> cross-variable structure without prescribing exact phase?

This is deliberately kept as one new mechanism so the result is interpretable.

## Arm

| Arm | Run number | New mechanism |
|---|---|---|
| Attractor rollout | `phase6_attractor_rollout_edim384` | rollout-stage spatial mean/variance/covariance matching |

Everything else follows the Phase 4/5 residual+rollout setup:

| Item | Value |
|---|---|
| Architecture | `sfno_walker_1deg_edim384_layers8` |
| Parameters | 147,776,272 |
| Data schedule | soft residual stages, then raw |
| Epochs | `10,15,20,25,35,45` = 150 |
| Rollout schedule | stages 1-3: 1 step, stage 4: 3 steps, stage 5: 6 steps, stage 6: 12 steps |
| Batch schedule | 16, 16, 16, 8, 4, 2 |

## Loss

Stages 1-3 use the normal field L2 loss. Stages 4-6 add a weak
`attractor_stats` term:

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

## Why This Differs From Phase 5

Phase 5 preserves spectral energy. Phase 6 preserves broader field-level
attractor statistics. It should be less tied to any specific Fourier band and
more directly aimed at the collapse we saw in Phase 4:

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
attractor statistic is too weak or too local.
