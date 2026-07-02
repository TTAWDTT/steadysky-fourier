# Phase 7 Closed-Loop Stability Plan

Phase 7 should be chosen after reading the completed Phase 6 120-month rollout
metrics. The working hypothesis is already clear enough to lock the next two
candidate mechanisms:

1. Phase 4 made long rollouts stable mostly by damping anomalies.
2. Phase 5 made spectral or energy collapse more expensive, but did not clearly
   improve skill over Phase 4.
3. Phase 6 tests attractor-level statistics and batch distribution matching.

If Phase 6 still cannot recover long-lead phase skill, Phase 7 should stop
adding stronger pointwise spectral penalties and instead train the model to be
stable under its own closed-loop dynamics.

## Arms

| Arm | Run number | New mechanism |
|---|---|---|
| Phase 7A Cycle consistency rollout | `phase7_cycle_rollout_edim384` | short forward rollout plus weak reverse/cycle consistency on coarse anomalies |
| Phase 7B Teacher-anchored rollout | `phase7_teacher_rollout_edim384` | residual+rollout student regularized toward a raw one-step teacher at early rollout leads |

Both arms keep the same SFNO architecture, 147,776,272 parameters, variables,
split, normalization, optimizer family, 150-epoch budget, and 120-month
post-training evaluation.

## Phase 7A Rationale

The smoothing failure suggests that the model has learned a stable low-energy
fixed region. A cycle-style constraint targets a different property: the
forecast should remain on a trajectory whose coarse state can be locally
reconciled with the starting state, rather than simply moving toward a damped
mean.

The constraint should be weak and coarse:

```text
L = L_field_rollout
  + lambda_cycle L_cycle(coarse_anomaly(x_t), coarse_anomaly(x_t_hat_from_rollout))
```

This is not meant to reconstruct exact high-frequency weather. It is meant to
discourage irreversible collapse of low-frequency anomaly state.

## Phase 7B Rationale

The raw baseline keeps more short-lead variability but drifts in long rollouts.
The rollout-trained residual model is stable but damped. A teacher-anchored arm
uses the raw one-step model as a short-lead variability anchor while the student
still trains with multistep rollout.

```text
L = L_field_rollout
  + lambda_teacher L(student_rollout_lead_1_to_3, stopgrad(raw_teacher_lead_1_to_3))
```

This tests whether the useful part of the raw model can be preserved without
letting its long-rollout instability dominate.

## Evaluation Gate

Phase 7 should only be launched after Phase 6 figures and metrics are checked.
The comparison table must include:

- raw baseline,
- Phase 4 residual+rollout,
- Phase 5 spectrum/energy rollout,
- Phase 6 attractor/distribution rollout,
- persistence,
- climatology.

Primary success criteria:

- maintain or improve Phase 4 120-month tos RMSE,
- improve tos 120-month ACC over Phase 4,
- raise tos and Nino3.4 amplitude above Phase 4 without spectral explosion,
- reduce the Nino3.4 positive-bias/damping pattern seen in earlier rollouts.

If Phase 6A already improves both RMSE and amplitude, Phase 7A should be the
first launch because it is the cleaner mathematical continuation. If Phase 6B
improves distributional amplitude but harms RMSE, Phase 7B should be prioritized
because it anchors local skill while keeping rollout training.
