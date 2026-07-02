# Phase 7 Closed-Loop Stability Plan

Phase 7 follows the completed Phase 6 120-month rollout metrics. The result
locks the next two candidate mechanisms:

1. Phase 4 made long rollouts stable mostly by damping anomalies.
2. Phase 5 made spectral or energy collapse more expensive, but did not clearly
   improve skill over Phase 4.
3. Phase 6A per-sample attractor statistics did not fix anomaly collapse.
4. Phase 6B batch-level distribution matching improved the formal 120-month
   Makani metrics and restored some Nino3.4 amplitude, but did not yet solve
   regional phase skill.

Phase 7 should therefore stop adding stronger pointwise spectral penalties and
instead train the model to be stable under its own closed-loop dynamics while
anchoring short-to-medium lead phase.

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

## Recommended Order

Phase 7B should run first. Phase 6B suggests that distributional realism helps,
but the remaining failure is phase and regional trajectory anchoring. A raw
one-step teacher is the most direct way to keep short-lead variability while
the student still learns multistep stability.

Phase 7A should run second. It is the cleaner mathematical continuation for
irreversible low-frequency collapse, but it is likely harder to implement
faithfully in the current Makani path.
