# Phase 7 Literature-Grounded Closed-Loop Plan

Phase 7 is chosen after Phase 6 and a second literature check. The important
Phase 6 result is that batch-level distribution matching was the first
follow-up to improve formal 120-month rollout metrics while partially restoring
Nino3.4 amplitude. The remaining failure is not "lack of energy" in the simple
sense. It is trajectory anchoring: recovered anomalies can still be mistimed or
regionally misplaced.

## Literature Signals

The relevant literature points to four constraints:

1. Long autoregressive forecasts should be evaluated as closed-loop dynamical
   systems, not only as repeated one-step predictions.
2. Teacher-forced training creates exposure bias; rollout-time states differ
   from train-time states. Scheduled sampling, DAgger-style state aggregation,
   and professor-forcing-style dynamics matching all target this mismatch.
3. For chaotic or weakly predictable long horizons, pointwise losses encourage
   conditional means and therefore damping. Invariant-measure or distributional
   losses are better aligned with long-run climate statistics.
4. Distributional realism alone is insufficient for forecast usefulness. Short
   and medium lead phase anchoring must remain strong, otherwise the model can
   look climate-realistic but lose regional trajectory skill.

Phase 7 therefore should deepen Phase 6B rather than pivot to a raw one-step
teacher. A raw teacher could preserve short-lead variability, but it would also
import the raw baseline's long-rollout instability and does not directly solve
the model-induced-state distribution problem.

## Locked Arms

| Arm | Run number | New mechanism | Primary question |
|---|---|---|---|
| Phase 7A Long distribution rollout | `phase7_long_distribution_rollout_edim384` | Phase 6B distribution loss with a longer rollout curriculum | Does matching the model-induced distribution over longer train rollouts reduce 120-month drift and damping? |
| Phase 7B Anchored distribution rollout | `phase7_anchored_distribution_rollout_edim384` | Phase 6B distribution loss plus a stronger short-lead pointwise/phase anchor | Can we keep Phase 6B's amplitude recovery while improving Nino3.4 phase and regional RMSE? |

Both arms keep:

- `sfno_walker_1deg_edim384_layers8`,
- 147,776,272 parameters,
- the same four variables: `tauu`, `tauv`, `tos`, `zos`,
- the same train/valid/test split and normalization,
- the same 150-epoch total budget unless memory forces a documented batch or
  gradient-accumulation change,
- the same 120-month post-training rollout evaluation.

## Phase 7A: Long Distribution Rollout

Phase 7A is the cleanest continuation of the positive Phase 6B result.

Current rollout-stage maximum is 12 months. That is still far from the
120-month evaluation horizon. Phase 7A extends the model-induced training
distribution while keeping the successful distribution matching idea:

| Stage | Epochs | Rollout steps | Distribution weight |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 0 |
| 2 | 15 | 1 | 0 |
| 3 | 20 | 3 | 0.02 |
| 4 | 25 | 6 | 0.035 |
| 5 | 35 | 12 | 0.05 |
| 6 | 45 | 24 | 0.07 |

Prediction:

- formal 120-month RMSE/ACC should match or beat Phase 6B,
- drift should decrease relative to Phase 6B,
- Nino3.4 amplitude should stay above Phase 4,
- if short/medium Nino3.4 RMSE worsens, the arm confirms that longer
  distributional training alone is not enough for phase anchoring.

## Phase 7B: Anchored Distribution Rollout

Phase 7B tests the missing ingredient suggested by the Phase 6 diagnostics:
short-to-medium lead phase anchoring.

The key idea is not to use a separate raw teacher as the main mechanism. The
anchor should be internal to the supervised rollout objective:

```text
L = L_short_field
  + lambda_dist L_feature_distribution
  + lambda_tendency L_short_tendency
```

where:

- `L_short_field` keeps the early rollout leads pointwise accurate,
- `L_feature_distribution` preserves the Phase 6B invariant-distribution
  benefit,
- `L_short_tendency` compares short-lead changes, especially for `tos`/`zos`,
  so the model is penalized for recovering amplitude with the wrong local
  tendency.

Recommended schedule:

| Stage | Epochs | Rollout steps | Distribution weight | Tendency weight |
|---:|---:|---:|---:|---:|
| 1 | 10 | 1 | 0 | 0 |
| 2 | 15 | 1 | 0 | 0 |
| 3 | 20 | 3 | 0.015 | 0.02 |
| 4 | 25 | 6 | 0.03 | 0.03 |
| 5 | 35 | 12 | 0.045 | 0.04 |
| 6 | 45 | 12 | 0.06 | 0.05 |

Prediction:

- Nino3.4 RMSE at 12/30/60/120 months should improve relative to Phase 6B,
- amplitude should remain well above Phase 4,
- formal Makani metrics should remain close to Phase 6B,
- if amplitude collapses back toward Phase 4, the tendency/field anchor is too
  strong and has restored the conditional-mean failure mode.

## Decision Order

Run Phase 7A first if only one GPU is available, because it is the most direct
test of whether Phase 6B simply needed longer model-induced rollout exposure.

If two GPUs are available, run Phase 7A and 7B in parallel. They are a clean
ablation pair:

- 7A asks whether longer distribution matching is sufficient.
- 7B asks whether phase anchoring must be added.

Do not launch a raw-teacher arm before these two. It is now a lower-priority
fallback, useful only if Phase 7B cannot be implemented cleanly in Makani.

## Evaluation

The comparison table must include:

- raw baseline,
- Phase 4 residual+rollout,
- Phase 6B distribution rollout,
- Phase 7A,
- Phase 7B,
- persistence,
- climatology.

Primary metrics:

- formal Makani 120-month tos RMSE, ACC, CRPS,
- Nino3.4 trajectory std and RMSE across leads,
- `tos` drift and masked RMSE curves,
- radial spectra at 1, 3, 6, 12, 30, 60, and 120 months,
- spatial field and error maps at the same leads.

Success requires both:

1. long-run metrics near or better than Phase 6B, and
2. better Nino3.4 phase/trajectory behavior than Phase 6B without returning to
   Phase 4-like amplitude collapse.
