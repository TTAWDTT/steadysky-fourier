# Phase 7 Literature Note

This note records the second literature pass used to decide Phase 7.

## Sources

| Source | Relevant signal for Phase 7 |
|---|---|
| *Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts* | Long AI weather rollouts should be analyzed through blow-up, drift, and loss of seasonality, not only short-range skill. |
| Bengio et al., 2015, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | Teacher-forced training differs from inference-time autoregression; models should gradually face their own generated states during training. |
| Ross et al., 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* | Sequential prediction violates i.i.d. assumptions because future observations depend on previous model actions; training on the induced state distribution is important. |
| Lamb et al., 2016, *Professor Forcing* | Matching training-time and sampling-time dynamics is a direct way to improve long generated sequences. |
| *Training neural operators to preserve invariant measures of chaotic attractors* | For chaotic dynamics, preserving invariant-measure statistics can be a better long-time objective than matching pointwise trajectories alone. |
| *Dynamics Stable Learning by Invariant Measure for Chaotic Systems* | Typical trajectory misfit losses can diverge over long horizons; adding invariant-measure learning targets stable long-run behavior. |
| *Thermalizer: Stable autoregressive neural emulation of climate* | Short-term validation loss can be weakly predictive of long-term stability, matching our Phase 6A/6B surprise. |

## Consequences

The literature does not support simply increasing Fourier-band or spectral-energy
penalties. Those objectives still ask for pointwise or local spectral matching,
and our Phase 3/5 results already show that this can restore amplitude without
improving phase skill.

The literature also makes a raw one-step teacher less attractive as the first
Phase 7 arm. A teacher can anchor short-lead variability, but unless it is tied
to the student-induced rollout distribution, it does not directly address the
training/inference distribution mismatch.

The strongest next test is therefore:

1. extend Phase 6B to longer train-time rollout exposure,
2. separately test whether a short-lead tendency/phase anchor can preserve
   regional trajectory skill without undoing Phase 6B's distributional gain.
