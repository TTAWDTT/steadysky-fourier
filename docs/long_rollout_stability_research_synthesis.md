# Long-Rollout Stability Research Synthesis

This note synthesizes the Phase 2-5 SteadySky results with the current
literature on long autoregressive rollouts in AI weather and chaotic dynamical
systems.

## Problem Statement

For an autoregressive model

```text
x_{t+1} = F_theta(x_t),
```

short-horizon training usually minimizes a pointwise loss such as

```text
E ||F_theta(x_t) - x_{t+1}||^2.
```

This is not the same as learning a long-horizon climate emulator. A useful
long-rollout model needs both:

1. **Trajectory skill while predictability remains**: the model should track
   phase and structure at leads where the initial condition still contains
   information.
2. **Invariant-statistics fidelity after phase predictability decays**: the
   model should remain on the right attractor, preserving seasonal structure,
   anomaly variance, spectra, and cross-variable relationships.

Our experiments show the same tension:

- Phase 2 residual schedules can improve some spatial-structure metrics, but
  do not solve long-lead RMSE.
- Phase 3 Fourier-band losses can restore amplitude, but damage RMSE and ACC.
- Phase 4 rollout training lowers long-lead RMSE, but mostly by damping
  anomaly energy.

So the refined problem is:

> Learn an autoregressive map whose iterates are stable and whose long-run
> distribution matches the data attractor, without collapsing to the conditional
> mean or injecting phase-incoherent spectral energy.

## Literature Signals

### Stability failure modes

Lehmann et al. (2026), *Can AI Weather Models Predict Beyond Two Weeks?*, frame
long-rollout failures as blow-up, drift, and loss of seasonality. They also
find that stability depends strongly on the handling of small spatio-temporal
scales: unstable models amplify high-frequency energy, while stable models
often behave like denoisers.

Relevance: our Phase 4 residual+rollout arm is a stable denoiser. It avoids
blow-up but loses anomaly amplitude.

Source: https://arxiv.org/abs/2605.30184

### MSE induces smoothing

The recipe-oriented analysis in *Mathematical Foundations of the AI Weather
Prediction Pipeline* argues that MSE-trained deterministic weather models
systematically lose spectral energy as lead time grows. The mechanism is the
conditional-mean/double-penalty effect: when phase uncertainty is large, the MSE
optimizer reduces amplitude instead of sampling a plausible phase. The paper
also argues that this energy deficit is loss-induced rather than
architecture-specific.

Relevance: this is the mathematical explanation for why our low-to-high and
rollout-MSE runs become smooth and climatology-like.

Source: https://arxiv.org/html/2604.01215v1

### Invariant measures are the right long-term target

Jiang et al. (2023), *Training neural operators to preserve invariant measures
of chaotic attractors*, argue that long-term pointwise RMSE is the wrong signal
for chaotic systems. They propose preserving invariant measures through
distributional losses such as optimal transport over summary statistics, or
contrastive feature losses.

Relevance: this gives a principled version of our emerging direction. We should
not only penalize pointwise rollout error; we should match the distribution of
long-rollout statistics.

Source: https://arxiv.org/html/2306.01187v3

### Spectral loss helps, but must be carefully balanced

FastNet uses a modified spherical-harmonic loss as a fine-tuning step after MSE.
The authors report that it reduces blurring and improves spectral fidelity, but
can hurt RMSE if too dominant. They explicitly treat it as a balancing act:
combat MSE blur while keeping long-lead accuracy.

Relevance: our Phase 3 Fourier loss probably over-constrained spectral content
without protecting phase. Phase 5's small rollout-stage energy regularizer is
closer to the FastNet recipe than Phase 3 was.

Source: https://arxiv.org/html/2509.17601v1

### Proper scoring and multiscale losses are promising

AIFS-CRPS trains autoregressively through increasing rollout lengths and uses
an almost-fair CRPS objective. The training stages explicitly move from
single-step to multi-step rollouts. A related ECMWF multi-scale loss paper
shows that adding scale-aware components to proper-score training improves
small-scale variability without hurting skill.

Relevance: deterministic MSE may be intrinsically pressured toward smoothing.
A probabilistic or distributional objective is likely the cleaner long-term
solution, especially if it is scale-aware.

Sources:

- https://www.nature.com/articles/s44387-026-00073-7
- https://arxiv.org/html/2506.10868v1

### Probabilistic spectral-fidelity models

FourCastNet 3 uses a probabilistic hidden-Markov formulation with stochastic
latent dynamics on the sphere. It reports realistic spectra at extended leads
and stable subseasonal rollouts.

Relevance: if deterministic energy constraints are not enough, a stochastic
transition model may be needed. Long-term climate-like fidelity may require
sampling from the conditional distribution, not predicting its mean.

Source: https://arxiv.org/html/2507.12144v1

## Current Best Hypothesis

The most promising direction is:

> **Invariant-statistics-constrained rollout training**: train with short
> pointwise rollout loss for phase skill, plus weak long-rollout statistical
> constraints that preserve anomaly energy, spectra, seasonal cycle, and
> low-frequency indices.

This is more precise than "Fourier curriculum" alone. The Fourier idea remains
valuable, but its role should shift from "replace the training data with
low-pass fields" to "define scale-aware statistics that the rollout must
preserve."

## Optimization Target

Define a composite objective:

```text
L = L_pointwise_short
  + lambda_stat L_rollout_statistics
  + lambda_spec L_spectral_fidelity
  + lambda_drift L_climate_drift
```

where:

- `L_pointwise_short`: L2 or L1 over short rollout windows, preserving phase
  where phase is predictable.
- `L_rollout_statistics`: distance between forecast and target distributions
  of summary features over rollout windows.
- `L_spectral_fidelity`: log spectral energy error in low/mid bands, not raw
  Fourier coefficient MSE.
- `L_climate_drift`: mean/variance/seasonal-cycle drift penalty.

The optimization goal should be explicitly Pareto-like, not single-metric:

| Metric | Desired direction |
|---|---|
| 120-month tos RMSE | no worse than Phase 4 residual+rollout |
| 120-month tos ACC | higher than Phase 4 residual+rollout |
| tos amplitude ratio | closer to 0.8-1.1, not 0.35 |
| Nino3.4 amplitude ratio | closer to persistence, without RMSE blow-up |
| spectral energy ratio | near 1 in low/mid bands, no high-k explosion |
| drift | lower absolute mean/variance drift than raw |

## Promising Ideas

### 1. Energy-preserving rollout regularization

This is the Phase 5 experiment now running. It adds a small loss matching
forecast and target log spectral energy during rollout stages.

Prediction:

- RMSE stays near Phase 4 residual+rollout.
- Amplitude ratio rises above 0.35.
- If the weight is too high, it will reproduce Phase 3: amplitude improves but
  ACC/RMSE degrade.

### 2. Distributional attractor matching

Use MMD, sliced Wasserstein, or Sinkhorn loss on rollout summary features:

- channel means and variances,
- Nino3.4 / basin indices,
- low/mid spectral energy,
- seasonal harmonic coefficients,
- cross-variable correlations.

This follows the invariant-measure literature more directly than pointwise
spectral losses.

Prediction:

- Better long-run climate statistics than Phase 4.
- Less phase damage than Fourier coefficient loss because the target is a
  distribution of statistics, not exact spectral phase.

### 3. Two-head deterministic + stochastic residual model

Keep the deterministic SFNO mean forecast, but add a small stochastic residual
head or learned perturbation model during rollout. Deterministic MSE gives the
conditional mean; the stochastic residual restores conditional variance.

Prediction:

- Better amplitude and spectra than deterministic MSE.
- RMSE may not improve, but CRPS/spread/spectral fidelity should.

This is a larger architectural step, so it should come after exhausting
training-objective variants.

### 4. Seasonal low-mode anchoring

Explicitly penalize drift in the first annual harmonic and low-frequency
regional indices over rollout windows. This targets the "loss of seasonality"
failure mode directly.

Prediction:

- Improved seasonal cycle and low-frequency stability.
- Might not help small-scale spectra unless combined with energy regularization.

### 5. Jacobian/spectral-radius stability control

Penalize local expansion of the learned map along rollout states, for example
with a Hutchinson estimate of Jacobian norm or finite-difference perturbation
growth.

Prediction:

- Reduces blow-up risk.
- By itself, likely worsens damping unless paired with invariant-statistics
  constraints.

## Recommended Near-Term Plan

1. Complete Phase 5 energy/spectrum rollout.
2. If Phase 5 improves amplitude without wrecking RMSE, sweep only the
   regularizer strength in a narrow range.
3. If Phase 5 still damps too much, pivot to attractor-statistic matching
   rather than stronger spectral losses.
4. Add evaluation metrics before adding more training arms:
   - spectral energy ratio area-under-error,
   - anomaly amplitude error,
   - seasonal harmonic error,
   - Nino3.4 distributional distance,
   - drift slope.

## Clean Research Claim

A defensible paper framing is not:

> Fourier curriculum makes weather models forecast longer.

The stronger and more accurate framing is:

> Long-rollout stability is a trade-off between autoregressive contraction and
> attractor fidelity. Low-frequency curricula and rollout training can stabilize
> the learned map, but without invariant-statistics constraints they collapse
> anomaly energy. Scale-aware invariant-statistics regularization is a promising
> way to preserve long-run climate structure while retaining rollout stability.

