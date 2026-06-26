# Formal Experiment Plan

## Phase 1: SFNO Causal Test

| Arm | Architecture | Training data injection | Loss | Evaluation |
|---|---|---|---|---|
| Raw baseline | SFNO / Makani | raw four-variable fields for all epochs | same Makani weighted L2 recipe | long rollout stability suite |
| Fourier layerwise | SFNO / Makani | cumulative low-to-high Fourier stages, ending raw | same Makani weighted L2 recipe | same long rollout stability suite |

Both arms must use identical data split, normalization stats, optimizer settings, scheduler settings, seed policy, and total optimizer update count.

## Data

Source variables:

- `tauu`
- `tauv`
- `tos`
- `zos`

Shape discovered from source files:

- time: 1980
- latitude: 180
- longitude: 360
- channel: 4

Chronological split:

- train: first 80%
- validation: next 10%
- test: final 10%

## First Implementation Steps

1. Convert full NetCDF data to Makani HDF5.
2. Freeze SFNO four-variable config.
3. Compute exact parameter count from frozen config.
4. Implement Fourier curriculum dataset generation over the training split only.
5. Lock benchmark-aligned long-rollout evaluation.
6. Launch paired formal training.

