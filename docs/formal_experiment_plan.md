# Formal Experiment Plan

## Phase 1: SFNO Causal Test

| Arm | Architecture | Training data injection | Loss | Evaluation |
|---|---|---|---|---|
| Raw baseline | SFNO / Makani, `sfno_walker_1deg_edim192_layers8` | raw four-variable fields for all epochs | same Makani weighted L2 recipe | long rollout stability suite |
| Fourier layerwise | SFNO / Makani, `sfno_walker_1deg_edim192_layers8` | cumulative low-to-high Fourier stages, ending raw | same Makani weighted L2 recipe | same long rollout stability suite |

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

Concrete split after conversion:

| Split | Index range | Samples |
|---|---:|---:|
| Train | `[0, 1584)` | 1584 |
| Validation | `[1584, 1782)` | 198 |
| Test | `[1782, 1980)` | 198 |

The prepared Makani dataset is:

```text
/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full
```

NaN policy:

- `tauu` and `tauv` are fully finite.
- `tos` and `zos` have a fixed valid-grid fraction of `0.6330092592592592`.
- Missing `tos/zos` cells are filled with train-split channel means before HDF5 writing so Makani loss does not become NaN.
- `valid_mask` is saved and must be used for masked evaluation of `tos/zos`.

Train-split fill values:

| Channel | Fill value |
|---|---:|
| tauu | 0.0138265491 |
| tauv | 0.0070770057 |
| tos | 13.8552541733 |
| zos | -0.2408056557 |

## Phase 1 Model Choice

| Config | Parameters | Role |
|---|---:|---|
| `sfno_walker_1deg_edim192_layers8` | 72,339,076 | Phase 1 primary formal model |
| `sfno_walker_1deg_edim384_layers8` | 289,332,484 | Larger-scale follow-up after the primary causal test |

The 72M parameter model is large enough to be non-toy while still plausible on two RTX 6000 Ada GPUs. The 289M parameter version is retained for scale validation, not for the first formal run.

## First Implementation Steps

1. Convert full NetCDF data to Makani HDF5.
2. Freeze SFNO four-variable config.
3. Compute exact parameter count from frozen config.
4. Implement Fourier curriculum dataset generation over the training split only.
5. Lock benchmark-aligned long-rollout evaluation.
6. Launch paired formal training.
