# Formal Experiment Plan

## Phase 1: SFNO Causal Test

| Arm | Architecture | Training data injection | Loss | Evaluation |
|---|---|---|---|---|
| Raw baseline | SFNO / Makani, `sfno_walker_1deg_edim384_layers8` | raw four-variable fields for all epochs | same Makani L2 recipe | long rollout stability suite |
| Fourier layerwise | SFNO / Makani, `sfno_walker_1deg_edim384_layers8` | cumulative low-to-high Fourier stages, ending raw | same Makani L2 recipe | same long rollout stability suite |

Both arms must use identical data split, normalization stats, optimizer settings, scheduler settings, seed policy, and total optimizer update count.

Loss note: Makani's official SFNO config uses `channel_weights: auto`, but that helper assumes ERA5-style variable names and pressure-level suffixes. It fails on the four Walker variables (`tauu`, `tauv`, `tos`, `zos`). Phase 1 therefore uses `channel_weights: constant` with `temp_diff_normalization: true` for both arms.

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
| `sfno_walker_1deg_edim384_layers8` | 147,776,272 | Phase 1 primary formal model |
| `sfno_walker_1deg_edim192_layers8` | 36,950,416 | Engineering preflight / emergency budget fallback only |

The 147.8M parameter model is selected for the first full-data causal test because the Makani Trainer preflight succeeded on RTX 6000 Ada. The 37.0M parameter version should not be reported as the main formal result unless the primary run becomes infeasible.

Parameter-count note: Makani counts complex-valued spectral weights by real-valued entries using `torch.view_as_real`, so these are the official counts to use in reports.

Training-time validation uses `valid_autoreg_steps: 19`, matching the style of the official SFNO recipe. Paper-aligned long-rollout stability metrics are run after training from saved checkpoints, not at every training epoch.

## First Implementation Steps

1. Convert full NetCDF data to Makani HDF5.
2. Freeze SFNO four-variable config.
3. Compute exact parameter count from frozen config.
4. Implement Fourier curriculum dataset generation over the training split only.
5. Lock benchmark-aligned long-rollout evaluation.
6. Launch paired formal training.
