# steadysky-fourier

Fourier layerwise data injection for stable long-rollout weather and climate models.

The project tests a single intervention: keep the model architecture and training recipe fixed, but change how training data is injected. The baseline sees raw fields throughout training. The Fourier curriculum starts from low-frequency reconstructions, cumulatively adds higher-frequency content, and ends on raw fields.

## Research Rule

Every formal comparison must hold constant:

- model architecture
- loss function
- optimizer and scheduler
- train/validation/test split
- normalization statistics
- initialization seed where possible
- total optimizer updates

Only the data injection schedule may differ.

## Current Formal Plan

Phase 1 uses **SFNO / FourCastNet2 via NVIDIA Makani** on the four provided 1-degree variables:

- `tauu`
- `tauv`
- `tos`
- `zos`

The first formal comparison is:

| Arm | Model | Data injection |
|---|---|---|
| Baseline | `sfno_walker_1deg_edim384_layers8` | raw four-variable fields throughout training |
| Ours | `sfno_walker_1deg_edim384_layers8` | cumulative low-to-high Fourier curriculum, ending on raw fields |

The Phase 1 model has **76,997,392 trainable parameters**. A smaller `edim192_layers8` configuration with **19,255,696 trainable parameters** is retained only for engineering preflight and emergency budget fallback, not as the main formal result.

Phase 2 will repeat the same Raw vs Fourier comparison on additional architectures after Phase 1 is complete.

## Repository Scope

This repository stores code, configs, protocols, and lightweight metadata. It does not store the source NetCDF data, generated HDF5 datasets, checkpoints, or rollout files.

The working compute path used for formal experiments is:

```text
/mnt/nvme1/lz/fourier_layerwise_weather
```

## Data Preparation

The expected local source data path is:

```text
D:\Github\WalkerNet\data_1x1
```

The formal conversion script is:

```bash
python scripts/prepare_walker_makani_full.py \
  --source-root /path/to/data_1x1 \
  --output-root /mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full
```

It writes Makani-compatible HDF5 splits, metadata, normalization statistics, and a manifest.

The current formal remote dataset is:

```text
/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full
```

It uses train/validation/test chronological splits of `1584/198/198` samples.
