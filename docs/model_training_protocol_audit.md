# Model Training Protocol Audit

This repository treats the following as locked for Phase 1:

- Primary architecture: **SFNO / FourCastNet2 via NVIDIA Makani**
- Baseline: raw data injection
- Intervention: Fourier low-to-high cumulative data injection
- Evaluation target: long-rollout stability, including blow-up, drift, loss of seasonality, and small-scale spectral ratio
- Dataset: four provided 1-degree variables, not full ERA5

The canonical Phase 1 protocol is split across:

- `docs/formal_experiment_plan.md`
- `docs/phase1_launch_manifest.md`
- `docs/phase1_batch_capacity.md`
