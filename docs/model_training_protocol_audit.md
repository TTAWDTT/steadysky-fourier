# Model Training Protocol Audit

See the canonical local audit saved in the original research workspace:

`outputs/model_training_protocol_audit.md`

This repository treats the following as locked for Phase 1:

- Primary architecture: **SFNO / FourCastNet2 via NVIDIA Makani**
- Baseline: raw data injection
- Intervention: Fourier low-to-high cumulative data injection
- Evaluation target: long-rollout stability, including blow-up, drift, loss of seasonality, and small-scale spectral ratio
- Dataset: four provided 1-degree variables, not full ERA5

