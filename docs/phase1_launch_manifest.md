# Phase 1 Launch Manifest

This document freezes the first formal training launch for the SFNO causal test.

## Fixed Experimental Contract

| Item | Value |
|---|---|
| Architecture | NVIDIA Makani SFNO / FourCastNet2 |
| Config | `sfno_walker_1deg_edim384_layers8` |
| Trainable parameters | 147,776,272 |
| Variables | `tauu`, `tauv`, `tos`, `zos` |
| Grid | 1-degree, 180 x 360 |
| Train samples | 1584 |
| Validation samples | 198 |
| Test samples | 198 |
| Loss | Makani L2, constant channel weights, temporal-difference normalization, squared |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Precision | bf16 |
| Default launcher processes | 1 process per node |
| Optional launcher processes | `STEADYSKY_NPROC_PER_NODE=2`, only after distributed smoke test passes |
| Per-process batch size | 16 |
| Global batch size | 16 |
| Total planned epochs | 150 |
| Training-time validation rollout | 19 autoregressive steps |
| Formal evaluation | Post-training long-rollout stability suite |

## Arms

| Arm | Run number | Training data schedule | Stage epochs | Total epochs |
|---|---|---|---|---:|
| Raw baseline | `phase1_raw_edim384` | `train_raw`, repeated at every stage boundary | `10,15,20,25,35,45` | 150 |
| Fourier curriculum | `phase1_fourier_edim384` | `train_lp004`, `train_lp008`, `train_lp016`, `train_lp032`, `train_lp064`, `train_raw` | `10,15,20,25,35,45` | 150 |

The raw baseline is intentionally staged even though the data are always raw. This keeps checkpoint boundaries and cumulative epoch targets aligned with the Fourier curriculum arm.

## Early-Stop Safety

Early stopping is a safety rule, not a scoring rule. It is applied identically to both arms after each completed stage.

| Check | Default |
|---|---:|
| Non-finite training loss, validation loss, or gradient norm | stop immediately |
| Validation-loss explosion | stop if latest validation loss exceeds `1e6` |
| Validation-loss plateau | stop after 8 validation points without `1e-4` improvement, after 20 validation points |

If neither arm triggers a divergence condition, the primary comparison is the strict 150-epoch paired result.

## Launch Commands

```bash
export STEADYSKY_WORK=/path/to/working/directory

bash scripts/smoke_phase1_makani_launch.sh
bash scripts/launch_phase1_pair.sh --dry-run
bash scripts/launch_phase1_pair.sh
```

Formal training should run from the repository copy under `$STEADYSKY_WORK/repos/steadysky-fourier`, with Makani under `$STEADYSKY_WORK/repos/makani`.

The first launch uses `STEADYSKY_NPROC_PER_NODE=1` because the initial two-process launch failed before training with a runtime-level `free(): invalid pointer` in rank 1. This does not change the model, loss, data schedule, or epoch budget; it only avoids an unstable distributed startup path. A two-process launch can be restored through the environment variable after the distributed smoke test passes.

The paired launcher runs `smoke_phase1_makani_launch.sh` before formal training unless `STEADYSKY_RUN_SMOKE=0` is set. The smoke test uses the same process count as the formal launcher by default.

The batch-size choice is documented in `docs/phase1_batch_capacity.md`.

Diagnostic entry points:

| Script | Purpose |
|---|---|
| `scripts/check_phase1_readiness.sh` | Verify paths, data stages, metadata, Makani environment, and GPU visibility |
| `scripts/smoke_phase1_makani_launch.sh` | Validate Makani edim384 initialization and validation with the selected launcher settings |
| `scripts/probe_phase1_batch_capacity.sh` | Run short real train+validation probes for candidate batch sizes |
| `scripts/run_phase1_training_schedule.sh` | Run one arm schedule directly; normally called by the paired launcher |
