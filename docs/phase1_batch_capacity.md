# Phase 1 Batch Capacity

This note records the batch-size decision for the single-process Phase 1 launch path.

## Meaning Of Batch Size

Makani reports both:

| Field | Meaning |
|---|---|
| `batch_size` | per-process batch size passed to `makani.train` |
| `global_batch_size` | `batch_size * world_size` |

Because Phase 1 currently uses `STEADYSKY_NPROC_PER_NODE=1`, the per-process batch size and global batch size are identical.

Each sample is a full 1-degree four-variable field, not a token or a small patch:

```text
4 channels x 180 latitude x 360 longitude
```

## Probe Result

The capacity probe used the edim384 SFNO, bf16, one process, a short real training run, validation enabled, and checkpoint writing disabled.

| Batch size | Global batch size | Status | Reported memory high watermark |
|---:|---:|---|---:|
| 8 | 8 | pass | 15.30 GB |
| 12 | 12 | pass | 17.71 GB |
| 16 | 16 | pass | 20.32 GB |
| 20 | 20 | pass | 18.95 GB |
| 24 | 24 | OOM | failed during forward/backward |

The selected formal value is:

```text
STEADYSKY_BATCH_SIZE=16
STEADYSKY_NPROC_PER_NODE=1
global_batch_size=16
```

Although batch 20 passed the short probe, batch 24 failed with CUDA OOM. Batch 16 leaves more operational margin for long runs, memory fragmentation, checkpointing, and the full validation cadence while still doubling the original stable global batch of 8.

Both raw and Fourier arms must use the same batch setting.
