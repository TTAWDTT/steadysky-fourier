# Phase 3 Loss-Curriculum Plan

Phase 3 tests the next step after the Phase 2 smoothing diagnosis: keep the model input and target on raw data for every epoch, and move the low-to-high Fourier curriculum into the loss.

This avoids treating low-pass reconstructions as if they were raw physical states.

## Arms

| GPU | Arm | Run number | Input | Target | Training change |
|---:|---|---|---|---|---|
| 0 | Fourier loss curriculum | `phase3_freq_loss_edim384` | raw | raw | stage-wise 2D Fourier-band loss |
| 1 | Fourier loss + anomaly-change | `phase3_freq_anom_edim384` | raw | raw | Fourier-band loss plus tendency-space Fourier loss |

Everything else stays aligned with Phase 1 and Phase 2: SFNO/FourCastNet2 via Makani, 147,776,272 parameters, same split, same optimizer, same batch size, same 150-epoch budget, and the same post-training 120-month rollout evaluation.

## Stage Weights

| Stage | Epochs | Field L2 | Low-k | Mid-k | High-k | Tendency/anomaly-change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.15 | 1.00 | 0.00 | 0.00 | 0.25 |
| 2 | 15 | 0.20 | 1.00 | 0.20 | 0.00 | 0.35 |
| 3 | 20 | 0.30 | 1.00 | 0.50 | 0.00 | 0.45 |
| 4 | 25 | 0.45 | 1.00 | 0.75 | 0.20 | 0.55 |
| 5 | 35 | 0.70 | 1.00 | 1.00 | 0.50 | 0.65 |
| 6 | 45 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 |

The `phase3_freq_loss_edim384` arm uses field L2 plus Fourier-band loss. The `phase3_freq_anom_edim384` arm adds a tendency-space Fourier-band loss. This tendency term compares `prediction - input` with `target - input`, so it gives the second arm a real anomaly-change signal rather than a climatology subtraction that would cancel algebraically under MSE.

## Evaluation

After training, both arms use the same long-rollout evaluation path:

```bash
bash scripts/run_phase1_long_rollout_eval.sh freq_loss 120
bash scripts/run_phase1_long_rollout_eval.sh freq_anom 120
```

The comparison should include raw, pure Fourier, mixed, residual, frequency-loss, frequency-anomaly, persistence, and climatology. The main success criteria are higher long-lead ACC and Nino3.4 correlation without collapsing anomaly amplitude or relying only on smoother RMSE.
