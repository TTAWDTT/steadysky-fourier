#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
PYTHON="${ROOT}/conda_makani/bin/python"
AE_LOG="${ROOT}/logs/phase10_wla_lite_autoencoder.log"
AE_CKPT="${STEADYSKY_WLA_LITE_CKPT:-${ROOT}/latent/wla_lite/wla_lite_best.pt}"
STAGE_EPOCHS="${1:-10,15,20,25,35,45}"

mkdir -p "${ROOT}/logs" "${ROOT}/latent/wla_lite"

if [[ ! -f "${AE_CKPT}" ]]; then
  echo "[$(date -Is)] training WLA-lite autoencoder -> ${AE_CKPT}" | tee -a "${AE_LOG}"
  CUDA_VISIBLE_DEVICES="${STEADYSKY_PHASE10_AE_GPU:-0}" "${PYTHON}" "${REPO}/scripts/train_wla_lite_autoencoder.py" \
    --data-root "${ROOT}/data/walker_ocean_1deg_full" \
    --output-dir "$(dirname "${AE_CKPT}")" \
    --epochs "${STEADYSKY_PHASE10_AE_EPOCHS:-80}" \
    --batch-size "${STEADYSKY_PHASE10_AE_BATCH_SIZE:-16}" \
    --latent-channels "${STEADYSKY_PHASE10_LATENT_CHANNELS:-32}" \
    --hidden-channels "${STEADYSKY_PHASE10_HIDDEN_CHANNELS:-64}" \
    --amp \
    2>&1 | tee -a "${AE_LOG}"
else
  echo "[$(date -Is)] reusing WLA-lite checkpoint: ${AE_CKPT}" | tee -a "${AE_LOG}"
fi

export STEADYSKY_WLA_LITE_CKPT="${AE_CKPT}"

CUDA_VISIBLE_DEVICES="${STEADYSKY_PHASE10A_GPU:-0}" bash "${REPO}/scripts/run_phase1_training_schedule.sh" \
  latent_distribution_rollout "${STAGE_EPOCHS}" \
  > "${ROOT}/logs/phase10_latent_distribution_rollout_launch.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES="${STEADYSKY_PHASE10B_GPU:-1}" bash "${REPO}/scripts/run_phase1_training_schedule.sh" \
  latent_recon_distribution_rollout "${STAGE_EPOCHS}" \
  > "${ROOT}/logs/phase10_latent_recon_distribution_rollout_launch.log" 2>&1 &
PID_B=$!

echo "[$(date -Is)] launched Phase10A pid=${PID_A}, Phase10B pid=${PID_B}"
wait "${PID_A}" "${PID_B}"
echo "[$(date -Is)] completed Phase10 pair"
