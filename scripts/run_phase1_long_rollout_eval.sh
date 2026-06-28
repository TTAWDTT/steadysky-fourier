#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG_SOURCE="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"

ARM="${1:?usage: run_phase1_long_rollout_eval.sh raw|fourier|mixed|residual|freq_loss|freq_anom|residual_soft|residual_rollout [rollout_months]}"
ROLLOUT_MONTHS="${2:-120}"
NPROC_PER_NODE="${STEADYSKY_EVAL_NPROC_PER_NODE:-1}"
BATCH_SIZE="${STEADYSKY_EVAL_BATCH_SIZE:-1}"
DATE_STEP_HOURS="${STEADYSKY_EVAL_DATE_STEP_HOURS:-8760}"
OUTPUT_MEMORY_BUFFER_SIZE="${STEADYSKY_EVAL_OUTPUT_MEMORY_BUFFER_SIZE:-0}"

if [[ "${ARM}" != "raw" && "${ARM}" != "fourier" && "${ARM}" != "mixed" && "${ARM}" != "residual" && "${ARM}" != "freq_loss" && "${ARM}" != "freq_anom" && "${ARM}" != "residual_soft" && "${ARM}" != "residual_rollout" ]]; then
  echo "ARM must be raw, fourier, mixed, residual, freq_loss, freq_anom, residual_soft, or residual_rollout" >&2
  exit 2
fi

if ! [[ "${ROLLOUT_MONTHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "rollout_months must be a positive integer, got: ${ROLLOUT_MONTHS}" >&2
  exit 3
fi

if (( ROLLOUT_MONTHS < 2 )); then
  echo "rollout_months must be at least 2, got: ${ROLLOUT_MONTHS}" >&2
  exit 3
fi

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_EVAL_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 4
fi

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_EVAL_BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 5
fi

if [[ "${ARM}" == "raw" || "${ARM}" == "fourier" ]]; then
  RUN_NUM="phase1_${ARM}_edim384"
elif [[ "${ARM}" == "mixed" || "${ARM}" == "residual" ]]; then
  RUN_NUM="phase2_${ARM}_edim384"
elif [[ "${ARM}" == "residual_soft" || "${ARM}" == "residual_rollout" ]]; then
  RUN_NUM="phase4_${ARM}_edim384"
else
  RUN_NUM="phase3_${ARM}_edim384"
fi
CONFIG="${ROOT}/configs/${RUN_NUM}_eval_rollout${ROLLOUT_MONTHS}.yaml"
LOG="${ROOT}/logs/${RUN_NUM}_eval_rollout${ROLLOUT_MONTHS}.log"

cd "${MAKANI}"
mkdir -p "${ROOT}/configs" "${ROOT}/logs" "${ROOT}/eval"

ROOT="${ROOT}" CONFIG_SOURCE="${CONFIG_SOURCE}" CONFIG="${CONFIG}" ROLLOUT_MONTHS="${ROLLOUT_MONTHS}" python - <<'PY'
import os
from pathlib import Path

root = os.environ["ROOT"]
source = Path(os.environ["CONFIG_SOURCE"])
target = Path(os.environ["CONFIG"])
rollout_months = int(os.environ["ROLLOUT_MONTHS"])
autoreg_steps = rollout_months - 1

txt = source.read_text()
txt = txt.replace("${STEADYSKY_WORK}", root)
txt = txt.replace("valid_autoreg_steps: 19", f"valid_autoreg_steps: {autoreg_steps}")
txt = txt.replace("save_raw_forecasts: !!bool True", "save_raw_forecasts: !!bool True")
target.write_text(txt)
print(target)
PY

echo "[$(date -Is)] ARM=${ARM} rollout_months=${ROLLOUT_MONTHS} nproc_per_node=${NPROC_PER_NODE} batch_size=${BATCH_SIZE} date_step_hours=${DATE_STEP_HOURS}" | tee -a "${LOG}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m makani.inference \
  --yaml_config="${CONFIG}" \
  --config="${CONFIG_NAME}" \
  --run_num="${RUN_NUM}" \
  --amp_mode=bf16 \
  --batch_size="${BATCH_SIZE}" \
  --load_checkpoint=legacy \
  --h_parallel_size=1 \
  --w_parallel_size=1 \
  --matmul_parallel_size=1 \
  --date_step="${DATE_STEP_HOURS}" \
  --output_channels tauu tauv tos zos \
  --output_file="${RUN_NUM}_rollout${ROLLOUT_MONTHS}_forecasts.h5" \
  --output_memory_buffer_size="${OUTPUT_MEMORY_BUFFER_SIZE}" \
  --metrics_file="${RUN_NUM}_rollout${ROLLOUT_MONTHS}_metrics.h5" \
  --bias_file="${RUN_NUM}_rollout${ROLLOUT_MONTHS}_bias.h5" \
  --spectrum_file="${RUN_NUM}_rollout${ROLLOUT_MONTHS}_spectrum.h5" \
  --zonal_spectrum_file="${RUN_NUM}_rollout${ROLLOUT_MONTHS}_zonal_spectrum.h5" \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Is)] completed ${ARM} rollout${ROLLOUT_MONTHS} evaluation" | tee -a "${ROOT}/logs/${RUN_NUM}_eval_rollout${ROLLOUT_MONTHS}_complete.log"
