#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"
NPROC_PER_NODE="${STEADYSKY_PROBE_NPROC_PER_NODE:-1}"
BATCHES_CSV="${STEADYSKY_PROBE_BATCHES:-8,12,16,20,24}"
TRAIN_SAMPLES="${STEADYSKY_PROBE_TRAIN_SAMPLES:-64}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_PROBE_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 2
fi

if ! [[ "${TRAIN_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_PROBE_TRAIN_SAMPLES must be a positive integer, got: ${TRAIN_SAMPLES}" >&2
  exit 3
fi

ln -sfn "${DATA}/train_raw" "${DATA}/train_current_probe"
mkdir -p "${ROOT}/configs" "${ROOT}/logs"

ROOT="${ROOT}" CONFIG="${CONFIG}" TRAIN_SAMPLES="${TRAIN_SAMPLES}" python - <<'PY'
import os
from pathlib import Path

root = os.environ["ROOT"]
src = Path(os.environ["CONFIG"])
train_samples = os.environ["TRAIN_SAMPLES"]
txt = src.read_text()
txt = txt.replace("${STEADYSKY_WORK}", root)
txt = txt.replace(
    f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_raw"',
    f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_current_probe"',
)
txt = txt.replace("max_epochs: 300", "max_epochs: 1")
txt = txt.replace("valid_autoreg_steps: 19", "valid_autoreg_steps: 1")
txt = txt.replace("n_train_samples_per_epoch: 1583", f"n_train_samples_per_epoch: {train_samples}")
out = Path(root) / "configs/probe_phase1_batch_capacity.yaml"
out.write_text(txt)
print(out)
PY

IFS=',' read -r -a BATCHES <<< "${BATCHES_CSV}"
SUMMARY="${ROOT}/logs/probe_phase1_batch_capacity_$(date -u +%Y%m%dT%H%M%SZ).tsv"
echo -e "batch_size\tglobal_batch_size\tstatus\tmax_gpu0_mib\tlog" | tee "${SUMMARY}"

cd "${MAKANI}"
for BATCH_SIZE in "${BATCHES[@]}"; do
  if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid batch size in STEADYSKY_PROBE_BATCHES: ${BATCH_SIZE}" >&2
    exit 4
  fi

  RUN_NUM="probe_phase1_edim384_np${NPROC_PER_NODE}_bs${BATCH_SIZE}"
  LOG="${ROOT}/logs/${RUN_NUM}.log"
  GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))
  echo "[$(date -Is)] probe batch_size=${BATCH_SIZE} global_batch_size=${GLOBAL_BATCH}" | tee "${LOG}"

  set +e
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m makani.train \
    --yaml_config="${ROOT}/configs/probe_phase1_batch_capacity.yaml" \
    --config="${CONFIG_NAME}" \
    --run_num="${RUN_NUM}" \
    --amp_mode=bf16 \
    --batch_size="${BATCH_SIZE}" \
    --h_parallel_size=1 \
    --w_parallel_size=1 \
    --matmul_parallel_size=1 \
    --multistep_count=1 \
    --save_checkpoint=none \
    2>&1 | tee -a "${LOG}"
  STATUS=${PIPESTATUS[0]}
  set -e

  MAX_GPU0_MIB=$(grep -E "memory footprint \\[GB\\]:" "${LOG}" | awk '{print int($NF * 1024)}' | tail -1)
  MAX_GPU0_MIB="${MAX_GPU0_MIB:-unknown}"
  if [[ "${STATUS}" -eq 0 ]]; then
    RESULT="ok"
  else
    RESULT="fail_${STATUS}"
  fi
  echo -e "${BATCH_SIZE}\t${GLOBAL_BATCH}\t${RESULT}\t${MAX_GPU0_MIB}\t${LOG}" | tee -a "${SUMMARY}"

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "Stopping after failed batch candidate: ${BATCH_SIZE}" | tee -a "${SUMMARY}"
    exit "${STATUS}"
  fi
done

echo "summary=${SUMMARY}"
