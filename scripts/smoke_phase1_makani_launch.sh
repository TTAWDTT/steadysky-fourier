#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"
NPROC_PER_NODE="${STEADYSKY_SMOKE_NPROC_PER_NODE:-1}"
BATCH_SIZE="${STEADYSKY_SMOKE_BATCH_SIZE:-16}"
RUN_NUM="smoke_phase1_edim384_np${NPROC_PER_NODE}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_SMOKE_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 2
fi

ln -sfn "${DATA}/train_raw" "${DATA}/train_current_smoke"
mkdir -p "${ROOT}/configs" "${ROOT}/logs"

ROOT="${ROOT}" CONFIG="${CONFIG}" python - <<'PY'
import os
from pathlib import Path

root = os.environ["ROOT"]
src = Path(os.environ["CONFIG"])
txt = src.read_text()
txt = txt.replace("${STEADYSKY_WORK}", root)
txt = txt.replace(
    f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_raw"',
    f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_current_smoke"',
)
txt = txt.replace("max_epochs: 300", "max_epochs: 1")
txt = txt.replace("valid_autoreg_steps: 19", "valid_autoreg_steps: 1")
out = Path(root) / "configs/smoke_phase1_edim384.yaml"
out.write_text(txt)
print(out)
PY

cd "${MAKANI}"
LOG="${ROOT}/logs/${RUN_NUM}.log"
echo "[$(date -Is)] smoke nproc_per_node=${NPROC_PER_NODE} batch_size=${BATCH_SIZE}" | tee "${LOG}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m makani.train \
  --yaml_config="${ROOT}/configs/smoke_phase1_edim384.yaml" \
  --config="${CONFIG_NAME}" \
  --run_num="${RUN_NUM}" \
  --amp_mode=bf16 \
  --batch_size="${BATCH_SIZE}" \
  --h_parallel_size=1 \
  --w_parallel_size=1 \
  --matmul_parallel_size=1 \
  --multistep_count=1 \
  --skip_training \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Is)] smoke complete" | tee -a "${LOG}"
