#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme1/lz/fourier_layerwise_weather"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"

ARM="${1:?usage: run_phase1_training_schedule.sh raw|fourier [epochs_per_stage]}"
EPOCHS_PER_STAGE="${2:-25}"

if [[ "${ARM}" != "raw" && "${ARM}" != "fourier" ]]; then
  echo "ARM must be raw or fourier" >&2
  exit 2
fi

if [[ "${ARM}" == "raw" ]]; then
  STAGES=(train_raw train_raw train_raw train_raw train_raw train_raw)
  RUN_NUM="phase1_raw_edim384"
else
  STAGES=(train_lp004 train_lp008 train_lp016 train_lp032 train_lp064 train_raw)
  RUN_NUM="phase1_fourier_edim384"
fi

cd "${MAKANI}"
mkdir -p "${ROOT}/runs/${CONFIG_NAME}/${RUN_NUM}" "${ROOT}/logs"

for IDX in "${!STAGES[@]}"; do
  STAGE="${STAGES[$IDX]}"
  TARGET="${DATA}/${STAGE}"
  if [[ ! -d "${TARGET}" ]]; then
    echo "Missing stage data: ${TARGET}" >&2
    exit 3
  fi

  ln -sfn "${TARGET}" "${DATA}/train_current_${ARM}"
  STAGE_END_EPOCH=$(( (IDX + 1) * EPOCHS_PER_STAGE ))
  LOG="${ROOT}/logs/${RUN_NUM}_stage$((IDX + 1))_${STAGE}.log"
  echo "[$(date -Is)] ARM=${ARM} stage=$((IDX + 1))/${#STAGES[@]} data=${STAGE} max_epochs=${STAGE_END_EPOCH}" | tee -a "${LOG}"

  python - <<PY
from pathlib import Path
src = Path("${CONFIG}")
txt = src.read_text()
txt = txt.replace('train_data_path: "/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full/train_raw"', 'train_data_path: "/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full/train_current_${ARM}"')
txt = txt.replace('max_epochs: 300', 'max_epochs: ${STAGE_END_EPOCH}')
out = Path("${ROOT}/configs/${RUN_NUM}_stage$((IDX + 1)).yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(txt)
print(out)
PY

  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 -m makani.train \
    --yaml_config="${ROOT}/configs/${RUN_NUM}_stage$((IDX + 1)).yaml" \
    --config="${CONFIG_NAME}" \
    --run_num="${RUN_NUM}" \
    --amp_mode=bf16 \
    --batch_size=4 \
    --h_parallel_size=1 \
    --w_parallel_size=1 \
    --matmul_parallel_size=1 \
    --multistep_count=1 \
    2>&1 | tee -a "${LOG}"
done

echo "[$(date -Is)] completed ${ARM} schedule" | tee -a "${ROOT}/logs/${RUN_NUM}_complete.log"
