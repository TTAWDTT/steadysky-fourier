#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"

ARM="${1:?usage: run_phase1_training_schedule.sh raw|fourier [comma_separated_stage_epochs]}"
STAGE_EPOCHS_CSV="${2:-10,15,20,25,35,45}"
EARLY_STOP_PATIENCE="${STEADYSKY_EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_POINTS="${STEADYSKY_EARLY_STOP_MIN_POINTS:-20}"
EARLY_STOP_MIN_DELTA="${STEADYSKY_EARLY_STOP_MIN_DELTA:-1e-4}"
EARLY_STOP_MAX_VALID_LOSS="${STEADYSKY_EARLY_STOP_MAX_VALID_LOSS:-1e6}"

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

IFS=',' read -r -a STAGE_EPOCHS <<< "${STAGE_EPOCHS_CSV}"
if [[ "${#STAGE_EPOCHS[@]}" -ne "${#STAGES[@]}" ]]; then
  echo "Expected ${#STAGES[@]} stage epoch values, got ${#STAGE_EPOCHS[@]}: ${STAGE_EPOCHS_CSV}" >&2
  exit 4
fi

cd "${MAKANI}"
mkdir -p "${ROOT}/runs/${CONFIG_NAME}/${RUN_NUM}" "${ROOT}/logs"

for IDX in "${!STAGES[@]}"; do
  STAGE="${STAGES[$IDX]}"
  STAGE_EPOCHS_THIS="${STAGE_EPOCHS[$IDX]}"
  TARGET="${DATA}/${STAGE}"
  if [[ ! -d "${TARGET}" ]]; then
    echo "Missing stage data: ${TARGET}" >&2
    exit 3
  fi

  ln -sfn "${TARGET}" "${DATA}/train_current_${ARM}"
  STAGE_END_EPOCH=0
  for J in $(seq 0 "${IDX}"); do
    STAGE_END_EPOCH=$(( STAGE_END_EPOCH + STAGE_EPOCHS[$J] ))
  done
  LOG="${ROOT}/logs/${RUN_NUM}_stage$((IDX + 1))_${STAGE}.log"
  echo "[$(date -Is)] ARM=${ARM} stage=$((IDX + 1))/${#STAGES[@]} data=${STAGE} stage_epochs=${STAGE_EPOCHS_THIS} max_epochs=${STAGE_END_EPOCH}" | tee -a "${LOG}"

  STAGE_INDEX=$((IDX + 1)) ROOT="${ROOT}" CONFIG="${CONFIG}" ARM="${ARM}" RUN_NUM="${RUN_NUM}" STAGE_END_EPOCH="${STAGE_END_EPOCH}" python - <<'PY'
import os
from pathlib import Path
root = os.environ["ROOT"]
src = Path(os.environ["CONFIG"])
arm = os.environ["ARM"]
run_num = os.environ["RUN_NUM"]
stage_index = os.environ["STAGE_INDEX"]
stage_end_epoch = os.environ["STAGE_END_EPOCH"]
txt = src.read_text()
txt = txt.replace("${STEADYSKY_WORK}", root)
txt = txt.replace(f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_raw"', f'train_data_path: "{root}/data/walker_ocean_1deg_full/train_current_{arm}"')
txt = txt.replace("max_epochs: 300", f"max_epochs: {stage_end_epoch}")
out = Path(root) / "configs" / f"{run_num}_stage{stage_index}.yaml"
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

  "${PYTHON}" "${REPO}/scripts/check_training_log.py" "${ROOT}"/logs/"${RUN_NUM}"_stage*.log \
    --patience "${EARLY_STOP_PATIENCE}" \
    --min-points "${EARLY_STOP_MIN_POINTS}" \
    --min-delta "${EARLY_STOP_MIN_DELTA}" \
    --max-valid-loss "${EARLY_STOP_MAX_VALID_LOSS}" \
    2>&1 | tee -a "${LOG}"
done

echo "[$(date -Is)] completed ${ARM} schedule" | tee -a "${ROOT}/logs/${RUN_NUM}_complete.log"
