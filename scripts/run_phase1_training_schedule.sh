#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"

ARM="${1:?usage: run_phase1_training_schedule.sh raw|fourier|mixed|residual|freq_loss|freq_anom [comma_separated_stage_epochs]}"
STAGE_EPOCHS_CSV="${2:-10,15,20,25,35,45}"
EARLY_STOP_PATIENCE="${STEADYSKY_EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_POINTS="${STEADYSKY_EARLY_STOP_MIN_POINTS:-20}"
EARLY_STOP_MIN_DELTA="${STEADYSKY_EARLY_STOP_MIN_DELTA:-1e-4}"
EARLY_STOP_MAX_VALID_LOSS="${STEADYSKY_EARLY_STOP_MAX_VALID_LOSS:-1e6}"
NPROC_PER_NODE="${STEADYSKY_NPROC_PER_NODE:-1}"
BATCH_SIZE="${STEADYSKY_BATCH_SIZE:-16}"
START_STAGE="${STEADYSKY_START_STAGE:-1}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 5
fi

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 6
fi

if ! [[ "${START_STAGE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEADYSKY_START_STAGE must be a positive integer, got: ${START_STAGE}" >&2
  exit 7
fi

if [[ "${ARM}" != "raw" && "${ARM}" != "fourier" && "${ARM}" != "mixed" && "${ARM}" != "residual" && "${ARM}" != "freq_loss" && "${ARM}" != "freq_anom" ]]; then
  echo "ARM must be raw, fourier, mixed, residual, freq_loss, or freq_anom" >&2
  exit 2
fi

if [[ "${ARM}" == "raw" ]]; then
  STAGES=(train_raw train_raw train_raw train_raw train_raw train_raw)
  RUN_NUM="phase1_raw_edim384"
elif [[ "${ARM}" == "fourier" ]]; then
  STAGES=(train_lp004 train_lp008 train_lp016 train_lp032 train_lp064 train_raw)
  RUN_NUM="phase1_fourier_edim384"
elif [[ "${ARM}" == "mixed" ]]; then
  STAGES=(train_mixed_lp004_r020 train_mixed_lp008_r030 train_mixed_lp016_r040 train_mixed_lp032_r050 train_mixed_lp064_r065 train_raw)
  RUN_NUM="phase2_mixed_edim384"
elif [[ "${ARM}" == "residual" ]]; then
  STAGES=(train_residual_lp004_l005 train_residual_lp008_l015 train_residual_lp016_l030 train_residual_lp032_l050 train_residual_lp064_l075 train_raw)
  RUN_NUM="phase2_residual_edim384"
elif [[ "${ARM}" == "freq_loss" ]]; then
  STAGES=(train_raw train_raw train_raw train_raw train_raw train_raw)
  RUN_NUM="phase3_freq_loss_edim384"
else
  STAGES=(train_raw train_raw train_raw train_raw train_raw train_raw)
  RUN_NUM="phase3_freq_anom_edim384"
fi

EXTERNAL_FOURIER_MARKER="${ROOT}/logs/phase1_fourier_external_run.marker"
if [[ "${ARM}" == "fourier" && -f "${EXTERNAL_FOURIER_MARKER}" && "${STEADYSKY_FORCE_FOURIER_RUN:-0}" != "1" ]]; then
  echo "Fourier arm is already handled by an external run: ${EXTERNAL_FOURIER_MARKER}"
  exit 0
fi

IFS=',' read -r -a STAGE_EPOCHS <<< "${STAGE_EPOCHS_CSV}"
if [[ "${#STAGE_EPOCHS[@]}" -ne "${#STAGES[@]}" ]]; then
  echo "Expected ${#STAGES[@]} stage epoch values, got ${#STAGE_EPOCHS[@]}: ${STAGE_EPOCHS_CSV}" >&2
  exit 4
fi

if (( START_STAGE > ${#STAGES[@]} )); then
  echo "STEADYSKY_START_STAGE=${START_STAGE} exceeds number of stages (${#STAGES[@]})" >&2
  exit 8
fi

cd "${MAKANI}"
RUN_DIR="${ROOT}/runs/${CONFIG_NAME}/${RUN_NUM}"
mkdir -p "${RUN_DIR}/training_checkpoints" "${ROOT}/logs"

if [[ "${ARM}" == "freq_loss" || "${ARM}" == "freq_anom" ]]; then
  "${PYTHON}" "${REPO}/scripts/install_makani_phase3_losses.py" --makani-root "${MAKANI}"
fi

for IDX in "${!STAGES[@]}"; do
  if (( IDX + 1 < START_STAGE )); then
    continue
  fi

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
  echo "[$(date -Is)] ARM=${ARM} stage=$((IDX + 1))/${#STAGES[@]} data=${STAGE} stage_epochs=${STAGE_EPOCHS_THIS} max_epochs=${STAGE_END_EPOCH} nproc_per_node=${NPROC_PER_NODE} batch_size=${BATCH_SIZE}" | tee -a "${LOG}"

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
if arm in {"freq_loss", "freq_anom"}:
    # Stage-wise frequency curriculum. Inputs and targets remain raw; only
    # supervision weights change across the same cumulative epoch boundaries.
    weights = {
        1: {"field": 0.15, "low": 1.00, "mid": 0.00, "high": 0.00, "anom": 0.25},
        2: {"field": 0.20, "low": 1.00, "mid": 0.20, "high": 0.00, "anom": 0.35},
        3: {"field": 0.30, "low": 1.00, "mid": 0.50, "high": 0.00, "anom": 0.45},
        4: {"field": 0.45, "low": 1.00, "mid": 0.75, "high": 0.20, "anom": 0.55},
        5: {"field": 0.70, "low": 1.00, "mid": 1.00, "high": 0.50, "anom": 0.65},
        6: {"field": 1.00, "low": 1.00, "mid": 1.00, "high": 1.00, "anom": 0.75},
    }[int(stage_index)]
    base_loss = f'''    losses:
    -   type: "l2"
        channel_weights: "constant"
        temp_diff_normalization: !!bool True
        relative_weight: {weights["field"]}
        parameters:
            squared: !!bool True
    -   type: "fourier2d"
        channel_weights: "constant"
        relative_weight: 1.0
        parameters:
            low_weight: {weights["low"]}
            mid_weight: {weights["mid"]}
            high_weight: {weights["high"]}
            low_max: 5
            mid_max: 20
'''
    if arm == "freq_anom":
        base_loss += f'''    -   type: "fourier2d"
        tendency: !!bool True
        channel_weights: "constant"
        relative_weight: {weights["anom"]}
        parameters:
            low_weight: {weights["low"]}
            mid_weight: {weights["mid"]}
            high_weight: {weights["high"]}
            low_max: 5
            mid_max: 20
'''
    original = '''    losses:
    -   type: "l2"
        channel_weights: "constant"
        temp_diff_normalization: !!bool True
        parameters:
            squared: !!bool True
'''
    if original not in txt:
        raise RuntimeError("Could not find base loss block to replace")
    txt = txt.replace(original, base_loss)
out = Path(root) / "configs" / f"{run_num}_stage{stage_index}.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(txt)
print(out)
PY

  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m makani.train \
    --yaml_config="${ROOT}/configs/${RUN_NUM}_stage$((IDX + 1)).yaml" \
    --config="${CONFIG_NAME}" \
    --run_num="${RUN_NUM}" \
    --amp_mode=bf16 \
    --batch_size="${BATCH_SIZE}" \
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
