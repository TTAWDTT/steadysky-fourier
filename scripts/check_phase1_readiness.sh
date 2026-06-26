#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG="${REPO}/configs/sfno_walker_1deg.yaml"

REQUIRED_DATA_DIRS=(
  metadata
  stats_raw
  train_raw
  valid_raw
  test_raw
  train_lp004
  train_lp008
  train_lp016
  train_lp032
  train_lp064
)

REQUIRED_FILES=(
  "${CONFIG}"
  "${REPO}/scripts/run_phase1_training_schedule.sh"
  "${REPO}/scripts/check_training_log.py"
  "${DATA}/metadata/data.json"
  "${DATA}/stats_raw/global_means.npy"
  "${DATA}/stats_raw/global_stds.npy"
  "${DATA}/stats_raw/time_diff_means.npy"
  "${DATA}/stats_raw/time_diff_stds.npy"
)

echo "STEADYSKY_WORK=${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing executable Python: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -d "${MAKANI}/makani" ]]; then
  echo "Missing Makani package directory: ${MAKANI}/makani" >&2
  exit 3
fi

for dir in "${REQUIRED_DATA_DIRS[@]}"; do
  if [[ ! -d "${DATA}/${dir}" ]]; then
    echo "Missing data directory: ${DATA}/${dir}" >&2
    exit 4
  fi
done

for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 5
  fi
done

"${PYTHON}" "${REPO}/scripts/check_training_log.py" --help >/dev/null

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["STEADYSKY_WORK"])
data = root / "data/walker_ocean_1deg_full"
with (data / "metadata/data.json").open("r", encoding="utf-8") as handle:
    metadata = json.load(handle)

required_channels = ["tauu", "tauv", "tos", "zos"]
coords = metadata.get("coords", {})
channels = metadata.get("channel_names") or metadata.get("channels") or coords.get("channel") or []
if channels and list(channels) != required_channels:
    raise SystemExit(f"Unexpected channel names: {channels}")

for split, filename in [("train_raw", "train.h5"), ("valid_raw", "valid.h5"), ("test_raw", "test.h5")]:
    path = data / split / filename
    if not path.exists():
        raise SystemExit(f"Missing HDF5 file: {path}")

for stage in ["train_lp004", "train_lp008", "train_lp016", "train_lp032", "train_lp064"]:
    path = data / stage / "train.h5"
    if not path.exists():
        raise SystemExit(f"Missing Fourier-stage HDF5 file: {path}")

print("metadata_and_hdf5=ok")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo "nvidia-smi not found; skipping GPU visibility check"
fi

df -h "${ROOT}" || true
echo "phase1_readiness=ok"
