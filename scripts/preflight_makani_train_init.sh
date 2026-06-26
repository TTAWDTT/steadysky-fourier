#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme1/lz/fourier_layerwise_weather"
REPO="${ROOT}/repos/steadysky-fourier"
MAKANI="${ROOT}/repos/makani"
DATA="${ROOT}/data/walker_ocean_1deg_full"
PYTHON="${ROOT}/conda_makani/bin/python"
CONFIG_NAME="sfno_walker_1deg_edim384_layers8"

ln -sfn "${DATA}/train_raw" "${DATA}/train_current_preflight"
mkdir -p "${ROOT}/configs" "${ROOT}/logs"
python - <<'PY'
from pathlib import Path
root = Path("/mnt/nvme1/lz/fourier_layerwise_weather")
src = root / "repos/steadysky-fourier/configs/sfno_walker_1deg.yaml"
txt = src.read_text()
txt = txt.replace('train_data_path: "/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full/train_raw"', 'train_data_path: "/mnt/nvme1/lz/fourier_layerwise_weather/data/walker_ocean_1deg_full/train_current_preflight"')
txt = txt.replace('max_epochs: 300', 'max_epochs: 1')
txt = txt.replace('valid_autoreg_steps: 119', 'valid_autoreg_steps: 1')
out = root / "configs/preflight_sfno_walker.yaml"
out.write_text(txt)
print(out)
PY

cd "${MAKANI}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 -m makani.train \
  --yaml_config="${ROOT}/configs/preflight_sfno_walker.yaml" \
  --config="${CONFIG_NAME}" \
  --run_num="preflight_edim384" \
  --amp_mode=bf16 \
  --batch_size=1 \
  --skip_training \
  2>&1 | tee "${ROOT}/logs/preflight_edim384.log"
