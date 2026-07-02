#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
STAGE_EPOCHS_CSV="${STEADYSKY_STAGE_EPOCHS:-10,15,20,25,35,45}"
ORDER="${STEADYSKY_PHASE7_ORDER:-long_distribution_rollout,anchored_distribution_rollout}"
RUN_SMOKE="${STEADYSKY_RUN_SMOKE:-0}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" != "" ]]; then
  echo "usage: launch_phase7_pair.sh [--dry-run]" >&2
  exit 2
fi

IFS=',' read -r -a ARMS <<< "${ORDER}"
for ARM in "${ARMS[@]}"; do
  if [[ "${ARM}" != "long_distribution_rollout" && "${ARM}" != "anchored_distribution_rollout" ]]; then
    echo "Invalid arm in STEADYSKY_PHASE7_ORDER: ${ARM}" >&2
    exit 3
  fi
done

cd "${REPO}"
bash scripts/check_phase1_readiness.sh
if [[ "${RUN_SMOKE}" == "1" ]]; then
  bash scripts/smoke_phase1_makani_launch.sh
fi

mkdir -p "${ROOT}/logs"
MANIFEST="${ROOT}/logs/phase7_pair_launch_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "launch_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "steadysky_work=${ROOT}"
  echo "repo=${REPO}"
  echo "stage_epochs=${STAGE_EPOCHS_CSV}"
  echo "order=${ORDER}"
  echo "run_smoke=${RUN_SMOKE}"
  echo "dry_run=${DRY_RUN}"
  echo
  echo "commands:"
  for IDX in "${!ARMS[@]}"; do
    GPU="${IDX}"
    echo "  CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_phase1_training_schedule.sh ${ARMS[$IDX]} ${STAGE_EPOCHS_CSV}"
  done
} | tee "${MANIFEST}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry_run_complete manifest=${MANIFEST}"
  exit 0
fi

PIDS=()
for IDX in "${!ARMS[@]}"; do
  ARM="${ARMS[$IDX]}"
  GPU="${IDX}"
  LOG="${ROOT}/logs/phase7_${ARM}_launcher_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "starting_arm=${ARM} gpu=${GPU} log=${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash scripts/run_phase1_training_schedule.sh "${ARM}" "${STAGE_EPOCHS_CSV}" > "${LOG}" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "phase7_pair_failed manifest=${MANIFEST}" >&2
  exit 1
fi

echo "phase7_pair_complete manifest=${MANIFEST}"
