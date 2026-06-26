#!/usr/bin/env bash
set -euo pipefail

ROOT="${STEADYSKY_WORK:?Set STEADYSKY_WORK to the experiment working directory}"
REPO="${ROOT}/repos/steadysky-fourier"
STAGE_EPOCHS_CSV="${STEADYSKY_STAGE_EPOCHS:-10,15,20,25,35,45}"
ORDER="${STEADYSKY_PHASE1_ORDER:-raw,fourier}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" != "" ]]; then
  echo "usage: launch_phase1_pair.sh [--dry-run]" >&2
  exit 2
fi

IFS=',' read -r -a ARMS <<< "${ORDER}"
for ARM in "${ARMS[@]}"; do
  if [[ "${ARM}" != "raw" && "${ARM}" != "fourier" ]]; then
    echo "Invalid arm in STEADYSKY_PHASE1_ORDER: ${ARM}" >&2
    exit 3
  fi
done

cd "${REPO}"
bash scripts/check_phase1_readiness.sh

mkdir -p "${ROOT}/logs"
MANIFEST="${ROOT}/logs/phase1_pair_launch_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "launch_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "steadysky_work=${ROOT}"
  echo "repo=${REPO}"
  echo "stage_epochs=${STAGE_EPOCHS_CSV}"
  echo "order=${ORDER}"
  echo "dry_run=${DRY_RUN}"
  echo
  echo "commands:"
  for ARM in "${ARMS[@]}"; do
    echo "  bash scripts/run_phase1_training_schedule.sh ${ARM} ${STAGE_EPOCHS_CSV}"
  done
} | tee "${MANIFEST}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry_run_complete manifest=${MANIFEST}"
  exit 0
fi

for ARM in "${ARMS[@]}"; do
  echo "starting_arm=${ARM}"
  bash scripts/run_phase1_training_schedule.sh "${ARM}" "${STAGE_EPOCHS_CSV}"
done

echo "phase1_pair_complete manifest=${MANIFEST}"
