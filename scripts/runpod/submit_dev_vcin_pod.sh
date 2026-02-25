#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

# Dev loop defaults prioritize availability and fast iteration over peak single-run throughput.
RUNPOD_CLOUD_TYPE="${DEV_RUNPOD_CLOUD_TYPE:-SECURE}"
RUNPOD_GPU_TYPE="${DEV_RUNPOD_GPU_TYPE:-NVIDIA A100 80GB PCIe}"

new_run_id() {
    printf 'dev-vcin-%s-%04d' "$(date +%Y%m%d-%H%M%S)" "$((RANDOM % 10000))"
}

if [[ "${FORCE_NEW_RUN_ID:-true}" == "true" ]]; then
    RUN_ID="$(new_run_id)"
    RUNPOD_POD_NAME="${RUN_ID}"
else
    RUN_ID="${RUN_ID:-$(new_run_id)}"
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUN_ID}}"
fi

RUNPOD_REPO_URL="${RUNPOD_REPO_URL:-https://github.com/kufirre/choral-source-separation.git}"
RUNPOD_GIT_REF="${RUNPOD_GIT_REF:-vcin-dev}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/choral-source-separation}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-processed/CSD_satb,processed/Cantoria_satb}"
RUNPOD_AUTO_TERMINATE_ON_EXIT="false"
ARTIFACT_SYNC_SECONDS="${ARTIFACT_SYNC_SECONDS:-300}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-40}"
RUNPOD_WAIT_READY="${RUNPOD_WAIT_READY:-true}"
RUNPOD_READY_TIMEOUT_SECONDS="${RUNPOD_READY_TIMEOUT_SECONDS:-1800}"
RUNPOD_READY_POLL_SECONDS="${RUNPOD_READY_POLL_SECONDS:-15}"

TRAIN_CMD_DEFAULT="set -euo pipefail; TMP_REPO=/tmp/choral-source-separation; rm -rf \"\$TMP_REPO\"; git clone --depth 1 --branch ${RUNPOD_GIT_REF} ${RUNPOD_REPO_URL} \"\$TMP_REPO\"; mkdir -p ${WORKSPACE_DIR}; cp -a \"\$TMP_REPO\"/. ${WORKSPACE_DIR}/; cd ${WORKSPACE_DIR}; bash scripts/runpod/bootstrap_train_env.sh || true; echo '[dev-pod] Ready. Repo synced at /workspace/choral-source-separation'; echo '[dev-pod] Run training manually, e.g. python train.py ...'; exec tail -f /dev/null"
TRAIN_CMD="${TRAIN_CMD:-${TRAIN_CMD_DEFAULT}}"

export RUN_ID
export RUNPOD_POD_NAME
export RUNPOD_AUTO_TERMINATE_ON_EXIT
export ARTIFACT_SYNC_SECONDS
export DATASET_GCS_PATHS
export RUNPOD_CONTAINER_DISK_GB
export RUNPOD_VOLUME_GB
export RUNPOD_CLOUD_TYPE
export RUNPOD_GPU_TYPE
export TRAIN_CMD
export RUN_PHASE="${RUN_PHASE:-dev}"
export RUN_TIER="${RUN_TIER:-T0}"
export RUNPOD_WAIT_READY
export RUNPOD_READY_TIMEOUT_SECONDS
export RUNPOD_READY_POLL_SECONDS

echo "[runpod-dev] Submitting long-lived dev pod RUN_ID=${RUN_ID}"
echo "[runpod-dev] RUNPOD_GIT_REF=${RUNPOD_GIT_REF}"
echo "[runpod-dev] TRAIN_CMD=${TRAIN_CMD}"

"${SCRIPT_DIR}/submit_runpod_pod.sh"
