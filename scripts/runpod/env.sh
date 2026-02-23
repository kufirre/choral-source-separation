#!/usr/bin/env bash
# shellcheck shell=bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../vertex/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/../vertex/env.sh"
fi

export RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
export RUNPOD_DRY_RUN="${RUNPOD_DRY_RUN:-false}"

export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
export RUNPOD_INTERRUPTIBLE="${RUNPOD_INTERRUPTIBLE:-false}"
export RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA H100 80GB HBM3}"
export RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
export RUNPOD_POD_NAME_PREFIX="${RUNPOD_POD_NAME_PREFIX:-vcin}"
export RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUNPOD_POD_NAME_PREFIX}-$(date +%Y%m%d-%H%M%S)}"

export RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-100}"
export RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-100}"
export RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
export RUNPOD_SUPPORT_PUBLIC_IP="${RUNPOD_SUPPORT_PUBLIC_IP:-true}"
export RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp}"
export RUNPOD_DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-}"
export RUNPOD_ALLOWED_CUDA_VERSIONS="${RUNPOD_ALLOWED_CUDA_VERSIONS:-12.1,12.2,12.3,12.4}"
export RUNPOD_MIN_VCPU_PER_GPU="${RUNPOD_MIN_VCPU_PER_GPU:-}"
export RUNPOD_MIN_RAM_PER_GPU_GB="${RUNPOD_MIN_RAM_PER_GPU_GB:-}"
export RUNPOD_CONTAINER_REGISTRY_AUTH_ID="${RUNPOD_CONTAINER_REGISTRY_AUTH_ID:-}"

export IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${TAG}}"
export RUN_ID="${RUN_ID:-${RUNPOD_POD_NAME}}"

# Optional: pass a GCP service-account key to enable gcloud storage sync in non-Vertex runtimes.
export GCP_SA_KEY_B64="${GCP_SA_KEY_B64:-}"
export GCP_SA_KEY_JSON="${GCP_SA_KEY_JSON:-}"
