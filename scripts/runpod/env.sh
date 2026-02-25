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

# Default to secure capacity for reliable iterative experimentation.
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
export RUNPOD_INTERRUPTIBLE="${RUNPOD_INTERRUPTIBLE:-false}"
export RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA A100-SXM4-80GB}"
export RUNPOD_GPU_CANDIDATES="${RUNPOD_GPU_CANDIDATES:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe}"
export RUNPOD_AUTO_PICK_GPU="${RUNPOD_AUTO_PICK_GPU:-true}"
export RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
export RUNPOD_POD_NAME_PREFIX="${RUNPOD_POD_NAME_PREFIX:-vcin}"
# Leave blank by default; submit wrappers generate a fresh unique value per launch.
export RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-}"

export RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
export RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-20}"
# Do not mount over /workspace; the image code lives at /workspace/choral-source-separation.
export RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
export RUNPOD_SUPPORT_PUBLIC_IP="${RUNPOD_SUPPORT_PUBLIC_IP:-true}"
export RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp}"
export RUNPOD_DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-}"
export RUNPOD_ALLOWED_CUDA_VERSIONS="${RUNPOD_ALLOWED_CUDA_VERSIONS:-}"
export RUNPOD_MIN_VCPU_PER_GPU="${RUNPOD_MIN_VCPU_PER_GPU:-}"
export RUNPOD_MIN_RAM_PER_GPU_GB="${RUNPOD_MIN_RAM_PER_GPU_GB:-}"
export RUNPOD_CONTAINER_REGISTRY_AUTH_ID="${RUNPOD_CONTAINER_REGISTRY_AUTH_ID:-}"
export RUNPOD_AUTO_TERMINATE_ON_EXIT="${RUNPOD_AUTO_TERMINATE_ON_EXIT:-true}"
export FORCE_NEW_RUN_ID="${FORCE_NEW_RUN_ID:-true}"
export RUNPOD_WAIT_READY="${RUNPOD_WAIT_READY:-true}"
# Secure cloud pods (especially H100/A100) regularly take >5 min to initialize runtime.
export RUNPOD_READY_TIMEOUT_SECONDS="${RUNPOD_READY_TIMEOUT_SECONDS:-1800}"
export RUNPOD_READY_POLL_SECONDS="${RUNPOD_READY_POLL_SECONDS:-15}"
export RUNPOD_READY_LOG_INTERVAL_SECONDS="${RUNPOD_READY_LOG_INTERVAL_SECONDS:-60}"

export IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${TAG}}"
# Leave blank by default; submit wrappers generate a fresh unique value per launch.
export RUN_ID="${RUN_ID:-}"

# Optional: pass a GCP service-account key to enable gcloud storage sync in non-Vertex runtimes.
export GCP_SA_KEY_B64="${GCP_SA_KEY_B64:-}"
export GCP_SA_KEY_JSON="${GCP_SA_KEY_JSON:-}"
# Runtime destination path inside container.
export GCP_SA_KEY_FILE="${GCP_SA_KEY_FILE:-/tmp/gcp-service-account.json}"
# Local source key path used by submit script to populate GCP_SA_KEY_B64.
export GCP_SA_KEY_SOURCE_FILE="${GCP_SA_KEY_SOURCE_FILE:-$HOME/.config/runpod/runpod-gar-pull-key.json}"
export SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-}"
