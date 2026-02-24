#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)}"
TRAIN_CMD="${TRAIN_CMD:-}"
DATA_ROOT="${DATA_ROOT:-/gcs_data}"
DATASET_GCS_PREFIX="${DATASET_GCS_PREFIX:-datasets}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-artifacts/${RUN_ID}}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
BOOTSTRAP_CKPT_URI="${BOOTSTRAP_CKPT_URI:-}"
BOOTSTRAP_CKPT_LOCAL_PATH="${BOOTSTRAP_CKPT_LOCAL_PATH:-${DATA_ROOT}/bootstrap/start_checkpoint.ckpt}"
ARTIFACT_SYNC_SECONDS="${ARTIFACT_SYNC_SECONDS:-60}"
GCP_SA_KEY_B64="${GCP_SA_KEY_B64:-}"
GCP_SA_KEY_JSON="${GCP_SA_KEY_JSON:-}"
GCP_SA_KEY_FILE="${GCP_SA_KEY_FILE:-/tmp/gcp-service-account.json}"
RUNPOD_AUTO_TERMINATE_ON_EXIT="${RUNPOD_AUTO_TERMINATE_ON_EXIT:-false}"
RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-}"
export GCP_SA_KEY_FILE
GCLOUD_AUTH_ACTIVE="false"

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

sync_dataset_path() {
    local raw_path="$1"
    local dataset_path
    local bucket_path
    local local_path
    local source_uri
    local target_dir

    dataset_path="$(trim "${raw_path}")"
    if [[ -z "${dataset_path}" ]]; then
        return 0
    fi
    dataset_path="${dataset_path#/}"

    if [[ "${dataset_path}" == "${DATASET_GCS_PREFIX}/"* ]]; then
        bucket_path="${dataset_path}"
        local_path="${dataset_path#${DATASET_GCS_PREFIX}/}"
    else
        bucket_path="${DATASET_GCS_PREFIX}/${dataset_path}"
        local_path="${dataset_path}"
    fi

    if [[ -z "${local_path}" ]]; then
        echo "[entrypoint] Skipping invalid DATASET_GCS_PATHS item: ${raw_path}" >&2
        return 0
    fi

    source_uri="gs://${GCS_DATA_BUCKET}/${bucket_path}"
    target_dir="${DATA_ROOT}/${local_path}"
    echo "[entrypoint] Downloading dataset path ${source_uri} -> ${target_dir}"
    if gcloud storage ls "${source_uri}" >/dev/null 2>&1; then
        mkdir -p "${target_dir}"
        gcloud storage rsync -r "${source_uri}" "${target_dir}"
    else
        echo "[entrypoint] Dataset path not found (${source_uri}); skipping download."
    fi
}

sync_artifacts_to_gcs() {
    if [[ "${GCLOUD_AUTH_ACTIVE}" != "true" ]]; then
        return
    fi
    if [[ -z "${GCS_ARTIFACT_BUCKET:-}" ]]; then
        return
    fi
    gcloud storage rsync -r "${LOCAL_ARTIFACT_DIR}" \
        "gs://${GCS_ARTIFACT_BUCKET}/artifacts/${RUN_ID}" >/dev/null 2>&1 || true
}

background_sync_loop() {
    while true; do
        sleep "${ARTIFACT_SYNC_SECONDS}"
        sync_artifacts_to_gcs
    done
}

terminate_runpod_pod() {
    if [[ "${RUNPOD_AUTO_TERMINATE_ON_EXIT}" != "true" ]]; then
        return 0
    fi
    if [[ -z "${RUNPOD_API_KEY}" || -z "${RUNPOD_POD_NAME}" ]]; then
        echo "[entrypoint] RUNPOD_AUTO_TERMINATE_ON_EXIT=true but RUNPOD_API_KEY/RUNPOD_POD_NAME is missing; skipping pod termination."
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "[entrypoint] curl not found; skipping pod termination."
        return 0
    fi

    local pods_json pod_id
    pods_json="$(curl -sS --fail-with-body \
        -X GET "${RUNPOD_API_BASE%/}/pods" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" 2>/dev/null || true)"
    if [[ -z "${pods_json}" ]]; then
        echo "[entrypoint] Could not query RunPod API for pod termination."
        return 0
    fi

    pod_id="$(RUNPOD_PODS_JSON="${pods_json}" RUNPOD_POD_NAME="${RUNPOD_POD_NAME}" python3 - <<'PY'
import json
import os

pod_name = os.getenv("RUNPOD_POD_NAME", "")
pod_id = ""
try:
    pods = json.loads(os.getenv("RUNPOD_PODS_JSON", "[]"))
except json.JSONDecodeError:
    pods = []

for pod in pods:
    if isinstance(pod, dict) and pod.get("name") == pod_name:
        pod_id = str(pod.get("id", ""))
        break

print(pod_id)
PY
)"
    if [[ -z "${pod_id}" ]]; then
        echo "[entrypoint] Pod ${RUNPOD_POD_NAME} not found in RunPod API response; skipping pod termination."
        return 0
    fi

    if curl -sS --fail-with-body \
        -X DELETE "${RUNPOD_API_BASE%/}/pods/${pod_id}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null 2>&1; then
        echo "[entrypoint] Requested RunPod termination for pod ${pod_id} (${RUNPOD_POD_NAME})."
    else
        echo "[entrypoint] Failed to terminate RunPod pod ${pod_id} (${RUNPOD_POD_NAME})."
    fi
}

cleanup_on_exit() {
    local exit_code=$?
    if [[ -n "${SYNC_PID:-}" ]]; then
        kill "${SYNC_PID}" 2>/dev/null || true
    fi
    sync_artifacts_to_gcs
    terminate_runpod_pod || true
    exit "${exit_code}"
}
trap cleanup_on_exit EXIT

activate_gcp_service_account() {
    local active_account

    if ! command -v gcloud >/dev/null 2>&1; then
        echo "[entrypoint] gcloud not found; skipping GCS sync."
        return
    fi

    if [[ -n "${GCP_SA_KEY_B64}" ]]; then
        python3 - <<'PY'
import base64
import os
key_file = os.getenv("GCP_SA_KEY_FILE", "/tmp/gcp-service-account.json")
with open(key_file, "wb") as f:
    f.write(base64.b64decode(os.environ["GCP_SA_KEY_B64"]))
PY
    elif [[ -n "${GCP_SA_KEY_JSON}" ]]; then
        printf '%s\n' "${GCP_SA_KEY_JSON}" > "${GCP_SA_KEY_FILE}"
    fi

    if [[ -s "${GCP_SA_KEY_FILE}" ]]; then
        gcloud auth activate-service-account --key-file="${GCP_SA_KEY_FILE}" >/dev/null
        export GOOGLE_APPLICATION_CREDENTIALS="${GCP_SA_KEY_FILE}"
        echo "[entrypoint] Activated GCP service account credentials."
    fi

    active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
    if [[ -n "${active_account}" ]]; then
        GCLOUD_AUTH_ACTIVE="true"
        echo "[entrypoint] Active gcloud account: ${active_account}"
    else
        echo "[entrypoint] No active gcloud account; skipping GCS sync operations."
    fi
}

mkdir -p "${LOCAL_ARTIFACT_DIR}"
mkdir -p "${LOCAL_ARTIFACT_DIR}/tb"
mkdir -p "${DATA_ROOT}"
mkdir -p data

activate_gcp_service_account

if [[ -n "${GCS_DATA_BUCKET:-}" ]]; then
    if [[ "${GCLOUD_AUTH_ACTIVE}" != "true" ]]; then
        echo "[entrypoint] GCS_DATA_BUCKET set but no gcloud auth; skipping dataset download."
    else
        if [[ -n "${DATASET_GCS_PATHS}" ]]; then
            echo "[entrypoint] Downloading scoped datasets: ${DATASET_GCS_PATHS}"
            IFS=',' read -r -a dataset_paths <<< "${DATASET_GCS_PATHS}"
            for dataset_path in "${dataset_paths[@]}"; do
                sync_dataset_path "${dataset_path}"
            done
        else
            DATASET_URI="gs://${GCS_DATA_BUCKET}/${DATASET_GCS_PREFIX}"
            echo "[entrypoint] DATASET_GCS_PATHS is empty; downloading full prefix ${DATASET_URI}"
            if gcloud storage ls "${DATASET_URI}" >/dev/null 2>&1; then
                gcloud storage rsync -r "${DATASET_URI}" "${DATA_ROOT}"
            else
                echo "[entrypoint] Dataset path not found (${DATASET_URI}); skipping download."
            fi
        fi
    fi
fi

if [[ "${GCLOUD_AUTH_ACTIVE}" == "true" && "${USE_CHECKPOINT}" == "true" && -n "${BOOTSTRAP_CKPT_URI}" ]]; then
    mkdir -p "$(dirname "${BOOTSTRAP_CKPT_LOCAL_PATH}")"
    echo "[entrypoint] Downloading bootstrap checkpoint from ${BOOTSTRAP_CKPT_URI}"
    if gcloud storage cp "${BOOTSTRAP_CKPT_URI}" "${BOOTSTRAP_CKPT_LOCAL_PATH}" >/dev/null 2>&1; then
        export START_CHECKPOINT="${START_CHECKPOINT:-${BOOTSTRAP_CKPT_LOCAL_PATH}}"
        echo "[entrypoint] Bootstrap checkpoint available at ${BOOTSTRAP_CKPT_LOCAL_PATH}"
    else
        echo "[entrypoint] Bootstrap checkpoint download failed; continuing without it."
    fi
fi

if [[ "${GCLOUD_AUTH_ACTIVE}" == "true" && -n "${GCS_ARTIFACT_BUCKET:-}" ]]; then
    echo "[entrypoint] Restoring artifacts from gs://${GCS_ARTIFACT_BUCKET}/artifacts/${RUN_ID}"
    gcloud storage rsync -r \
        "gs://${GCS_ARTIFACT_BUCKET}/artifacts/${RUN_ID}" \
        "${LOCAL_ARTIFACT_DIR}" || true
fi

if [[ "${GCLOUD_AUTH_ACTIVE}" == "true" && -n "${GCS_ARTIFACT_BUCKET:-}" ]]; then
    background_sync_loop &
    SYNC_PID=$!
    echo "[entrypoint] Background sync started (PID ${SYNC_PID})"
else
    echo "[entrypoint] Background sync disabled (missing gcloud auth or GCS_ARTIFACT_BUCKET)."
fi
echo "[entrypoint] RUN_ID=${RUN_ID}"

if [[ -x "scripts/vertex/capture_run_metadata.sh" ]]; then
    scripts/vertex/capture_run_metadata.sh "${LOCAL_ARTIFACT_DIR}/meta" || true
fi

if [[ -n "${TRAIN_CMD}" ]]; then
    echo "[entrypoint] Running TRAIN_CMD: ${TRAIN_CMD}"
    bash -lc "${TRAIN_CMD}"
else
    echo "[entrypoint] TRAIN_CMD is required."
    exit 1
fi
