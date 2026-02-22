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

cleanup_on_exit() {
    local exit_code=$?
    if [[ -n "${SYNC_PID:-}" ]]; then
        kill "${SYNC_PID}" 2>/dev/null || true
    fi
    sync_artifacts_to_gcs
    exit "${exit_code}"
}
trap cleanup_on_exit EXIT

mkdir -p "${LOCAL_ARTIFACT_DIR}"
mkdir -p "${LOCAL_ARTIFACT_DIR}/tb"
mkdir -p "${DATA_ROOT}"
mkdir -p data

if [[ -n "${GCS_DATA_BUCKET:-}" ]]; then
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

if [[ "${USE_CHECKPOINT}" == "true" && -n "${BOOTSTRAP_CKPT_URI}" ]]; then
    mkdir -p "$(dirname "${BOOTSTRAP_CKPT_LOCAL_PATH}")"
    echo "[entrypoint] Downloading bootstrap checkpoint from ${BOOTSTRAP_CKPT_URI}"
    if gcloud storage cp "${BOOTSTRAP_CKPT_URI}" "${BOOTSTRAP_CKPT_LOCAL_PATH}" >/dev/null 2>&1; then
        export START_CHECKPOINT="${START_CHECKPOINT:-${BOOTSTRAP_CKPT_LOCAL_PATH}}"
        echo "[entrypoint] Bootstrap checkpoint available at ${BOOTSTRAP_CKPT_LOCAL_PATH}"
    else
        echo "[entrypoint] Bootstrap checkpoint download failed; continuing without it."
    fi
fi

if [[ -n "${GCS_ARTIFACT_BUCKET:-}" ]]; then
    echo "[entrypoint] Restoring artifacts from gs://${GCS_ARTIFACT_BUCKET}/artifacts/${RUN_ID}"
    gcloud storage rsync -r \
        "gs://${GCS_ARTIFACT_BUCKET}/artifacts/${RUN_ID}" \
        "${LOCAL_ARTIFACT_DIR}" || true
fi

background_sync_loop &
SYNC_PID=$!
echo "[entrypoint] Background sync started (PID ${SYNC_PID})"
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
