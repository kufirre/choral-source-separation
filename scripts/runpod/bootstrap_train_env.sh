#!/usr/bin/env bash
set -euo pipefail

REQUIRE_GCLOUD_AUTH="${REQUIRE_GCLOUD_AUTH:-true}"

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

sync_dataset_path() {
    local raw_path="$1"
    local prefix dataset_path bucket_path local_path source_uri target_dir

    dataset_path="$(trim "${raw_path}")"
    if [[ -z "${dataset_path}" ]]; then
        return 0
    fi
    dataset_path="${dataset_path#/}"

    prefix="${DATASET_GCS_PREFIX:-datasets}"
    prefix="${prefix#/}"
    prefix="${prefix%/}"

    if [[ -n "${prefix}" ]]; then
        if [[ "${dataset_path}" == "${prefix}/"* ]]; then
            bucket_path="${dataset_path}"
            local_path="${dataset_path#${prefix}/}"
        else
            bucket_path="${prefix}/${dataset_path}"
            local_path="${dataset_path}"
        fi
    else
        bucket_path="${dataset_path}"
        local_path="${dataset_path}"
    fi

    source_uri="gs://${GCS_DATA_BUCKET}/${bucket_path}"
    target_dir="${DATA_ROOT:-/gcs_data}/${local_path}"

    echo "[runpod-bootstrap] Downloading ${source_uri} -> ${target_dir}"
    if ! gcloud storage ls "${source_uri}" >/dev/null 2>&1; then
        echo "[runpod-bootstrap] Dataset path not found (${source_uri}); skipping."
        return 0
    fi
    mkdir -p "${target_dir}"
    gcloud storage rsync -r "${source_uri}" "${target_dir}"
}

activate_gcp_service_account() {
    local key_file
    local active_account
    key_file="${GCP_SA_KEY_FILE:-/tmp/gcp-service-account.json}"

    if ! command -v gcloud >/dev/null 2>&1; then
        echo "[runpod-bootstrap] gcloud not found; cannot perform dataset/checkpoint sync." >&2
        return 1
    fi

    if [[ -n "${GCP_SA_KEY_B64:-}" ]]; then
        GCP_SA_KEY_FILE="${key_file}" python3 - <<'PY'
import base64
import os

key_file = os.getenv("GCP_SA_KEY_FILE", "/tmp/gcp-service-account.json")
with open(key_file, "wb") as f:
    f.write(base64.b64decode(os.environ["GCP_SA_KEY_B64"]))
PY
    elif [[ -n "${GCP_SA_KEY_JSON:-}" ]]; then
        printf '%s\n' "${GCP_SA_KEY_JSON}" > "${key_file}"
    fi

    if [[ -s "${key_file}" ]]; then
        gcloud auth activate-service-account --key-file="${key_file}" >/dev/null
        export GOOGLE_APPLICATION_CREDENTIALS="${key_file}"
        active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
        if [[ -n "${active_account}" ]]; then
            echo "[runpod-bootstrap] Activated GCP service account: ${active_account}"
            return 0
        fi
    fi

    echo "[runpod-bootstrap] No active gcloud account after auth setup." >&2
    return 1
}

download_bootstrap_checkpoint() {
    local checkpoint_path
    checkpoint_path="${BOOTSTRAP_CKPT_LOCAL_PATH:-${DATA_ROOT:-/gcs_data}/bootstrap/start_checkpoint.ckpt}"

    if [[ "${USE_CHECKPOINT:-false}" != "true" ]]; then
        return 0
    fi
    if [[ -z "${BOOTSTRAP_CKPT_URI:-}" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${checkpoint_path}")"
    echo "[runpod-bootstrap] Downloading checkpoint ${BOOTSTRAP_CKPT_URI} -> ${checkpoint_path}"
    if ! gcloud storage ls "${BOOTSTRAP_CKPT_URI}" >/dev/null 2>&1; then
        echo "[runpod-bootstrap] Checkpoint not found (${BOOTSTRAP_CKPT_URI}); skipping."
        return 0
    fi
    gcloud storage cp "${BOOTSTRAP_CKPT_URI}" "${checkpoint_path}"
}

main() {
    if ! activate_gcp_service_account; then
        if [[ "${REQUIRE_GCLOUD_AUTH}" == "true" ]]; then
            echo "[runpod-bootstrap] GCP auth is required but unavailable." >&2
            exit 1
        fi
        echo "[runpod-bootstrap] Continuing without gcloud sync because REQUIRE_GCLOUD_AUTH=false."
        return 0
    fi

    if [[ -n "${GCS_DATA_BUCKET:-}" && -n "${DATASET_GCS_PATHS:-}" ]]; then
        IFS=',' read -r -a dataset_paths <<< "${DATASET_GCS_PATHS}"
        for dataset_path in "${dataset_paths[@]}"; do
            sync_dataset_path "${dataset_path}"
        done
    fi

    download_bootstrap_checkpoint
}

main "$@"
