#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

PROJECT_ID="${PROJECT_ID:-ai-training-484220}"
REGION="${REGION:-europe-west4}"
GAR_REPO="${GAR_REPO:-}"
IMAGE_NAME="${IMAGE_NAME:-choral-source-separation}"
TAG="${TAG:-latest}"
JOB_NAME="${JOB_NAME:-${IMAGE_NAME}-$(date +%Y%m%d-%H%M%S)}"

MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-1g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_H100_80GB}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
REPLICA_COUNT="${REPLICA_COUNT:-1}"
SCHEDULING_STRATEGY="${SCHEDULING_STRATEGY:-SPOT}"
SCHEDULING_MAX_WAIT_DURATION="${SCHEDULING_MAX_WAIT_DURATION:-}"

GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-}"
GCS_ARTIFACT_BUCKET="${GCS_ARTIFACT_BUCKET:-}"
RUN_ID="${RUN_ID:-${JOB_NAME}}"
TRAIN_CMD="${TRAIN_CMD:-}"

ENABLE_WEB_ACCESS="${ENABLE_WEB_ACCESS:-true}"
ENABLE_DASHBOARD_ACCESS="${ENABLE_DASHBOARD_ACCESS:-true}"

VERTEX_SA_EMAIL="${VERTEX_SA_EMAIL:-}"
NETWORK="${NETWORK:-}"
DATASET_GCS_PREFIX="${DATASET_GCS_PREFIX:-datasets}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-}"
DATA_ROOT="${DATA_ROOT:-/gcs_data}"
ARTIFACT_SYNC_SECONDS="${ARTIFACT_SYNC_SECONDS:-60}"
BOOTSTRAP_CKPT_URI="${BOOTSTRAP_CKPT_URI:-}"
DRY_RUN="${DRY_RUN:-false}"

yaml_quote() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

if [[ -z "${GAR_REPO}" ]]; then
    echo "GAR_REPO is required." >&2
    exit 1
fi

if [[ -z "${GCS_DATA_BUCKET}" || -z "${GCS_ARTIFACT_BUCKET}" ]]; then
    echo "GCS_DATA_BUCKET and GCS_ARTIFACT_BUCKET are required." >&2
    exit 1
fi

if [[ -z "${TRAIN_CMD}" ]]; then
    echo "TRAIN_CMD is required." >&2
    exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required." >&2
    exit 1
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" submit
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${TAG}"
CONFIG_FILE="$(mktemp -t choral_source_separation_vertex_XXXXXX.yaml)"
VERTEX_AI_JOB_YAML="$(yaml_quote "true")"
GCS_DATA_BUCKET_YAML="$(yaml_quote "${GCS_DATA_BUCKET}")"
GCS_ARTIFACT_BUCKET_YAML="$(yaml_quote "${GCS_ARTIFACT_BUCKET}")"
RUN_ID_YAML="$(yaml_quote "${RUN_ID}")"
TRAIN_CMD_YAML="$(yaml_quote "${TRAIN_CMD}")"
DATASET_GCS_PREFIX_YAML="$(yaml_quote "${DATASET_GCS_PREFIX}")"
DATASET_GCS_PATHS_YAML="$(yaml_quote "${DATASET_GCS_PATHS}")"
DATA_ROOT_YAML="$(yaml_quote "${DATA_ROOT}")"
ARTIFACT_SYNC_SECONDS_YAML="$(yaml_quote "${ARTIFACT_SYNC_SECONDS}")"
IMAGE_URI_YAML="$(yaml_quote "${IMAGE_URI}")"

cat > "${CONFIG_FILE}" <<EOF_CONFIG
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    replicaCount: ${REPLICA_COUNT}
    containerSpec:
      imageUri: ${IMAGE_URI}
      env:
        - name: VERTEX_AI_JOB
          value: ${VERTEX_AI_JOB_YAML}
        - name: GCS_DATA_BUCKET
          value: ${GCS_DATA_BUCKET_YAML}
        - name: GCS_ARTIFACT_BUCKET
          value: ${GCS_ARTIFACT_BUCKET_YAML}
        - name: RUN_ID
          value: ${RUN_ID_YAML}
        - name: TRAIN_CMD
          value: ${TRAIN_CMD_YAML}
        - name: IMAGE_URI
          value: ${IMAGE_URI_YAML}
        - name: DATASET_GCS_PREFIX
          value: ${DATASET_GCS_PREFIX_YAML}
        - name: DATASET_GCS_PATHS
          value: ${DATASET_GCS_PATHS_YAML}
        - name: DATA_ROOT
          value: ${DATA_ROOT_YAML}
        - name: ARTIFACT_SYNC_SECONDS
          value: ${ARTIFACT_SYNC_SECONDS_YAML}
EOF_CONFIG

append_env() {
    local name="$1"
    local value="$2"
    if [[ -n "${value}" ]]; then
        local value_yaml
        value_yaml="$(yaml_quote "${value}")"
        printf "        - name: %s\n          value: %s\n" "${name}" "${value_yaml}" >> "${CONFIG_FILE}"
    fi
}

append_env "HF_TOKEN" "${HF_TOKEN:-}"
append_env "USE_CHECKPOINT" "${USE_CHECKPOINT:-}"
append_env "BOOTSTRAP_CKPT_URI" "${BOOTSTRAP_CKPT_URI}"

if [[ -n "${SCHEDULING_STRATEGY}" ]]; then
    printf "scheduling:\n  strategy: %s\n" "${SCHEDULING_STRATEGY}" >> "${CONFIG_FILE}"
    if [[ -n "${SCHEDULING_MAX_WAIT_DURATION}" ]]; then
        printf "  maxWaitDuration: %s\n" "${SCHEDULING_MAX_WAIT_DURATION}" >> "${CONFIG_FILE}"
    fi
fi

if [[ -n "${VERTEX_SA_EMAIL}" ]]; then
    printf "serviceAccount: %s\n" "$(yaml_quote "${VERTEX_SA_EMAIL}")" >> "${CONFIG_FILE}"
fi

if [[ -n "${NETWORK}" ]]; then
    printf "network: %s\n" "$(yaml_quote "${NETWORK}")" >> "${CONFIG_FILE}"
fi

GCLOUD_EXTRA_FLAGS=()
if [[ "${ENABLE_WEB_ACCESS}" == "true" ]]; then
    GCLOUD_EXTRA_FLAGS+=(--enable-web-access)
fi
if [[ "${ENABLE_DASHBOARD_ACCESS}" == "true" ]]; then
    GCLOUD_EXTRA_FLAGS+=(--enable-dashboard-access)
fi

echo "Submitting Vertex custom job: ${JOB_NAME}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] Submission skipped. Generated config:"
    awk '
        /name: HF_TOKEN/ {
            print
            getline
            sub(/value:.*/, "value: \"***REDACTED***\"")
            print
            next
        }
        { print }
    ' "${CONFIG_FILE}"
    rm -f "${CONFIG_FILE}"
    exit 0
fi

gcloud ai custom-jobs create \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --display-name "${JOB_NAME}" \
    --config "${CONFIG_FILE}" \
    "${GCLOUD_EXTRA_FLAGS[@]}"

rm -f "${CONFIG_FILE}"
