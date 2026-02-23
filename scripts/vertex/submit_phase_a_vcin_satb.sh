#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

RUN_ID="${RUN_ID:-phase-a-vcin-satb-$(date +%Y%m%d-%H%M%S)}"
JOB_NAME="${JOB_NAME:-${RUN_ID}}"

MODEL_TYPE="${MODEL_TYPE:-vcin}"
CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a.yaml}"
DATASET_TYPE="${DATASET_TYPE:-4}"
TRAIN_DATA_PATHS="${TRAIN_DATA_PATHS:-/gcs_data/processed/CSD_satb}"
VALID_DATA_PATHS="${VALID_DATA_PATHS:-/gcs_data/processed/Cantoria_satb}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-processed/CSD_satb,processed/Cantoria_satb}"
RESULTS_PATH="${RESULTS_PATH:-artifacts/${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE_IDS="${DEVICE_IDS:-0}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
CHECKPOINT_LOAD_FLAGS="${CHECKPOINT_LOAD_FLAGS:---load_only_compatible_weights}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ "${START_CHECKPOINT}" == gs://* ]]; then
    export BOOTSTRAP_CKPT_URI="${BOOTSTRAP_CKPT_URI:-${START_CHECKPOINT}}"
    START_CHECKPOINT="/gcs_data/bootstrap/start_checkpoint.ckpt"
fi

CONFIG_PATH_CHECK="${CONFIG_PATH}"
if [[ "${CONFIG_PATH_CHECK}" != /* ]]; then
    CONFIG_PATH_CHECK="${PROJECT_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH_CHECK}" ]]; then
    echo "Config path does not exist: ${CONFIG_PATH_CHECK}" >&2
    exit 1
fi

TRAIN_CMD_DEFAULT="python train.py --model_type ${MODEL_TYPE} --config_path ${CONFIG_PATH} --results_path ${RESULTS_PATH} --dataset_type ${DATASET_TYPE} --data_path ${TRAIN_DATA_PATHS} --valid_path ${VALID_DATA_PATHS} --num_workers ${NUM_WORKERS} --device_ids ${DEVICE_IDS} ${EXTRA_TRAIN_ARGS}"
if [[ -n "${START_CHECKPOINT}" ]]; then
    TRAIN_CMD_DEFAULT="${TRAIN_CMD_DEFAULT} --start_check_point ${START_CHECKPOINT} ${CHECKPOINT_LOAD_FLAGS}"
fi
TRAIN_CMD="${TRAIN_CMD:-${TRAIN_CMD_DEFAULT}}"

export RUN_ID
export JOB_NAME
export TRAIN_CMD
export USE_CHECKPOINT
export DATASET_GCS_PATHS

"${SCRIPT_DIR}/validate_env.sh" submit

echo "Submitting RUN_ID=${RUN_ID}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "TRAIN_CMD=${TRAIN_CMD}"

"${PROJECT_ROOT}/scripts/vertex/submit_vertex_job.sh"
