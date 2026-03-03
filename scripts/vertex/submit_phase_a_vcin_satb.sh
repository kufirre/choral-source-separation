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
CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a_staged.yaml}"
DATASET_TYPE="${DATASET_TYPE:-4}"
TRAIN_DATA_PATHS="${TRAIN_DATA_PATHS:-/gcs_data/processed/CSD_satb /gcs_data/processed/ChoralSynth_satb /gcs_data/processed/jaCappella_satb}"
VALID_DATA_PATHS="${VALID_DATA_PATHS:-/gcs_data/processed/Cantoria_satb}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-processed/CSD_satb,processed/ChoralSynth_satb,processed/jaCappella_satb,processed/Cantoria_satb}"
RESULTS_PATH="${RESULTS_PATH:-artifacts/${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DEVICE_IDS="${DEVICE_IDS:-0}"
ALLOW_BLEED_DATASETS="${ALLOW_BLEED_DATASETS:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
RESUME_MODE="${RESUME_MODE:-bootstrap}" # bootstrap|resume
BOOTSTRAP_LOAD_FLAGS="${BOOTSTRAP_LOAD_FLAGS---partial_backbone_load}"
RESUME_LOAD_FLAGS="${RESUME_LOAD_FLAGS:---load_optimizer --load_scheduler --load_epoch --load_best_metric --load_all_metrics --load_all_losses}"

if [[ "${RESUME_MODE}" != "bootstrap" && "${RESUME_MODE}" != "resume" ]]; then
    echo "Invalid RESUME_MODE='${RESUME_MODE}'. Expected 'bootstrap' or 'resume'." >&2
    exit 1
fi

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

assert_phase_a_dataset_policy() {
    if [[ "${ALLOW_BLEED_DATASETS}" == "true" ]]; then
        return 0
    fi

    local joined lowered
    joined="${TRAIN_DATA_PATHS} ${VALID_DATA_PATHS} ${DATASET_GCS_PATHS}"
    lowered="$(printf '%s' "${joined}" | tr '[:upper:]' '[:lower:]')"

    if [[ "${lowered}" == *"dagstuhl"* || "${lowered}" == *"esmuc"* || "${lowered}" == *"cmuc"* ]]; then
        echo "Phase A dataset policy violation: bleed-heavy datasets detected in dataset paths." >&2
        echo "Blocked tokens: dagstuhl, esmuc, cmuc" >&2
        echo "Set ALLOW_BLEED_DATASETS=true only if this is an intentional override." >&2
        exit 1
    fi
}

assert_phase_a_dataset_policy

DATALOADER_ARGS="--num_workers ${NUM_WORKERS}"
if [[ "${PIN_MEMORY}" == "true" ]]; then
    DATALOADER_ARGS="${DATALOADER_ARGS} --pin_memory"
fi
if [[ "${PERSISTENT_WORKERS}" == "true" && "${NUM_WORKERS}" != "0" ]]; then
    DATALOADER_ARGS="${DATALOADER_ARGS} --persistent_workers"
fi
if [[ -n "${PREFETCH_FACTOR}" && "${NUM_WORKERS}" != "0" ]]; then
    DATALOADER_ARGS="${DATALOADER_ARGS} --prefetch_factor ${PREFETCH_FACTOR}"
fi

TRAIN_CMD_DEFAULT="python train.py --model_type ${MODEL_TYPE} --config_path ${CONFIG_PATH} --results_path ${RESULTS_PATH} --dataset_type ${DATASET_TYPE} --data_path ${TRAIN_DATA_PATHS} --valid_path ${VALID_DATA_PATHS} ${DATALOADER_ARGS} --device_ids ${DEVICE_IDS} ${EXTRA_TRAIN_ARGS}"
if [[ -n "${START_CHECKPOINT}" ]]; then
    TRAIN_CMD_DEFAULT="${TRAIN_CMD_DEFAULT} --start_check_point ${START_CHECKPOINT}"
    if [[ "${RESUME_MODE}" == "resume" ]]; then
        TRAIN_CMD_DEFAULT="${TRAIN_CMD_DEFAULT} ${RESUME_LOAD_FLAGS}"
    elif [[ -n "${BOOTSTRAP_LOAD_FLAGS}" ]]; then
        TRAIN_CMD_DEFAULT="${TRAIN_CMD_DEFAULT} ${BOOTSTRAP_LOAD_FLAGS}"
    fi
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
