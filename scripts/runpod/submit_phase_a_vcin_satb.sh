#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

new_run_id() {
    printf 'phase-a-vcin-satb-%s-%04d' "$(date +%Y%m%d-%H%M%S)" "$((RANDOM % 10000))"
}

if [[ "${FORCE_NEW_RUN_ID:-true}" == "true" ]]; then
    RUN_ID="${RUN_ID:-$(new_run_id)}"
    RUNPOD_POD_NAME="${RUN_ID}"
else
    RUN_ID="${RUN_ID:-$(new_run_id)}"
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUN_ID}}"
fi

MODEL_TYPE="${MODEL_TYPE:-vcin}"
CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a_stable.yaml}"
DATASET_TYPE="${DATASET_TYPE:-4}"
TRAIN_DATA_PATHS="${TRAIN_DATA_PATHS:-/gcs_data/processed/CSD_satb /gcs_data/processed/ChoralSynth_satb /gcs_data/processed/jaCappella_satb}"
VALID_DATA_PATHS="${VALID_DATA_PATHS:-/gcs_data/processed/Cantoria_satb}"
DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-processed/CSD_satb,processed/ChoralSynth_satb,processed/jaCappella_satb,processed/Cantoria_satb}"
RESULTS_PATH="${RESULTS_PATH:-artifacts/${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DEVICE_IDS="${DEVICE_IDS:-0}"
ALLOW_BLEED_DATASETS="${ALLOW_BLEED_DATASETS:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
CHECKPOINT_LOAD_FLAGS="${CHECKPOINT_LOAD_FLAGS:---load_only_compatible_weights}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
BOOTSTRAP_CMD="${BOOTSTRAP_CMD:-bash scripts/runpod/bootstrap_train_env.sh}"
RUNPOD_REPO_URL="${RUNPOD_REPO_URL:-https://github.com/kufirre/choral-source-separation.git}"
RUNPOD_GIT_REF="${RUNPOD_GIT_REF:-vcin-dev}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/choral-source-separation}"
RUN_PHASE="${RUN_PHASE:-A}"
RUN_TIER="${RUN_TIER:-T3}"
QUALITY_GATE_MODE="${QUALITY_GATE_MODE:-warn}"
QUALITY_GATE_MIN_BEST_SDR="${QUALITY_GATE_MIN_BEST_SDR:-0.8}"
QUALITY_GATE_MAX_DROP_FROM_BEST="${QUALITY_GATE_MAX_DROP_FROM_BEST:-1.0}"
QUALITY_GATE_MIN_EVALS="${QUALITY_GATE_MIN_EVALS:-2}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-20}"
RUNPOD_RETRY_ON_STARTUP_TIMEOUT="${RUNPOD_RETRY_ON_STARTUP_TIMEOUT:-true}"
RUNPOD_FALLBACK_GPU_TYPE="${RUNPOD_FALLBACK_GPU_TYPE:-NVIDIA A100 80GB PCIe}"
RUNPOD_FALLBACK_CLOUD_TYPE="${RUNPOD_FALLBACK_CLOUD_TYPE:-SECURE}"

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

TRAIN_ARGS="--model_type ${MODEL_TYPE} --config_path ${CONFIG_PATH} --results_path ${RESULTS_PATH} --dataset_type ${DATASET_TYPE} --data_path ${TRAIN_DATA_PATHS} --valid_path ${VALID_DATA_PATHS} ${DATALOADER_ARGS} --device_ids ${DEVICE_IDS} ${EXTRA_TRAIN_ARGS}"
if [[ -n "${START_CHECKPOINT}" ]]; then
    TRAIN_ARGS="${TRAIN_ARGS} --start_check_point ${START_CHECKPOINT} ${CHECKPOINT_LOAD_FLAGS}"
fi

POST_TRAIN_CHECK_CMD="bash scripts/runpod/post_train_checks.sh --train-log ${RESULTS_PATH}/train.log --results-path ${RESULTS_PATH} --mode ${QUALITY_GATE_MODE} --min-best-sdr ${QUALITY_GATE_MIN_BEST_SDR} --max-drop-from-best ${QUALITY_GATE_MAX_DROP_FROM_BEST} --min-evals ${QUALITY_GATE_MIN_EVALS}"

TRAIN_CMD_DEFAULT="set -euo pipefail; cleanup(){ code=\$?; if [[ \"\${RUNPOD_AUTO_TERMINATE_ON_EXIT:-false}\" == \"true\" ]]; then bash scripts/runpod/terminate_self.sh || true; fi; exit \$code; }; trap cleanup EXIT; TMP_REPO=/tmp/choral-source-separation; rm -rf \"\$TMP_REPO\"; git clone --depth 1 --branch ${RUNPOD_GIT_REF} ${RUNPOD_REPO_URL} \"\$TMP_REPO\"; mkdir -p ${WORKSPACE_DIR}; cp -a \"\$TMP_REPO\"/. ${WORKSPACE_DIR}/; cd ${WORKSPACE_DIR}; ${BOOTSTRAP_CMD}; python train.py ${TRAIN_ARGS} 2>&1 | tee -a ${RESULTS_PATH}/train.log; ${POST_TRAIN_CHECK_CMD}"
TRAIN_CMD="${TRAIN_CMD:-${TRAIN_CMD_DEFAULT}}"

export RUN_ID
export RUNPOD_POD_NAME
export TRAIN_CMD
export DATASET_GCS_PATHS
export USE_CHECKPOINT
export CONFIG_PATH
export TRAIN_DATA_PATHS
export VALID_DATA_PATHS
export RESULTS_PATH
export START_CHECKPOINT
export RUNPOD_REPO_URL
export RUNPOD_GIT_REF
export RUN_PHASE
export RUN_TIER
export QUALITY_GATE_MODE
export QUALITY_GATE_MIN_BEST_SDR
export QUALITY_GATE_MAX_DROP_FROM_BEST
export QUALITY_GATE_MIN_EVALS
export RUNPOD_VOLUME_GB

echo "[runpod] Submitting pod RUN_ID=${RUN_ID}"
echo "[runpod] MODEL_TYPE=${MODEL_TYPE}"
echo "[runpod] CONFIG_PATH=${CONFIG_PATH}"
echo "[runpod] RUN_PHASE=${RUN_PHASE}"
echo "[runpod] RUN_TIER=${RUN_TIER}"
echo "[runpod] QUALITY_GATE_MODE=${QUALITY_GATE_MODE}"
echo "[runpod] RUNPOD_VOLUME_GB=${RUNPOD_VOLUME_GB}"
echo "[runpod] TRAIN_CMD=${TRAIN_CMD}"

submit_with_optional_retry() {
    if "${SCRIPT_DIR}/submit_runpod_pod.sh"; then
        return 0
    fi

    if [[ "${RUNPOD_RETRY_ON_STARTUP_TIMEOUT}" != "true" ]]; then
        return 1
    fi

    echo "[runpod] Initial submit/startup failed. Retrying once with fallback runtime profile..."
    echo "[runpod] Fallback cloud=${RUNPOD_FALLBACK_CLOUD_TYPE}, gpu=${RUNPOD_FALLBACK_GPU_TYPE}"
    export RUNPOD_CLOUD_TYPE="${RUNPOD_FALLBACK_CLOUD_TYPE}"
    export RUNPOD_GPU_TYPE="${RUNPOD_FALLBACK_GPU_TYPE}"
    "${SCRIPT_DIR}/submit_runpod_pod.sh"
}

submit_with_optional_retry
