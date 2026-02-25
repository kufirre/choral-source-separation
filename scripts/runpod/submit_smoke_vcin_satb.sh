#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

new_run_id() {
    printf 'smoke-vcin-satb-%s-%04d' "$(date +%Y%m%d-%H%M%S)" "$((RANDOM % 10000))"
}

if [[ "${FORCE_NEW_RUN_ID:-true}" == "true" ]]; then
    RUN_ID="$(new_run_id)"
    RUNPOD_POD_NAME="${RUN_ID}"
else
    RUN_ID="${RUN_ID:-$(new_run_id)}"
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUN_ID}}"
fi
export RUN_ID
export RUNPOD_POD_NAME
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_smoke_test.yaml}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
export RUN_PHASE="${RUN_PHASE:-smoke}"
export RUN_TIER="${RUN_TIER:-T1}"
export QUALITY_GATE_MODE="${QUALITY_GATE_MODE:-none}"
export TRAIN_DATA_PATHS="${TRAIN_DATA_PATHS:-/gcs_data/processed/CSD_satb}"
export VALID_DATA_PATHS="${VALID_DATA_PATHS:-/gcs_data/processed/Cantoria_satb}"
export DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-processed/CSD_satb,processed/Cantoria_satb}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh"
