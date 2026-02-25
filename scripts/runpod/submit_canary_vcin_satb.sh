#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

new_run_id() {
    printf 'canary-vcin-satb-%s-%04d' "$(date +%Y%m%d-%H%M%S)" "$((RANDOM % 10000))"
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
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a_diag_core_2e50s.yaml}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
export RUN_PHASE="${RUN_PHASE:-canary}"
export RUN_TIER="${RUN_TIER:-T2}"
export QUALITY_GATE_MODE="${QUALITY_GATE_MODE:-strict}"
export QUALITY_GATE_MIN_BEST_SDR="${QUALITY_GATE_MIN_BEST_SDR:-0.7}"
export QUALITY_GATE_MAX_DROP_FROM_BEST="${QUALITY_GATE_MAX_DROP_FROM_BEST:-0.8}"
export QUALITY_GATE_MIN_EVALS="${QUALITY_GATE_MIN_EVALS:-2}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh"
