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

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh"
