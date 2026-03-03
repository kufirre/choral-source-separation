#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-phase-a-direct4-$(date +%Y%m%d-%H%M%S)}"
export JOB_NAME="${JOB_NAME:-${RUN_ID}}"
export MODEL_TYPE="${MODEL_TYPE:-vcin_direct4}"
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_direct4_baseline.yaml}"
# Direct4 baseline checkpoints are fully compatible; do not partial-load by default.
export BOOTSTRAP_LOAD_FLAGS="${BOOTSTRAP_LOAD_FLAGS:-}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh" "$@"
