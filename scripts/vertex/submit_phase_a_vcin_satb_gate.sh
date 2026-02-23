#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-phase-a-vcin-gate-1e1000s-$(date +%Y%m%d-%H%M%S)}"
export JOB_NAME="${JOB_NAME:-${RUN_ID}}"
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a_gate_1e1000s.yaml}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh"
