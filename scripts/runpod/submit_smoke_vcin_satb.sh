#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-smoke-vcin-satb-$(date +%Y%m%d-%H%M%S)}"
export RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUN_ID}}"
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_smoke_test.yaml}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh"
