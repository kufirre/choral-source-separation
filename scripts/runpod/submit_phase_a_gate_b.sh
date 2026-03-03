#!/usr/bin/env bash
set -euo pipefail

# Gate B: Minimal over-separation test (M=10 -> V=4, reconstruction losses only).
# This is the first VCIN config that should be run after Gate A (Direct-4) passes.
# Uses partial backbone loading by default to bootstrap from a Direct-4 checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-phase-a-gate-b-$(date +%Y%m%d-%H%M%S)}"
export MODEL_TYPE="${MODEL_TYPE:-vcin}"
export CONFIG_PATH="${CONFIG_PATH:-configs/vcin/config_vcin_satb_phase_a_gate_b.yaml}"
# Gate B uses partial backbone loading: 4-output Direct-4 -> 10-output VCIN.
export BOOTSTRAP_LOAD_FLAGS="${BOOTSTRAP_LOAD_FLAGS---partial_backbone_load}"

"${SCRIPT_DIR}/submit_phase_a_vcin_satb.sh" "$@"
