#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

# Values are kept for run labeling/experiment tracking unless EXTRA_TRAIN_ARGS
# consumes them explicitly.
LAMBDA_VALUES=(${LAMBDA_VALUES:-0.000 0.001 0.002})
RUN_PREFIX="${RUN_PREFIX:-phase-b-vcin-satb-mixit}"
BASE_EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SWEEP_ARG_TEMPLATE="${SWEEP_ARG_TEMPLATE:-}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-true}"

if [[ -z "${START_CHECKPOINT:-}" && -z "${BOOTSTRAP_CKPT_URI:-}" ]]; then
    echo "Set START_CHECKPOINT or BOOTSTRAP_CKPT_URI before running the sweep." >&2
    exit 1
fi

if [[ -z "${SWEEP_ARG_TEMPLATE}" && "${BASE_EXTRA_TRAIN_ARGS}" != *"{value}"* ]]; then
    cat >&2 <<'EOF'
Sweep is not configured with any varying train argument.
Set one of:
  1) SWEEP_ARG_TEMPLATE='--your_flag {value}'
  2) EXTRA_TRAIN_ARGS containing '{value}' placeholder.
EOF
    exit 1
fi

index=0
for lambda in "${LAMBDA_VALUES[@]}"; do
    index=$((index + 1))
    lambda_tag="$(printf '%s' "${lambda}" | tr -d '.')"
    run_id="${RUN_PREFIX}-l${lambda_tag}-$(date +%Y%m%d-%H%M%S)-${index}"
    sweep_extra_args="${BASE_EXTRA_TRAIN_ARGS}"

    if [[ -n "${SWEEP_ARG_TEMPLATE}" ]]; then
        rendered_arg="${SWEEP_ARG_TEMPLATE//\{value\}/${lambda}}"
        sweep_extra_args="${sweep_extra_args} ${rendered_arg}"
    else
        sweep_extra_args="${sweep_extra_args//\{value\}/${lambda}}"
    fi

    echo "Submitting sweep run ${run_id} with EXTRA_TRAIN_ARGS=${sweep_extra_args}"

    RUN_ID="${run_id}" \
    JOB_NAME="${run_id}" \
    PHASE_B_MAGNITUDE_PENALTY_COEF="${lambda}" \
    EXTRA_TRAIN_ARGS="${sweep_extra_args}" \
    "${PROJECT_ROOT}/scripts/vertex/submit_phase_b_vcin_satb_mixit.sh"
done
