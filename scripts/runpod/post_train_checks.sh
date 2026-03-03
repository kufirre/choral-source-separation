#!/usr/bin/env bash
set -euo pipefail

TRAIN_LOG=""
RESULTS_PATH=""
MODE="${QUALITY_GATE_MODE:-warn}"
MIN_BEST_SDR="${QUALITY_GATE_MIN_BEST_SDR:-0.8}"
MAX_DROP="${QUALITY_GATE_MAX_DROP_FROM_BEST:-1.0}"
MIN_EVALS="${QUALITY_GATE_MIN_EVALS:-2}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-log)
            TRAIN_LOG="$2"
            shift 2
            ;;
        --results-path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --min-best-sdr)
            MIN_BEST_SDR="$2"
            shift 2
            ;;
        --max-drop-from-best)
            MAX_DROP="$2"
            shift 2
            ;;
        --min-evals)
            MIN_EVALS="$2"
            shift 2
            ;;
        *)
            echo "[post-train-checks] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${TRAIN_LOG}" || -z "${RESULTS_PATH}" ]]; then
    echo "[post-train-checks] --train-log and --results-path are required." >&2
    exit 1
fi

if [[ "${MODE}" == "none" ]]; then
    echo "[post-train-checks] Quality gate disabled (mode=none)."
    exit 0
fi

OUT_JSON="${RESULTS_PATH}/quality_gate.json"
echo "[post-train-checks] Running quality gate (mode=${MODE})"
echo "[post-train-checks] train_log=${TRAIN_LOG}"
echo "[post-train-checks] out_json=${OUT_JSON}"

set +e
python3 scripts/runpod/quality_gate.py \
    --train-log "${TRAIN_LOG}" \
    --out-json "${OUT_JSON}" \
    --min-best-sdr "${MIN_BEST_SDR}" \
    --max-drop-from-best "${MAX_DROP}" \
    --min-evals "${MIN_EVALS}"
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
    echo "[post-train-checks] Quality gate PASSED."
    exit 0
fi

if [[ "${MODE}" == "strict" ]]; then
    echo "[post-train-checks] Quality gate FAILED in strict mode." >&2
    exit "${status}"
fi

echo "[post-train-checks] Quality gate FAILED in warn mode; continuing." >&2
exit 0
