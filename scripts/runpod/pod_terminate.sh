#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" terminate
fi

RUNPOD_DIR="${PROJECT_ROOT}/.runpod"
POD_ID="${1:-${POD_ID:-}}"
if [[ -z "${POD_ID}" && -f "${RUNPOD_DIR}/last_pod_id" ]]; then
    POD_ID="$(cat "${RUNPOD_DIR}/last_pod_id")"
fi

if [[ -z "${POD_ID}" ]]; then
    echo "Usage: $0 <pod_id>" >&2
    exit 1
fi

curl -sS --fail-with-body \
    -X DELETE "${RUNPOD_API_BASE%/}/pods/${POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null

echo "[runpod] Terminated pod ${POD_ID}"
