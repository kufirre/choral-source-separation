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

terminate_body_file="$(mktemp)"
http_code="$(curl -sS \
    -o "${terminate_body_file}" \
    -w "%{http_code}" \
    -X DELETE "${RUNPOD_API_BASE%/}/pods/${POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}")"

case "${http_code}" in
    200|202|204)
        echo "[runpod] Terminated pod ${POD_ID}"
        ;;
    404)
        echo "[runpod] Pod ${POD_ID} is already terminated (404)."
        ;;
    *)
        echo "[runpod] Failed to terminate pod ${POD_ID} (HTTP ${http_code})." >&2
        cat "${terminate_body_file}" >&2 || true
        rm -f "${terminate_body_file}"
        exit 1
        ;;
esac

rm -f "${terminate_body_file}"
