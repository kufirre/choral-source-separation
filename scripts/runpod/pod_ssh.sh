#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STATUS_OUTPUT="$("${SCRIPT_DIR}/pod_status.sh" "${1:-}")"
echo "${STATUS_OUTPUT}"

HOST="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F= '/^PUBLIC_IP=/{print $2}')"
PORT="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F= '/^SSH_PORT=/{print $2}')"

if [[ -z "${HOST}" || -z "${PORT}" ]]; then
    echo "[runpod] Could not determine SSH host/port." >&2
    exit 1
fi

SSH_CMD="ssh -o StrictHostKeyChecking=no -p ${PORT} root@${HOST}"
echo "SSH_CMD=${SSH_CMD}"

if [[ "${RUNPOD_SSH_EXECUTE:-false}" == "true" ]]; then
    exec ssh -o StrictHostKeyChecking=no -p "${PORT}" "root@${HOST}"
fi
