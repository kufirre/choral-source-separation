#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STATUS_OUTPUT="$("${SCRIPT_DIR}/pod_status.sh" "${1:-}")"
echo "${STATUS_OUTPUT}"

HOST="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F= '/^PUBLIC_IP=/{print $2}')"
PORT="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F= '/^SSH_PORT=/{print $2}')"
HOST_ID="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F= '/^MACHINE_HOST_ID=/{print $2}')"
SSH_KEY_PATH="${RUNPOD_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
RUNPOD_PREFER_PROXY_SSH="${RUNPOD_PREFER_PROXY_SSH:-false}"
SSH_KEY_ARG=()
if [[ -n "${SSH_KEY_PATH}" && -f "${SSH_KEY_PATH}" ]]; then
    SSH_KEY_ARG=(-i "${SSH_KEY_PATH}")
fi

if [[ "${RUNPOD_PREFER_PROXY_SSH}" == "true" && -n "${HOST_ID}" ]]; then
    SSH_CMD=(ssh -tt -o StrictHostKeyChecking=no "${SSH_KEY_ARG[@]}" "${HOST_ID}@ssh.runpod.io")
elif [[ -n "${HOST}" && -n "${PORT}" ]]; then
    SSH_CMD=(ssh -o StrictHostKeyChecking=no "${SSH_KEY_ARG[@]}" -p "${PORT}" "root@${HOST}")
elif [[ -n "${HOST_ID}" ]]; then
    SSH_CMD=(ssh -tt -o StrictHostKeyChecking=no "${SSH_KEY_ARG[@]}" "${HOST_ID}@ssh.runpod.io")
else
    echo "[runpod] Could not determine SSH connection details." >&2
    exit 1
fi

echo "SSH_CMD=${SSH_CMD[*]}"

if [[ "${RUNPOD_SSH_EXECUTE:-false}" == "true" ]]; then
    exec "${SSH_CMD[@]}"
fi
