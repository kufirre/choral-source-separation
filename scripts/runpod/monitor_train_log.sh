#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

RUNPOD_DIR="${PROJECT_ROOT}/.runpod"
POD_ID="${1:-${POD_ID:-}}"
RUN_ID="${2:-${RUN_ID:-}}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-15}"
TAIL_LINES="${TAIL_LINES:-120}"
SSH_KEY_PATH="${RUNPOD_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"

if [[ -z "${POD_ID}" && -f "${RUNPOD_DIR}/last_pod_id" ]]; then
    POD_ID="$(cat "${RUNPOD_DIR}/last_pod_id")"
fi
if [[ -z "${RUN_ID}" && -f "${RUNPOD_DIR}/last_run_id" ]]; then
    RUN_ID="$(cat "${RUNPOD_DIR}/last_run_id")"
fi

if [[ -z "${POD_ID}" ]]; then
    echo "Usage: $0 <pod_id> [run_id]" >&2
    exit 1
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "Run id is required (arg #2 or .runpod/last_run_id)." >&2
    exit 1
fi

SSH_KEY_ARG=()
if [[ -n "${SSH_KEY_PATH}" && -f "${SSH_KEY_PATH}" ]]; then
    SSH_KEY_ARG=(-i "${SSH_KEY_PATH}")
fi

echo "[runpod-monitor] POD_ID=${POD_ID}"
echo "[runpod-monitor] RUN_ID=${RUN_ID}"

while true; do
    status_output="$(RUNPOD_SKIP_VALIDATE=true "${SCRIPT_DIR}/pod_status.sh" "${POD_ID}")"
    echo "${status_output}"
    desired_status="$(printf '%s\n' "${status_output}" | awk -F= '/^DESIRED_STATUS=/{print $2}')"
    host="$(printf '%s\n' "${status_output}" | awk -F= '/^PUBLIC_IP=/{print $2}')"
    port="$(printf '%s\n' "${status_output}" | awk -F= '/^SSH_PORT=/{print $2}')"
    host_id="$(printf '%s\n' "${status_output}" | awk -F= '/^MACHINE_HOST_ID=/{print $2}')"

    if [[ "${desired_status}" == "TERMINATED" || "${desired_status}" == "STOPPED" ]]; then
        echo "[runpod-monitor] Pod is ${desired_status}; exiting monitor."
        exit 1
    fi

    if [[ -n "${host}" && -n "${port}" ]]; then
        ssh_target=(root@"${host}")
        ssh_base=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_KEY_ARG[@]}" -p "${port}" "${ssh_target[@]}")
    elif [[ -n "${host_id}" ]]; then
        ssh_target=("${host_id}@ssh.runpod.io")
        ssh_base=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_KEY_ARG[@]}" "${ssh_target[@]}")
    else
        echo "[runpod-monitor] Runtime not yet reachable over SSH; waiting ${STATUS_INTERVAL_SECONDS}s."
        sleep "${STATUS_INTERVAL_SECONDS}"
        continue
    fi

    if "${ssh_base[@]}" "bash -lc 'test -f /workspace/choral-source-separation/artifacts/${RUN_ID}/train.log'" >/dev/null 2>&1; then
        echo "[runpod-monitor] Streaming /workspace/choral-source-separation/artifacts/${RUN_ID}/train.log"
        exec "${ssh_base[@]}" "bash -lc 'tail -n ${TAIL_LINES} -f /workspace/choral-source-separation/artifacts/${RUN_ID}/train.log'"
    fi

    echo "[runpod-monitor] SSH reachable but train.log not ready yet (or SSH command failed); waiting ${STATUS_INTERVAL_SECONDS}s."
    sleep "${STATUS_INTERVAL_SECONDS}"
done
