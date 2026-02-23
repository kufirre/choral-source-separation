#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" status
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

RESPONSE="$(curl -sS --fail-with-body \
    -X GET "${RUNPOD_API_BASE%/}/pods/${POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}")"

printf '%s\n' "${RESPONSE}" > "${RUNPOD_DIR}/last_pod_status.json"

python3 - <<'PY' "${RUNPOD_DIR}/last_pod_status.json"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    pod = json.load(f)

def get_ssh_port(port_mappings):
    if not isinstance(port_mappings, dict):
        return ""
    ssh = port_mappings.get("22")
    if isinstance(ssh, list) and ssh:
        return str(ssh[0].get("hostPort", ""))
    return ""

print(f"POD_ID={pod.get('id', '')}")
print(f"NAME={pod.get('name', '')}")
print(f"DESIRED_STATUS={pod.get('desiredStatus', '')}")
print(f"COST_PER_HR={pod.get('costPerHr', '')}")
print(f"GPU_DISPLAY_NAME={pod.get('machine', {}).get('gpuDisplayName', '')}")
print(f"PUBLIC_IP={pod.get('machine', {}).get('podHostId', '')}")
print(f"SSH_PORT={get_ssh_port(pod.get('portMappings'))}")
print(f"IMAGE_NAME={pod.get('imageName', '')}")
PY
