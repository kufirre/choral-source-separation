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
    -X POST "https://api.runpod.io/graphql" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { id name desiredStatus imageName machineId lastStatusChange machine { podHostId gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}")"

printf '%s\n' "${RESPONSE}" > "${RUNPOD_DIR}/last_pod_status.json"

python3 - <<'PY' "${RUNPOD_DIR}/last_pod_status.json"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    resp = json.load(f)

errors = resp.get("errors")
if errors:
    raise SystemExit(f"GraphQL error: {errors[0].get('message', errors[0])}")

pod = resp.get("data", {}).get("pod")
if pod is None:
    raise SystemExit("Pod not found (or already terminated).")

runtime = pod.get("runtime") or {}
ports = runtime.get("ports") or []
ssh_ip = ""
ssh_port = ""
http_ip = ""
http_port = ""
for p in ports:
    ptype = str(p.get("type", "")).lower()
    priv = p.get("privatePort")
    if ptype == "tcp" and priv == 22:
        ssh_ip = str(p.get("ip", ""))
        ssh_port = str(p.get("publicPort", ""))
    if ptype == "http":
        http_ip = str(p.get("ip", ""))
        http_port = str(p.get("publicPort", ""))

print(f"POD_ID={pod.get('id', '')}")
print(f"NAME={pod.get('name', '')}")
print(f"DESIRED_STATUS={pod.get('desiredStatus', '')}")
print(f"GPU_DISPLAY_NAME={pod.get('machine', {}).get('gpuDisplayName', '')}")
print(f"MACHINE_HOST_ID={pod.get('machine', {}).get('podHostId', '')}")
print(f"MACHINE_ID={pod.get('machineId', '')}")
print(f"LAST_STATUS_CHANGE={pod.get('lastStatusChange', '')}")
print(f"UPTIME_SECONDS={runtime.get('uptimeInSeconds', '')}")
print(f"PUBLIC_IP={ssh_ip}")
print(f"SSH_PORT={ssh_port}")
print(f"HTTP_IP={http_ip}")
print(f"HTTP_PORT={http_port}")
print(f"IMAGE_NAME={pod.get('imageName', '')}")
PY
