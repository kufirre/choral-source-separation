#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" status
fi

RESPONSE="$(curl -sS --fail-with-body \
    -X GET "${RUNPOD_API_BASE%/}/pods" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}")"

RUNPOD_PODS_RESPONSE="${RESPONSE}" python3 - <<'PY'
import json
import os
pods = json.loads(os.environ["RUNPOD_PODS_RESPONSE"])
if not pods:
    print("[runpod] No active pods.")
    raise SystemExit(0)
for pod in pods:
    pod_id = pod.get("id", "")
    name = pod.get("name", "")
    status = pod.get("desiredStatus", "")
    cost = pod.get("costPerHr", "")
    gpu = pod.get("machine", {}).get("gpuDisplayName", "")
    print(f"{pod_id}\t{status}\t{cost}\t{gpu}\t{name}")
PY
