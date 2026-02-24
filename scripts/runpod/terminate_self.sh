#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUNPOD_AUTO_TERMINATE_ON_EXIT:-false}" != "true" ]]; then
    exit 0
fi

if [[ -z "${RUNPOD_API_KEY:-}" || -z "${RUNPOD_POD_NAME:-}" ]]; then
    echo "[runpod-self-terminate] RUNPOD_API_KEY or RUNPOD_POD_NAME missing; skipping."
    exit 0
fi

RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"

pods_json="$(curl -sS --fail-with-body \
    -X GET "${RUNPOD_API_BASE%/}/pods" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}")"

pod_id="$(RUNPOD_PODS_JSON="${pods_json}" RUNPOD_POD_NAME="${RUNPOD_POD_NAME}" python3 - <<'PY'
import json
import os

pod_name = os.getenv("RUNPOD_POD_NAME", "")
pod_id = ""
try:
    pods = json.loads(os.getenv("RUNPOD_PODS_JSON", "[]"))
except json.JSONDecodeError:
    pods = []

for pod in pods:
    if isinstance(pod, dict) and pod.get("name") == pod_name:
        pod_id = str(pod.get("id", ""))
        break

print(pod_id)
PY
)"

if [[ -z "${pod_id}" ]]; then
    echo "[runpod-self-terminate] Pod ${RUNPOD_POD_NAME} not found in API response; skipping."
    exit 0
fi

curl -sS --fail-with-body \
    -X DELETE "${RUNPOD_API_BASE%/}/pods/${pod_id}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null

echo "[runpod-self-terminate] Requested termination for pod ${pod_id} (${RUNPOD_POD_NAME})."
