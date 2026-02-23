#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" submit
fi

TRAIN_CMD="${TRAIN_CMD:-}"
if [[ -z "${TRAIN_CMD}" ]]; then
    echo "TRAIN_CMD is required." >&2
    exit 1
fi

RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-${RUNPOD_POD_NAME_PREFIX}-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-${RUNPOD_POD_NAME}}"

build_payload() {
    python3 - <<'PY'
import json
import os

def as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

def as_int(value: str, default: int) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)

def split_csv(value: str):
    if value is None:
        return []
    out = []
    for part in value.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out

runtime_env_keys = [
    "RUN_ID",
    "TRAIN_CMD",
    "GCS_DATA_BUCKET",
    "GCS_ARTIFACT_BUCKET",
    "DATA_ROOT",
    "DATASET_GCS_PREFIX",
    "DATASET_GCS_PATHS",
    "ARTIFACT_SYNC_SECONDS",
    "USE_CHECKPOINT",
    "BOOTSTRAP_CKPT_URI",
    "HF_TOKEN",
    "START_CHECKPOINT",
    "GCP_SA_KEY_B64",
    "GCP_SA_KEY_JSON",
]

runtime_env = {}
for k in runtime_env_keys:
    v = os.getenv(k)
    if v is not None and v != "":
        runtime_env[k] = v

payload = {
    "name": os.environ["RUNPOD_POD_NAME"],
    "imageName": os.environ["IMAGE_URI"],
    "cloudType": os.environ["RUNPOD_CLOUD_TYPE"],
    "gpuCount": as_int(os.getenv("RUNPOD_GPU_COUNT"), 1),
    "gpuTypeIds": [os.environ["RUNPOD_GPU_TYPE"]],
    "containerDiskInGb": as_int(os.getenv("RUNPOD_CONTAINER_DISK_GB"), 100),
    "volumeInGb": as_int(os.getenv("RUNPOD_VOLUME_GB"), 100),
    "volumeMountPath": os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace"),
    "supportPublicIp": as_bool(os.getenv("RUNPOD_SUPPORT_PUBLIC_IP"), True),
    "interruptible": as_bool(os.getenv("RUNPOD_INTERRUPTIBLE"), False),
    "ports": split_csv(os.getenv("RUNPOD_PORTS")),
    "allowedCudaVersions": split_csv(os.getenv("RUNPOD_ALLOWED_CUDA_VERSIONS")),
    "env": runtime_env,
}

if os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID"):
    payload["containerRegistryAuthId"] = os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID")

if os.getenv("RUNPOD_MIN_VCPU_PER_GPU"):
    payload["minVCPUPerGPU"] = as_int(os.getenv("RUNPOD_MIN_VCPU_PER_GPU"), 0)

if os.getenv("RUNPOD_MIN_RAM_PER_GPU_GB"):
    payload["minRAMPerGPU"] = as_int(os.getenv("RUNPOD_MIN_RAM_PER_GPU_GB"), 0)

data_centers = split_csv(os.getenv("RUNPOD_DATA_CENTER_IDS"))
if data_centers:
    payload["dataCenterIds"] = data_centers

print(json.dumps(payload))
PY
}

PAYLOAD="$(build_payload)"
RUNPOD_DIR="${PROJECT_ROOT}/.runpod"
mkdir -p "${RUNPOD_DIR}"

if [[ "${RUNPOD_DRY_RUN}" == "true" ]]; then
    echo "[runpod] Dry run enabled. Request payload:"
    RUNPOD_PAYLOAD="${PAYLOAD}" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["RUNPOD_PAYLOAD"])
for k in ("HF_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY_JSON"):
    if isinstance(payload.get("env"), dict) and k in payload["env"]:
        payload["env"][k] = "***REDACTED***"
print(json.dumps(payload, indent=4))
PY
    exit 0
fi

RESPONSE="$(curl -sS --fail-with-body \
    -X POST "${RUNPOD_API_BASE%/}/pods" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}")"

printf '%s\n' "${RESPONSE}" > "${RUNPOD_DIR}/last_pod_response.json"

POD_ID="$(python3 - <<'PY' "${RUNPOD_DIR}/last_pod_response.json"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("id", ""))
PY
)"

if [[ -z "${POD_ID}" ]]; then
    echo "[runpod] Pod creation response did not contain an id." >&2
    cat "${RUNPOD_DIR}/last_pod_response.json" >&2
    exit 1
fi

printf '%s\n' "${POD_ID}" > "${RUNPOD_DIR}/last_pod_id"
printf '%s\n' "${RUN_ID}" > "${RUNPOD_DIR}/last_run_id"

echo "[runpod] Pod submitted successfully."
echo "  POD_ID=${POD_ID}"
echo "  RUN_ID=${RUN_ID}"
echo "  IMAGE_URI=${IMAGE_URI}"
echo "  TRAIN_CMD=${TRAIN_CMD}"
