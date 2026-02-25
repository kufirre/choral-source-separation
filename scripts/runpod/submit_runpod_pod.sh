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

if [[ -z "${RUNPOD_POD_NAME:-}" ]]; then
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME_PREFIX}-$(date +%Y%m%d-%H%M%S)-$((RANDOM % 10000))"
fi
RUN_ID="${RUN_ID:-${RUNPOD_POD_NAME}}"
RUN_PLATFORM="${RUN_PLATFORM:-runpod}"
RUN_PHASE="${RUN_PHASE:-}"
RUN_TIER="${RUN_TIER:-}"
RUN_LEDGER_FILE="${RUN_LEDGER_FILE:-${PROJECT_ROOT}/artifacts/run_ledger.jsonl}"
RUNPOD_WAIT_READY="${RUNPOD_WAIT_READY:-true}"
RUNPOD_READY_TIMEOUT_SECONDS="${RUNPOD_READY_TIMEOUT_SECONDS:-1800}"
RUNPOD_READY_POLL_SECONDS="${RUNPOD_READY_POLL_SECONDS:-15}"
RUNPOD_READY_LOG_INTERVAL_SECONDS="${RUNPOD_READY_LOG_INTERVAL_SECONDS:-60}"
RUNPOD_TERMINATE_ON_STARTUP_TIMEOUT="${RUNPOD_TERMINATE_ON_STARTUP_TIMEOUT:-true}"
RUNPOD_AUTO_PICK_GPU="${RUNPOD_AUTO_PICK_GPU:-true}"

select_gpu_type_by_stock() {
    if [[ "${RUNPOD_AUTO_PICK_GPU}" != "true" ]]; then
        return 0
    fi

    local candidate_csv secure_cloud min_vcpu min_ram cuda_version selected
    candidate_csv="${RUNPOD_GPU_CANDIDATES:-${RUNPOD_GPU_TYPE}}"
    secure_cloud="false"
    if [[ "${RUNPOD_CLOUD_TYPE}" == "SECURE" ]]; then
        secure_cloud="true"
    fi

    min_vcpu="${RUNPOD_MIN_VCPU_PER_GPU:-8}"
    min_ram="${RUNPOD_MIN_RAM_PER_GPU_GB:-30}"
    cuda_version="${RUNPOD_REQUIRED_CUDA_VERSION:-12.1}"

    selected="$(RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
        RUNPOD_GPU_CANDIDATES="${candidate_csv}" \
        RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}" \
        RUNPOD_SECURE_CLOUD="${secure_cloud}" \
        RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}" \
        RUNPOD_MIN_VCPU="${min_vcpu}" \
        RUNPOD_MIN_RAM="${min_ram}" \
        RUNPOD_REQUIRED_CUDA_VERSION="${cuda_version}" \
        python3 - <<'PY'
import json
import os
import urllib.request

api_key = os.getenv("RUNPOD_API_KEY", "").strip()
candidates = [x.strip() for x in os.getenv("RUNPOD_GPU_CANDIDATES", "").split(",") if x.strip()]
if not api_key or not candidates:
    print("")
    raise SystemExit(0)

query = """
query SelectGpu(
  $gpuId: String!
  $gpuCount: Int!
  $secureCloud: Boolean!
  $minDisk: Int!
  $minVcpuCount: Int!
  $minMemoryInGb: Int!
  $cudaVersion: String!
) {
  gpuTypes(input: {id: $gpuId}) {
    id
    displayName
    memoryInGb
    lowestPrice(
      input: {
        gpuCount: $gpuCount
        secureCloud: $secureCloud
        minDisk: $minDisk
        minVcpuCount: $minVcpuCount
        minMemoryInGb: $minMemoryInGb
        cudaVersion: $cudaVersion
      }
    ) {
      uninterruptablePrice
      stockStatus
    }
  }
}
"""

stock_rank = {
    "high": 4,
    "available": 3,
    "medium": 3,
    "low": 1,
    "none": 0,
    "outofstock": 0,
    "out_of_stock": 0,
}

best_gpu = ""
best_score = -10**9

for idx, gpu_id in enumerate(candidates):
    payload = {
        "query": query,
        "variables": {
            "gpuId": gpu_id,
            "gpuCount": int(os.getenv("RUNPOD_GPU_COUNT", "1")),
            "secureCloud": os.getenv("RUNPOD_SECURE_CLOUD", "false").lower() == "true",
            "minDisk": int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "50")),
            "minVcpuCount": int(os.getenv("RUNPOD_MIN_VCPU", "8")),
            "minMemoryInGb": int(os.getenv("RUNPOD_MIN_RAM", "30")),
            "cudaVersion": os.getenv("RUNPOD_REQUIRED_CUDA_VERSION", "12.1"),
        },
    }
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[runpod] GPU option {gpu_id} query_error={exc}", file=os.sys.stderr)
        continue

    obj = json.loads(raw)
    if obj.get("errors"):
        print(f"[runpod] GPU option {gpu_id} graphql_errors={obj['errors']}", file=os.sys.stderr)
        continue

    gpu_types = ((obj.get("data") or {}).get("gpuTypes")) or []
    item = gpu_types[0] if gpu_types else {}
    lowest = item.get("lowestPrice") or {}
    stock = str(lowest.get("stockStatus") or "").strip().lower()
    price = lowest.get("uninterruptablePrice")

    stock_score = stock_rank.get(stock, 0)
    price_penalty = float(price) if isinstance(price, (int, float)) else 9999.0
    score = stock_score * 100000 - int(price_penalty * 1000) - idx
    print(f"[runpod] GPU option {gpu_id} stock={stock or 'unknown'} price={price}", file=os.sys.stderr)

    if score > best_score:
        best_score = score
        best_gpu = gpu_id

if not best_gpu:
    best_gpu = candidates[0]
print(best_gpu)
PY
    )"

    selected="$(echo "${selected}" | tr -d '\n' | tr -d '\r')"
    if [[ -n "${selected}" ]]; then
        RUNPOD_GPU_TYPE="${selected}"
        echo "[runpod] Auto-selected GPU type: ${RUNPOD_GPU_TYPE}"
    else
        echo "[runpod] GPU auto-pick returned empty; keeping requested GPU type: ${RUNPOD_GPU_TYPE}"
    fi
}

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
    "RUNPOD_POD_NAME",
    "RUNPOD_AUTO_TERMINATE_ON_EXIT",
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
    "GCP_SA_KEY_FILE",
    "SSH_PUBLIC_KEY",
]

runtime_env = {}
for k in runtime_env_keys:
    v = os.getenv(k)
    if v is not None and v != "":
        runtime_env[k] = v

auto_terminate = as_bool(os.getenv("RUNPOD_AUTO_TERMINATE_ON_EXIT"), False)
if auto_terminate:
    runtime_env["RUNPOD_API_BASE"] = os.getenv("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
    if os.getenv("RUNPOD_API_KEY"):
        runtime_env["RUNPOD_API_KEY"] = os.getenv("RUNPOD_API_KEY")

if (
    "GCP_SA_KEY_B64" not in runtime_env
    and "GCP_SA_KEY_JSON" not in runtime_env
):
    key_file = os.getenv("GCP_SA_KEY_SOURCE_FILE", "")
    if not key_file:
        key_file = os.getenv("GCP_SA_KEY_FILE", "")
    if key_file and os.path.exists(key_file):
        import base64
        with open(key_file, "rb") as f:
            runtime_env["GCP_SA_KEY_B64"] = base64.b64encode(f.read()).decode("utf-8")

volume_gb = as_int(os.getenv("RUNPOD_VOLUME_GB"), 100)
payload = {
    "name": os.environ["RUNPOD_POD_NAME"],
    "imageName": os.environ["IMAGE_URI"],
    "cloudType": os.environ["RUNPOD_CLOUD_TYPE"],
    "gpuCount": as_int(os.getenv("RUNPOD_GPU_COUNT"), 1),
    "gpuTypeIds": [os.environ["RUNPOD_GPU_TYPE"]],
    "containerDiskInGb": as_int(os.getenv("RUNPOD_CONTAINER_DISK_GB"), 100),
    "supportPublicIp": as_bool(os.getenv("RUNPOD_SUPPORT_PUBLIC_IP"), True),
    "interruptible": as_bool(os.getenv("RUNPOD_INTERRUPTIBLE"), False),
    "ports": split_csv(os.getenv("RUNPOD_PORTS")),
    "env": runtime_env,
}

if volume_gb > 0:
    payload["volumeInGb"] = volume_gb
    payload["volumeMountPath"] = os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace")

cuda_versions = split_csv(os.getenv("RUNPOD_ALLOWED_CUDA_VERSIONS"))
if cuda_versions:
    payload["allowedCudaVersions"] = cuda_versions

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

append_ledger() {
    local status="$1"
    local job_id="${2:-}"
    local notes="${3:-}"
    local ledger_tool="${SCRIPT_DIR}/run_ledger.py"

    if [[ ! -f "${ledger_tool}" ]]; then
        return 0
    fi

    python3 "${ledger_tool}" append \
        --ledger "${RUN_LEDGER_FILE}" \
        --run-id "${RUN_ID}" \
        --platform "${RUN_PLATFORM}" \
        --phase "${RUN_PHASE}" \
        --tier "${RUN_TIER}" \
        --status "${status}" \
        --job-id "${job_id}" \
        --config-path "${CONFIG_PATH:-}" \
        --train-data-paths "${TRAIN_DATA_PATHS:-}" \
        --valid-data-paths "${VALID_DATA_PATHS:-}" \
        --dataset-gcs-paths "${DATASET_GCS_PATHS:-}" \
        --results-path "${RESULTS_PATH:-}" \
        --image-uri "${IMAGE_URI:-}" \
        --gpu-type "${RUNPOD_GPU_TYPE:-}" \
        --gpu-count "${RUNPOD_GPU_COUNT:-}" \
        --cloud-type "${RUNPOD_CLOUD_TYPE:-}" \
        --repo-url "${RUNPOD_REPO_URL:-}" \
        --git-ref "${RUNPOD_GIT_REF:-}" \
        --start-checkpoint "${START_CHECKPOINT:-}" \
        --use-checkpoint "${USE_CHECKPOINT:-}" \
        --notes "${notes}" \
        --extra "pod_name=${RUNPOD_POD_NAME:-}" \
        --extra "interruptible=${RUNPOD_INTERRUPTIBLE:-}" \
        --extra "volume_mount_path=${RUNPOD_VOLUME_MOUNT_PATH:-}" \
        || true
}

pod_runtime_state() {
    local pod_id="$1"
    local response

    if ! response="$(curl -sS --fail-with-body \
        -X POST "https://api.runpod.io/graphql" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"query { pod(input: {podId: \\\"${pod_id}\\\"}) { id desiredStatus lastStatusChange runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}")"; then
        echo "0|unknown|api_error||"
        return 0
    fi

    RUNPOD_RESPONSE="${response}" python3 - <<'PY'
import json
import os

resp = json.loads(os.environ["RUNPOD_RESPONSE"])
errors = resp.get("errors") or []
if errors:
    print("0|unknown|graphql_error||")
    raise SystemExit(0)

pod = (resp.get("data") or {}).get("pod")
if pod is None:
    print("0|not_found|pod_not_found||")
    raise SystemExit(0)

desired = str(pod.get("desiredStatus") or "unknown").strip()
runtime = pod.get("runtime")
if runtime is None:
    print(f"0|{desired}|runtime_null||")
    raise SystemExit(0)

ssh_ip = ""
ssh_port = ""
for p in runtime.get("ports") or []:
    ptype = str(p.get("type", "")).lower()
    if ptype == "tcp" and p.get("privatePort") == 22:
        ip = str(p.get("ip", "")).strip()
        pub = str(p.get("publicPort", "")).strip()
        if ip and pub:
            ssh_ip = ip
            ssh_port = pub
            break

if ssh_ip and ssh_port:
    print(f"1|{desired}|runtime_ready|{ssh_ip}|{ssh_port}")
else:
    print(f"0|{desired}|runtime_no_public_ssh||")
PY
}

wait_for_pod_ready() {
    local pod_id="$1"
    local timeout_s="$2"
    local poll_s="$3"
    local elapsed=0
    local log_interval_s="${RUNPOD_READY_LOG_INTERVAL_SECONDS}"
    local last_log_elapsed=-999999
    local state ready desired reason ssh_ip ssh_port

    while (( elapsed < timeout_s )); do
        state="$(pod_runtime_state "${pod_id}")"
        IFS='|' read -r ready desired reason ssh_ip ssh_port <<< "${state}"

        if [[ "${ready}" == "1" ]]; then
            echo "[runpod] Pod runtime is ready (pod_id=${pod_id}, elapsed=${elapsed}s, ssh=${ssh_ip}:${ssh_port})."
            append_ledger "startup_ready" "${pod_id}" "runtime_initialized ssh=${ssh_ip}:${ssh_port}"
            return 0
        fi

        if (( elapsed - last_log_elapsed >= log_interval_s )); then
            echo "[runpod] Waiting for runtime (pod_id=${pod_id}, elapsed=${elapsed}s, desired_status=${desired}, reason=${reason})"
            last_log_elapsed="${elapsed}"
        fi

        case "${desired}" in
            TERMINATED|STOPPED|FAILED|CANCELED|CANCELLED|DELETED|NOT_FOUND|not_found)
                echo "[runpod] Pod entered terminal state before runtime became ready (pod_id=${pod_id}, desired_status=${desired}, reason=${reason})." >&2
                append_ledger "startup_failed" "${pod_id}" "terminal_state_before_runtime_ready:${desired}:${reason}"
                return 1
                ;;
        esac

        sleep "${poll_s}"
        elapsed=$((elapsed + poll_s))
    done

    echo "[runpod] Pod runtime was not ready within ${timeout_s}s (pod_id=${pod_id})." >&2
    append_ledger "startup_timeout" "${pod_id}" "runtime_not_initialized"

    if [[ "${RUNPOD_TERMINATE_ON_STARTUP_TIMEOUT}" == "true" ]]; then
        curl -sS --fail-with-body \
            -X DELETE "${RUNPOD_API_BASE%/}/pods/${pod_id}" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null || true
        echo "[runpod] Terminated pod ${pod_id} after startup timeout."
        append_ledger "startup_timeout_terminated" "${pod_id}" "auto_terminated_after_startup_timeout"
    fi

    return 1
}

select_gpu_type_by_stock
PAYLOAD="$(build_payload)"
RUNPOD_DIR="${PROJECT_ROOT}/.runpod"
mkdir -p "${RUNPOD_DIR}"

if [[ "${RUNPOD_DRY_RUN}" == "true" ]]; then
    echo "[runpod] Dry run enabled. Request payload:"
    RUNPOD_PAYLOAD="${PAYLOAD}" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["RUNPOD_PAYLOAD"])
for k in ("HF_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY_JSON", "RUNPOD_API_KEY"):
    if isinstance(payload.get("env"), dict) and k in payload["env"]:
        payload["env"][k] = "***REDACTED***"
print(json.dumps(payload, indent=4))
PY
    append_ledger "dry_run_submitted" "" "runpod_dry_run=true"
    exit 0
fi

RESPONSE="$(curl -sS --fail-with-body \
    -X POST "${RUNPOD_API_BASE%/}/pods" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}")"

RUNPOD_RESPONSE="${RESPONSE}" python3 - <<'PY' > "${RUNPOD_DIR}/last_pod_response.json"
import json
import os

response = json.loads(os.environ["RUNPOD_RESPONSE"])
env = response.get("env")
if isinstance(env, dict):
    for key in ("HF_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY_JSON", "RUNPOD_API_KEY"):
        if key in env:
            env[key] = "***REDACTED***"
print(json.dumps(response))
PY

POD_ID="$(RUNPOD_RESPONSE="${RESPONSE}" python3 - <<'PY'
import json
import os
data = json.loads(os.environ["RUNPOD_RESPONSE"])
print(data.get("id", ""))
PY
)"

if [[ -z "${POD_ID}" ]]; then
    echo "[runpod] Pod creation response did not contain an id." >&2
    cat "${RUNPOD_DIR}/last_pod_response.json" >&2
    append_ledger "submit_failed" "" "runpod_response_missing_pod_id"
    exit 1
fi

printf '%s\n' "${POD_ID}" > "${RUNPOD_DIR}/last_pod_id"
printf '%s\n' "${RUN_ID}" > "${RUNPOD_DIR}/last_run_id"
append_ledger "submitted" "${POD_ID}" "pod_created"

echo "[runpod] Pod submitted successfully."
echo "  POD_ID=${POD_ID}"
echo "  RUN_ID=${RUN_ID}"
echo "  IMAGE_URI=${IMAGE_URI}"
echo "  TRAIN_CMD=${TRAIN_CMD}"

if [[ "${RUNPOD_WAIT_READY}" == "true" ]]; then
    wait_for_pod_ready "${POD_ID}" "${RUNPOD_READY_TIMEOUT_SECONDS}" "${RUNPOD_READY_POLL_SECONDS}"
fi
