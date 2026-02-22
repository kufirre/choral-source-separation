#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-run-unknown}"
OUT_DIR="${1:-artifacts/${RUN_ID}/meta}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/run_metadata.txt"

{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_id=${RUN_ID}"
    echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python_version=$(python -V 2>&1 || echo unknown)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "image_uri=${IMAGE_URI:-unset}"
    echo "project_id=${PROJECT_ID:-unset}"
    echo "region=${REGION:-unset}"
    echo "gcs_data_bucket=${GCS_DATA_BUCKET:-unset}"
    echo "gcs_artifact_bucket=${GCS_ARTIFACT_BUCKET:-unset}"
} > "${OUT_FILE}"

python - <<'PY' >> "${OUT_FILE}" 2>/dev/null || true
import platform
print(f"platform={platform.platform()}")
try:
    import torch
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
except Exception:
    print("torch_version=unavailable")
PY

echo "[metadata] Wrote ${OUT_FILE}"
