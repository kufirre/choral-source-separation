#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <GCS_BUCKET> <RUN_ID> [local_tb_dir]" >&2
    exit 1
fi

GCS_BUCKET="$1"
RUN_ID="$2"
LOCAL_TB_DIR="${3:-artifacts/${RUN_ID}/tb}"
REMOTE_TB_DIR="gs://${GCS_BUCKET}/artifacts/${RUN_ID}/tb"

mkdir -p "${LOCAL_TB_DIR}"
echo "[tb-sync] Syncing ${REMOTE_TB_DIR} -> ${LOCAL_TB_DIR}"
gcloud storage rsync -r "${REMOTE_TB_DIR}" "${LOCAL_TB_DIR}"
