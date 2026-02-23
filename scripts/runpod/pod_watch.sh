#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POD_ID="${1:-${POD_ID:-}}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"

if [[ -z "${POD_ID}" ]]; then
    echo "Usage: $0 <pod_id>" >&2
    exit 1
fi

while true; do
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    "${SCRIPT_DIR}/pod_status.sh" "${POD_ID}"
    echo "----"
    sleep "${INTERVAL_SECONDS}"
done
