#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ai-training-484220}"
REGION="${REGION:-europe-west4}"
BUCKET_NAME="${BUCKET_NAME:-}"
RUN_ID="${RUN_ID:-}"

usage() {
    echo "Usage: $0 [list|describe|logs|stream|checkpoints|get-checkpoint]"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 stream <JOB_ID>"
    echo "  BUCKET_NAME=my-bucket RUN_ID=my-run $0 checkpoints"
    exit 1
}

list_jobs() {
    gcloud ai custom-jobs list \
        --project "${PROJECT_ID}" \
        --region "${REGION}" \
        --sort-by "~createTime" \
        --limit 20 \
        --format "table(name,displayName,state,createTime,endTime)"
}

describe_job() {
    local job_id="${1:-}"
    if [[ -z "${job_id}" ]]; then
        echo "JOB_ID required." >&2
        exit 1
    fi
    gcloud ai custom-jobs describe "${job_id}" \
        --project "${PROJECT_ID}" \
        --region "${REGION}"
}

job_logs() {
    local job_id="${1:-}"
    if [[ -z "${job_id}" ]]; then
        echo "JOB_ID required." >&2
        exit 1
    fi
    gcloud logging read "(resource.type=\"aiplatform.googleapis.com/CustomJob\" AND labels.job_id=\"${job_id}\") OR (resource.type=\"ml_job\" AND resource.labels.job_id=\"${job_id}\")" \
        --project "${PROJECT_ID}" \
        --limit 500 \
        --order asc \
        --format "value(textPayload)"
}

stream_logs() {
    local job_id="${1:-}"
    if [[ -z "${job_id}" ]]; then
        echo "JOB_ID required." >&2
        exit 1
    fi
    gcloud ai custom-jobs stream-logs "${job_id}" \
        --project "${PROJECT_ID}" \
        --region "${REGION}"
}

checkpoints() {
    if [[ -z "${BUCKET_NAME}" || -z "${RUN_ID}" ]]; then
        echo "BUCKET_NAME and RUN_ID are required for checkpoint commands." >&2
        exit 1
    fi
    gcloud storage ls -l "gs://${BUCKET_NAME}/artifacts/${RUN_ID}/checkpoints/*" || true
}

get_checkpoint() {
    local remote_path="${1:-}"
    if [[ -z "${remote_path}" ]]; then
        echo "Remote checkpoint URI required." >&2
        exit 1
    fi
    mkdir -p checkpoints
    gcloud storage cp "${remote_path}" checkpoints/
}

if [[ $# -eq 0 ]]; then
    usage
fi

cmd="$1"
shift

case "${cmd}" in
    list)
        list_jobs
        ;;
    describe)
        describe_job "${1:-}"
        ;;
    logs)
        job_logs "${1:-}"
        ;;
    stream)
        stream_logs "${1:-}"
        ;;
    checkpoints)
        checkpoints
        ;;
    get-checkpoint)
        get_checkpoint "${1:-}"
        ;;
    *)
        usage
        ;;
esac
