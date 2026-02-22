#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

MODE="${1:-submit}"

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[validate-env] Missing required variable: ${name}" >&2
        return 1
    fi
    return 0
}

check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[validate-env] Required command not found: ${cmd}" >&2
        return 1
    fi
    return 0
}

main() {
    local -a required=(PROJECT_ID REGION GAR_REPO IMAGE_NAME TAG)

    case "${MODE}" in
        build)
            required+=(DOCKERFILE_PATH)
            check_command gcloud
            ;;
        submit)
            required+=(GCS_DATA_BUCKET GCS_ARTIFACT_BUCKET)
            check_command gcloud
            ;;
        *)
            echo "Usage: $0 [build|submit]" >&2
            exit 1
            ;;
    esac

    local failed=0
    for var_name in "${required[@]}"; do
        if ! require_var "${var_name}"; then
            failed=1
        fi
    done

    if [[ "${failed}" -ne 0 ]]; then
        exit 1
    fi

    echo "[validate-env] OK (${MODE})"
    echo "  PROJECT_ID=${PROJECT_ID}"
    echo "  REGION=${REGION}"
    echo "  GAR_REPO=${GAR_REPO}"
    echo "  IMAGE_NAME=${IMAGE_NAME}"
    echo "  TAG=${TAG}"
    echo "  GCS_DATA_BUCKET=${GCS_DATA_BUCKET}"
    echo "  GCS_ARTIFACT_BUCKET=${GCS_ARTIFACT_BUCKET}"
}

main "$@"
