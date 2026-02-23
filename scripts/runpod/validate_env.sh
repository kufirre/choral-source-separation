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
        echo "[runpod-validate] Missing required variable: ${name}" >&2
        return 1
    fi
    return 0
}

check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[runpod-validate] Required command not found: ${cmd}" >&2
        return 1
    fi
    return 0
}

main() {
    local -a required=(RUNPOD_API_BASE IMAGE_URI RUNPOD_GPU_TYPE RUNPOD_GPU_COUNT RUNPOD_CLOUD_TYPE)
    local failed=0

    check_command curl || failed=1
    check_command python3 || failed=1

    case "${MODE}" in
        submit)
            if [[ "${RUNPOD_DRY_RUN:-false}" != "true" ]]; then
                required+=(RUNPOD_API_KEY)
            fi
            ;;
        status|terminate)
            if [[ "${RUNPOD_DRY_RUN:-false}" != "true" ]]; then
                required+=(RUNPOD_API_KEY)
            fi
            ;;
        *)
            echo "Usage: $0 [submit|status|terminate]" >&2
            exit 1
            ;;
    esac

    for var_name in "${required[@]}"; do
        if ! require_var "${var_name}"; then
            failed=1
        fi
    done

    if [[ "${failed}" -ne 0 ]]; then
        exit 1
    fi

    echo "[runpod-validate] OK (${MODE})"
    echo "  RUNPOD_API_BASE=${RUNPOD_API_BASE}"
    echo "  IMAGE_URI=${IMAGE_URI}"
    echo "  RUNPOD_CLOUD_TYPE=${RUNPOD_CLOUD_TYPE}"
    echo "  RUNPOD_GPU_TYPE=${RUNPOD_GPU_TYPE}"
    echo "  RUNPOD_GPU_COUNT=${RUNPOD_GPU_COUNT}"
}

main "$@"
