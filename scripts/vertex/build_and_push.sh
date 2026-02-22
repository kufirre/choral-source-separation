#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

usage() {
    cat <<'EOF'
Build and push the Vertex training image.

Usage:
  build_and_push.sh [--mode local|cloud] [--dockerfile <path>] [--help]

Environment:
  PROJECT_ID        GCP project id
  REGION            GCP region
  GAR_REPO          Artifact Registry repository (required)
  IMAGE_NAME        Image name
  TAG               Image tag
  CACHE_TAG         Cache source tag (default: latest)
  DOCKER_PLATFORM   Docker target platform (default: linux/amd64)
  DOCKERFILE_PATH   Dockerfile path relative to repo root (default: scripts/Dockerfile)
  BUILD_MODE        Build backend: local or cloud (default: cloud)
EOF
}

PROJECT_ID="${PROJECT_ID:-ai-training-484220}"
REGION="${REGION:-europe-west4}"
GAR_REPO="${GAR_REPO:-}"
IMAGE_NAME="${IMAGE_NAME:-choral-source-separation}"
TAG="${TAG:-latest}"
CACHE_TAG="${CACHE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BUILD_MODE="${BUILD_MODE:-cloud}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-scripts/Dockerfile}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${GAR_REPO}" ]]; then
    echo "GAR_REPO is required." >&2
    exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required." >&2
    exit 1
fi

if [[ -x "${SCRIPT_DIR}/validate_env.sh" ]]; then
    "${SCRIPT_DIR}/validate_env.sh" build
fi

if [[ "${DOCKERFILE_PATH}" = /* ]]; then
    DOCKERFILE_ABS="${DOCKERFILE_PATH}"
else
    DOCKERFILE_ABS="${PROJECT_ROOT}/${DOCKERFILE_PATH}"
fi

if [[ ! -f "${DOCKERFILE_ABS}" ]]; then
    echo "Dockerfile not found: ${DOCKERFILE_ABS}" >&2
    exit 1
fi

DOCKERFILE_REL="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${DOCKERFILE_ABS}" "${PROJECT_ROOT}")"
if [[ "${DOCKERFILE_REL}" == ../* ]]; then
    echo "Dockerfile must be inside repo root for cloud builds: ${DOCKERFILE_ABS}" >&2
    exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${TAG}"
CACHE_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${CACHE_TAG}"
echo "Using dockerfile: ${DOCKERFILE_REL}"
echo "Image URI: ${IMAGE_URI}"
echo "Cache image URI: ${CACHE_IMAGE_URI}"

case "${BUILD_MODE}" in
    local)
        if ! command -v docker >/dev/null 2>&1; then
            echo "docker is required for local mode." >&2
            exit 1
        fi
        gcloud auth configure-docker "${REGION}-docker.pkg.dev"
        docker pull "${CACHE_IMAGE_URI}" >/dev/null 2>&1 || true
        docker build \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --cache-from "${CACHE_IMAGE_URI}" \
            --platform "${DOCKER_PLATFORM}" \
            -f "${DOCKERFILE_ABS}" \
            -t "${IMAGE_URI}" \
            "${PROJECT_ROOT}"
        if [[ "${CACHE_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
            docker tag "${IMAGE_URI}" "${CACHE_IMAGE_URI}"
        fi
        docker push "${IMAGE_URI}"
        if [[ "${CACHE_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
            docker push "${CACHE_IMAGE_URI}"
        fi
        ;;
    cloud)
        build_config_file="$(mktemp)"
        trap 'rm -f "${build_config_file}"' EXIT
        cat >"${build_config_file}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    entrypoint: bash
    args:
      - -c
      - docker pull ${CACHE_IMAGE_URI} || true
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - --build-arg=BUILDKIT_INLINE_CACHE=1
      - --cache-from=${CACHE_IMAGE_URI}
      - --platform=${DOCKER_PLATFORM}
      - -f
      - ${DOCKERFILE_REL}
      - -t
      - ${IMAGE_URI}
EOF
        if [[ "${CACHE_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
            cat >>"${build_config_file}" <<EOF
      - -t
      - ${CACHE_IMAGE_URI}
EOF
        fi
        cat >>"${build_config_file}" <<EOF
      - .
images:
  - ${IMAGE_URI}
EOF
        if [[ "${CACHE_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
            cat >>"${build_config_file}" <<EOF
  - ${CACHE_IMAGE_URI}
EOF
        fi
        gcloud builds submit \
            --project "${PROJECT_ID}" \
            --region "${REGION}" \
            --config "${build_config_file}" \
            "${PROJECT_ROOT}"
        ;;
    *)
        echo "Invalid BUILD_MODE: ${BUILD_MODE}. Use local or cloud." >&2
        exit 1
        ;;
esac

echo "Pushed image: ${IMAGE_URI}"
