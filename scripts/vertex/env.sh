#!/usr/bin/env bash
# shellcheck shell=bash

# Shared defaults for Vertex training scripts.
# Override any value by exporting it before running submit/build scripts.

export PROJECT_ID="${PROJECT_ID:-ai-training-484220}"
export REGION="${REGION:-europe-west4}"

# Artifact Registry
export GAR_REPO="${GAR_REPO:-choral-separation}"
export IMAGE_NAME="${IMAGE_NAME:-choral-source-separation}"
export TAG="${TAG:-latest}"
export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DOCKERFILE_PATH="${DOCKERFILE_PATH:-scripts/Dockerfile}"
export CLOUD_CACHE_BACKEND="${CLOUD_CACHE_BACKEND:-docker}"
export KANIKO_CACHE_REPO="${KANIKO_CACHE_REPO:-}"
export KANIKO_CACHE_TTL="${KANIKO_CACHE_TTL:-336h}"

# Vertex compute defaults
export MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-1g}"
export ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_H100_80GB}"
export ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
export REPLICA_COUNT="${REPLICA_COUNT:-1}"
export SCHEDULING_STRATEGY="${SCHEDULING_STRATEGY:-SPOT}"
export ENABLE_WEB_ACCESS="${ENABLE_WEB_ACCESS:-true}"
export ENABLE_DASHBOARD_ACCESS="${ENABLE_DASHBOARD_ACCESS:-true}"

# Buckets
export GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-csmamba2-484220-ew4-20260206-500242}"
export GCS_ARTIFACT_BUCKET="${GCS_ARTIFACT_BUCKET:-csmamba2-484220-ew4-20260206-500242}"

# Optional Vertex fields
export VERTEX_SA_EMAIL="${VERTEX_SA_EMAIL:-}"
export NETWORK="${NETWORK:-}"

# Container runtime options
export DATASET_GCS_PREFIX="${DATASET_GCS_PREFIX:-datasets}"
# Optional comma-separated paths relative to DATASET_GCS_PREFIX, e.g.
# "processed/CSD_satb,processed/Cantoria_satb". Empty means sync all datasets/.
export DATASET_GCS_PATHS="${DATASET_GCS_PATHS:-}"
export DATA_ROOT="${DATA_ROOT:-/gcs_data}"
export ARTIFACT_SYNC_SECONDS="${ARTIFACT_SYNC_SECONDS:-60}"
export BOOTSTRAP_CKPT_URI="${BOOTSTRAP_CKPT_URI:-}"
