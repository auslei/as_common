#!/usr/bin/env bash
set -euo pipefail

# Unified Docker production build script for Innovation-Child services.
# Intended to be invoked from a project root where a Dockerfile lives,
# with ic_common added as a git submodule.
#
# Usage:
#   ./ic_common/scripts/build_docker_prod.sh IMAGE_TAG [OPTIONS]
#
# Arguments:
#   IMAGE_TAG             Docker image name and tag, e.g. ic_vectordb:latest (required)
#
# Options:
#   --multiplatform       Build for linux/amd64 AND linux/arm64.
#                         Saves as ./images/{name}-amd64.tar and ./images/{name}-arm64.tar.
#                         Without this flag, builds only the native platform and saves as
#                         ./images/{name}.tar.
#   --build-arg K=V       Pass a build argument to docker build (repeatable).
#   --dockerfile PATH     Dockerfile to use (default: Dockerfile).
#   --target TARGET       Build stage target (default: production).
#
# GPU:
#   NVIDIA GPU is auto-detected on Linux via nvidia-smi.
#   When detected, --gpus all is added to the sanity-test container run.
#   Build-time CUDA args (e.g. INSTALL_CUDA) must be passed via --build-arg.
#
# Sanity test:
#   The native-platform image is loaded into the local Docker daemon, a container
#   is started, and we verify it either stays running or exits with an expected
#   config-missing code (1 or 3).  The image is left loaded; the test container
#   is always removed.

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
IMAGE_TAG=""
MULTIPLATFORM=false
BUILD_ARGS=()
DOCKERFILE="Dockerfile"
BUILD_TARGET="production"

if [[ $# -ge 1 && "$1" != --* ]]; then
    IMAGE_TAG="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multiplatform)
            MULTIPLATFORM=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS+=("--build-arg" "$2")
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: build_docker_prod.sh IMAGE_TAG [--multiplatform] [--build-arg K=V] [--dockerfile PATH] [--target TARGET]"
            exit 1
            ;;
    esac
done

if [[ -z "${IMAGE_TAG}" ]]; then
    echo "Error: IMAGE_TAG is required."
    echo "Usage: build_docker_prod.sh IMAGE_TAG [--multiplatform] [--build-arg K=V]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

# Strip registry prefix and tag to get a bare name for tar filenames.
# e.g. "auslei/investor_iq_core_app:latest" -> "investor_iq_core_app"
IMAGE_BASENAME="${IMAGE_TAG%%:*}"
IMAGE_BASENAME="${IMAGE_BASENAME##*/}"

# Exit codes that indicate a missing runtime config, not a broken image.
#   1 = generic Python/RuntimeError (e.g. missing config.yaml)
#   3 = uvicorn application startup failed
EXPECTED_EXIT_CODES=(1 3)

# ---------------------------------------------------------------------------
# GPU detection (affects sanity-test runtime, not build time)
# ---------------------------------------------------------------------------
GPU_FLAGS=()
if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — sanity test will run with --gpus all."
    GPU_FLAGS=("--gpus" "all")
fi

# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------
case "$(uname -m)" in
    x86_64)        NATIVE_PLATFORM="linux/amd64" ;;
    arm64|aarch64) NATIVE_PLATFORM="linux/arm64" ;;
    *)
        echo "Unsupported host architecture: $(uname -m)"
        exit 1
        ;;
esac

if ${MULTIPLATFORM}; then
    PLATFORMS=("linux/amd64" "linux/arm64")
else
    PLATFORMS=("${NATIVE_PLATFORM}")
fi

# ---------------------------------------------------------------------------
# Buildx builder
# ---------------------------------------------------------------------------
if ! docker buildx inspect multiplatform-builder &>/dev/null; then
    echo "Creating multiplatform-builder..."
    docker buildx create --name multiplatform-builder --use
else
    docker buildx use multiplatform-builder
fi

mkdir -p ./images

# ---------------------------------------------------------------------------
# Build each platform and save to tar
# ---------------------------------------------------------------------------
NATIVE_TAR=""

for PLATFORM in "${PLATFORMS[@]}"; do
    ARCH="${PLATFORM#*/}"   # amd64 or arm64

    if ${MULTIPLATFORM}; then
        TAR_PATH="./images/${IMAGE_BASENAME}-${ARCH}.tar"
    else
        TAR_PATH="./images/${IMAGE_BASENAME}.tar"
    fi

    echo "========================================"
    echo "Building ${IMAGE_TAG} for ${PLATFORM}..."
    echo "========================================"

    docker buildx build \
        --platform "${PLATFORM}" \
        --target "${BUILD_TARGET}" \
        -f "${DOCKERFILE}" \
        "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
        -t "${IMAGE_TAG}" \
        --output "type=docker,dest=${TAR_PATH}" \
        .

    echo "Saved → ${TAR_PATH}"

    if [[ "${PLATFORM}" == "${NATIVE_PLATFORM}" ]]; then
        NATIVE_TAR="${TAR_PATH}"
    fi
done

# ---------------------------------------------------------------------------
# Sanity test — load native image and verify the container starts
# ---------------------------------------------------------------------------
echo "========================================"
echo "Sanity testing ${IMAGE_TAG} (${NATIVE_PLATFORM})..."
echo "========================================"

echo "Loading ${NATIVE_TAR}..."
docker load -i "${NATIVE_TAR}"

# Do NOT use --rm so we can retrieve logs after a crash.
CONTAINER_ID=$(docker run -d "${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"}" "${IMAGE_TAG}")

echo "Waiting 5s for container ${CONTAINER_ID} to initialise..."
sleep 5

if docker ps -q --no-trunc | grep -q "${CONTAINER_ID}"; then
    echo "Container is running — sanity test passed."
    docker stop "${CONTAINER_ID}"
else
    EXIT_CODE=$(docker inspect "${CONTAINER_ID}" --format='{{.State.ExitCode}}')
    if [[ " ${EXPECTED_EXIT_CODES[*]} " == *" ${EXIT_CODE} "* ]]; then
        echo "Container exited with code ${EXIT_CODE} (config/env not supplied — expected in isolated build). Image OK."
    else
        echo "ERROR: Container crashed with exit code ${EXIT_CODE} — sanity test FAILED."
        echo "--- Container logs ---"
        docker logs "${CONTAINER_ID}"
        docker rm "${CONTAINER_ID}"
        exit 1
    fi
fi

docker rm "${CONTAINER_ID}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "========================================"
echo "Done. Image(s) ready under ./images/:"
for PLATFORM in "${PLATFORMS[@]}"; do
    ARCH="${PLATFORM#*/}"
    if ${MULTIPLATFORM}; then
        echo "  ./images/${IMAGE_BASENAME}-${ARCH}.tar"
    else
        echo "  ./images/${IMAGE_BASENAME}.tar"
    fi
done
echo "========================================"
