#!/bin/bash
# =============================================================================
# BUILD SCRIPT
# Build Docker images for services
# Usage: ./scripts/ci/build.sh [service_name] [--all] [--push]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration (can be overridden by environment)
REGISTRY_URL="${REGISTRY_URL:-ghcr.io/olsisoft/viralify}"
IMAGE_TAG="${IMAGE_TAG:-$(get_short_sha)}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_ALL="${BUILD_ALL:-false}"
PUSH_AFTER_BUILD="${PUSH_AFTER_BUILD:-false}"

cd "$PROJECT_ROOT"

# Parse arguments
SERVICE_FILTER=""
for arg in "$@"; do
    case $arg in
        --all)
            BUILD_ALL="true"
            ;;
        --push)
            PUSH_AFTER_BUILD="true"
            ;;
        --*)
            # Skip other flags
            ;;
        *)
            SERVICE_FILTER="$arg"
            ;;
    esac
done

log_info "Build configuration:"
log_info "  Registry: $REGISTRY_URL"
log_info "  Tag: $IMAGE_TAG"
log_info "  Platform: $PLATFORM"

# =============================================================================
# Build Functions
# =============================================================================

build_service() {
    local service="$1"
    local dockerfile_path=""
    local context_path=""
    local image_name="${REGISTRY_URL}/${service}:${IMAGE_TAG}"

    # Determine paths
    if [[ "$service" == "frontend" ]]; then
        dockerfile_path="frontend/Dockerfile"
        context_path="frontend"
    else
        dockerfile_path="services/${service}/Dockerfile"
        context_path="services/${service}"
    fi

    if [[ ! -f "$dockerfile_path" ]]; then
        log_warning "Dockerfile not found: $dockerfile_path"
        return 1
    fi

    log_info "Building $service -> $image_name"

    # Build with Docker Buildx for better caching
    local build_args=(
        "docker" "build"
        "-t" "$image_name"
        "-t" "${REGISTRY_URL}/${service}:latest"
        "-f" "$dockerfile_path"
        "--platform" "$PLATFORM"
    )

    # Add build args from environment
    if [[ -n "${BUILD_DATE:-}" ]]; then
        build_args+=("--build-arg" "BUILD_DATE=$BUILD_DATE")
    fi
    build_args+=("--build-arg" "GIT_COMMIT=$(get_short_sha)")
    build_args+=("--build-arg" "GIT_BRANCH=$(get_branch)")

    # Add cache configuration for CI
    if is_ci; then
        build_args+=("--cache-from" "type=gha")
        build_args+=("--cache-to" "type=gha,mode=max")
    fi

    build_args+=("$context_path")

    # Execute build
    "${build_args[@]}" || return $?

    log_success "Built: $image_name"

    # Push if requested
    if [[ "$PUSH_AFTER_BUILD" == "true" ]]; then
        log_info "Pushing $image_name..."
        docker push "$image_name" || return $?
        docker push "${REGISTRY_URL}/${service}:latest" || return $?
        log_success "Pushed: $image_name"
    fi

    return 0
}

# =============================================================================
# Main
# =============================================================================
main() {
    local services_to_build=()
    local built_count=0
    local failed_count=0

    # Determine which services to build
    if [[ -n "$SERVICE_FILTER" ]]; then
        services_to_build+=("$SERVICE_FILTER")
    elif [[ "$BUILD_ALL" == "true" ]]; then
        # Build all services with Dockerfiles
        while IFS= read -r service; do
            services_to_build+=("$service")
        done < <(get_services)
        services_to_build+=("frontend")
    else
        # Build only modified services
        while IFS= read -r service; do
            [[ -n "$service" ]] && services_to_build+=("$service")
        done < <(get_modified_services)
    fi

    if [[ ${#services_to_build[@]} -eq 0 ]]; then
        log_info "No services to build"
        exit 0
    fi

    log_info "Services to build: ${services_to_build[*]}"

    # Build each service
    for service in "${services_to_build[@]}"; do
        if build_service "$service"; then
            ((built_count++))
        else
            ((failed_count++))
            log_error "Failed to build: $service"
        fi
    done

    echo ""
    log_info "Build Summary: $built_count succeeded, $failed_count failed"

    if [[ $failed_count -gt 0 ]]; then
        exit 1
    fi

    log_success "All builds completed!"
}

main "$@"
