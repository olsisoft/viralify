#!/bin/bash
# =============================================================================
# PUSH SCRIPT
# Push Docker images to registry
# Usage: ./scripts/ci/push.sh [service_name] [--all]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
REGISTRY_URL="${REGISTRY_URL:-ghcr.io/olsisoft/viralify}"
IMAGE_TAG="${IMAGE_TAG:-$(get_short_sha)}"
PUSH_ALL="${PUSH_ALL:-false}"

cd "$PROJECT_ROOT"

# Parse arguments
SERVICE_FILTER=""
for arg in "$@"; do
    case $arg in
        --all)
            PUSH_ALL="true"
            ;;
        --*)
            ;;
        *)
            SERVICE_FILTER="$arg"
            ;;
    esac
done

log_info "Push configuration:"
log_info "  Registry: $REGISTRY_URL"
log_info "  Tag: $IMAGE_TAG"

# =============================================================================
# Push Functions
# =============================================================================

push_service() {
    local service="$1"
    local image_name="${REGISTRY_URL}/${service}:${IMAGE_TAG}"
    local latest_tag="${REGISTRY_URL}/${service}:latest"

    # Check if image exists locally
    if ! docker image inspect "$image_name" &> /dev/null; then
        log_warning "Image not found locally: $image_name"
        return 1
    fi

    log_info "Pushing $image_name..."

    # Push with tag
    retry 3 5 "docker push $image_name" || return $?

    # Push latest tag
    docker tag "$image_name" "$latest_tag"
    retry 3 5 "docker push $latest_tag" || return $?

    log_success "Pushed: $image_name"
    return 0
}

# =============================================================================
# Registry Login
# =============================================================================

login_registry() {
    log_info "Logging into registry..."

    # GitHub Container Registry
    if [[ "$REGISTRY_URL" == ghcr.io/* ]]; then
        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
        elif [[ -n "${CR_PAT:-}" ]]; then
            echo "$CR_PAT" | docker login ghcr.io -u "${GITHUB_USERNAME:-olsisoft}" --password-stdin
        else
            log_warning "No GitHub token found, assuming already logged in"
        fi
    # Docker Hub
    elif [[ "$REGISTRY_URL" == docker.io/* ]] || [[ ! "$REGISTRY_URL" == */* ]]; then
        if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
            echo "$DOCKER_PASSWORD" | docker login -u "${DOCKER_USERNAME:-}" --password-stdin
        fi
    # AWS ECR
    elif [[ "$REGISTRY_URL" == *.dkr.ecr.*.amazonaws.com/* ]]; then
        aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" | \
            docker login --username AWS --password-stdin "$REGISTRY_URL"
    # Google Container Registry
    elif [[ "$REGISTRY_URL" == gcr.io/* ]] || [[ "$REGISTRY_URL" == *.gcr.io/* ]]; then
        gcloud auth configure-docker --quiet
    fi

    log_success "Registry login successful"
}

# =============================================================================
# Main
# =============================================================================
main() {
    local services_to_push=()
    local pushed_count=0
    local failed_count=0

    # Login to registry
    login_registry

    # Determine which services to push
    if [[ -n "$SERVICE_FILTER" ]]; then
        services_to_push+=("$SERVICE_FILTER")
    elif [[ "$PUSH_ALL" == "true" ]]; then
        while IFS= read -r service; do
            services_to_push+=("$service")
        done < <(get_services)
        services_to_push+=("frontend")
    else
        while IFS= read -r service; do
            [[ -n "$service" ]] && services_to_push+=("$service")
        done < <(get_modified_services)
    fi

    if [[ ${#services_to_push[@]} -eq 0 ]]; then
        log_info "No services to push"
        exit 0
    fi

    log_info "Services to push: ${services_to_push[*]}"

    # Push each service
    for service in "${services_to_push[@]}"; do
        if push_service "$service"; then
            ((pushed_count++))
        else
            ((failed_count++))
        fi
    done

    echo ""
    log_info "Push Summary: $pushed_count succeeded, $failed_count failed"

    if [[ $failed_count -gt 0 ]]; then
        exit 1
    fi

    log_success "All images pushed!"
}

main "$@"
