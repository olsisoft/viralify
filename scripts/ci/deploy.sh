#!/bin/bash
# =============================================================================
# DEPLOY SCRIPT
# Deploy to Kubernetes cluster (staging or production)
# Usage: ./scripts/ci/deploy.sh <environment> [--dry-run]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
ENVIRONMENT="${1:-staging}"
DRY_RUN="${DRY_RUN:-false}"
IMAGE_TAG="${IMAGE_TAG:-$(get_short_sha)}"
REGISTRY_URL="${REGISTRY_URL:-ghcr.io/olsisoft/viralify}"

# Server configuration (from environment or defaults)
STAGING_HOST="${STAGING_HOST:-}"
STAGING_USER="${STAGING_USER:-root}"
PRODUCTION_HOST="${PRODUCTION_HOST:-}"
PRODUCTION_USER="${PRODUCTION_USER:-root}"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN="true"
            ;;
        staging|production)
            ENVIRONMENT="$arg"
            ;;
    esac
done

cd "$PROJECT_ROOT"

log_info "Deploy configuration:"
log_info "  Environment: $ENVIRONMENT"
log_info "  Image Tag: $IMAGE_TAG"
log_info "  Dry Run: $DRY_RUN"

# =============================================================================
# Deployment Methods
# =============================================================================

# Deploy using SSH + Docker Compose (current setup)
deploy_docker_compose() {
    local host="$1"
    local user="$2"
    local env_name="$3"

    log_info "Deploying to $host via SSH..."

    if [[ -z "$host" ]]; then
        log_error "No host configured for $env_name"
        log_info "Set ${env_name^^}_HOST environment variable"
        exit 1
    fi

    local ssh_cmd="ssh -o StrictHostKeyChecking=no ${user}@${host}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute on $host:"
        log_info "  1. Pull latest images"
        log_info "  2. Run /rebuild.sh (main server)"
        log_info "  3. Run /setup-worker.sh (workers)"
        return 0
    fi

    # Execute deployment commands
    log_info "Pulling latest images..."
    $ssh_cmd "cd /opt/viralify && docker compose pull" || true

    log_info "Rebuilding main server..."
    $ssh_cmd "/rebuild.sh" || {
        log_error "Failed to run /rebuild.sh"
        return 1
    }

    log_info "Setting up workers..."
    $ssh_cmd "/setup-worker.sh" || {
        log_error "Failed to run /setup-worker.sh"
        return 1
    }

    log_success "Deployment to $host completed!"
}

# Deploy using Kubernetes
deploy_kubernetes() {
    local namespace="viralify-${ENVIRONMENT}"

    require_command kubectl

    log_info "Deploying to Kubernetes namespace: $namespace"

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    local kubectl_cmd="kubectl"
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl_cmd="kubectl --dry-run=client"
    fi

    # Update image tags in deployments
    local services=(
        "frontend"
        "api-gateway"
        "course-generator"
        "presentation-generator"
        "media-generator"
        "visual-generator"
        "vqv-hallu"
        "maestro-engine"
        "nexus-engine"
    )

    for service in "${services[@]}"; do
        local image="${REGISTRY_URL}/${service}:${IMAGE_TAG}"
        log_info "Updating $service to $image..."

        $kubectl_cmd -n "$namespace" set image "deployment/${service}" \
            "${service}=${image}" || log_warning "Failed to update $service"
    done

    # Wait for rollout
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Waiting for rollout to complete..."
        for service in "${services[@]}"; do
            kubectl -n "$namespace" rollout status "deployment/${service}" \
                --timeout=300s || log_warning "Rollout timeout for $service"
        done
    fi

    log_success "Kubernetes deployment completed!"
}

# Deploy using Helm
deploy_helm() {
    local namespace="viralify-${ENVIRONMENT}"
    local release_name="viralify-${ENVIRONMENT}"
    local values_file="k8s/values-${ENVIRONMENT}.yaml"

    require_command helm

    log_info "Deploying with Helm to namespace: $namespace"

    local helm_cmd="helm upgrade --install"
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_cmd="$helm_cmd --dry-run"
    fi

    $helm_cmd "$release_name" ./k8s/helm/viralify \
        --namespace "$namespace" \
        --create-namespace \
        --set image.tag="$IMAGE_TAG" \
        --set image.registry="$REGISTRY_URL" \
        -f "$values_file" || return $?

    log_success "Helm deployment completed!"
}

# =============================================================================
# Main
# =============================================================================
main() {
    case "$ENVIRONMENT" in
        staging)
            log_info "Deploying to STAGING..."
            deploy_docker_compose "$STAGING_HOST" "$STAGING_USER" "staging"
            ;;
        production)
            log_info "Deploying to PRODUCTION..."

            # Production safety check
            if [[ "$DRY_RUN" != "true" ]]; then
                log_warning "⚠️  You are about to deploy to PRODUCTION!"
                log_warning "Press Ctrl+C within 5 seconds to cancel..."
                sleep 5
            fi

            deploy_docker_compose "$PRODUCTION_HOST" "$PRODUCTION_USER" "production"
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        helm)
            deploy_helm
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Valid environments: staging, production, kubernetes, helm"
            exit 1
            ;;
    esac
}

main "$@"
