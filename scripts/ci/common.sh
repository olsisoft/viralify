#!/bin/bash
# =============================================================================
# COMMON CI/CD FUNCTIONS
# Portable functions used across all CI scripts
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
require_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        exit 1
    fi
}

# Load environment variables from file
load_env() {
    local env_file="${1:-.env}"
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment from $env_file"
        set -a
        source "$env_file"
        set +a
    fi
}

# Get the list of services with Dockerfiles
get_services() {
    find services -maxdepth 2 -name "Dockerfile" -exec dirname {} \; | \
        sed 's|services/||' | sort
}

# Get modified services (for incremental builds)
get_modified_services() {
    local base_ref="${1:-origin/master}"
    local services=""

    # Get list of changed files
    local changed_files=$(git diff --name-only "$base_ref" HEAD 2>/dev/null || echo "")

    if [[ -z "$changed_files" ]]; then
        # No changes detected, return all services
        get_services
        return
    fi

    # Check each service for changes
    for service in $(get_services); do
        if echo "$changed_files" | grep -q "^services/$service/"; then
            echo "$service"
        fi
    done

    # Check if frontend changed
    if echo "$changed_files" | grep -q "^frontend/"; then
        echo "frontend"
    fi
}

# Build Docker image tag
build_image_tag() {
    local service="$1"
    local registry="${REGISTRY_URL:-ghcr.io/olsisoft/viralify}"
    local tag="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

    echo "${registry}/${service}:${tag}"
}

# Check if running in CI environment
is_ci() {
    [[ "${CI:-false}" == "true" ]] || \
    [[ -n "${GITHUB_ACTIONS:-}" ]] || \
    [[ -n "${GITLAB_CI:-}" ]] || \
    [[ -n "${JENKINS_URL:-}" ]]
}

# Get current git branch
get_branch() {
    git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
}

# Get current git commit SHA
get_commit_sha() {
    git rev-parse HEAD 2>/dev/null || echo "unknown"
}

# Get short commit SHA
get_short_sha() {
    git rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

# Retry command with exponential backoff
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-5}"
    local command="${@:3}"
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Attempt $attempt/$max_attempts: $command"
        if eval "$command"; then
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            log_warning "Command failed, retrying in ${delay}s..."
            sleep "$delay"
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done

    log_error "Command failed after $max_attempts attempts"
    return 1
}

# Export functions for use in subshells
export -f log_info log_success log_warning log_error
export -f require_command load_env get_services get_modified_services
export -f build_image_tag is_ci get_branch get_commit_sha get_short_sha retry
