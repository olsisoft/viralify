#!/bin/bash
# =============================================================================
# SECURITY SCAN SCRIPT
# Run security scans on code and Docker images
# Usage: ./scripts/ci/security-scan.sh [--images] [--code] [--dependencies]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
SCAN_IMAGES="${SCAN_IMAGES:-true}"
SCAN_CODE="${SCAN_CODE:-true}"
SCAN_DEPS="${SCAN_DEPS:-true}"
SEVERITY_THRESHOLD="${SEVERITY_THRESHOLD:-HIGH}"
REGISTRY_URL="${REGISTRY_URL:-ghcr.io/olsisoft/viralify}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

cd "$PROJECT_ROOT"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --images)
            SCAN_IMAGES="true"
            SCAN_CODE="false"
            SCAN_DEPS="false"
            ;;
        --code)
            SCAN_CODE="true"
            SCAN_IMAGES="false"
            SCAN_DEPS="false"
            ;;
        --dependencies)
            SCAN_DEPS="true"
            SCAN_IMAGES="false"
            SCAN_CODE="false"
            ;;
    esac
done

log_info "Security scan configuration:"
log_info "  Scan Images: $SCAN_IMAGES"
log_info "  Scan Code: $SCAN_CODE"
log_info "  Scan Dependencies: $SCAN_DEPS"
log_info "  Severity Threshold: $SEVERITY_THRESHOLD"

# =============================================================================
# Image Scanning (Trivy)
# =============================================================================
scan_images() {
    log_info "Scanning Docker images for vulnerabilities..."

    if ! command -v trivy &> /dev/null; then
        log_warning "Trivy not installed. Installing..."
        # Install trivy
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi

    local exit_code=0
    local services=(
        "frontend"
        "api-gateway"
        "course-generator"
        "presentation-generator"
        "media-generator"
    )

    for service in "${services[@]}"; do
        local image="${REGISTRY_URL}/${service}:${IMAGE_TAG}"
        log_info "Scanning $image..."

        trivy image \
            --severity "$SEVERITY_THRESHOLD,CRITICAL" \
            --exit-code 1 \
            --ignore-unfixed \
            "$image" || {
                log_warning "Vulnerabilities found in $image"
                exit_code=1
            }
    done

    return $exit_code
}

# =============================================================================
# Code Scanning (Bandit for Python, ESLint security for JS)
# =============================================================================
scan_code() {
    log_info "Scanning code for security issues..."

    local exit_code=0

    # Python security scan with Bandit
    if command -v bandit &> /dev/null; then
        log_info "Running Bandit (Python security)..."
        bandit -r services/ \
            -ll \
            --exclude "*/tests/*,*/.venv/*" \
            -f json \
            -o bandit-report.json || {
                log_warning "Bandit found security issues"
                exit_code=1
            }
    else
        log_warning "Bandit not installed (pip install bandit)"
    fi

    # Check for secrets with gitleaks
    if command -v gitleaks &> /dev/null; then
        log_info "Running Gitleaks (secret detection)..."
        gitleaks detect --source . --verbose || {
            log_error "Secrets detected in code!"
            exit_code=1
        }
    else
        log_warning "Gitleaks not installed"
    fi

    return $exit_code
}

# =============================================================================
# Dependency Scanning
# =============================================================================
scan_dependencies() {
    log_info "Scanning dependencies for vulnerabilities..."

    local exit_code=0

    # Python dependencies with pip-audit
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit..."
        for req_file in $(find services -name "requirements.txt"); do
            log_info "Checking $req_file..."
            pip-audit -r "$req_file" --ignore-vuln GHSA-xxxx || {
                log_warning "Vulnerabilities in $req_file"
                exit_code=1
            }
        done
    else
        log_warning "pip-audit not installed (pip install pip-audit)"
    fi

    # Node.js dependencies with npm audit
    if [[ -d "frontend" ]]; then
        log_info "Running npm audit..."
        cd frontend
        npm audit --audit-level=high || {
            log_warning "Vulnerabilities in npm dependencies"
            exit_code=1
        }
        cd "$PROJECT_ROOT"
    fi

    return $exit_code
}

# =============================================================================
# Main
# =============================================================================
main() {
    local overall_exit=0

    if [[ "$SCAN_IMAGES" == "true" ]]; then
        scan_images || overall_exit=1
    fi

    if [[ "$SCAN_CODE" == "true" ]]; then
        scan_code || overall_exit=1
    fi

    if [[ "$SCAN_DEPS" == "true" ]]; then
        scan_dependencies || overall_exit=1
    fi

    echo ""
    if [[ $overall_exit -eq 0 ]]; then
        log_success "All security scans passed!"
    else
        log_error "Security issues found. Please review and fix."
        exit 1
    fi
}

main "$@"
