#!/bin/bash
# =============================================================================
# TEST SCRIPT
# Run tests for all services
# Usage: ./scripts/ci/test.sh [service_name] [--coverage]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_FILTER="${1:-}"
COVERAGE_MODE=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --coverage)
            COVERAGE_MODE="--coverage"
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

log_info "Starting tests..."

# =============================================================================
# Python Tests
# =============================================================================
test_python_service() {
    local service="$1"
    local service_path="services/$service"

    if [[ ! -d "$service_path" ]]; then
        log_warning "Service not found: $service"
        return 1
    fi

    if [[ ! -d "$service_path/tests" ]]; then
        log_warning "No tests found for: $service"
        return 0
    fi

    log_info "Testing $service..."

    cd "$service_path"

    local pytest_args="-v --tb=short"

    if [[ -n "$COVERAGE_MODE" ]]; then
        pytest_args="$pytest_args --cov=. --cov-report=xml --cov-report=term-missing"
    fi

    # Run pytest
    python -m pytest tests/ $pytest_args || return $?

    cd "$PROJECT_ROOT"
}

# =============================================================================
# Frontend Tests
# =============================================================================
test_frontend() {
    log_info "Testing frontend..."

    if [[ ! -d "frontend" ]]; then
        log_warning "Frontend directory not found"
        return 0
    fi

    cd frontend

    # Install dependencies if needed
    if [[ ! -d "node_modules" ]]; then
        log_info "Installing frontend dependencies..."
        npm ci --silent
    fi

    # Run tests if they exist
    if grep -q '"test"' package.json; then
        if [[ -n "$COVERAGE_MODE" ]]; then
            npm run test -- --coverage --passWithNoTests || return $?
        else
            npm run test -- --passWithNoTests || return $?
        fi
    else
        log_warning "No test script found in frontend/package.json"
    fi

    cd "$PROJECT_ROOT"
}

# =============================================================================
# Main
# =============================================================================
main() {
    local failed_services=()
    local passed_count=0
    local failed_count=0

    if [[ -n "$SERVICE_FILTER" ]] && [[ "$SERVICE_FILTER" != "--coverage" ]]; then
        # Test specific service
        if [[ "$SERVICE_FILTER" == "frontend" ]]; then
            test_frontend || failed_services+=("frontend")
        else
            test_python_service "$SERVICE_FILTER" || failed_services+=("$SERVICE_FILTER")
        fi
    else
        # Test all services with tests
        local services_with_tests=(
            "course-generator"
            "presentation-generator"
            "vqv-hallu"
        )

        for service in "${services_with_tests[@]}"; do
            if test_python_service "$service"; then
                ((passed_count++))
            else
                failed_services+=("$service")
                ((failed_count++))
            fi
        done

        # Test frontend
        if test_frontend; then
            ((passed_count++))
        else
            failed_services+=("frontend")
            ((failed_count++))
        fi
    fi

    echo ""
    log_info "Test Summary: $passed_count passed, $failed_count failed"

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Failed services: ${failed_services[*]}"
        exit 1
    else
        log_success "All tests passed!"
        exit 0
    fi
}

main "$@"
