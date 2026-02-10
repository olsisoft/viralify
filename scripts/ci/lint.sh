#!/bin/bash
# =============================================================================
# LINT SCRIPT
# Run linters for Python and TypeScript code
# Usage: ./scripts/ci/lint.sh [--fix]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIX_MODE="${1:-}"

cd "$PROJECT_ROOT"

log_info "Starting linting..."

# =============================================================================
# Python Linting
# =============================================================================
lint_python() {
    log_info "Linting Python code..."

    local python_dirs="services"
    local exit_code=0

    # Check if ruff is available (faster than flake8)
    if command -v ruff &> /dev/null; then
        log_info "Using ruff for Python linting"
        if [[ "$FIX_MODE" == "--fix" ]]; then
            ruff check "$python_dirs" --fix || exit_code=$?
            ruff format "$python_dirs" || exit_code=$?
        else
            ruff check "$python_dirs" || exit_code=$?
            ruff format --check "$python_dirs" || exit_code=$?
        fi
    elif command -v flake8 &> /dev/null; then
        log_info "Using flake8 for Python linting"
        flake8 "$python_dirs" \
            --max-line-length=120 \
            --extend-ignore=E501,W503 \
            --exclude=__pycache__,*.pyc,.git,venv,.venv \
            || exit_code=$?
    else
        log_warning "No Python linter found (install ruff or flake8)"
    fi

    return $exit_code
}

# =============================================================================
# TypeScript/JavaScript Linting
# =============================================================================
lint_typescript() {
    log_info "Linting TypeScript/JavaScript code..."

    local exit_code=0

    if [[ -d "frontend" ]]; then
        cd frontend

        # Install dependencies if needed
        if [[ ! -d "node_modules" ]]; then
            log_info "Installing frontend dependencies..."
            npm ci --silent
        fi

        # ESLint
        if [[ -f ".eslintrc.json" ]] || [[ -f "eslint.config.mjs" ]]; then
            log_info "Running ESLint..."
            if [[ "$FIX_MODE" == "--fix" ]]; then
                npx eslint . --ext .ts,.tsx,.js,.jsx --fix || exit_code=$?
            else
                npx eslint . --ext .ts,.tsx,.js,.jsx || exit_code=$?
            fi
        fi

        # TypeScript type checking
        log_info "Running TypeScript type check..."
        npx tsc --noEmit --skipLibCheck || exit_code=$?

        cd ..
    fi

    return $exit_code
}

# =============================================================================
# Main
# =============================================================================
main() {
    local python_exit=0
    local ts_exit=0

    lint_python || python_exit=$?
    lint_typescript || ts_exit=$?

    if [[ $python_exit -eq 0 ]] && [[ $ts_exit -eq 0 ]]; then
        log_success "All linting passed!"
        exit 0
    else
        log_error "Linting failed (Python: $python_exit, TypeScript: $ts_exit)"
        exit 1
    fi
}

main "$@"
