# =============================================================================
# VIRALIFY MAKEFILE
# Portable CI/CD interface - works locally and in any CI system
# =============================================================================

.PHONY: help install lint test build push deploy security clean

# Default registry and tag (can be overridden)
REGISTRY_URL ?= ghcr.io/olsisoft/viralify
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")
ENVIRONMENT ?= staging

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "$(BLUE)Viralify CI/CD Commands$(NC)"
	@echo "========================"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make install         Install dependencies"
	@echo "  make lint            Run linters (Python + TypeScript)"
	@echo "  make lint-fix        Run linters with auto-fix"
	@echo "  make test            Run all tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo ""
	@echo "$(GREEN)Build & Deploy:$(NC)"
	@echo "  make build           Build modified Docker images"
	@echo "  make build-all       Build ALL Docker images"
	@echo "  make push            Push images to registry"
	@echo "  make deploy-stg      Deploy to staging"
	@echo "  make deploy-prod     Deploy to production"
	@echo ""
	@echo "$(GREEN)Security:$(NC)"
	@echo "  make security        Run all security scans"
	@echo "  make security-images Scan Docker images only"
	@echo "  make security-code   Scan code only"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make clean           Clean build artifacts"
	@echo "  make docker-up       Start local Docker Compose"
	@echo "  make docker-down     Stop local Docker Compose"
	@echo "  make logs            View service logs"
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  REGISTRY_URL    Docker registry (default: ghcr.io/olsisoft/viralify)"
	@echo "  IMAGE_TAG       Image tag (default: git short SHA)"
	@echo "  ENVIRONMENT     Deploy target (default: staging)"
	@echo ""

# =============================================================================
# Development
# =============================================================================
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@cd frontend && npm ci
	@pip install ruff bandit pip-audit pytest pytest-cov pytest-asyncio

lint:
	@echo "$(BLUE)Running linters...$(NC)"
	@bash scripts/ci/lint.sh

lint-fix:
	@echo "$(BLUE)Running linters with auto-fix...$(NC)"
	@bash scripts/ci/lint.sh --fix

test:
	@echo "$(BLUE)Running tests...$(NC)"
	@bash scripts/ci/test.sh

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@bash scripts/ci/test.sh --coverage

test-service:
	@echo "$(BLUE)Testing service: $(SERVICE)$(NC)"
	@bash scripts/ci/test.sh $(SERVICE)

# =============================================================================
# Build
# =============================================================================
build:
	@echo "$(BLUE)Building modified images...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/build.sh

build-all:
	@echo "$(BLUE)Building ALL images...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/build.sh --all

build-service:
	@echo "$(BLUE)Building service: $(SERVICE)$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/build.sh $(SERVICE)

build-push:
	@echo "$(BLUE)Building and pushing images...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/build.sh --all --push

# =============================================================================
# Push
# =============================================================================
push:
	@echo "$(BLUE)Pushing images to registry...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/push.sh --all

push-service:
	@echo "$(BLUE)Pushing service: $(SERVICE)$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/push.sh $(SERVICE)

# =============================================================================
# Deploy
# =============================================================================
deploy-stg:
	@echo "$(BLUE)Deploying to STAGING...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/deploy.sh staging

deploy-prod:
	@echo "$(BLUE)Deploying to PRODUCTION...$(NC)"
	@REGISTRY_URL=$(REGISTRY_URL) IMAGE_TAG=$(IMAGE_TAG) bash scripts/ci/deploy.sh production

deploy-dry-run:
	@echo "$(BLUE)Deploy dry-run...$(NC)"
	@bash scripts/ci/deploy.sh $(ENVIRONMENT) --dry-run

# =============================================================================
# Security
# =============================================================================
security:
	@echo "$(BLUE)Running security scans...$(NC)"
	@bash scripts/ci/security-scan.sh

security-images:
	@echo "$(BLUE)Scanning Docker images...$(NC)"
	@bash scripts/ci/security-scan.sh --images

security-code:
	@echo "$(BLUE)Scanning code...$(NC)"
	@bash scripts/ci/security-scan.sh --code

security-deps:
	@echo "$(BLUE)Scanning dependencies...$(NC)"
	@bash scripts/ci/security-scan.sh --dependencies

# =============================================================================
# Docker Compose (Local)
# =============================================================================
docker-up:
	@echo "$(BLUE)Starting Docker Compose...$(NC)"
	@docker compose up -d

docker-down:
	@echo "$(BLUE)Stopping Docker Compose...$(NC)"
	@docker compose down

docker-logs:
	@docker compose logs -f $(SERVICE)

logs:
	@docker compose logs -f --tail=100

# =============================================================================
# Clean
# =============================================================================
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf coverage/ htmlcov/ .coverage.* 2>/dev/null || true
	@rm -rf frontend/.next frontend/node_modules/.cache 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

# =============================================================================
# CI Targets (called by GitHub Actions)
# =============================================================================
ci-lint: lint
ci-test: test
ci-build: build-all
ci-push: push
ci-deploy-staging: deploy-stg
ci-deploy-production: deploy-prod
ci-security: security
