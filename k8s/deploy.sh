#!/bin/bash
# Viralify Kubernetes Deployment Script
# Usage: ./deploy.sh [create|update|delete]

set -e

NAMESPACE="viralify"
ACTION=${1:-create}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_info "Prerequisites OK"
}

# Build and push Docker images
build_images() {
    log_info "Building Docker images..."

    # List of services to build
    SERVICES=(
        "frontend"
        "api-gateway"
        "course-generator"
        "presentation-generator"
        "media-generator"
        "visual-generator"
        "diagrams-generator"
        "maestro-engine"
        "vqv-hallu"
    )

    for service in "${SERVICES[@]}"; do
        log_info "Building viralify/$service..."
        docker build -t viralify/$service:latest ./services/$service 2>/dev/null || \
        docker build -t viralify/$service:latest ./$service 2>/dev/null || \
        log_warn "Could not build $service (may not have Dockerfile)"
    done

    log_info "Images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    log_warn "Make sure you're logged into your container registry"
    log_warn "Example: docker login registry.digitalocean.com"

    SERVICES=(
        "frontend"
        "api-gateway"
        "course-generator"
        "presentation-generator"
        "media-generator"
        "visual-generator"
        "diagrams-generator"
        "maestro-engine"
        "vqv-hallu"
    )

    for service in "${SERVICES[@]}"; do
        docker push viralify/$service:latest || log_warn "Could not push $service"
    done
}

# Create namespace and apply configs
deploy_base() {
    log_info "Creating namespace and base configs..."

    kubectl apply -f namespace/
    kubectl apply -f configmaps/
    kubectl apply -f secrets/
    kubectl apply -f storage/

    log_info "Base configuration applied"
}

# Deploy databases
deploy_databases() {
    log_info "Deploying databases..."

    kubectl apply -f databases/

    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=60s || true
    kubectl wait --for=condition=ready pod -l app=rabbitmq -n $NAMESPACE --timeout=60s || true

    log_info "Databases deployed"
}

# Deploy application services
deploy_services() {
    log_info "Deploying application services..."

    kubectl apply -f services/

    log_info "Waiting for services to be ready..."
    sleep 10

    # Wait for critical services
    kubectl wait --for=condition=ready pod -l app=course-generator -n $NAMESPACE --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=presentation-generator -n $NAMESPACE --timeout=120s || true

    log_info "Services deployed"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."

    # Check if nginx-ingress is installed
    if ! kubectl get ns ingress-nginx &> /dev/null; then
        log_warn "Installing nginx-ingress controller..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/do/deploy.yaml
        sleep 30
    fi

    # Check if cert-manager is installed
    if ! kubectl get ns cert-manager &> /dev/null; then
        log_warn "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        sleep 30
    fi

    kubectl apply -f ingress/

    log_info "Ingress deployed"
}

# Deploy autoscaling
deploy_hpa() {
    log_info "Deploying autoscaling policies..."
    kubectl apply -f hpa/
    log_info "HPA deployed"
}

# Full deployment
deploy_all() {
    check_prerequisites
    deploy_base
    deploy_databases
    deploy_services
    deploy_ingress
    deploy_hpa

    log_info "=========================================="
    log_info "Deployment complete!"
    log_info "=========================================="
    log_info ""
    log_info "Check status with:"
    log_info "  kubectl get pods -n $NAMESPACE"
    log_info "  kubectl get svc -n $NAMESPACE"
    log_info "  kubectl get ingress -n $NAMESPACE"
    log_info ""
    log_info "View logs with:"
    log_info "  kubectl logs -f deployment/presentation-generator -n $NAMESPACE"
}

# Update deployment
update_deployment() {
    log_info "Updating deployment..."
    kubectl apply -f configmaps/
    kubectl apply -f services/
    kubectl apply -f hpa/

    # Restart deployments to pick up changes
    kubectl rollout restart deployment -n $NAMESPACE

    log_info "Deployment updated"
}

# Delete deployment
delete_deployment() {
    log_warn "This will delete all Viralify resources!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deleting deployment..."
        kubectl delete -f hpa/ --ignore-not-found
        kubectl delete -f ingress/ --ignore-not-found
        kubectl delete -f services/ --ignore-not-found
        kubectl delete -f databases/ --ignore-not-found
        kubectl delete -f storage/ --ignore-not-found
        kubectl delete -f secrets/ --ignore-not-found
        kubectl delete -f configmaps/ --ignore-not-found
        kubectl delete namespace $NAMESPACE --ignore-not-found
        log_info "Deployment deleted"
    else
        log_info "Cancelled"
    fi
}

# Show status
show_status() {
    log_info "Viralify Status"
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n $NAMESPACE -o wide
    echo ""
    echo "=== Services ==="
    kubectl get svc -n $NAMESPACE
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n $NAMESPACE
    echo ""
    echo "=== HPA ==="
    kubectl get hpa -n $NAMESPACE
    echo ""
    echo "=== PVC ==="
    kubectl get pvc -n $NAMESPACE
}

# Main
case $ACTION in
    create)
        deploy_all
        ;;
    update)
        update_deployment
        ;;
    delete)
        delete_deployment
        ;;
    status)
        show_status
        ;;
    build)
        build_images
        ;;
    push)
        push_images
        ;;
    *)
        echo "Usage: $0 [create|update|delete|status|build|push]"
        echo ""
        echo "Commands:"
        echo "  create  - Full deployment (namespace, databases, services, ingress, hpa)"
        echo "  update  - Update services and configmaps"
        echo "  delete  - Delete all Viralify resources"
        echo "  status  - Show deployment status"
        echo "  build   - Build Docker images locally"
        echo "  push    - Push images to registry"
        exit 1
        ;;
esac
