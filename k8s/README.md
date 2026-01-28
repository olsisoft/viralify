# Viralify Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Viralify to a Kubernetes cluster.

## Prerequisites

1. **Kubernetes Cluster** (DigitalOcean, AWS EKS, GKE, etc.)
2. **kubectl** configured to access your cluster
3. **Docker Registry** for storing images (Docker Hub, DigitalOcean Container Registry, etc.)

## Quick Start

### 1. Create DigitalOcean Kubernetes Cluster

```bash
# Install doctl
brew install doctl  # or snap install doctl

# Authenticate
doctl auth init

# Create cluster (Toronto region, 2 nodes with 4 vCPUs each)
doctl kubernetes cluster create viralify-cluster \
  --region tor1 \
  --node-pool "name=workers;size=s-4vcpu-8gb;count=2"

# Configure kubectl
doctl kubernetes cluster kubeconfig save viralify-cluster
```

### 2. Update Secrets

Edit `secrets/app-secrets.yaml` and replace all `CHANGE_ME_` values with your actual credentials:

```bash
# Edit the file
nano secrets/app-secrets.yaml

# Or use sed to replace (example)
sed -i 's/CHANGE_ME_sk-xxx/sk-your-actual-key/g' secrets/app-secrets.yaml
```

### 3. Update Domain

Edit `ingress/ingress.yaml` and replace `olsitec.com` with your domain.

### 4. Build and Push Images

```bash
# Login to your registry
docker login registry.digitalocean.com

# Build images
./deploy.sh build

# Tag for your registry
docker tag viralify/presentation-generator:latest registry.digitalocean.com/your-registry/presentation-generator:latest

# Push
docker push registry.digitalocean.com/your-registry/presentation-generator:latest
```

### 5. Deploy

```bash
chmod +x deploy.sh
./deploy.sh create
```

## Directory Structure

```
k8s/
├── namespace/          # Namespace definition
├── configmaps/         # Non-sensitive configuration
├── secrets/            # API keys, passwords (edit before deploying!)
├── storage/            # PersistentVolumeClaims
├── databases/          # PostgreSQL, Redis, RabbitMQ
├── services/           # Application deployments
├── ingress/            # External routing + TLS
├── hpa/                # Horizontal Pod Autoscaler
├── deploy.sh           # Deployment script
└── README.md           # This file
```

## Commands

```bash
# Full deployment
./deploy.sh create

# Update services (after code changes)
./deploy.sh update

# Check status
./deploy.sh status

# Delete everything
./deploy.sh delete

# View logs
kubectl logs -f deployment/presentation-generator -n viralify

# Scale manually
kubectl scale deployment presentation-generator --replicas=3 -n viralify

# View resource usage
kubectl top pods -n viralify

# Enter a pod
kubectl exec -it deployment/presentation-generator -n viralify -- bash
```

## Scaling

The HPA (Horizontal Pod Autoscaler) automatically scales these services:

| Service | Min | Max | CPU Target |
|---------|-----|-----|------------|
| presentation-generator | 1 | 5 | 70% |
| media-generator | 1 | 3 | 70% |
| course-generator | 1 | 3 | 80% |
| frontend | 2 | 5 | 70% |
| api-gateway | 2 | 5 | 70% |

## Storage

PersistentVolumeClaims use DigitalOcean Block Storage (`do-block-storage`):

| PVC | Size | Purpose |
|-----|------|---------|
| postgres-pvc | 20Gi | Database |
| redis-pvc | 5Gi | Cache/sessions |
| videos-pvc | 50Gi | Generated videos |
| presentations-pvc | 50Gi | Slides, assets |
| diagrams-pvc | 10Gi | Generated diagrams |

## SSL/TLS

The ingress uses cert-manager with Let's Encrypt for automatic SSL certificates.

Make sure to:
1. Update the email in `ingress/ingress.yaml`
2. Point your domain's DNS to the LoadBalancer IP

```bash
# Get LoadBalancer IP
kubectl get svc -n ingress-nginx
```

## Monitoring

For production, consider adding:

```bash
# Prometheus + Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# View Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod <pod-name> -n viralify
kubectl logs <pod-name> -n viralify --previous
```

### Database connection issues
```bash
# Check if postgres is ready
kubectl exec -it statefulset/postgres -n viralify -- pg_isready

# Check connection from another pod
kubectl exec -it deployment/course-generator -n viralify -- \
  python -c "import asyncpg; print('OK')"
```

### Storage issues
```bash
kubectl get pvc -n viralify
kubectl describe pvc <pvc-name> -n viralify
```

## Cost Estimation (DigitalOcean)

| Resource | Spec | Monthly Cost |
|----------|------|--------------|
| Kubernetes Control Plane | Free | $0 |
| Worker Node 1 | s-4vcpu-8gb | ~$48 |
| Worker Node 2 | s-4vcpu-8gb | ~$48 |
| Block Storage | ~200GB | ~$20 |
| Load Balancer | 1 | ~$12 |
| **Total** | | **~$128/month** |

This replaces a single large server and provides better reliability with auto-scaling.
