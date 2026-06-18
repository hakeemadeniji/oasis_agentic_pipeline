# Deployment Guide

Comprehensive deployment guide for the OASIS Agentic Pipeline covering Docker, Kubernetes, and various deployment scenarios.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Deployment](#local-development-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Scaling Strategies](#scaling-strategies)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum for models and data
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- kubectl (for Kubernetes deployment)
- Helm 3.x (optional)
- Git

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/oasis-agentic-pipeline.git
cd oasis-agentic-pipeline

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

## Local Development Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Redis: localhost:6379
```

### Development Mode

```bash
# Start development environment with hot reload
docker-compose --profile development up

# Run tests
docker-compose exec api pytest

# Access shell in container
docker-compose exec api bash
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop api
```

## Docker Deployment

### Building Images

```bash
# Build production image
docker build --target production -t oasis-pipeline:latest .

# Build GPU-enabled image
docker build --target gpu -t oasis-pipeline:gpu .

# Build development image
docker build --target development -t oasis-pipeline:dev .
```

### Running Containers

```bash
# Run production container
docker run -d \
  --name oasis-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  oasis-pipeline:latest

# Run with GPU support
docker run -d \
  --name oasis-api-gpu \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  oasis-pipeline:gpu
```

### Docker Compose Production

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=4
```

### Health Checks

```bash
# Check container health
docker ps --filter name=oasis-api

# Check health endpoint
curl http://localhost:8000/health

# View container logs
docker logs oasis-api
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Creating Namespace

```bash
# Create namespace
kubectl create namespace oasis-pipeline

# Set default namespace
kubectl config set-context --current --namespace=oasis-pipeline
```

### Deploying Application

```bash
# Apply all configurations
kubectl apply -f k8s/

# Deploy specific components
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
```

### Using Helm Chart

```bash
# Create Helm chart
helm create oasis-chart k8s/helm

# Install chart
helm install oasis oasis-chart \
  --namespace oasis-pipeline \
  --set image.tag=latest \
  --set resources.requests.memory=2Gi \
  --set replicaCount=3

# Upgrade chart
helm upgrade oasis oasis-chart \
  --namespace oasis-pipeline \
  --set image.tag=v1.0.1

# Uninstall chart
helm uninstall oasis --namespace oasis-pipeline
```

### Managing Deployments

```bash
# Check deployment status
kubectl get deployments
kubectl get pods

# View logs
kubectl logs -f deployment/oasis-api

# Scale deployment
kubectl scale deployment oasis-api --replicas=5

# Rollback deployment
kubectl rollout undo deployment/oasis-api

# Check rollout status
kubectl rollout status deployment/oasis-api
```

### ConfigMaps and Secrets

```bash
# Create ConfigMap from file
kubectl create configmap oasis-config \
  --from-file=config.yaml \
  --namespace=oasis-pipeline

# Create Secret from file
kubectl create secret generic oasis-secrets \
  --from-file=.env \
  --namespace=oasis-pipeline

# View ConfigMap
kubectl get configmap oasis-config -o yaml

# View Secret (decoded)
kubectl get secret oasis-secrets -o yaml
```

## Cloud Deployment

### AWS Deployment

#### EKS (Elastic Kubernetes Service)

```bash
# Install eksctl
curl --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name oasis-cluster \
  --region us-west-2 \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5

# Deploy application
kubectl apply -f k8s/
```

#### EC2 with Docker

```bash
# Launch EC2 instance with Docker AMI
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy application
git clone https://github.com/your-org/oasis-agentic-pipeline.git
cd oasis-agentic-pipeline
docker-compose up -d
```

### Google Cloud Platform (GCP)

#### GKE (Google Kubernetes Engine)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create GKE cluster
gcloud container clusters create oasis-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5

# Get cluster credentials
gcloud container clusters get-credentials oasis-cluster \
  --zone us-central1-a

# Deploy application
kubectl apply -f k8s/
```

### Azure Deployment

#### AKS (Azure Kubernetes Service)

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create AKS cluster
az aks create \
  --resource-group oasis-rg \
  --name oasis-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5 \
  --generate-ssh-keys

# Get cluster credentials
az aks get-credentials \
  --resource-group oasis-rg \
  --name oasis-cluster

# Deploy application
kubectl apply -f k8s/
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'oasis-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboards

```bash
# Access Grafana
http://localhost:3000

# Default credentials
Username: admin
Password: admin

# Add Prometheus data source
Configuration → Data Sources → Add Prometheus
URL: http://prometheus:9090

# Import dashboard
Dashboards → Import → Dashboard ID (e.g., 1860 for Node Exporter)
```

### Log Aggregation

```bash
# View logs from all pods
kubectl logs -l app=oasis-api --all-containers=true

# Stream logs
kubectl logs -f deployment/oasis-api

# View logs for specific pod
kubectl logs oasis-api-7d8f9b8c-x7k9p
```

### Health Monitoring

```bash
# Check health endpoint
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- curl http://localhost:8000/health

# Monitor pod health
kubectl get pods -w

# Check resource usage
kubectl top nodes
kubectl top pods
```

## Scaling Strategies

### Horizontal Pod Autoscaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: oasis-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: oasis-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

### Database Scaling

```yaml
# Read replicas
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: postgres
        env:
        - name: POSTGRES_REPLICATION_MODE
          value: replica
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs oasis-api
kubectl logs deployment/oasis-api

# Check resource limits
kubectl describe pod oasis-api-7d8f9b8c-x7k9p

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

#### High Memory Usage

```bash
# Check memory usage
docker stats
kubectl top pods

# Restart container
docker restart oasis-api
kubectl rollout restart deployment/oasis-api
```

#### Connection Issues

```bash
# Check network connectivity
docker network inspect oasis-network
kubectl get services
kubectl describe service oasis-api-service

# Check DNS resolution
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- nslookup redis
```

#### Performance Issues

```bash
# Check resource usage
kubectl top nodes
kubectl top pods

# Check pod metrics
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- top

# Check application logs for errors
kubectl logs deployment/oasis-api --tail=100 | grep ERROR
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/oasis-api LOG_LEVEL=debug

# Access container shell
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- /bin/bash

# Run diagnostic commands
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- python -m pytest
```

## Backup and Recovery

### Data Backup

```bash
# Backup volumes
docker run --rm \
  -v oasis-data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar czf /backups/oasis-data-$(date +%Y%m%d).tar.gz /data

# Backup PostgreSQL
kubectl exec -it postgres-0 -- pg_dump -U oasis_user oasis_pipeline > backup.sql
```

### Disaster Recovery

```bash
# Restore volumes
docker run --rm \
  -v oasis-data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar xzf /backups/oasis-data-20260617.tar.gz -C /

# Restore PostgreSQL
kubectl exec -i postgres-0 -- psql -U oasis_user oasis_pipeline < backup.sql
```

## Security Best Practices

1. **Use Secrets Management**: Never commit secrets to git
2. **Enable RBAC**: Configure proper Kubernetes RBAC
3. **Network Policies**: Implement network segmentation
4. **Image Scanning**: Scan images for vulnerabilities
5. **Regular Updates**: Keep dependencies updated
6. **Audit Logging**: Enable audit logging for compliance

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/oasis-api \
  oasis-api=oasis-pipeline:v1.1.0

# Monitor rollout
kubectl rollout status deployment/oasis-api

# Rollback if needed
kubectl rollout undo deployment/oasis-api
```

### Certificate Management

```bash
# Update TLS certificates
kubectl create secret tls oasis-tls \
  --cert=path/to/cert.crt \
  --key=path/to/cert.key \
  --namespace=oasis-pipeline
```

### Resource Cleanup

```bash
# Clean up old Docker images
docker image prune -a

# Clean up unused resources
kubectl delete pods --field-selector=status.phase=Succeeded
kubectl delete pods --field-selector=status.phase=Failed
```

This deployment guide provides comprehensive instructions for deploying the OASIS Agentic Pipeline in various environments. For specific deployment scenarios or issues, refer to the troubleshooting documentation or contact the support team.