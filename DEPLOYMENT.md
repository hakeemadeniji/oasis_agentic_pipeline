# OASIS Agentic Pipeline - Deployment Guide

**Version:** 1.0.0  
**Last Updated:** June 10, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Scaling & Performance](#scaling--performance)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Security Considerations](#security-considerations)

---

## Overview

This guide covers deployment options for the OASIS Agentic Pipeline, from local development to production environments. The system supports multiple deployment strategies:

- **Local Development**: Direct Python execution
- **Docker**: Containerized single-node deployment
- **Docker Compose**: Multi-service orchestration
- **Kubernetes**: Production-grade orchestration (optional)

### Architecture Components

- **API Service**: FastAPI backend (port 8000)
- **Dashboard**: Streamlit frontend (port 8501)
- **Monitoring**: Prometheus + Grafana (ports 9090, 3000)
- **Logging**: Loki + Promtail (port 3100)
- **Cache**: Redis (port 6379)
- **Proxy**: Nginx (ports 80, 443)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB
- OS: Linux, macOS, or Windows with WSL2

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: NVIDIA with 6GB+ VRAM (optional, for faster inference)

### Software Requirements

- **Python**: 3.11+ (3.14 recommended)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.30+

---

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/oasis-agentic-pipeline.git
cd oasis-agentic-pipeline
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file:

```bash
# Application
APP_NAME=oasis-pipeline
LOG_LEVEL=INFO
WORKERS=4

# Paths
DATA_DIR=./data
LOG_DIR=./logs

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 5. Prepare Data

```bash
# Create data directories
mkdir -p data/oasis_raw data/processed_tensors data/vector_store data/batch_results

# Download OASIS dataset (if not already present)
# Place MRI images in data/oasis_raw/[class_name]/
# Place CSV files in data/oasis_raw/
```

### 6. Run Services

**API Server:**
```bash
python src/api/main.py
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Dashboard:**
```bash
streamlit run src/orchestrator/dashboard_enhanced.py
```

**Access:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t oasis-pipeline:latest .
```

### 2. Run API Container

```bash
docker run -d \
  --name oasis-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e LOG_LEVEL=INFO \
  oasis-pipeline:latest
```

### 3. Run Dashboard Container

```bash
docker run -d \
  --name oasis-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  oasis-pipeline:latest \
  streamlit run src/orchestrator/dashboard_enhanced.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Verify Deployment

```bash
# Check container status
docker ps

# View logs
docker logs oasis-api
docker logs oasis-dashboard

# Test API
curl http://localhost:8000/health
```

---

## Production Deployment

### Using Docker Compose

#### 1. Configure Environment

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    environment:
      - LOG_LEVEL=WARNING
      - WORKERS=8
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G

  dashboard:
    deploy:
      replicas: 2
```

#### 2. Deploy Stack

```bash
# Start all services
docker-compose up -d

# Start with production overrides
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale API service
docker-compose up -d --scale api=3
```

#### 3. Verify Deployment

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### Using Kubernetes (Optional)

#### 1. Create Kubernetes Manifests

**Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oasis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oasis-api
  template:
    metadata:
      labels:
        app: oasis-api
    spec:
      containers:
      - name: api
        image: oasis-pipeline:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: oasis-api-service
spec:
  selector:
    app: oasis-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 2. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/oasis-api
```

---

## Monitoring & Logging

### Prometheus Metrics

**Access:** http://localhost:9090

**Key Metrics:**
- `oasis_api_requests_total` - Total API requests
- `oasis_diagnoses_total` - Total diagnoses
- `oasis_agent_executions_total` - Agent executions
- `oasis_diagnosis_confidence` - Confidence scores

**Example Queries:**
```promql
# Request rate
rate(oasis_api_requests_total[5m])

# Error rate
rate(oasis_api_requests_total{status=~"5.."}[5m])

# Average diagnosis confidence
avg(oasis_diagnosis_confidence)

# Agent execution duration (95th percentile)
histogram_quantile(0.95, rate(oasis_agent_execution_duration_seconds_bucket[5m]))
```

### Grafana Dashboards

**Access:** http://localhost:3000  
**Default Credentials:** admin / oasis_admin_2026

**Pre-configured Dashboards:**
1. **System Overview**: Overall health and performance
2. **API Metrics**: Request rates, latencies, errors
3. **Diagnosis Analytics**: Confidence scores, approval rates
4. **Agent Performance**: Execution times, success rates

### Loki Logs

**Access:** http://localhost:3100

**Query Examples:**
```logql
# All API logs
{job="oasis-api"}

# Error logs only
{job="oasis-api"} |= "ERROR"

# Diagnosis events
{job="oasis-api-json"} | json | event_type="diagnosis"

# High latency requests
{job="oasis-api-json"} | json | duration_ms > 2000
```

### Alerting

Alerts are configured in `monitoring/alerting-rules.yml`:

- **Critical**: API down, model not loaded
- **Warning**: High error rate, slow responses, low confidence

**Alert Channels:**
- Slack (configure webhook in docker-compose.yml)
- Email (configure SMTP settings)
- PagerDuty (optional)

---

## Scaling & Performance

### Horizontal Scaling

**Docker Compose:**
```bash
# Scale API service
docker-compose up -d --scale api=5

# Scale with load balancer
docker-compose -f docker-compose.yml -f docker-compose.lb.yml up -d
```

**Kubernetes:**
```bash
# Manual scaling
kubectl scale deployment oasis-api --replicas=5

# Auto-scaling
kubectl autoscale deployment oasis-api --min=3 --max=10 --cpu-percent=70
```

### Performance Optimization

**1. Enable GPU Acceleration:**
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**2. Optimize Worker Count:**
```bash
# Calculate optimal workers
workers = (2 * CPU_cores) + 1

# Set in environment
export WORKERS=9  # For 4-core system
```

**3. Enable Caching:**
```python
# Use Redis for caching
REDIS_ENABLED=true
REDIS_TTL=3600
```

**4. Batch Processing:**
```bash
# Use parallel processing
python src/api/batch_processor.py --csv data.csv --parallel --workers 8
```

### Load Balancing

**Nginx Configuration:**
```nginx
upstream oasis_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://oasis_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Backup & Recovery

### Data Backup

**1. Automated Backup Script:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/oasis-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/data.tar.gz data/

# Backup logs
tar -czf $BACKUP_DIR/logs.tar.gz logs/

# Backup database (if applicable)
# pg_dump oasis_db > $BACKUP_DIR/database.sql

echo "Backup completed: $BACKUP_DIR"
```

**2. Schedule with Cron:**
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### Disaster Recovery

**1. Restore from Backup:**
```bash
# Stop services
docker-compose down

# Restore data
tar -xzf backup/data.tar.gz -C ./

# Restart services
docker-compose up -d
```

**2. Database Recovery:**
```bash
# Restore database
psql oasis_db < backup/database.sql
```

---

## Troubleshooting

### Common Issues

#### 1. API Not Starting

**Symptoms:** Container exits immediately

**Solutions:**
```bash
# Check logs
docker logs oasis-api

# Verify dependencies
pip install -r requirements.txt

# Check port availability
netstat -an | grep 8000
```

#### 2. Out of Memory

**Symptoms:** Container killed, OOM errors

**Solutions:**
```bash
# Increase memory limit
docker-compose up -d --scale api=1 --memory=8g

# Monitor memory usage
docker stats

# Optimize batch size
export MAX_BATCH_SIZE=50
```

#### 3. Slow Inference

**Symptoms:** High latency, timeouts

**Solutions:**
```bash
# Enable GPU
docker run --gpus all ...

# Reduce batch size
# Increase worker count
# Enable model caching
```

#### 4. Connection Refused

**Symptoms:** Cannot connect to API

**Solutions:**
```bash
# Check firewall
sudo ufw allow 8000

# Verify network
docker network inspect oasis-network

# Check service status
docker-compose ps
```

### Debug Mode

Enable debug logging:
```bash
# Environment variable
export LOG_LEVEL=DEBUG

# Docker
docker run -e LOG_LEVEL=DEBUG ...

# View detailed logs
docker-compose logs -f --tail=100 api
```

---

## Security Considerations

### 1. Authentication

**Implement API Key Authentication:**
```python
# Add to main.py
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/diagnose")
async def diagnose(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    # ...
```

### 2. HTTPS/TLS

**Configure SSL in Nginx:**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://api:8000;
    }
}
```

### 3. Data Encryption

**Encrypt sensitive data:**
```bash
# Encrypt data directory
gpg --encrypt --recipient admin@example.com data.tar.gz

# Decrypt
gpg --decrypt data.tar.gz.gpg > data.tar.gz
```

### 4. Network Security

**Firewall Rules:**
```bash
# Allow only necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Block direct API access
```

### 5. Secrets Management

**Use Docker Secrets:**
```yaml
services:
  api:
    secrets:
      - api_key
      - db_password

secrets:
  api_key:
    file: ./secrets/api_key.txt
  db_password:
    file: ./secrets/db_password.txt
```

---

## CI/CD Integration

### GitHub Actions

The project includes automated CI/CD pipeline (`.github/workflows/ci-cd.yml`):

**Triggers:**
- Push to `main` or `develop`
- Pull requests
- Release tags

**Pipeline Stages:**
1. **Lint**: Code quality checks
2. **Test**: Unit and integration tests
3. **Security**: Vulnerability scanning
4. **Build**: Docker image creation
5. **Deploy**: Staging/production deployment

**Manual Deployment:**
```bash
# Trigger deployment
gh workflow run ci-cd.yml -f environment=production
```

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor system health
- Check error logs
- Verify backup completion

**Weekly:**
- Review performance metrics
- Update dependencies
- Clean old logs

**Monthly:**
- Security updates
- Capacity planning
- Performance optimization

### Update Procedure

```bash
# 1. Pull latest changes
git pull origin main

# 2. Rebuild images
docker-compose build

# 3. Rolling update
docker-compose up -d --no-deps --build api

# 4. Verify health
curl http://localhost:8000/health

# 5. Rollback if needed
docker-compose up -d --no-deps api:previous-tag
```

---

## Support

For issues and questions:

- **Documentation**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/your-org/oasis-pipeline/issues
- **Email**: support@example.com

---

**Last Updated:** June 10, 2026  
**Version:** 1.0.0  
**Maintained by:** OASIS Development Team
