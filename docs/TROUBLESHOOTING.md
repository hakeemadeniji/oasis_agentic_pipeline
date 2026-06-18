# Troubleshooting Guide

Comprehensive troubleshooting guide for the OASIS Agentic Pipeline covering common issues, diagnostics, and solutions.

## Table of Contents

1. [Getting Help](#getting-help)
2. [Common Issues](#common-issues)
3. [Diagnostics](#diagnostics)
4. [Performance Issues](#performance-issues)
5. [Deployment Issues](#deployment-issues)
6. [Data Issues](#data-issues)
7. [Security Issues](#security-issues)
8. [Recovery Procedures](#recovery-procedures)

## Getting Help

### Before Asking for Help

1. Check this troubleshooting guide
2. Review logs and error messages
3. Check system requirements
4. Verify configuration
5. Search existing issues

### Collecting Diagnostic Information

```bash
# System information
python --version
pip list
docker version
kubectl version

# Application logs
docker logs oasis-api
kubectl logs deployment/oasis-api

# System resources
free -h
df -h
top
```

### Support Channels

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check docs/ directory
- **Community**: Join our Discord/Slack
- **Email**: support@oasis-pipeline.com

## Common Issues

### Installation Issues

#### Python Version Incompatibility

**Problem**: `SyntaxError` or import errors after installation

**Solution**:
```bash
# Check Python version
python --version  # Should be 3.10+

# Upgrade Python if needed
# Use pyenv or conda to manage versions
pyenv install 3.10.12
pyenv local 3.10.12
```

#### Dependency Conflicts

**Problem**: `pip install` fails with dependency conflicts

**Solution**:
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# If conflicts persist, use pip-tools
pip install pip-tools
pip-compile requirements.txt
pip-sync
```

#### CUDA/GPU Issues

**Problem**: CUDA not available or GPU not detected

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Startup Issues

#### Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows

# Or use different port
export API_PORT=8001
```

#### Missing Environment Variables

**Problem**: Application fails with missing configuration

**Solution**:
```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env

# Verify variables are set
echo $API_HOST
echo $ANTHROPIC_API_KEY
```

#### Model Files Not Found

**Problem**: Model loading fails with file not found error

**Solution**:
```bash
# Check models directory
ls -la models/

# Download models if missing
python scripts/download_models.py

# Set correct model path
export MODEL_PATH=/path/to/models
```

### Runtime Issues

#### Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
export BATCH_SIZE=16

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use CPU instead
export DEVICE=cpu
```

#### Slow Performance

**Problem**: Diagnosis takes too long

**Solution**:
```bash
# Enable GPU
export DEVICE=cuda

# Enable caching
export ENABLE_CACHE=true

# Reduce image resolution
export IMAGE_SIZE=128

# Check system resources
htop
nvidia-smi
```

#### LLM Provider Timeout

**Problem**: LLM requests timing out

**Solution**:
```bash
# Increase timeout
export LLM_TIMEOUT=60

# Check API key
echo $ANTHROPIC_API_KEY

# Test connection
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY"
```

## Diagnostics

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/status

# Database connection
kubectl exec -it postgres-0 -- psql -U oasis_user -d oasis_pipeline -c "SELECT 1"
```

### Log Analysis

```bash
# View error logs
docker logs oasis-api 2>&1 | grep ERROR
kubectl logs deployment/oasis-api | grep ERROR

# View warning logs
docker logs oasis-api 2>&1 | grep WARNING

# Tail logs in real-time
docker logs -f oasis-api
kubectl logs -f deployment/oasis-api
```

### Performance Profiling

```python
# Enable profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run your code
result = your_function()

profiler.disable()
profiler.dump_stats('profile.prof')

# Analyze results
pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)
```

### Memory Profiling

```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Memory leak detection
import tracemalloc
tracemalloc.start()

# Run your code
your_function()

# Print memory snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Performance Issues

### High CPU Usage

**Diagnosis**:
```bash
# Check CPU usage
top
htop
kubectl top pods
```

**Solutions**:
1. Reduce number of concurrent requests
2. Implement caching
3. Optimize image preprocessing
4. Use GPU for inference
5. Scale horizontally

### High Memory Usage

**Diagnosis**:
```bash
# Check memory usage
free -h
docker stats
kubectl top pods
```

**Solutions**:
1. Reduce batch size
2. Clear cache periodically
3. Use memory-efficient data loading
4. Implement lazy loading
5. Restart application

### Slow API Response

**Diagnosis**:
```bash
# Check response time
time curl http://localhost:8000/health

# Check database queries
kubectl logs deployment/oasis-api | grep "slow query"
```

**Solutions**:
1. Enable caching
2. Optimize database queries
3. Use async processing
4. Implement connection pooling
5. Scale API servers

## Deployment Issues

### Docker Issues

#### Container Won't Start

**Diagnosis**:
```bash
# Check container logs
docker logs oasis-api

# Check container status
docker ps -a

# Inspect container
docker inspect oasis-api
```

**Solutions**:
1. Check Docker daemon is running
2. Verify image exists locally
3. Check port conflicts
4. Review environment variables
5. Check volume mounts

#### Image Build Fails

**Diagnosis**:
```bash
# Build with no cache
docker build --no-cache -t oasis-pipeline .

# Build with verbose output
docker build --progress=plain -t oasis-pipeline .
```

**Solutions**:
1. Check Docker daemon disk space
2. Verify network connectivity
3. Check base image availability
4. Review Dockerfile syntax
5. Clear Docker cache: `docker system prune -a`

### Kubernetes Issues

#### Pod Won't Start

**Diagnosis**:
```bash
# Check pod status
kubectl get pods
kubectl describe pod oasis-api-7d8f9b8c-x7k9p

# Check pod logs
kubectl logs oasis-api-7d8f9b8c-x7k9p

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

**Solutions**:
1. Check resource limits
2. Verify image pull secrets
3. Check node availability
4. Review pod configuration
5. Check service account permissions

#### Service Not Accessible

**Diagnosis**:
```bash
# Check service status
kubectl get services
kubectl describe service oasis-api-service

# Check endpoints
kubectl get endpoints oasis-api-service

# Test connectivity
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- curl http://localhost:8000/health
```

**Solutions**:
1. Check service selector
2. Verify pod labels
3. Check network policies
4. Review ingress configuration
5. Check DNS resolution

### Network Issues

#### Connection Refused

**Diagnosis**:
```bash
# Check if service is listening
netstat -tlnp | grep 8000

# Check firewall
sudo ufw status
sudo iptables -L

# Test connectivity
telnet localhost 8000
```

**Solutions**:
1. Check if application is running
2. Verify port binding
3. Check firewall rules
4. Review network configuration
5. Check service discovery

#### DNS Resolution Failure

**Diagnosis**:
```bash
# Check DNS resolution
nslookup oasis-api-service
kubectl exec -it oasis-api-7d8f9b8c-x7k9p -- nslookup redis

# Check CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns
```

**Solutions**:
1. Check CoreDNS pods
2. Verify service DNS names
3. Check network policies
4. Review kube-dns configuration
5. Check pod DNS configuration

## Data Issues

### Missing Data Files

**Diagnosis**:
```bash
# Check data directory
ls -la data/

# Check file permissions
ls -la data/oasis_raw/

# Verify data integrity
python scripts/validate_data.py
```

**Solutions**:
1. Download OASIS dataset
2. Extract data files
3. Set correct file permissions
4. Verify data structure
5. Update configuration paths

### Database Connection Issues

**Diagnosis**:
```bash
# Test database connection
psql -h localhost -U oasis_user -d oasis_pipeline

# Check database logs
kubectl logs postgres-0

# Check connection string
echo $DATABASE_URL
```

**Solutions**:
1. Verify database is running
2. Check connection string
3. Review network policies
4. Check credentials
5. Verify database schema

### Model Loading Issues

**Diagnosis**:
```bash
# Check model files
ls -lh models/

# Verify model integrity
python -c "import torch; torch.load('models/vision_model.pth')"

# Check model version
python scripts/check_model_version.py
```

**Solutions**:
1. Re-download models
2. Verify model format
3. Check PyTorch version compatibility
4. Review model configuration
5. Update model paths

## Security Issues

### Authentication Failures

**Diagnosis**:
```bash
# Test authentication
curl -X POST http://localhost:8000/api/auth/token \
  -d "username=admin&password=test"

# Check JWT configuration
echo $JWT_SECRET
echo $ACCESS_TOKEN_EXPIRE_MINUTES
```

**Solutions**:
1. Verify credentials
2. Check JWT secret
3. Review token expiration
4. Check user database
5. Verify authentication configuration

### Authorization Failures

**Diagnosis**:
```bash
# Check user roles
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Check role configuration
echo $ENABLE_AUTH
```

**Solutions**:
1. Verify user roles
2. Check RBAC configuration
3. Review endpoint permissions
4. Check token claims
5. Update user permissions

### Rate Limiting Issues

**Diagnosis**:
```bash
# Check rate limit headers
curl -I http://localhost:8000/health

# Check rate limit configuration
echo $ENABLE_RATE_LIMITING
echo $RATE_LIMIT_DEFAULT
```

**Solutions**:
1. Adjust rate limits
2. Increase quotas for specific users
3. Implement caching
4. Use API keys for higher limits
5. Review rate limit configuration

## Recovery Procedures

### Application Crash

**Steps**:
1. Collect logs and error information
2. Check system resources
3. Restart application
4. Monitor for recurrence
5. If recurring, investigate root cause

```bash
# Restart Docker container
docker restart oasis-api

# Restart Kubernetes deployment
kubectl rollout restart deployment/oasis-api
```

### Data Corruption

**Steps**:
1. Identify corrupted data
2. Restore from backup
3. Validate restored data
4. Investigate corruption cause
5. Implement preventive measures

```bash
# Restore from backup
docker run --rm \
  -v oasis-data:/data \
  -v ./backups:/backups \
  alpine tar xzf /backups/oasis-data-20260617.tar.gz -C /
```

### System Outage

**Steps**:
1. Identify affected services
2. Check infrastructure status
3. Restart critical services
4. Monitor recovery
5. Document incident

```bash
# Check all services
docker-compose ps
kubectl get pods -A

# Restart all services
docker-compose restart
kubectl rollout restart deployment --all
```

### Rollback Procedure

**Steps**:
1. Identify last stable version
2. Stop current deployment
3. Deploy previous version
4. Verify functionality
5. Monitor for issues

```bash
# Docker rollback
docker stop oasis-api
docker run -d --name oasis-api oasis-pipeline:v1.0.0

# Kubernetes rollback
kubectl rollout undo deployment/oasis-api
kubectl rollout status deployment/oasis-api
```

## Advanced Troubleshooting

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
export DEBUG=true

# Run with debug flags
python -m uvicorn src.api.main:app --reload --log-level debug
```

### Interactive Debugging

```python
# Add debugpy to requirements.txt
# Run with debugger
python -m debugpy --listen 5678 --wait-for-client -m uvicorn src.api.main:app
```

### Performance Analysis

```bash
# Profile application
python -m cProfile -o profile.prof -m uvicorn src.api.main:app
python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"
```

### Network Tracing

```bash
# Trace network calls
tcpdump -i any port 8000 -w trace.pcap

# Analyze with Wireshark
wireshark trace.pcap
```

When all else fails, collect comprehensive diagnostic information and contact support with:

- Error messages and stack traces
- System information
- Configuration details
- Logs
- Steps to reproduce the issue

This troubleshooting guide should help resolve most common issues. For persistent problems, please contact the support team with detailed diagnostic information.