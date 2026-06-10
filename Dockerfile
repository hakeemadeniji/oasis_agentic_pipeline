# OASIS Agentic Pipeline - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with Python and system dependencies
FROM python:3.14-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 oasis && \
    mkdir -p /app /app/data /app/logs && \
    chown -R oasis:oasis /app

WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Switch to non-root user
USER oasis

# Copy application code
COPY --chown=oasis:oasis . .

# Create necessary directories
RUN mkdir -p data/oasis_raw \
    data/processed_tensors \
    data/vector_store \
    data/batch_results \
    logs \
    .bob/tmp

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# For Streamlit dashboard: streamlit run src/orchestrator/dashboard_enhanced.py --server.port 8501 --server.address 0.0.0.0
# For batch processing: python src/api/batch_processor.py --csv data/patients.csv
