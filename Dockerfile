# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY service.py ./

# Create wheels from dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r <(grep -v "^#" <(grep dependencies -A 100 pyproject.toml | sed '1d') | sed 's/^ *"//' | sed 's/".*//' | grep -E "bentoml|pandas|scikit-learn|lightgbm|xgboost|catboost|fastapi|numpy|pydantic|python-dotenv")

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels

# Copy application code and config
COPY service.py bento.yaml ./
COPY pyproject.toml ./

# Install Python dependencies from wheels and project
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    pip install --no-cache-dir -e . 2>&1 | grep -v "Running setup.py"

# Create non-root user for security
RUN useradd -m -u 1000 bentoml && chown -R bentoml:bentoml /app
USER bentoml

# Expose BentoML API port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:3000/healthz')" || exit 1

# Set environment variables
ENV BENTOML_HOME=/app/.bentoml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run BentoML service
CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
