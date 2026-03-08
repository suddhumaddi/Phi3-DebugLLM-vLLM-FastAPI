# =============================================================
# DebugLLM Backend Service - Dockerfile
# =============================================================
# Multi-stage build for production FastAPI backend.
# Stage 1: Install dependencies
# Stage 2: Minimal runtime image

# --- Stage 1: Dependency builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Production runtime ---
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY deploy/backend_app.py ./backend_app.py
COPY config/model_config.yaml ./config/model_config.yaml

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
