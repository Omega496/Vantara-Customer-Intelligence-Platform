# Production Dockerfile for Vantara Customer Intelligence Platform
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for psycopg2 and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/processed /app/models_artifacts /app/docs /app/logs && \
    chown -R appuser:appuser /app

# Copy application codebase
COPY --chown=appuser:appuser . .

USER appuser

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Default entrypoint starts FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
