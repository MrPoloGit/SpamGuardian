# ── SpamGuardian – inference microservice (CPU) ──────────────────────────────
# For GPU support swap the base image for nvidia/cuda:<version>-runtime-ubuntu22.04
# and replace requirements-inference.txt with a +cuXXX torch wheel.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app.py .

# Model weights are mounted at runtime; create the directory so the mount point exists.
RUN mkdir -p /app/model

EXPOSE 8000

# 1 worker keeps the model in a single process; raise if you have enough RAM
# to load multiple copies. Timeout is generous because DistilBERT cold-starts slowly.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
