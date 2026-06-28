FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements-inference.txt

COPY api/ ./api/
COPY training/ ./training/

ENV PYTHONUNBUFFERED=1 \
    BIRDBRAIN_MODELS_CONFIG=api/models.yaml \
    BIRDBRAIN_LABELS_PATH=api/labels.json \
    BIRDBRAIN_DATA_DIR=data/raw/CUB_200_2011 \
    BIRDBRAIN_MAX_UPLOAD_MB=10

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
