# Dockerfile for TinyMetatron SLM (CPU-only) — Hugging Face Docker Space.
# Python 3.13-slim base per IMPLEMENTATION_CONTRACT.md section 2 / README.md.
FROM python:3.13-slim

WORKDIR /app

# Environment: UTF-8 stdio, import path, runtime paths, deploy mode.
# TMT_DEPLOY_MODE=demo -> read-only public Space (training/data-write disabled).
# TMT_DB_PATH / TMT_CHECKPOINT_DIR point at image-layer /app paths so the demo
# works with NO attached storage. Override both to a mounted /data volume for a
# stateful private deployment.
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    TMT_DB_PATH=/app/metatron.db \
    TMT_CHECKPOINT_DIR=/app/ckpt \
    TMT_DEPLOY_MODE=demo

# CPU-only PyTorch wheel (no CUDA runtime -> smaller image).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir fastapi==0.115.6 uvicorn[standard]==0.34.0 pydantic==2.10.4

# Copy the flat repo (no package) into /app.
COPY . /app

# Ensure runtime directories for data and checkpoints exist.
RUN mkdir -p /app/data /app/ckpt

# Build-time training: seed a small corpus and train a checkpoint so the Space
# ships with a real active model in /app/ckpt + an active row in /app/metatron.db.
# /generate finds the active checkpoint at runtime with no extra config.
# (~20-40s on a CPU builder for a 6.35M model; no large file committed to the repo.)
RUN python manage_data.py generate --domain cybersecurity --count 400 \
        --db_path /app/metatron.db \
    && python train_db.py --steps 300 --domain cybersecurity \
        --min_quality 0.4 --db_path /app/metatron.db --checkpoint_dir /app/ckpt

# Hugging Face Docker Spaces expect the app on port 7860 (app_port in README).
EXPOSE 7860

# Serve the FastAPI app (module-level `app` in api.py) on all interfaces.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]