# Build:  docker build -t aifsi .
# Run:    docker run -p 8000:8000 -v aifsi-data:/app/data aifsi      then open http://localhost:8000
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AIFSI_DB=/app/data/aifsi.db
WORKDIR /app

# CPU-only PyTorch first (much smaller than the default GPU build), then the rest.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train the model while building, so the image starts instantly and needs no setup.
RUN python ml/train.py

RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app
VOLUME /app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
