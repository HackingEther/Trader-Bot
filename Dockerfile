FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir . 

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000 9090

CMD ["sh", "-c", "uvicorn trader.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
