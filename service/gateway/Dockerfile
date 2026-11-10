# Gateway image: no agent code, no database drivers. Build context: the repository root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app

WORKDIR /app

COPY service/gateway/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY service/__init__.py service/__init__.py
COPY service/common service/common
COPY service/gateway service/gateway

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["uvicorn", "service.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
