# Backend image: CultPass, Ticket, Knowledge and Agent services, and the migrate and seed jobs.
# One image, different commands (see docker-compose.yml). Build context: the repository root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/solution:/app \
    UDAHUB_EVENT_LOG=/app/logs/events.jsonl

WORKDIR /app

COPY service/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY solution/agentic solution/agentic
# Models plus the committed SQLite files, which the seed job copies into Postgres
COPY solution/data solution/data
COPY service/__init__.py service/__init__.py
COPY service/common service/common
COPY service/cultpass_service service/cultpass_service
COPY service/ticket_service service/ticket_service
COPY service/knowledge_service service/knowledge_service
COPY service/agent_service service/agent_service
COPY service/scripts service/scripts
COPY service/migrations service/migrations
COPY service/alembic.ini service/alembic.ini

RUN useradd --create-home --uid 10001 app && mkdir -p /app/logs && chown app /app/logs
USER app

EXPOSE 8000
