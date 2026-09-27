# syntax=docker/dockerfile:1
# The Project Agent's service image (agent-project#1): private on Docker Hub as
# oramasearch/agent-project. The retrieval corpus is a named build context, `corpus`, pointing at
# agent-project-corpus/corpus, so the image carries the precedent it cannot run without.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home agent
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=corpus project-sft-v1.jsonl /app/corpus/project-sft-v1.jsonl
ENV PATH=/app/.venv/bin:$PATH \
    PROJECT_CORPUS=/app/corpus/project-sft-v1.jsonl \
    TRACE_DIR=/tmp/agent-project-traces \
    PORT=8080
USER 10001
EXPOSE 8080
ENTRYPOINT ["agent-project-serve"]
