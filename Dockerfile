# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/data/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/data/cache/sentence-transformers

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
# Install CPU-only torch first to avoid pulling ~7GB of CUDA wheels as a
# transitive dep of sentence-transformers. sentence-transformers will then
# accept the already-installed torch instead of resolving the GPU build.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
 && uv pip install --python /opt/venv/bin/python -r pyproject.toml

COPY . .

RUN useradd --create-home --uid 1000 runner \
 && mkdir -p /app/data/raw /app/data/cache /app/results/cache /app/results/plots /app/results/predictions \
 && chown -R runner:runner /app /opt/venv
USER runner

CMD ["python", "-m", "eval.run_eval", "--methods", "knn"]