# The worker image: every train, score and export job runs in the image built
# for its commit (ghcr.io/duytq5/duytq5-leaf-al-platform-worker:{sha}, built by
# the worker-image workflow on every merge to main). The locked torch wheel
# bundles CUDA 13, so the host needs only an NVIDIA driver for CUDA 13 and the
# NVIDIA container toolkit.
#
#   docker build -f worker/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) .

FROM python:3.13-slim-bookworm

# Links the GHCR package to this repo.
LABEL org.opencontainers.image.source=https://github.com/duytq5/duytq5-leaf-al-platform

RUN pip install --no-cache-dir uv==0.11.32

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # Pretrained weights land in the agent's cache volume, downloaded once.
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    WORKER_CACHE_DIR=/cache \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dependencies first, so a code-only change reuses this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra worker --no-install-project

COPY api ./api
COPY cli ./cli
COPY common ./common
COPY db ./db
COPY infra ./infra
COPY lambdas ./lambdas
COPY recipes ./recipes
COPY selection ./selection
COPY worker ./worker
RUN uv sync --locked --no-dev --extra worker

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

CMD ["python", "-m", "worker.run_job"]
