# The image viewer (deploy/ec2/docker-compose.yml: viewer-sync and viewer-app).
# Built from the repo root so the viewer can use the platform's packages.
FROM python:3.11-slim

ENV FIFTYONE_DO_NOT_TRACK=true \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY api api
COPY cli cli
COPY common common
COPY db db
COPY recipes recipes
COPY selection selection
COPY worker worker
COPY lambdas lambdas
COPY infra infra
COPY viewer viewer
RUN pip install --no-cache-dir . "fiftyone==1.22.1"

EXPOSE 5151
