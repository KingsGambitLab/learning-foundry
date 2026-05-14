# syntax=docker/dockerfile:1.6

# --- Stage: build the lab-tutor extension ---------------------------------
FROM node:20-bookworm-slim AS lab-tutor-build
WORKDIR /build
# Dependency layer — cached independently of source edits
COPY extensions/lab-tutor/package.json extensions/lab-tutor/package-lock.json ./
RUN npm ci --no-audit --no-fund
# Source layer
COPY extensions/lab-tutor/ ./
RUN npm run package \
 && test -f /build/lab-tutor.vsix

# --- Final image ----------------------------------------------------------
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates dumb-init nodejs npm \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN pip install --no-cache-dir \
    "fastapi>=0.136.1,<0.137.0" \
    "uvicorn>=0.46.0,<0.47.0"

COPY --from=lab-tutor-build /build/lab-tutor.vsix /opt/lab-tutor/lab-tutor.vsix
RUN mkdir -p /opt/lab-tutor/extensions \
 && code-server --extensions-dir /opt/lab-tutor/extensions \
                --install-extension /opt/lab-tutor/lab-tutor.vsix

WORKDIR /workspace

EXPOSE 8080 8000

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "--user-data-dir", "/tmp/code-server", "/workspace"]
