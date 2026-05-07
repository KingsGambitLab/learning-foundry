FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates dumb-init \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN pip install --no-cache-dir \
    "fastapi>=0.136.1,<0.137.0" \
    "uvicorn>=0.46.0,<0.47.0"

WORKDIR /workspace

EXPOSE 8080 8000

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "--user-data-dir", "/tmp/code-server", "/workspace"]
