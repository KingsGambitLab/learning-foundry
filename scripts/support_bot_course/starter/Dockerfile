FROM python:3.11-bookworm

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────
# DO NOT EDIT THIS BLOCK. It is Docker-layer-cached: it only rebuilds
# when requirements.txt changes. Editing this (or requirements.txt)
# forces a multi-minute reinstall + model re-download on EVERY
# submission. The full toolset + the MiniLM model are pre-installed.
# ─────────────────────────────────────────────────────────────────────
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
# ─────────────────────────────────────────────────────────────────────

COPY . .

RUN chmod +x .coursegen/runtime/install.sh .coursegen/runtime/verify.sh .coursegen/runtime/run.sh 2>/dev/null || true

# Unbuffered stdout so request logs are captured (good container hygiene;
# also required by the extended observability assignment).
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["./.coursegen/runtime/run.sh"]
