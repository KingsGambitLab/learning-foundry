#!/bin/bash
set -euo pipefail

# Run the widget injection (no-op if LAB_TUTOR_BASE_URL is unset).
python3 /usr/local/bin/lab-tutor-inject.py || true

# Hand off to dumb-init with the original code-server command, allowing
# the launcher's CMD (passed via docker run) to take effect.
exec /usr/bin/dumb-init -- "$@"
