#!/bin/bash
# Azure App Service / Oryx: when SCM_DO_BUILD_DURING_DEPLOYMENT=true,
# the application code is extracted to /tmp/<uid>/ at runtime, NOT
# /home/site/wwwroot. The path is exposed via the APP_PATH env var.
# The antenv venv lives at $APP_PATH/antenv.
#
# /home/site/wwwroot only contains: output.tar.zst, oryx-manifest.toml,
# requirements.txt, hostingstart.html, and a stub antenv directory.
# DO NOT cd there — DocuMind/ won't be found.

set -e

# Resolve the application directory
if [ -n "$APP_PATH" ] && [ -d "$APP_PATH" ]; then
    APP_DIR="$APP_PATH"
    echo "[startup] Using APP_PATH from env: $APP_DIR"
else
    # Fallback: find antenv dynamically and use its parent
    ANTENV_PATH=$(find /tmp -maxdepth 2 -name "antenv" -type d 2>/dev/null | head -1)
    if [ -z "$ANTENV_PATH" ]; then
        echo "[startup] FATAL: APP_PATH not set and no antenv found in /tmp"
        echo "[startup] Contents of /tmp:"
        ls /tmp/ 2>/dev/null | head -20
        exit 1
    fi
    APP_DIR=$(dirname "$ANTENV_PATH")
    echo "[startup] APP_PATH not set, derived APP_DIR: $APP_DIR"
fi

ANTENV="$APP_DIR/antenv"

if [ ! -d "$ANTENV" ]; then
    echo "[startup] FATAL: antenv not found at $ANTENV"
    echo "[startup] Contents of $APP_DIR:"
    ls -la "$APP_DIR" 2>/dev/null
    exit 1
fi

cd "$APP_DIR"

echo "[startup] APP_DIR contents:"
ls -la "$APP_DIR" | head -25

if [ ! -d "$APP_DIR/DocuMind" ]; then
    echo "[startup] FATAL: DocuMind/ folder not found at $APP_DIR"
    echo "[startup] Did the build complete successfully? Check Oryx build logs."
    exit 1
fi

# Activate the venv
export PATH="$ANTENV/bin:$PATH"
export PYTHONPATH="$APP_DIR:$PYTHONPATH"

echo "[startup] gunicorn binary: $(which gunicorn)"
echo "[startup] python version: $(python --version 2>&1)"
echo "[startup] PYTHONPATH: $PYTHONPATH"

# Pre-validate the import. Catches Python-level errors before gunicorn workers
# spawn — gives a clean traceback instead of cryptic "worker failed to boot".
echo "[startup] Validating app import..."
python -c "from DocuMind.api.main import app; print('[startup] App import OK')" || {
    echo "[startup] FATAL: Cannot import DocuMind.api.main"
    exit 1
}

echo "[startup] Launching gunicorn..."

exec gunicorn \
    --bind=0.0.0.0:8000 \
    --timeout 600 \
    --workers 2 \
    --access-logfile '-' \
    --error-logfile '-' \
    -k uvicorn.workers.UvicornWorker \
    DocuMind.api.main:app