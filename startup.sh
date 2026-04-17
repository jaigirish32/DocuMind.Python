#!/bin/bash
APP_DIR=$(find /tmp -maxdepth 1 -name "8de*" -type d | head -1)
export PYTHONPATH=$APP_DIR:$PYTHONPATH
gunicorn --bind=0.0.0.0 --timeout 600 -k uvicorn.workers.UvicornWorker DocuMind.api.main:app