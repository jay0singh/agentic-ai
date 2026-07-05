#!/usr/bin/env bash
# Hugging Face Spaces entrypoint: run the API privately on localhost and
# expose only the Streamlit UI on the Space's public port.
set -e

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export API_BASE="http://127.0.0.1:8000"

uvicorn api:app --host 127.0.0.1 --port 8000 &

# CORS/XSRF must be off behind Hugging Face's reverse proxy, otherwise the
# browser's file-upload requests are rejected with 403.
exec streamlit run streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
