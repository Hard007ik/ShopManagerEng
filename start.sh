#!/bin/bash
set -e

# Start the OpenEnv API server in the background (port 8000)
cd /app/env
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for the API to be ready
echo "[start] Waiting for API server on port 8000..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "[start] API server ready."
        break
    fi
    sleep 1
done

# Start Streamlit UI (port 7860 — HF Spaces default)
echo "[start] Launching Streamlit UI on port 7860..."
exec streamlit run ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#7c3aed" \
    --theme.backgroundColor "#0f0c29" \
    --theme.secondaryBackgroundColor "#302b63" \
    --theme.textColor "#e2e8f0"
