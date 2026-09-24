#!/usr/bin/env bash
# Usage: serving/start-decider.sh [model] [port]
# The graph-grid defaults below keep decider-2b inside 8 GB of VRAM (see docs/spike-notes.md); override via env.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="$here/.venv/bin/uvicorn"; [ -x "$bin" ] || bin="$here/.venv/Scripts/uvicorn.exe"
export DECIDER_GRAPH_TOKEN_BUDGET="${DECIDER_GRAPH_TOKEN_BUDGET:-8192}"
export DECIDER_T_BUCKETS="${DECIDER_T_BUCKETS:-256,512,768,1024,1536,2048,3072,4096,6144,8192}"
export DECIDER_B_BUCKETS="${DECIDER_B_BUCKETS:-1,4,16}"
DECIDER_MODEL="${1:-Mapika/decider-2b}" exec "$bin" decider.serve:app --host 127.0.0.1 --port "${2:-8000}"
