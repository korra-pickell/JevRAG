param([string]$Model = "Mapika/decider-2b", [int]$Port = 8000)
# The graph-grid defaults below keep decider-2b inside 8 GB of VRAM (see docs/spike-notes.md); override via env.
if (-not $env:DECIDER_GRAPH_TOKEN_BUDGET) { $env:DECIDER_GRAPH_TOKEN_BUDGET = "8192" }
if (-not $env:DECIDER_T_BUCKETS) { $env:DECIDER_T_BUCKETS = "256,512,768,1024,1536,2048,3072,4096,6144,8192" }
if (-not $env:DECIDER_B_BUCKETS) { $env:DECIDER_B_BUCKETS = "1,4,16" }
$env:DECIDER_MODEL = $Model
& "$PSScriptRoot\.venv\Scripts\uvicorn.exe" decider.serve:app --host 127.0.0.1 --port $Port
