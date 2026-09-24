# Serving decider locally

JevRAGRank's benchmarks talk to a [decider](https://github.com/Mapika/decider) server over the Jev wire API (`POST /v1/systemone`).

## Port convention

| model | port |
|---|---|
| `Mapika/decider-2b` | 8000 (default) |
| `Mapika/decider-0.8b` | 8001 (model-size ablation) |

On an 8 GB GPU, run only one at a time.

## Native (Windows or Linux), recommended

```bash
uv venv serving/.venv --python cpython-3.12.13 --managed-python   # not an anaconda base: its vcruntime breaks torch on Windows
uv pip install --python serving/.venv "decider-ai[serve]" --torch-backend cu126
uv pip install --python serving/.venv triton-windows              # Windows only: enables the flash-linear-attention kernels
```

Start the server from Git Bash:

```bash
bash serving/start-decider.sh                          # decider-2b on 127.0.0.1:8000
bash serving/start-decider.sh Mapika/decider-0.8b 8001
```

Or from PowerShell:

```powershell
serving\start-decider.ps1
serving\start-decider.ps1 -Model Mapika/decider-0.8b -Port 8001
```

- The first start downloads the weights (about 4 GB for 2b).
- Start-up then captures 18 CUDA graphs, which takes about 2 minutes.
- The server is ready when `curl -s http://127.0.0.1:8000/health` returns `"ok": true`.

The start scripts default `DECIDER_GRAPH_TOKEN_BUDGET=8192`, `DECIDER_T_BUCKETS=256,...,8192` and `DECIDER_B_BUCKETS=1,4,16`:

- decider's default grid (89 graphs) overflows 8 GB of VRAM.
- On Windows the overflow spills into system memory and becomes very slow.

Set these variables yourself to override them. See `docs/spike-notes.md` for the measurements.

To stop the server, press Ctrl+C in its terminal. From PowerShell:

```powershell
Get-CimInstance Win32_Process | ? { $_.Name -eq 'uvicorn.exe' -and $_.CommandLine -like '*decider.serve*' } | % { taskkill /PID $_.ProcessId /T /F }
```

## Docker (fallback, untested on this machine)

```bash
docker compose -f serving/docker-compose.yml up -d --build
```

This serves `DECIDER_MODEL` (default decider-2b) on port 8000 and mounts the host Hugging Face cache.

## Probe

```bash
uv run --no-project --with httpx python serving/probe.py --url http://127.0.0.1:8000 --save tests/fixtures/decider
```

The probe prints the answer shapes and p50 latencies for 12×score, 12×noul and one 64-option choice, and checks the 255/256-option limit.

## GPU sharing

**Stop decider before running `dense+llm`.** decider-2b (about 5.5–7.5 GB) and Qwen3.5-2B do not fit on an 8 GB GPU together. Don't run any other GPU job (embedding, cross-encoder) while decider is up, either: it pushes decider into shared memory and ruins timings.
