from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

import httpx

PACKAGES = ["jevragrank", "torch", "transformers", "sentence-transformers", "bm25s", "ranx",
            "httpx", "numpy"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def environment(jev_url: str | None = None) -> dict:
    """Best-effort description of the machine and software; never raises."""
    info: dict = {"python": platform.python_version(), "platform": platform.platform(),
                  "packages": {}}
    for p in PACKAGES:
        try:
            info["packages"][p] = version(p)
        except PackageNotFoundError:
            pass
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    if gpu:
        fields = [x.strip() for x in gpu.splitlines()[0].split(",")]
        if len(fields) == 3:
            info.update(gpu=fields[0], driver=fields[1], gpu_memory=fields[2])
    try:
        import torch

        info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001 - ImportError, or a broken CUDA/DLL install
        pass
    info["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    if jev_url:
        for path in ("/health", "/v1/models"):
            try:
                resp = httpx.get(jev_url.rstrip("/") + path, timeout=5)
                resp.raise_for_status()
                info["jev_server"] = resp.json()
                break
            except (httpx.HTTPError, httpx.InvalidURL, ValueError):
                continue
    return info
