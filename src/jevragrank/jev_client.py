from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}
CACHE_MODES = ("readwrite", "write", "off")


class JevError(Exception):
    """Base class for Jev client errors."""


class JevConnectionError(JevError):
    """The server could not be reached."""


class JevResponseError(JevError):
    """The server answered, but not in the expected shape."""


class JevRequestError(JevError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Jev server rejected the request (HTTP {status}): {message}")
        self.status = status


@dataclass
class JevStats:
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(data, dict):
        return str(data.get("detail") or data.get("error") or data)[:300]
    return str(data)[:300]


class JevClient:
    """Async client for any Jev-compatible `POST /v1/systemone` server."""

    def __init__(self, base_url: str = "http://localhost:8000", *, model: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0, max_retries: int = 3,
                 concurrency: int = 16, cache_dir: str | Path | None = None,
                 cache_mode: str = "readwrite", transport: httpx.AsyncBaseTransport | None = None,
                 backoff: float = 0.5):
        if cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {CACHE_MODES}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_mode = cache_mode
        self.backoff = backoff
        self.stats = JevStats()
        self._transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None
        self._checked = False

    def _loop_state(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                             headers=headers, transport=self._transport)
            self._sem = asyncio.Semaphore(self.concurrency)
            self._loop = loop
            self._checked = False
        return self._client, self._sem

    async def check(self) -> None:
        client, _ = self._loop_state()
        try:
            await client.get("/v1/models")
        except httpx.TransportError as e:
            raise JevConnectionError(
                f"Can't reach a Jev server at {self.base_url} ({e.__class__.__name__}). "
                "Start one (see serving/README.md) or pass the right base_url.") from e
        self._checked = True

    def _cache_path(self, body: dict) -> Path | None:
        if self.cache_dir is None or self.cache_mode == "off":
            return None
        ns = self.model or self.base_url
        raw = json.dumps({"ns": ns, "body": body}, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(raw.encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    async def ask(self, state: Any, questions: dict) -> dict:
        body: dict[str, Any] = {"state": state, "questions": questions}
        if self.model:
            body["model"] = self.model
        path = self._cache_path(body)
        if path is not None and self.cache_mode == "readwrite" and path.exists():
            self.stats.cache_hits += 1
            return json.loads(path.read_text("utf-8"))
        client, sem = self._loop_state()
        if not self._checked:
            await self.check()
        async with sem:
            answers = await self._post(client, body)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(answers), "utf-8")
            os.replace(tmp, path)
        return answers

    async def _post(self, client: httpx.AsyncClient, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            self.stats.requests += 1
            try:
                resp = await client.post("/v1/systemone", json=body)
            except httpx.TransportError as e:
                err: JevError = JevConnectionError(f"Request to {self.base_url} failed: {e!r}")
            else:
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        raise JevResponseError(
                            f"Response is not JSON: {resp.text[:200]}") from e
                    answers = data.get("answers") if isinstance(data, dict) else None
                    if not isinstance(answers, dict):
                        raise JevResponseError(
                            f"Response has no 'answers' object: {str(data)[:200]}")
                    return answers
                err = JevRequestError(resp.status_code, _error_message(resp))
                if resp.status_code not in RETRY_STATUS:
                    raise err
            if attempt == self.max_retries:
                raise err
            self.stats.retries += 1
            await asyncio.sleep(self.backoff * 2**attempt * (1 + random.random()))
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
