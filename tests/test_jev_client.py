import asyncio

import httpx
import pytest

from jevragrank.jev_client import JevClient, JevConnectionError, JevRequestError, JevResponseError

NOUL = {"a": {"type": "noul", "instructions": "Is it?\n\nPassage:\nvitamin d"}}


def client(fake, **kw):
    return JevClient("http://fake", transport=httpx.MockTransport(fake.handler), backoff=0.0, **kw)


async def test_ask_posts_state_and_questions(fake_jev):
    answers = await client(fake_jev).ask("Query: vitamin d", NOUL)
    assert answers["a"]["noul"] == 1.0
    assert fake_jev.bodies == [{"state": "Query: vitamin d", "questions": NOUL}]
    assert fake_jev.models_hits == 1


async def test_model_is_sent_when_set(fake_jev):
    await client(fake_jev, model="decider-2b").ask("Query: x", NOUL)
    assert fake_jev.bodies[0]["model"] == "decider-2b"


async def test_readwrite_cache_serves_repeat_requests(fake_jev, tmp_path):
    c = client(fake_jev, cache_dir=tmp_path)
    first = await c.ask("Query: x", NOUL)
    second = await c.ask("Query: x", NOUL)
    assert first == second
    assert len(fake_jev.bodies) == 1
    assert c.stats.cache_hits == 1


async def test_write_mode_never_reads_cache(fake_jev, tmp_path):
    c = client(fake_jev, cache_dir=tmp_path, cache_mode="write")
    await c.ask("Query: x", NOUL)
    await c.ask("Query: x", NOUL)
    assert len(fake_jev.bodies) == 2
    assert any(tmp_path.rglob("*.json"))


async def test_cache_is_namespaced_by_server(fake_jev, tmp_path):
    await client(fake_jev, cache_dir=tmp_path).ask("Query: x", NOUL)
    other = JevClient("http://other", transport=httpx.MockTransport(fake_jev.handler),
                      cache_dir=tmp_path, backoff=0.0)
    await other.ask("Query: x", NOUL)
    assert len(fake_jev.bodies) == 2


async def test_retries_transient_errors(fake_jev):
    fake_jev.fail_next = [503, 429]
    c = client(fake_jev)
    await c.ask("Query: x", NOUL)
    assert c.stats.retries == 2


async def test_gives_up_after_max_retries(fake_jev):
    fake_jev.fail_next = [503] * 4
    with pytest.raises(JevRequestError) as e:
        await client(fake_jev, max_retries=3).ask("Query: x", NOUL)
    assert e.value.status == 503


async def test_client_errors_are_not_retried(fake_jev):
    fake_jev.fail_next = [422]
    c = client(fake_jev)
    with pytest.raises(JevRequestError, match="boom"):
        await c.ask("Query: x", NOUL)
    assert c.stats.retries == 0


async def test_unreachable_server_raises_connection_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    c = JevClient("http://nowhere", transport=httpx.MockTransport(refuse), backoff=0.0)
    with pytest.raises(JevConnectionError, match="http://nowhere"):
        await c.ask("Query: x", NOUL)


async def test_response_without_answers_is_rejected():
    def bad(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"model": "x"})

    c = JevClient("http://x", transport=httpx.MockTransport(bad), backoff=0.0)
    with pytest.raises(JevResponseError):
        await c.ask("Query: x", NOUL)


def test_client_survives_separate_event_loops(fake_jev):
    c = client(fake_jev)
    asyncio.run(c.ask("Query: x", NOUL))
    asyncio.run(c.ask("Query: y", NOUL))
    assert len(fake_jev.bodies) == 2


def test_invalid_cache_mode():
    with pytest.raises(ValueError):
        JevClient(cache_mode="sometimes")
