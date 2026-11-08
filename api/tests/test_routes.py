"""HTTP-level checks for the routes added on top of the starter contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("CACHE_BACKEND", "none")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_readiness_is_200_with_named_checks_when_everything_is_ready(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) >= {"retriever", "redis"}


def test_readiness_is_503_naming_the_failed_dependency(client: TestClient) -> None:
    from app.main import app
    from app.services.cache import SemanticCache
    from tests.fakes import FakeRedis

    app.state.pipeline.cache = SemanticCache(FakeRedis(up=False), 0.95, 60, 10)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["failed"] == ["redis"]


def test_liveness_stays_200_when_a_dependency_is_down(client: TestClient) -> None:
    from app.main import app
    from app.services.cache import SemanticCache
    from tests.fakes import FakeRedis

    app.state.pipeline.cache = SemanticCache(FakeRedis(up=False), 0.95, 60, 10)
    assert client.get("/healthz").status_code == 200


def test_ask_reports_the_cache_state_in_a_header(client: TestClient) -> None:
    r = client.post("/api/ask", json={"question": "What is the attendance requirement?"})
    assert r.headers["X-Cache"] == "MISS"
    assert r.json()["cached"] is False


def test_history_marks_a_follow_up_and_is_accepted(client: TestClient) -> None:
    r = client.post("/api/ask", json={"question": "And in summer?",
                                      "history": ["What is the attendance requirement?"]})
    assert r.status_code == 200


def test_metrics_expose_the_rag_signals(client: TestClient) -> None:
    client.post("/api/ask", json={"question": "Count this request"})
    text = client.get("/metrics").text
    for name in ("rahbar_retrieval_seconds", "rahbar_generation_seconds",
                 "rahbar_cache_hits_total", "rahbar_abstentions_total",
                 "rahbar_fallbacks_total", "rahbar_tokens_in"):
        assert name in text
