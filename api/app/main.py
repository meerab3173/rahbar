"""The API process. It reads the index; it never builds it (section 4.2).

Run locally with stubs:   make dev     (then open http://localhost:8000)
Run in the container:     uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .logjson import configure_logging, request_id_var
from .providers.llm import get_provider
from .retrieval import get_retriever
from .routes import ask, health, metrics
from .services.answer import Pipeline
from .services.cache import get_cache

log = logging.getLogger("rahbar.api")
APP_VERSION = os.environ.get("APP_VERSION", "dev")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    retriever = get_retriever(settings)
    provider = get_provider(settings)
    cache = get_cache(settings.cache_backend, settings.redis_url, settings.cache_threshold,
                      settings.cache_ttl_seconds, settings.cache_max_entries)
    app.state.pipeline = Pipeline(
        settings=settings, retriever=retriever, provider=provider,
        embedder=getattr(retriever, "embedder", None), cache=cache,
    )
    app.state.retriever, app.state.provider = retriever, provider
    # The embedding pin (4.4): compare the active collection's embed_model metadata with
    # settings.embed_model and log any mismatch now; /readyz keeps failing until it is
    # fixed. This process only reads the index. The indexer Job writes it.
    verify_pin_at_boot = getattr(retriever, "verify_pin_at_boot", None)
    if verify_pin_at_boot is not None:
        verify_pin_at_boot()
    log.info(
        "api started",
        extra={"retriever": settings.retriever, "llm_provider": settings.llm_provider,
               "embed_model": settings.embed_model, "top_k": settings.top_k,
               "cache_backend": settings.cache_backend, "temperature": settings.temperature},
    )
    yield


app = FastAPI(title="RAHBAR API", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        response.headers["X-App-Version"] = APP_VERSION   # visible in a rollout demo (4.6)
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")   # route template, not the raw URL
        elapsed = time.perf_counter() - start
        metrics.REQUESTS.labels(path=path, status=str(status)).inc()
        metrics.LATENCY.labels(path=path).observe(elapsed)
        if path not in ("/healthz", "/readyz", "/metrics"):
            log.info("request", extra={"path": path, "status": status,
                                       "ms": round(elapsed * 1000, 1)})
        request_id_var.reset(token)


app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(ask.router)

# Development only: `make dev` serves the chat UI from this process, so you can try it
# without a cluster. On Kubernetes the ui Deployment serves it and this stays unset.
_ui_dir = os.environ.get("SERVE_UI_DIR")
if _ui_dir:
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
