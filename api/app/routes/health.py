"""Liveness and readiness (section 4.3). They answer different questions.

/healthz: is the process alive? It touches nothing else, so a slow Chroma can never make
the kubelet restart healthy API pods (a restart storm).
/readyz: can this pod take traffic now? 200 only if Chroma is reachable, the active
collection exists and is non-empty, its embedding model matches EMBED_MODEL (the pin,
section 4.4), and Redis is reachable. Otherwise 503, naming what failed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..services.readiness import readiness_checks

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    checks = readiness_checks(request.app.state.pipeline)
    failed = [name for name, result in checks.items() if not result.startswith("ok")]
    if len(failed) > 0:
        response.status_code = 503
        return {"status": "not ready", "failed": failed, "checks": checks}
    return {"status": "ready", "checks": checks}
