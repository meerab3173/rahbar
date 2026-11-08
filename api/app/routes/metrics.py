"""GET /metrics in Prometheus text format (section 4.8). The metrics live in telemetry.py."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..telemetry import LATENCY, REQUESTS

router = APIRouter()

__all__ = ["router", "REQUESTS", "LATENCY"]


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
