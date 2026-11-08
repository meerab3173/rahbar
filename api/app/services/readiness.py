"""The readiness checks behind /readyz, kept out of the route so they can be unit-tested."""

from __future__ import annotations

from .answer import Pipeline


def readiness_checks(pipe: Pipeline) -> dict[str, str]:
    """Every value starts with "ok" when that dependency is ready."""
    checks = dict(pipe.retriever.readiness())
    if pipe.cache is not None:
        checks["redis"] = pipe.cache.ping()
    else:
        checks["redis"] = "ok (cache disabled: CACHE_BACKEND=none)"
    # The collection must be non-empty: count() is checked inside the retriever, which
    # reports "empty: ... 0 chunks" when len(collection) == 0.
    return checks
