"""The retriever interface. The rest of the API only ever sees this."""

from __future__ import annotations

from typing import Protocol

from ..schemas import Passage


class Retriever(Protocol):
    name: str

    def retrieve(
        self,
        question: str,
        k: int,
        *,
        vector: list[float] | None = None,
        include_superseded: bool = False,
    ) -> list[Passage]:
        """Return up to k passages, best first. `vector` is the question's embedding when
        the caller already has it (the cache needed it first); `include_superseded` is
        True only when the user explicitly asked about an older version."""
        ...

    def readiness(self) -> dict[str, str]:
        """One entry per dependency check: "ok" or a reason it is not ready."""
        ...

    def generation(self) -> str:
        """An id that changes whenever the index is rebuilt (cache invalidation, 3.3)."""
        ...
