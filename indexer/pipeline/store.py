"""Chroma plumbing shared by the ingestion Job: the client, the wait, and the pointer.

The only code in the whole system that writes to Chroma is the ingestion Job
(pipeline.run -> pipeline.sync). Assignment 02 rebuilt the whole index into a new
``<collection>-v<unix time>`` collection on every run; Assignment 03 (Part A2) syncs one
collection per configuration incrementally instead (see sync.py). What stays the same:

  * the one-record pointer collection ``<collection>-active`` names the collection the
    api reads; the api re-reads it every 10 seconds, so a switch needs no restart;
  * the pointer records the embedding model and dimension (the pin) and a generation,
    which the api's semantic cache keys on;
  * old ``<collection>-v*`` collections from Assignment 02 are deleted once the pointer
    has moved on (the previous target is kept for one run, for a manual rollback).
"""

from __future__ import annotations

import time
from typing import Any

from .config import IndexerSettings

POINTER_SUFFIX = "-active"
POINTER_ID = "active"


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def wait_for_chroma(settings: IndexerSettings, timeout: float = 300.0) -> Any:
    """The Job can start before chroma-0 is ready (image pull, PVC attach). Wait for it
    here instead of burning backoffLimit retries."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            client = get_client(settings)
            client.heartbeat()
            return client
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(5)


def _names(client: Any) -> list[str]:
    """list_collections() returns names in some client versions, objects in others."""
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


def flip_pointer(client: Any, settings: IndexerSettings, name: str, dimension: int,
                 count: int, generation: str = "") -> None:
    pointer = client.get_or_create_collection(
        name=settings.collection + POINTER_SUFFIX, metadata={"purpose": "active-index"})
    pointer.upsert(
        ids=[POINTER_ID], embeddings=[[1.0]], documents=[name],
        metadatas=[{"collection": name, "embed_model": settings.embed_model,
                    "dimension": dimension, "chunks": count,
                    "generation": generation or str(int(time.time())),
                    "built_at": int(time.time())}],
    )


def cleanup(client: Any, settings: IndexerSettings, keep: list[str]) -> list[str]:
    prefix = settings.collection + "-v"
    removed = []
    for name in _names(client):
        if name.startswith(prefix) and name not in keep:
            client.delete_collection(name)
            removed.append(name)
    return removed
