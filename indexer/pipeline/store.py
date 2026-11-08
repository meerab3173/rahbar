"""Stage 5: Store. The only code in the whole system that writes to Chroma.

What happens to the old index while the API is serving from it (ADR 0001):

  1. Build into a NEW collection, ``<collection>-v<unix time>``, whose metadata records
     the embedding model and vector dimension (the pin, section 4.4).
  2. Verify its count equals the number of chunks written.
  3. Flip the one-record pointer collection ``<collection>-active`` to the new name.
     The API re-reads the pointer every 10 seconds, so it switches without a restart,
     and because the cache key contains the collection name, every cached answer from
     the old index stops matching at the same moment (section 3.3).
  4. Delete versions older than the previous one (kept for a quick manual rollback).

A Job that dies halfway leaves an orphan ``-v`` collection the API never reads; the next
run cleans it up. Readers never see a half-built index.
"""

from __future__ import annotations

import time
from typing import Any

from .config import IndexerSettings
from .models import Chunk

ADD_BATCH = 256
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
                 count: int) -> None:
    pointer = client.get_or_create_collection(
        name=settings.collection + POINTER_SUFFIX, metadata={"purpose": "active-index"})
    pointer.upsert(
        ids=[POINTER_ID], embeddings=[[1.0]], documents=[name],
        metadatas=[{"collection": name, "embed_model": settings.embed_model,
                    "dimension": dimension, "chunks": count,
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


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings,
          client: Any = None) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        raise ValueError("no chunks to store: refusing to publish an empty index")
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids")
    client = client or wait_for_chroma(settings)
    dimension = len(vectors[0])

    previous = None
    try:
        got = client.get_collection(settings.collection + POINTER_SUFFIX).get(
            ids=[POINTER_ID], include=["metadatas"])
        metas = got.get("metadatas") or []
        previous = metas[0].get("collection") if metas and metas[0] else None
    except Exception:  # first run: there is no pointer yet
        previous = None

    name = f"{settings.collection}-v{int(time.time())}"
    collection = client.create_collection(name=name, metadata={
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,
        "dimension": dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "built_at": int(time.time()),
    })
    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i:i + ADD_BATCH]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i + ADD_BATCH],
            metadatas=[c.metadata() for c in batch],
        )
    count = int(collection.count())
    if count != len(chunks):
        raise RuntimeError(f"wrote {len(chunks)} chunks but the collection holds {count}")
    flip_pointer(client, settings, name, dimension, count)
    cleanup(client, settings, keep=[n for n in (name, previous) if n])
    return count
