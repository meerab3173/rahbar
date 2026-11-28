"""Stage 5: Update the retrieval sources (Assignment 03, A2; RAGOps paper 4.2.4).

The vector store is brought in line with the data lake's catalog, document by document,
instead of being rebuilt:

  unchanged   the collection already holds this doc_key with the same file hash: nothing
              is embedded (an unchanged corpus embeds 0 chunks)
  relabelled  same hash, but the document was archived or restored: only the chunks'
              `superseded` metadata changes; nothing is embedded
  replaced    the hash changed: that document's old chunks are deleted and its new chunks
              embedded; no other document is touched
  added       a new doc_key: chunked, embedded, added
  deleted     a doc_key the catalog no longer has: every chunk with that doc_key is
              deleted, so nothing is left orphaned (proved by the count per doc_id)

The collection name comes from the embedding model and the chunking settings
(IndexerSettings.collection_name), and the same pointer as Assignment 02 tells the api
which collection to read. The pointer records a `generation`, which changes only when
the collection's content changes; the api's semantic cache keys on it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .chunk import chunk
from .config import IndexerSettings
from .lake import Entry
from .models import Chunk, Document
from .store import POINTER_ID, POINTER_SUFFIX, cleanup, flip_pointer

ADD_BATCH = 256
PAGE = 1000

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class IndexedDoc:
    ids: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hashes(self) -> set[str]:
        return {str(m.get("doc_sha256", "")) for m in self.metadatas}

    @property
    def superseded(self) -> set[bool]:
        return {bool(m.get("superseded", False)) for m in self.metadatas}


@dataclass
class SyncResult:
    collection: str
    embedded: int = 0           # chunks embedded this run
    deleted: int = 0            # chunks deleted this run
    kept: int = 0               # chunks left exactly as they were
    relabelled: int = 0         # chunks whose superseded flag changed (no embedding)
    total: int = 0              # chunks in the collection afterwards
    per_doc: dict[str, str] = field(default_factory=dict)        # doc_key -> what happened
    counts_by_doc_id: dict[str, int] = field(default_factory=dict)
    counts_by_doc_key: dict[str, int] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)            # doc_keys not in the catalog
    generation: str = ""
    pointer_flipped: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.embedded or self.deleted or self.relabelled)


def get_collection(client: Any, name: str) -> Any | None:
    try:
        return client.get_collection(name)
    except Exception:   # chromadb raises NotFoundError / ValueError for a missing collection
        return None


def index_of(collection: Any) -> dict[str, IndexedDoc]:
    """doc_key -> the chunk ids and metadata the collection holds for it."""
    found: dict[str, IndexedDoc] = defaultdict(IndexedDoc)
    if collection is None:
        return {}
    offset = 0
    while True:
        got = collection.get(include=["metadatas"], limit=PAGE, offset=offset)
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        for cid, meta in zip(ids, metas):
            meta = dict(meta or {})
            key = str(meta.get("doc_key") or f"(no doc_key) {meta.get('source', '?')}")
            found[key].ids.append(str(cid))
            found[key].metadatas.append(meta)
        if len(ids) < PAGE:
            break
        offset += PAGE
    return dict(found)


def counts(index: dict[str, IndexedDoc]) -> tuple[dict[str, int], dict[str, int]]:
    by_id: dict[str, int] = defaultdict(int)
    by_key: dict[str, int] = {}
    for key, doc in sorted(index.items()):
        by_key[key] = len(doc.ids)
        doc_id = str(doc.metadatas[0].get("doc_id") or key.split("@")[0]) if doc.metadatas else key
        by_id[doc_id] += len(doc.ids)
    return dict(sorted(by_id.items())), by_key


def chunks_for(entry: Entry, document: Document, settings: IndexerSettings) -> list[Chunk]:
    document.doc_id = entry.doc_id
    document.sha256 = entry.sha256
    document.doc_version = entry.version
    document.superseded = entry.superseded
    return chunk(document, settings.chunk_size, settings.chunk_overlap,
                 strategy=settings.chunk_strategy)


def sync(client: Any, desired: list[tuple[Entry, Document | None]],
         settings: IndexerSettings, embed_fn: EmbedFn, run_id: str) -> SyncResult:
    name = settings.collection_name
    result = SyncResult(collection=name)
    collection = get_collection(client, name)
    before = index_of(collection)
    wanted = {e.doc_key: (e, d) for e, d in desired}

    # deleted documents (and anything else the catalog does not know): no orphans
    for key, held in before.items():
        if key not in wanted and collection is not None:
            collection.delete(ids=held.ids)
            result.deleted += len(held.ids)
            result.per_doc[key] = f"deleted ({len(held.ids)} chunks)"

    to_add: list[Chunk] = []
    for key, (entry, document) in sorted(wanted.items()):
        found = before.get(key)
        if found is not None and collection is not None and found.hashes == {entry.sha256}:
            held = found
            if held.superseded != {entry.superseded}:
                metas = [{**m, "superseded": entry.superseded} for m in held.metadatas]
                collection.update(ids=held.ids, metadatas=metas)
                result.relabelled += len(held.ids)
                result.per_doc[key] = (f"relabelled superseded={entry.superseded} "
                                       f"({len(held.ids)} chunks, 0 embedded)")
            else:
                result.kept += len(held.ids)
                result.per_doc[key] = f"unchanged ({len(held.ids)} chunks, 0 embedded)"
            continue
        if document is None:
            # verified earlier, failed this run, and the last good version is kept
            if found is not None:
                result.kept += len(found.ids)
                result.per_doc[key] = f"kept last good version ({len(found.ids)} chunks)"
            continue
        old = 0
        if found is not None and collection is not None:
            collection.delete(ids=found.ids)
            old = len(found.ids)
            result.deleted += old
        pieces = chunks_for(entry, document, settings)
        to_add.extend(pieces)
        result.per_doc[key] = (f"{'replaced' if old else 'added'} "
                               f"({old} deleted, {len(pieces)} embedded)")

    ids = [c.id for c in to_add]
    if len(set(ids)) != len(ids):
        raise ValueError("two documents produced the same chunk id; rename one file")
    if to_add:
        vectors = embed_fn([c.text for c in to_add])
        if len(vectors) != len(to_add):
            raise RuntimeError(f"{len(to_add)} chunks but {len(vectors)} vectors")
        dimension = len(vectors[0])
        if collection is None:
            collection = client.create_collection(name=name, metadata={
                "hnsw:space": "cosine", "embed_model": settings.embed_model,
                "dimension": dimension, "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "chunk_strategy": settings.chunk_strategy, "created_at": int(time.time())})
        held_dim = (collection.metadata or {}).get("dimension")
        if held_dim and int(held_dim) != dimension:
            raise RuntimeError(f"{name} holds {held_dim}-d vectors; the model gave {dimension}")
        for i in range(0, len(to_add), ADD_BATCH):
            batch = to_add[i:i + ADD_BATCH]
            collection.add(ids=[c.id for c in batch], documents=[c.text for c in batch],
                           embeddings=vectors[i:i + ADD_BATCH],
                           metadatas=[c.metadata() for c in batch])
        result.embedded = len(to_add)

    after = index_of(collection)
    result.counts_by_doc_id, result.counts_by_doc_key = counts(after)
    result.orphans = sorted(k for k in after if k not in wanted)
    result.total = int(collection.count()) if collection is not None else 0
    expected = result.kept + result.relabelled + result.embedded
    if collection is not None and result.total != expected:
        raise RuntimeError(f"expected {expected} chunks in {name}, found {result.total}")
    if result.orphans:
        raise RuntimeError(f"orphan chunks left for {result.orphans}")

    if settings.activate and collection is not None:
        result.generation, result.pointer_flipped = _point(client, settings, collection,
                                                            result, run_id)
    return result


def _point(client: Any, settings: IndexerSettings, collection: Any, result: SyncResult,
           run_id: str) -> tuple[str, bool]:
    """Point the api at this collection. A new generation only when the content changed,
    so the semantic cache survives a run that changed nothing."""
    previous: dict[str, Any] = {}
    pointer = get_collection(client, settings.collection + POINTER_SUFFIX)
    if pointer is not None:
        got = pointer.get(ids=[POINTER_ID], include=["metadatas"])
        metas = got.get("metadatas") or []
        previous = dict(metas[0] or {}) if metas else {}
    same_target = previous.get("collection") == collection.name
    if same_target and not result.changed and previous.get("generation"):
        return str(previous["generation"]), False
    dimension = int((collection.metadata or {}).get("dimension") or 0)
    flip_pointer(client, settings, collection.name, dimension, result.total,
                 generation=run_id)
    keep = [n for n in (collection.name, previous.get("collection")) if n]
    cleanup(client, settings, keep=[str(n) for n in keep])
    return run_id, True
