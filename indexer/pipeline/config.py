"""Indexer configuration, from the same ConfigMap the API reads (section 4.1)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    embed_model: str        # EMBED_MODEL: must equal the API's value (section 4.4)
    chroma_host: str
    chroma_port: int
    collection: str
    corpus_dir: str         # where the documents and _index.json sit inside the Job's pod
    chunk_size: int
    chunk_overlap: int
    ollama_url: str
    chunk_strategy: str = "structure"   # CHUNK_STRATEGY: structure | fixed (stage B demo)
    ocr: bool = True                    # OCR: false indexes scans without OCR (stage A demo)
    # Assignment 03, Part A: the data lake and the verification stage.
    datalake_dir: str = "/datalake"     # DATALAKE_DIR: inbox, lake, archive, quarantine, state
    index_collection: str = ""          # INDEX_COLLECTION: "" = derived from model + chunking
    activate: bool = True               # INDEX_ACTIVATE: flip the api's pointer after a run
    near_dup_threshold: float = 0.95    # NEAR_DUP_THRESHOLD: fingerprint cosine (docs/DATA-LAKE.md)
    min_text_chars: int = 200           # MIN_TEXT_CHARS: less extracted text = "empty"
    max_garbled_ratio: float = 0.10     # MAX_GARBLED_RATIO: share of unreadable characters

    @property
    def collection_name(self) -> str:
        """The one collection this configuration writes to, e.g. rahbar-nomic-embed-text-c200o30.

        One collection per (embedding model, chunking) means re-running on an unchanged
        corpus finds every chunk already there, and a changed setting (a Part C candidate)
        gets its own collection instead of overwriting the live one.
        """
        if self.index_collection:
            return self.index_collection
        model = re.sub(r"[^a-z0-9]+", "-", self.embed_model.lower()).strip("-")
        tag = "f" if self.chunk_strategy == "fixed" else "c"
        return f"{self.collection}-{model}-{tag}{self.chunk_size}o{self.chunk_overlap}"[:63]


def get_settings() -> IndexerSettings:
    return IndexerSettings(
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=int(os.environ.get("CHROMA_PORT", 8000)),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        corpus_dir=os.environ.get("CORPUS_DIR", "/corpus"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", 400)),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 50)),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        chunk_strategy=os.environ.get("CHUNK_STRATEGY", "structure"),
        ocr=_flag("OCR", True),
        datalake_dir=os.environ.get("DATALAKE_DIR", "/datalake"),
        index_collection=os.environ.get("INDEX_COLLECTION", ""),
        activate=_flag("INDEX_ACTIVATE", True),
        near_dup_threshold=float(os.environ.get("NEAR_DUP_THRESHOLD", 0.95)),
        min_text_chars=int(os.environ.get("MIN_TEXT_CHARS", 200)),
        max_garbled_ratio=float(os.environ.get("MAX_GARBLED_RATIO", 0.10)),
    )


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}
