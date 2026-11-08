"""Runtime configuration, read once from environment variables.

On the cluster these variables come from the rahbar-config ConfigMap and the
rahbar-secret Secret (section 4.1). Nothing here is baked into the image, so one
image runs in every environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Which implementations to load. "stub" needs no cluster, no model, no network.
    retriever: str = "stub"          # RETRIEVER: stub | chroma
    llm_provider: str = "stub"       # LLM_PROVIDER: stub | ollama-host | ollama-incluster | hosted

    # The embedding pin (section 4.4). The indexer reads the same ConfigMap key.
    embed_model: str = "nomic-embed-text"

    # Vector store
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    collection: str = "rahbar"

    # Retrieval
    top_k: int = 5
    relevance_floor: float = 0.45    # RELEVANCE_FLOOR: below this top similarity, abstain
    hybrid_weight: float = 0.15      # HYBRID_WEIGHT: exact-term boost in the rerank; 0 = off
    candidate_multiplier: int = 4    # CANDIDATES: fetch k * this from Chroma, then rerank

    # Generation
    temperature: float = 0.1
    llm_model: str = "llama3.2:3b"
    llm_timeout: float = 60.0        # LLM_TIMEOUT seconds, per call
    ollama_url: str = "http://host.minikube.internal:11434"
    hosted_base_url: str = ""        # e.g. https://api.groq.com/openai/v1
    llm_api_key: str = field(default="", repr=False)   # from the Secret; never log it

    # Semantic cache (section 3.3)
    cache_backend: str = "none"      # CACHE_BACKEND: none | redis
    redis_url: str = "redis://redis:6379/0"
    cache_threshold: float = 0.95    # CACHE_THRESHOLD: cosine similarity for a hit
    cache_ttl_seconds: int = 86400
    cache_max_entries: int = 200

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings(
        retriever=os.environ.get("RETRIEVER", "stub"),
        llm_provider=os.environ.get("LLM_PROVIDER", "stub"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=_int("CHROMA_PORT", 8000),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        top_k=_int("TOP_K", 5),
        relevance_floor=_float("RELEVANCE_FLOOR", 0.45),
        hybrid_weight=_float("HYBRID_WEIGHT", 0.15),
        candidate_multiplier=_int("CANDIDATES", 4),
        temperature=_float("TEMPERATURE", 0.1),
        llm_model=os.environ.get("LLM_MODEL", "llama3.2:3b"),
        llm_timeout=_float("LLM_TIMEOUT", 60.0),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        hosted_base_url=os.environ.get("HOSTED_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        cache_backend=os.environ.get("CACHE_BACKEND", "none"),
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        cache_threshold=_float("CACHE_THRESHOLD", 0.95),
        cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 86400),
        cache_max_entries=_int("CACHE_MAX_ENTRIES", 200),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


# Kept for tests that build Settings by hand.
__all__ = ["Settings", "get_settings", "_bool"]
