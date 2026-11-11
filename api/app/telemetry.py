"""Prometheus metrics (section 4.8), defined once and imported where they are recorded.

Request count and latency per route, plus the RAG signals: retrieval and generation
latency, cache hits and misses (rahbar_cache_hit_ratio, or hits / (hits + misses) in
PromQL), abstentions, fallbacks, dropped citations, and tokens in and out per request.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
RETRIEVAL_SECONDS = Histogram(
    "rahbar_retrieval_seconds", "Embedding plus vector search latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
GENERATION_SECONDS = Histogram(
    "rahbar_generation_seconds", "LLM generation latency, including the one retry",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
CACHE_HITS = Counter("rahbar_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("rahbar_cache_misses_total", "Semantic cache misses")
CACHE_HIT_RATIO = Gauge("rahbar_cache_hit_ratio", "Cache hits / lookups since this pod started")
ABSTENTIONS = Counter("rahbar_abstentions_total", "Answers with answered_from_context=false",
                      ["reason"])
FALLBACKS = Counter("rahbar_fallbacks_total", "Canned fallback responses", ["reason"])
RETRIES = Counter("rahbar_generation_retries_total", "Validation failures that were retried")
CITATIONS_DROPPED = Counter("rahbar_citations_dropped_total",
                            "Model citations that matched no retrieved chunk")
TOKENS_IN = Histogram("rahbar_tokens_in", "Prompt tokens per request",
                      buckets=(100, 250, 500, 1000, 2000, 4000, 8000))
TOKENS_OUT = Histogram("rahbar_tokens_out", "Completion tokens per request",
                       buckets=(25, 50, 100, 200, 400, 800, 1600))
TOP_SCORE = Histogram("rahbar_top_similarity", "Similarity of the best retrieved chunk",
                      buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))

_lookups = {"hits": 0, "total": 0}


def record_cache(hit: bool) -> None:
    (CACHE_HITS if hit else CACHE_MISSES).inc()
    _lookups["total"] += 1
    _lookups["hits"] += int(hit)
    CACHE_HIT_RATIO.set(_lookups["hits"] / _lookups["total"])
