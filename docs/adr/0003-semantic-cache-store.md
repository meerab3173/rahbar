# ADR 0003: The semantic cache lives in Redis, keyed by index generation

## Status
Accepted · 2026-11-06

## Context
Section 3.3 allows Redis or a second Chroma collection. Two constraints decide it:
only the indexer Job writes to Chroma (ADR 0001), and the NetworkPolicy lets only the
api reach Redis. A cache is written on the hot path, by the API.

## Decision
Redis (`k8s/base/redis.yaml`: Deployment + PVC, append-only file, `Recreate`).
`api/app/services/cache.py` stores, per index generation `g` (the active collection's
name), one hash `rahbar:cache:<g>` of entries `{question, float32 vector, response,
exact terms}`. A lookup scans at most `CACHE_MAX_ENTRIES` (200) vectors and takes the best
cosine similarity; a hit needs similarity ≥ `CACHE_THRESHOLD` **and** the same exact terms
(numbers, clause numbers, letter grades, "summer/regular", codes; `guards.exact_signature`),
because the embedding blurs "B-" and "B+". Never cached: follow-ups, old-version
questions, abstentions, fallbacks, answers that lost a citation in validation.

**Invalidation on re-index:** the key contains the generation, so the moment the Job
flips the pointer every old entry stops matching; the API then deletes the other
generations' hashes (`invalidate_other_generations`). A 24 h TTL is a second line.

Contract change: `AskResponse.cached: bool` and the `X-Cache: HIT|MISS` header, which the
starter UI already displays.

## Options considered
- **Second Chroma collection**: native vector search, but the API would become a Chroma
  writer, which breaks the one-writer rule and widens the chroma NetworkPolicy.
- **Redis Stack vector index (RediSearch)**: the right answer above a few thousand
  entries; not needed for 200, and the plain `redis` image is smaller and pinned.
- **Exact-string cache**: zero false hits but almost no hits; students rephrase.

## Consequences
O(N) lookup is fine for N ≤ 200 (≈0.6 MB read per lookup) and would need RediSearch at
scale. Redis down → readiness fails and pods leave the Service; we chose that (the
handout's readiness rule) over silently serving without a cache.
