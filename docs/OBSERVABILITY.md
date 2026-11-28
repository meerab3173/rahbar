# Observability and traceability (Assignment 03, Part D and F4)

RAGOps paper 3.2.3, 5.1 and Table 2. Every question to `POST /api/ask` leaves one trace
record. The trace id is the request id (`X-Request-ID`), so the HTTP log line, the trace
and the response share one id.

## D1. The trace record (`api/app/tracing.py`)

| Field | What it holds |
| --- | --- |
| `trace_id`, `timestamp`, `pod` | the request id, UTC time to the millisecond, the api pod |
| `query`, `rewritten_query` | the question; for a follow-up, the text actually searched (previous question + this one), else `null` |
| `embedding` | `model` and `version`: the Ollama tag and digest (`nomic-embed-text:latest@0a109f422b47`) |
| `collection` | `name`, `generation` (changes whenever an ingestion run changes the content), and the model, dimension and chunking the index was built with |
| `filters` | the Chroma `where` clause, e.g. `{"superseded": false}`; `{}` when an older version was asked for |
| `candidates` | every chunk the vector search returned, in vector order: chunk id, `doc_id`, `doc_key`, document version, superseded flag, similarity, rank score |
| `retrieved` | the top-k given to the model, in that order, with document id, version and scores |
| `reranked` | the chunk ids after the exact-term rerank |
| `prompt` | `sha256` of the exact messages, and one hash per attempt (a retry is a second attempt) |
| `model` | provider, model, temperature, context size, output format, timeout |
| `response`, `citations`, `outcome` | the answer, whether it came from the context, its citations; abstention reason, tokens |
| `guardrails` | what each rail did: `cache_bypass`, `superseded_filter`, `relevance_floor`, `injection_scan`, `schema_validation`, `injection_echo`, `model_declined`, `citation_grounding` |
| `latency_ms` | `embed`, `cache_lookup`, `retrieve`, `generate`, `guards`, `total` |
| `cache` | `hit`, `similarity`, `threshold`; a hit names the trace that first produced the answer |

Where it goes (F4): one JSON log line from the logger `rahbar.trace` on standard output
(`kubectl logs`), and `/traces/traces-<pod>.jsonl` on the `traces` PersistentVolumeClaim.
Each exact prompt is stored once, as `/traces/prompts/<sha256>.json`.

## D2. The trace tool (`api/app/tracetool.py`)

```bash
make trace ID=<trace id>        # kubectl exec deploy/api -- python -m app.tracetool show <id>
make traces N=20                # the latest traces, one line each
python api/app/tracetool.py --dir traces show <id>    # on a copy pulled with make traces-pull
```

`show` prints every field above, then the exact system and user messages of each attempt,
rebuilt from the stored prompt by its hash: the evidence passages exactly as the model
saw them, and the question.

## D3. Metrics

On `/metrics` (`make metrics` port-forwards it; F4), next to the Assignment 02 metrics:

| Signal | Metric |
| --- | --- |
| top-1 retrieval similarity distribution | `rahbar_top_similarity` (histogram) |
| abstention rate | `rahbar_answers_total{abstained="true"}` / `rahbar_answers_total` |
| cache hit rate | `rahbar_cache_hit_ratio`; `rahbar_cache_hits_total`, `rahbar_cache_misses_total` |
| p95 latency per stage | `histogram_quantile(0.95, rahbar_stage_seconds_bucket{stage=...})` |
| injection attempts flagged | `rahbar_injection_flagged_total` |

Counters start at zero when a pod starts, so the report numbers come from the trace log
instead: `make trace-summary SINCE=2h` (`tracetool summary --since 2h [--json]`) prints the
same five signals over any time window, from the records on the volume.

## D4. Guardrail logging

The injection defence from Assignment 02 now writes its action into the trace.
`injection_scan` flags a retrieved chunk that carries an instruction-like line (it stays
in the evidence as data); `injection_echo` blocks an answer that repeats the planted
claim. Batch B's `b10-counselling-services.md` is the test document.
