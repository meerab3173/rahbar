# Failure analysis

Four stages broken on purpose, one at a time, then restored. Each break is one switch,
so it can be repeated by anyone. The *(fixtures)* numbers were measured offline with
`eval/tune_chunking.py` on the CI fixture corpus (hash-bow-768, deterministic); the
**[measure]** cells are the same experiment on the real corpus with Ollama.
From the RAG lecture: only F is the model's fault, so A to E are checked first.

## Stage A · Extraction
- **What we broke:** indexed the scanned ORIC policy without OCR: `OCR=false` in the
  ConfigMap, then `make index` (offline: `eval/tune_chunking.py --no-ocr`).
- **Symptom:** questions about the start-up policy (e.g. "What equity can a faculty
  member hold in a start-up?") abstain or answer from unrelated chunks; the policy's
  chunks simply do not exist. Unit-tested: `test_without_ocr_the_scanned_page_is_lost`.
  **[measure]** chunk count for that file drops from N to 0.
- **Diagnosis, as if we had not caused it:** the indexer's per-document log line
  `{"msg": "chunked", "source": "faculty-entrepreneurship-…pdf", "pages": 8,
  "ocr_pages": 0, "empty_pages": 8, "chunks": 0}`: pages > 0 but chunks = 0 is the
  signature of a missing text layer; `pdffonts` on the file confirms an empty font list.
  In serving, the `rahbar_top_similarity` histogram shifts low and
  `rahbar_abstentions_total{reason="below_relevance_floor"}` rises for that topic.
- **Remedy:** OCR pages with < 25 characters of text (`extract.py`, default `OCR=true`),
  and alert on `empty_pages > 0` in the indexer log.

## Stage B · Chunking
- **What we broke:** fixed-size 100-token windows, no overlap: `CHUNK_STRATEGY=fixed`,
  `CHUNK_SIZE=100`, `CHUNK_OVERLAP=0`.
- **Symptom:** *(fixtures)* recall@5 fell from **1.0 to 0.0** for every question type.
  The windows do find the right words, but they no longer know which section they are
  in (`section="(fixed window)"`), so no retrieved chunk matches a golden section and
  no citation can name a clause. On the real corpus the Generative AI policy's
  "Unacceptable Uses" list is also cut mid-item. **[measure]** recall@5:
  `python eval/tune_chunking.py --strategy fixed --configs 100:0`.
- **Diagnosis:** recall@5 by type in `eval/results.json` drops across *all* types at once
  (a retrieval-only change would hit some types more than others); the per-request log's
  `chunk_ids` end in `-f0001…` and the citations name "(fixed window)".
- **Remedy:** structure-aware chunking with section labels and sentence overlap (ADR 0002).

## Stage C · Retrieval
- **What we broke:** `EMBED_MODEL` changed on the API only (`make demo-pin`).
- **Symptom:** without the guardrail, retrieval would return confident nonsense with no
  error anywhere (vectors from two models are not comparable). With it, the new pod sits
  at **0/1 READY** while the three old pods keep serving (maxUnavailable 0), and the log
  says `"embedding pin check failed"`, `"conflict": "embedding model mismatch: index
  'rahbar-v…' was built with 'nomic-embed-text' (dimension 768), but this API has
  EMBED_MODEL='all-minilm'"`. **[measure]** paste `docs/evidence/pin-guard.txt`.
- **Diagnosis:** `kubectl get pods` (0/1 READY) → `kubectl describe pod` (Events:
  "Readiness probe failed: HTTP probe failed with statuscode: 503") → `kubectl logs`
  (the conflict line). `curl /readyz` names `embed_model` in `failed`.
- **Remedy:** the pin check in `api/app/retrieval/chroma.py` (`pin_conflict`), and
  `kubectl rollout undo deployment/api`. Second variant measured: `TOP_K=1` drops
  *(fixtures)* the injection question's relevant attendance chunk out of the top 1
  (recall@5 effectively becomes recall@1 = 0.8).

## Stage E · Source
- **What we broke:** removed the superseded filter (`--no-filter`; in the api, a
  retriever that always passes `include_superseded=True`) and asked the version-sensitive
  questions.
- **Symptom:** *(fixtures)* recall@3 fell from **1.0 to 0.8**: the 2023 edition's
  "1. Attendance" (75 %) outranks or crowds the 2026 one (80 %), and the stub answer quotes
  the repealed number. On the real corpus, q12 "What attendance is required in summer
  semester courses?" can retrieve the 2025 handbook's summer rules, which have no 80 %
  clause. **[measure]** `--no-filter` recall@3 for the version questions.
- **Diagnosis:** the per-request log's `chunk_ids` include `academic-rules-handbook-jun-2025-…`
  for a question that did not ask about 2025; the response's `retrieved` list shows
  `superseded` sources. A prompt cannot fix this: the model faithfully quotes what it is
  given.
- **Remedy:** metadata filtering `where={"superseded": False}` (retrieval/chroma.py) unless
  `wants_old_version()` sees "previous / 2025 handbook / older rules" (guards.py).

## Not broken here (two of six were optional)
- **D · Prompt** (remove "use the supplied context"): expected symptom is answers from
  general knowledge about *other* universities on the must-abstain questions, which the
  abstention gate (100 %) would catch.
- **F · Generation** (temperature 1.2): expected symptom is validation retries
  (`rahbar_generation_retries_total`) and fallbacks rising, not wrong retrieval.
