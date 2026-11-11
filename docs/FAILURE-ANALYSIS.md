# Failure analysis

Four stages broken on purpose, one at a time, then restored. Each break is one switch, so
anyone can repeat it. Stages A, B and E were measured with `eval/tune_chunking.py` on the
real corpus (Ollama, `nomic-embed-text`), using the same retrieval, filter and rerank as
the api; stage C was run on the live cluster. Reference point, the shipped configuration
(200/30, filter on, OCR on): **recall@5 0.658**, recall@3 0.632, 1,038 chunks.
Raw outputs: `docs/evidence/failure-*.txt`, `pin-guard.txt`.
From the RAG lecture: only F is the model's fault, so A to E are checked first.

## Stage A · Extraction
- **What we broke:** indexed without OCR (`OCR=false` in the ConfigMap; offline:
  `tune_chunking.py --no-ocr`).
- **Symptom:** the index shrank from **1,038 to 993 chunks**. The scanned ORIC
  *Faculty Entrepreneurship and Technical Start-ups Policy* (8 image-only pages, 26
  chunks) disappeared completely, along with the text of the 9 image pages in the two
  handbooks. **recall@5 stayed at 0.658**, because no golden question asks about the
  start-up policy. The failure was silent: no error, no change in the headline metric, and
  every question about that policy would now be refused or answered from the wrong text.
- **Diagnosis, as if we had not caused it:** the indexer's per-document log line. With OCR
  the policy logs `"pages": 8, "ocr_pages": 8, "chunks": 26`. Without it, its chunks drop
  to 0 while its pages stay at 8, the signature of a missing text layer. `pdffonts` on the
  file confirms there are no fonts. The total `"chunks"` in the `chunking done` line also
  drops by 45.
- **Remedy:** OCR every page with fewer than 25 characters of text (`extract.py`,
  default `OCR=true`); watch `empty_pages` and per-document chunk counts in the indexer
  log; and add a golden question for every document, because **an eval only catches what
  it asks about**.

## Stage B · Chunking
- **What we broke:** fixed-size 100-token windows with no overlap
  (`CHUNK_STRATEGY=fixed`, `CHUNK_SIZE=100`, `CHUNK_OVERLAP=0`).
- **Symptom:** **recall@5 fell from 0.658 to 0.105** (recall@3 0.632 → 0.105). By type:
  ordinary 0.773 → 0.091, version 0.667 → 0.0, injection 1.0 → 0.0; exact stayed at 0.25.
  The windows contain the right words, but they no longer know which section they are in
  (`section="(fixed window)"`), so they cannot be matched to a clause, cannot be cited
  usefully, and a 100-word window cuts the Generative AI policy's "Unacceptable Uses"
  list mid-item.
- **Diagnosis:** recall@5 by type in `eval/results.json` drops across *all* types at once
  (a retrieval-only change would hit some types more than others); the per-request
  log's `chunk_ids` end in `-f0001…` and citations name "(fixed window)".
- **Remedy:** structure-aware chunking with section labels and sentence overlap
  (ADR 0002).

## Stage C · Retrieval
- **What we broke:** `EMBED_MODEL=all-minilm` on the api only, with the index still built
  with `nomic-embed-text` (`kubectl set env deployment/api …`).
- **Symptom:** without the guardrail, retrieval would compare vectors from two different
  models and return confident nonsense with no error. With it, the new pod sat at **0/1
  READY** while the three old pods kept serving (`maxUnavailable: 0`), so users saw no
  change.
- **Diagnosis** (`docs/evidence/pin-guard.txt`): `kubectl get pods` (new pod 0/1) →
  `kubectl describe pod` (*"Readiness probe failed: HTTP probe failed with statuscode:
  503"*) → `kubectl logs` on the new pod: *"embedding pin check failed", "conflict":
  "embedding model mismatch: index 'rahbar-v1794388406' was built with 'nomic-embed-text'
  (dimension 768), but this API has EMBED_MODEL='all-minilm'"*.
- **Remedy:** the pin check (`api/app/retrieval/chroma.py`, `pin_conflict`), and
  `kubectl rollout undo deployment/api`, which restored all three pods to ready.

## Stage E · Source
- **What we broke:** removed the superseded filter (`--no-filter`; in the api, a
  retriever that always passes `include_superseded=True`).
- **Symptom:** **recall@5 fell from 0.658 to 0.632**; ordinary 0.773 → 0.727. Chunks from
  the repealed Jun 2025 handbook (174 of them, near-duplicates of the 2026 text) compete
  for the same top-5 slots and push a current section out. Where they win, the model
  faithfully quotes the old rule, e.g. the 2025 summer rules, which have no 80 %
  attendance clause.
- **Diagnosis:** the per-request log's `chunk_ids` include
  `academic-rules-handbook-jun-2025-…` for a question that did not ask about 2025, and
  the response's `retrieved` list shows superseded sources. A better prompt cannot fix
  this: the model answers from what it is given.
- **Remedy:** metadata filtering `where={"superseded": False}` (retrieval/chroma.py)
  unless `wants_old_version()` sees "previous / 2025 handbook / older rules" (guards.py).
  With the filter, golden q14 ("Under the previous 2025 handbook…") still retrieves the
  2025 text, because it asks for it.

## Not broken here (two of six were optional)
- **D · Prompt** (remove "use the supplied context"): expected symptom is answers from
  general knowledge on the must-abstain questions. The relevance floor now stops both
  before the model, so the abstention gate alone would not catch this; questions just
  above the floor would.
- **F · Generation** (temperature 1.2): expected symptom is validation retries
  (`rahbar_generation_retries_total`) and fallbacks rising, not worse retrieval.
