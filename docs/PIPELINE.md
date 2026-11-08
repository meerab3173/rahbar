# Indexing and retrieval pipeline

> Numbers marked **[measure]** come from running the commands next to them against the
> real corpus with Ollama (`make tune`, `make cache-threshold`, `make eval`); paste the
> printed tables in place. Numbers marked *(fixtures)* were measured offline on the CI
> fixture corpus with `hash-bow-768` and are what CI's gate uses.

## Extraction (`indexer/pipeline/extract.py`)
- **PDF:** pdfplumber text layer, page by page (`x_tolerance=1.5`). A page with fewer than
  25 characters is treated as having no text layer, rasterised at 300 dpi (pdf2image /
  poppler) and OCR'd with Tesseract (`eng`); the page is marked `ocr=True` and counted in
  the `ocr_pages` log field. Tables found by pdfplumber are appended as pipe rows.
- **HTML** (Fee Structure, Admissions FAQ): visible text; nav, header, footer, script and
  style dropped; every `<tr>` becomes a `| … |` row. **Markdown** (CI fixtures): front
  matter removed.
- **Needed OCR:** the ORIC *Faculty Entrepreneurship and Technical Start-ups Policy* (all
  pages). **[measure]** its `ocr_pages` count from the indexer log:
  `kubectl logs job/indexer -n rahbar | grep faculty-entrepreneurship`.
- **Mangled table, before and after.** A table without ruling lines (like clause 2.3's
  grade points) is missed by pdfplumber's table finder, and its text layer comes out as
  one run of tokens. In our test PDF (`indexer/tests/data/sample-handbook.pdf`, built with
  that layout; **[measure]** check the real one with `make sections` and the chunk text):

  | Before (raw text joined into a paragraph) | After (row rule in `chunk.py`) |
  | --- | --- |
  | `2.3 Grade points are awarded as follows: Grade Points A+ 4.00 B- 2.67 C 2.00` | `2.3 Grade points are awarded as follows:` / `A+ 4.00` / `B- 2.67` / `C 2.00`, one row per line |

  A short line that ends in a number (`_ROWLIKE`) is kept as its own line, so "B- 2.67"
  stays a pair; the repeated header row "Grade Points" is not mistaken for a new heading.
- **Could not recover:** multi-column tables in the annual reports whose columns are
  separated by whitespace only come out row by row but with merged cells where a cell
  wraps; the OCR'd policy loses bold/italic emphasis and any signature/stamp text.

## Cleaning (`indexer/pipeline/clean.py`)
Rules applied:
1. A line that appears (digits masked) on ≥ 50 % of a document's pages is a running
   header or footer and is removed ("NATIONAL UNIVERSITY OF COMPUTER AND EMERGING
   SCIENCES", "Undergraduate Academic Rules and Regulations 2026 17").
2. Bare page numbers ("17", "Page 17 of 53", "IV") are removed.
3. Table-of-contents pages are dropped: they share every keyword with every question and
   answer none of them (dot leaders, or ≥ 70 % of lines ending in a page number on the
   first six pages).
4. Words hyphenated across a line break are re-joined ("ineli-\ngible" → "ineligible").
5. Bullet glyphs (•, ▪, Wingdings private-use characters) become "- ".

Deliberately **not** stripped: headings and clause numbers (chunk labels and citations
depend on them), table rows, and text that looks like an instruction to the model
("Ignore all previous instructions…" in the fixture). Retrieved text is data; it is
handled at query time (delimiters + the echo guard), not hidden at index time.

## Chunking (`indexer/pipeline/chunk.py`, ADR 0002)
Unit: **tokens = whitespace-separated words.** Sections (heading stack, label = chapter ›
deepest heading), then paragraphs (a clause number, list item or table row starts a new
one), then sentences. Overlap = whole trailing sentences, never across a heading.

**Shipped: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=50`** (ConfigMap), justified below.

Three configurations, recall@5 on `eval/golden.yaml` (18 scored questions; the 2
must-abstain questions have no relevant passages), from `make tune`:

| Size (tokens) | Overlap (tokens) | recall@3 | recall@5 | Chunks stored | Amplification |
| --- | --- | --- | --- | --- | --- |
| 200 | 30 | **[measure]** | **[measure]** | **[measure]** | **[measure]** |
| 400 | 50 | **[measure]** | **[measure]** | **[measure]** | **[measure]** |
| 800 | 100 | **[measure]** | **[measure]** | **[measure]** | **[measure]** |

*(fixtures, hash-bow-768)*: 40:0 → recall@5 1.0, 15 chunks; 80:15 → 1.0, 11 chunks;
200:30 → 1.0, 11 chunks. The fixture sections are all shorter than 80 words, so they
cannot separate the settings; that is why the choice is made on the real corpus.

**Why 400/50:** ship the smallest size whose recall@5 is within one question of the best
(smaller chunks give the 3B model less irrelevant text and cheaper prompts: top_k 5 × 400
words ≈ 2.6k tokens, inside `num_ctx 8192`), and the overlap that keeps a two-sentence
clause whole across a boundary. If 800/100 measures clearly higher, ship it and say so.

**Storage amplification** = stored tokens / original tokens (the lecture's 150/120 = 1.25).
Ours is above 1 for two reasons: the overlap, and the section label prefixed to every
chunk. **[measure]** at 400/50 (the indexer logs it: `"storage_amplification"`).

## Embedding (`indexer/pipeline/embed.py`, `api/app/embedding.py`)
- Model **`nomic-embed-text`** via Ollama `/api/embed`, **768 dimensions**, batches of 32.
  Task prefixes as the model was trained: `search_document: ` at index time,
  `search_query: ` at query time.
- **The pin lives in one place:** `EMBED_MODEL` in `k8s/base/configmap.yaml`, read by
  both the indexer and the api through `envFrom`.
- **Enforced in code:** the indexer writes `embed_model` and `dimension` into the
  collection metadata (`store.py`); the api compares them with its own `EMBED_MODEL` at
  boot and on every readiness probe and query (`retrieval/chroma.py`, `pin_conflict`), logs
  `"embedding pin check failed"` with both values, and `/readyz` returns 503. A
  dimension mismatch on a query is refused as well.
- CI uses `hash-bow-768`, a deterministic feature-hashing embedding with no model, so the
  e2e job runs offline. Both copies are pinned to the same fingerprint by tests.

## Retrieval
Top-k = 5 from Chroma (cosine; similarity = 1 − distance), with
`where={"superseded": False}` unless the question explicitly asks about an older version.
We fetch 4 × k candidates and rerank with `similarity + 0.15 × exact-term match`
(clause numbers, decimals, amounts, codes found verbatim in the chunk): a light hybrid
step for what embeddings blur. **Relevance floor** `0.45`: if the best similarity is
below it, the api abstains without calling the model. **[measure]** confirm with
`make tune` (`--floor-report` prints the top score per question): the floor must sit above
both must-abstain questions and below every answerable one. *(fixtures: abstain 0.078,
lowest answerable 0.181 → CI floor 0.12.)*

## Semantic cache (`api/app/services/cache.py`, ADR 0003)
Hit = cosine ≥ `CACHE_THRESHOLD` **and** the same exact terms in both questions.
Thresholds tried with `make cache-threshold` on `eval/cache_pairs.yaml` (6 paraphrase
pairs, 6 look-alike pairs with different answers):

| Threshold | True hits | False hits | Verdict |
| --- | --- | --- | --- |
| 0.90 | **[measure]** | **[measure]** | **[measure]** |
| 0.93 | **[measure]** | **[measure]** | **[measure]** |
| 0.95 | **[measure]** | **[measure]** | **[measure]** |

The false hit that motivated the exact-term guard, measured *(hash-bow-768)*: "What is
the grade point for a B-?" vs "…for a B+?" score **1.000**, so *no* threshold can
separate them by similarity alone: the embedding's tokenizer drops "+" and "-". With the
guard, 0 false hits at every threshold. Shipped: **0.95**, or the lowest threshold with
zero false hits in the table above.

## Evaluation method (`eval/run_eval.py`)
- **recall@k:** for each golden `(source, section)` pair, it counts as found if any of the
  top-k retrieved passages has the same source (case-insensitive) and a section label that
  *contains* the golden section text (labels are "chapter > heading", golden sections name
  the heading). Per question: found / relevant; reported as the mean over questions that
  have relevant passages, overall and by type. An empty golden section matches any chunk of
  that source (used for the HTML pages, whose headings we do not control).
- **Citation validity:** pooled over the run: citations whose `(source, section)` equals a
  passage in that response's `retrieved` list / all citations.
- **Abstention correctness:** must-abstain questions with `answered_from_context: false`
  / must-abstain questions. Must be 1.0.
- **Gate (section 5.4):** recall@5 more than 5 points below `eval/baseline.json`, any
  must-abstain answered, or citation validity < 95 % → exit 1. CI uses the same code on
  the fixtures against `eval/fixtures/baseline.fixture.json`.
