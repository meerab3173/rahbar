# Indexing and retrieval pipeline

All numbers below were measured on our minikube stack and laptop (Ollama with
`nomic-embed-text` and `llama3.2:3b`) on 11 Nov 2026. Raw outputs are in `docs/evidence/`
(`chunking.txt`, `floor-200.txt`, `hybrid-off.txt`, `cache-threshold.txt`,
`failure-*.txt`) and `eval/results.json`.

## Extraction (`indexer/pipeline/extract.py`)
- **PDF:** pdfplumber text layer, page by page (`x_tolerance=1.5`). A page with fewer than
  25 characters is treated as having no text layer: rasterised at 300 dpi (pdf2image /
  poppler) and OCR'd with Tesseract (`eng`); the page is marked `ocr=True` and counted in
  the `ocr_pages` log field. Tables found by pdfplumber are appended as pipe rows.
- **HTML** (Fee Structure, Admissions FAQ): visible text; nav, header, footer, script and
  style dropped; every `<tr>` becomes a `| … |` row. **Markdown** (CI fixtures): front
  matter removed.
- **Needed OCR:** the ORIC *Faculty Entrepreneurship and Technical Start-ups Policy*,
  **8 of 8 pages** (scanned, no fonts); also 5 image pages in the 2026 Academic Rules
  Handbook and 4 in the Student Handbook (17 OCR pages in total, from the indexer log).
- **Mangled table, before and after.** A table without ruling lines, like clause 2.3's
  grade points, is missed by pdfplumber's table finder, and its text layer comes out as
  one run of tokens. Reproduced in our test PDF (`indexer/tests/data/sample-handbook.pdf`,
  same layout):

  | Before (raw text joined into a paragraph) | After (row rule in `chunk.py`) |
  | --- | --- |
  | `2.3 Grade points are awarded as follows: Grade Points A+ 4.00 B- 2.67 C 2.00` | `2.3 Grade points are awarded as follows:` / `A+ 4.00` / `B- 2.67` / `C 2.00`, one row per line |

  A short line that ends in a number (`_ROWLIKE`) is kept as its own line, so "B- 2.67"
  stays a pair; the repeated header row "Grade Points" is not mistaken for a new heading.
- **Could not recover:**
  - The Student Handbook prints some chapter headings on the same line as other text
    ("Academic Dishonesty IV - MIGRATION", "Turnitin Reports VIII - COMPUTER LAB RULES"),
    so after page 18 its section labels stay under "II - FINANCIAL RULES". The text is
    indexed, but the labels are wrong for that document. Golden q11 and q17 match on the
    library table's label ("Patron Loan"), and q10 on the document only.
  - Multi-column annual-report tables come out row by row, but cells that wrap merge.
  - OCR loses emphasis and any stamp or signature text.

## Cleaning (`indexer/pipeline/clean.py`)
Rules applied:
1. A line that appears (digits masked) on ≥ 50 % of a document's pages is a running
   header or footer and is removed ("NATIONAL UNIVERSITY OF COMPUTER AND EMERGING
   SCIENCES", "Undergraduate Academic Rules and Regulations 2026 17").
2. Bare page numbers ("17", "Page 17 of 53", "IV") are removed.
3. Table-of-contents pages are dropped: they share every keyword with every question and
   answer none (dot leaders, or ≥ 70 % of lines ending in a page number on pages 1–6).
4. Words hyphenated across a line break are re-joined ("ineli-\ngible" → "ineligible").
5. Bullet glyphs (•, ▪, Wingdings private-use characters) become "- ".

Deliberately **not** stripped: headings and clause numbers (labels and citations depend
on them), table rows, and text that looks like an instruction to the model. Retrieved text
is data; it is handled at query time (delimiters plus the echo guard), not hidden at index
time.

## Chunking (`indexer/pipeline/chunk.py`, ADR 0002)
Unit: **tokens = whitespace-separated words.** Sections first (heading stack; the label is
chapter › deepest heading), then paragraphs (a clause number, list item or table row
starts a new one), then sentences. Overlap = whole trailing sentences, never across a
heading.

**Shipped: `CHUNK_SIZE=200`, `CHUNK_OVERLAP=30`** (ConfigMap).

Three configurations, on the 19 golden questions that have relevant passages (`make tune`,
`docs/evidence/chunking.txt`):

| Size (tokens) | Overlap (tokens) | recall@3 | recall@5 | Chunks stored | Amplification |
| --- | --- | --- | --- | --- | --- |
| **200** | **30** | **0.632** | **0.658** | 1,038 | 1.113 |
| 400 | 50 | 0.605 | 0.632 | 777 | 1.056 |
| 800 | 100 | 0.605 | 0.632 | 696 | 1.039 |

| Size:overlap | recall@5 ordinary (11) | recall@5 version (3) | recall@5 exact (4) | recall@5 injection (1) |
| --- | --- | --- | --- | --- |
| 200:30 | **0.773** | 0.667 | 0.25 | 1.0 |
| 400:50 | 0.727 | 0.667 | 0.25 | 1.0 |
| 800:100 | 0.727 | 0.667 | 0.25 | 1.0 |

**Why 200/30:** it has the best recall@5 and recall@3, and the gain is entirely on ordinary
questions (one more relevant section found). It is also the smallest, so the 3B model gets
shorter prompts (5 × 200 words) and answers faster. The cost is 261 more chunks than
400/50 and amplification 1.113 instead of 1.056. Larger chunks did not help exact or
version questions at all, so there was no reason to pay their longer prompts.

**Storage amplification** = stored tokens / original tokens (the lecture's 150/120 =
1.25). Ours is **1.113** at 200/30, logged by the indexer as `storage_amplification`.
Two causes: the 30-token overlap, and the section label prefixed to every chunk.

## Embedding (`indexer/pipeline/embed.py`, `api/app/embedding.py`)
- Model **`nomic-embed-text`** via Ollama `/api/embed`, **768 dimensions**, batches of 32,
  with the model's task prefixes (`search_document: ` / `search_query: `). Embedding the
  1,038 chunks took **729 s** on the laptop CPU; the whole indexer run took **859 s**.
- **The pin lives in one place:** `EMBED_MODEL` in `k8s/base/configmap.yaml`, read by both
  the indexer and the api through `envFrom`.
- **Enforced in code:** the indexer writes `embed_model` and `dimension` into the
  collection metadata (`store.py`); the api compares them with its own `EMBED_MODEL` at
  boot and on every readiness probe and query (`retrieval/chroma.py`, `pin_conflict`).
  Demonstrated (`docs/evidence/pin-guard.txt`): with `EMBED_MODEL=all-minilm` the new pod
  stayed **0/1**, logged *"embedding model mismatch: index 'rahbar-v1794388406' was built
  with 'nomic-embed-text' (dimension 768), but this API has EMBED_MODEL='all-minilm'"*,
  and the kubelet reported *"Readiness probe failed: … statuscode: 503"*, while the three
  old pods kept serving.
- CI uses `hash-bow-768`, a deterministic feature-hashing embedding with no model, so the
  e2e job runs offline; both copies are pinned to the same fingerprint by tests.

## Retrieval
Top-k = 5 from Chroma (cosine; similarity = 1 − distance), with
`where={"superseded": False}` unless the question explicitly asks about an older version.
We fetch 4 × k candidates and rerank with `similarity + 0.15 × exact-term match`
(clause numbers, decimals, amounts, codes found verbatim in the chunk).

**Relevance floor: `0.69`.** Top similarity per golden question at 200/30
(`docs/evidence/floor-200.txt`):

| | Lowest / highest |
| --- | --- |
| Must-abstain questions (q19 hostel, q20 transport) | 0.640, **0.686** |
| Answerable questions (19) | **0.699** (q17 library fine) … 0.885 |

The floor sits in the 0.013 gap between them, so both absent-topic questions are refused
**before the model is called** (0.1–0.4 s in the eval), and no answerable question is
blocked. The margin is thin, so the model's own instruction to decline when the evidence
does not answer is kept as the second safeguard. With the old floor of 0.45 the hostel
question scored 0.618 and was declined only by the model.

## Semantic cache (`api/app/services/cache.py`, ADR 0003)
A hit needs cosine ≥ `CACHE_THRESHOLD` **and** the same exact terms in both questions
(numbers, clause numbers, letter grades with signs, "summer/regular", course codes).
Measured with `make cache-threshold` on `eval/cache_pairs.yaml` (6 paraphrase pairs with
the same answer, 6 look-alike pairs with different answers):

| Threshold | Without guard: true / false hits | With guard: true / false hits | Verdict (with guard) |
| --- | --- | --- | --- |
| 0.85 | 3/6 · **4** | 2/6 · **1** | ruled out: "deadline to withdraw" ~ "deadline to add or drop" (0.854) |
| **0.88** | 3/6 · 3 | **2/6 · 0** | **shipped: lowest with zero false hits** |
| 0.90 | 2/6 · 2 | 1/6 · 0 | safe, fewer hits |
| 0.94 | 0/6 · 0 | 0/6 · 0 | safe without the guard, but useless (no hits) |

The false hit that motivated the guard: "What is the grade point for a **B-**?" and
"…for a **B+**?" score **0.926**, higher than five of the six true paraphrases, but have
different answers (2.67 and 3.33). Without the guard the lowest safe threshold is 0.94,
which serves 0 of 6 paraphrases. With it, 0.88 is safe and serves 2 of 6. In the golden
run, q07 ("minimum CGPA required to graduate with a bachelor's degree") was served from
the cache entry of "What is the minimum CGPA required to graduate?" in **0.37 s**.

Never cached: follow-ups, old-version questions, abstentions, fallbacks, and answers that
lost a citation. Invalidation: the cache key contains the active index generation, so a
re-index makes every old entry unreachable; the old hashes are then deleted.

## Evaluation method (`eval/run_eval.py`)
- **recall@k:** a golden `(source, section)` pair is found if any of the top-k retrieved
  passages has the same source and a section label that **contains** the golden section
  text (labels are "chapter > heading"). An empty golden section matches any chunk of
  that source. Per question: found / relevant; reported as the mean over the questions
  that have relevant passages, overall and by type.
- **Citation validity:** pooled over the run: citations whose `(source, section)` equals a
  passage in that response's `retrieved` list / all citations.
- **Abstention correctness:** must-abstain questions with `answered_from_context: false`
  / must-abstain questions. Must be 1.0.
- **Gate (section 5.4):** recall@5 more than 5 points below `eval/baseline.json`, any
  must-abstain answered, or citation validity < 95 % → exit 1. CI uses the same code on
  the fixtures against `eval/fixtures/baseline.fixture.json`.

**Baseline (end-to-end through the Ingress, `eval/baseline.json`):** recall@3 **0.579**,
recall@5 **0.658**, citation validity **1.0**, abstention **1.0**; recall@5 by type:
ordinary 0.773, version 0.667, exact 0.25, injection 1.0. Gate passed.
