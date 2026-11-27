# The data lake: verify and update (Assignment 03, Part A)

RAGOps paper, sections 4.2.1 to 4.2.4: ingest, verify, update the data lake, update the
retrieval sources. In RAHBAR all four run in one Kubernetes Job, the ingestion Job
(`k8s/base/indexer-job.yaml`, re-run weekly by `k8s/base/reindex-cronjob.yaml`). It is
the only writer to the vector store.

```
corpus/sources.yaml ──fetch──▶ /datalake/sources ─┐
kubectl cp (make batch) ─────▶ /datalake/inbox/<batch> ─┤
documents already in the lake ───────────────────┤
                                                 ▼
                         extract + clean ─▶ VERIFY (4 checks) ─▶ verify-<run>.json
                                                 │
            pass ─▶ lake/<doc_id>/<version>/     │  older version ─▶ archive/<doc_id>/<version>/
            fail ─▶ quarantine/<run>/<file> + <file>.reason.json
                                                 ▼
                     state/catalog.json ─▶ SYNC the collection ─▶ ingest-<run>.json
```

Everything lives on the `datalake` PersistentVolumeClaim (`k8s/base/datalake-pvc.yaml`),
so it survives pod restarts and Job re-runs.

| Path on the PVC | What it holds |
| --- | --- |
| `sources/` | the fetched corpus (download cache; `_index.json` carries each file's metadata) |
| `inbox/<batch>/` | new documents waiting for the next run; `_meta.yaml` describes files without front matter |
| `lake/<doc_id>/<version>/` | verified, current documents |
| `archive/<doc_id>/<version>/` | verified, superseded versions: kept and restorable |
| `quarantine/<run_id>/` | documents that failed verification, each with `<file>.reason.json` |
| `state/catalog.json` | one record per document version: the source of truth the vector store is synced to |
| `state/fingerprints.json` | cached document fingerprints (so an unchanged corpus embeds nothing) |
| `control/` | queued requests (restore, release, delete); `control/applied/` keeps their outcomes |
| `reports/` | `verify-<run_id>` and `ingest-<run_id>`, as JSON and as text |

## A1. Verification (`indexer/pipeline/verify.py`)

Runs after extraction and before chunking. Every document gets a pass or fail, with a
reason, for each check.

| Check | Fails when | Then |
| --- | --- | --- |
| quality | the parser raised (failed extraction); fewer than `MIN_TEXT_CHARS` characters (empty); more than `MAX_GARBLED_RATIO` of the characters unreadable (garbled) | quarantine |
| completeness | missing source, title, date or version, or page count; or truncated: a PDF without its `%%EOF` trailer, HTML without `</html>`, text that stops mid-sentence | quarantine |
| uniqueness | the same SHA-256 as a document before it (exact duplicate), or fingerprint cosine >= `NEAR_DUP_THRESHOLD` with a document that has a different `doc_id` (near duplicate) | quarantine |
| recency | a newer version of the same `doc_id` is in the lake (or the metadata marks it superseded) | archive: kept, restorable, excluded from retrieval by `superseded: true` |

The fingerprint is the mean embedding of three 1,500-character windows (start, middle,
end) with the index's own model, cached by SHA-256. Versions of one document are not
compared for near-duplication; they are recency's job.

Thresholds are ConfigMap values (`k8s/base/configmap.yaml`), set from measurements on
our corpus; the verification report prints each document's garbled ratio and its nearest
other document with the cosine, which is the evidence for the values.

Planted instructions are not a verification check: retrieved text is data, and the
query-time guard handles them (Part D4). `corpus/batch-b/b10-counselling-services.md`
tests that.

## A2. Incremental update (`indexer/pipeline/sync.py`)

One collection per configuration: `rahbar-<embedding model>-c<size>o<overlap>`, for
example `rahbar-nomic-embed-text-c200o30`. Every chunk carries `doc_id`, `doc_key`
(`<doc_id>@<version>`) and `doc_sha256`. Each run compares the catalog with the
collection, document by document:

| Case | What happens | Chunks embedded |
| --- | --- | --- |
| unchanged (same `doc_key`, same SHA-256) | nothing | 0 |
| archived or restored | only the `superseded` flag of its chunks changes | 0 |
| changed (same `doc_key`, new SHA-256) | its old chunks are deleted, its new chunks added; no other document is touched | its own |
| new | chunked, embedded, added | its own |
| deleted | every chunk with its `doc_key` is deleted | 0 |

After the sync the Job counts chunks per `doc_id` and fails if any chunk belongs to a
`doc_key` the catalog does not have (an orphan). The pointer collection
(`rahbar-active`, as in Assignment 02) gets a new `generation` only when the content
changed; the api's semantic cache keys on it, so a changed document never serves a
cached answer built from its old chunks.

## Commands

| What | Command |
| --- | --- |
| Run the ingestion Job and print its reports | `make ingest` |
| Add a batch of new documents | `make batch DIR=corpus/batch-b`, then `make ingest` |
| Where every document version is | `make lake-status` |
| The latest verification / ingestion report | `make lake-report KIND=verify` / `KIND=ingest` |
| Roll back to an archived version | `make restore DOC=<doc_id>@<version>`, then `make ingest` |
| Undo the rollback | `make release DOC_ID=<doc_id>`, then `make ingest` |
| Delete a document and all its chunks | `make delete-doc DOC=<doc_id>@<version>`, then `make ingest` |
| Copy reports and the catalog into the repository | `make lake-pull` (then commit `eval/results/datalake/`) |

Requests are queued on the PVC by the `datalake-shell` pod and applied by the next
ingestion run, so the Job stays the only writer. The shell pod has no network egress.
