# ADR 0001: The indexer is a Job; the API only reads

## Status
Accepted · 2026-11-03 (store-and-flip added 2026-11-06)

## Context
Indexing runs once, and again when the material changes; querying runs on every question.
Our API runs as 3 replicas behind one Service, and Chroma keeps its index on one
ReadWriteOnce volume (`k8s/base/chroma-statefulset.yaml`). Extracting, OCR-ing and
embedding ~200 pages takes minutes and far more memory than answering a question.

If indexing ran in the API's startup:
1. **Three writers, one volume.** Three pods would each delete and rebuild the same
   collection at the same time, each halfway through a different document. The result is
   an index that no single run produced, and nothing would error.
2. **Every restart re-indexes.** A rolling update, an OOMKill, a node drain or a readiness
   blip would each pay the full extraction and embedding cost again, and hit the
   university website again.
3. **Readiness becomes a lie.** Either the pod says "ready" before the index exists and
   serves empty retrievals, or it takes minutes to become ready, the startup probe's
   60-second budget (`failureThreshold: 30 × periodSeconds: 2`) runs out, and the kubelet
   kills it halfway: a rolling update swaps three working pods for three that are still
   chunking PDFs.
4. **No re-index without a redeploy.** A new handbook would need a new API rollout.

## Decision
- A **Job** (`k8s/base/indexer-job.yaml`, `restartPolicy: OnFailure`, `backoffLimit: 3`)
  is the only process that writes to Chroma. A **CronJob** (`reindex`, Sundays 03:00 PKT,
  `concurrencyPolicy: Forbid`) runs the same image and pod spec weekly. Re-indexing by hand
  is one command with no redeploy and no downtime:
  `kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar`.
- An init container fetches the corpus listed in `corpus/sources.yaml` into an `emptyDir`
  and compares it with the committed `corpus/MANIFEST.md`; the main container runs
  extract → clean → chunk → embed → store and exits.
- **Store and flip** (`indexer/pipeline/store.py`): each run builds a *new* collection
  `rahbar-v<unix time>` with `embed_model` and `dimension` in its metadata, verifies the
  count, then upserts one record in the pointer collection `rahbar-active`. The API
  re-reads the pointer at most every 10 s (`api/app/retrieval/chroma.py`, `_resolve`), so
  it switches to the new index without a restart. The previous version is kept for a
  manual rollback; older ones are deleted.
- The API's readiness (`/readyz`) is 200 only when Chroma answers, the active collection
  exists and is non-empty, its `embed_model` equals the API's `EMBED_MODEL`, and Redis
  answers. Until the Job has run, API pods are withheld from the Service, by design.

## Options considered
- **Index in API startup**: rejected for the four reasons above (and it is a −10).
- **A leader-elected replica that indexes**: still couples a minutes-long batch task to a
  serving pod's lifecycle and needs RBAC for a Lease; a Job is the Kubernetes object made
  for run-to-completion work.
- **Delete-and-rebuild the live collection in the Job**: simpler, but for the whole
  rebuild the API would read a half-written index. Store-and-flip costs one tiny extra
  collection.
- **Bake the corpus into the image**: reproducible, but a weekly CronJob would re-index
  identical bytes and a re-published PDF would need an image rebuild. Fetching at Job
  start (with the manifest check) keeps "re-index" a one-command operation.

## Consequences
- Easier: rollouts are fast and never touch the index; re-index is a command; a failed
  index run never replaces a good one.
- Harder: a fresh cluster has a window where API pods are Running but 0/1 READY until the
  Job finishes. The README's apply order says so.
- Watch for: the Job's memory (OCR and embedding peaks, ENGINEERING-NOTES Q4); orphan
  `-v` collections if a Job is killed mid-write (the next run deletes them); the Job
  manifest is immutable, so re-applying with a new image needs
  `kubectl delete job indexer -n rahbar` first.
