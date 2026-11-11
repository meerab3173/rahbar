# Engineering notes

Line numbers refer to the files as committed.

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?
Three pods start together (`k8s/base/api-deployment.yaml:8`, `replicas: 3`) and each
would delete and rebuild the collection on the single RWO volume of `chroma-0`
(`chroma-statefulset.yaml:64`, `volumeClaimTemplates`). Their `add` calls interleave, so
the collection ends up with chunks from three half-finished runs, or one pod deletes the
collection while another is writing: an index no run produced, served without an error.
Every rolling update (`maxSurge: 1`, line 15) would add a fourth writer, and every restart
would pay the full cost again; ours is 859 s per index (OCR on 17 pages plus 729 s of
embedding), far beyond the startup probe's 60 s, so the kubelet would kill each pod
mid-index. The object that prevents it is the **Job** (`indexer-job.yaml:5`, `kind: Job`,
`restartPolicy: OnFailure` at line 20): the only writer, run once, with
`concurrencyPolicy: Forbid` on the CronJob (`reindex-cronjob.yaml:11`) so two indexers
never overlap. The api has no write call at all; `indexer/pipeline/store.py:115` is the
only `add`, and it writes a new versioned collection that is switched on only when
complete (ADR 0001).

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?
Liveness would then call `/readyz`, which waits on Chroma. With `periodSeconds: 20` and
`failureThreshold: 3` (`api-deployment.yaml:57-58`), the kubelet kills each container after
about 60 s of a slow Chroma. All three pods restart at roughly the same time. The startup
probe gives each 60 s (`failureThreshold: 30 × periodSeconds: 2`, lines 53-54), Chroma is
still slow, so they are killed again: a restart storm, turning a 90 s slowdown into a full
outage with CrashLoopBackOff on top. Meanwhile, readiness on `/healthz` would keep sending
traffic to pods that cannot answer. As built, `/healthz` touches nothing
(`api/app/routes/health.py:20-22`), so a slow Chroma only marks pods unready (removed from
the Service, not restarted), and they return the moment Chroma recovers. We saw this when
we deleted `chroma-0`: the api pods stayed Running, and the next question was answered
from the same index once Chroma was back (`docs/evidence/pod-delete.txt`).

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?
`hpa.yaml` scales on CPU utilisation = usage / `requests.cpu` (100m,
`api-deployment.yaml:46`). Ours reads **`cpu: 4%/70%`** (`docs/evidence/get-hpa.txt`), although each golden
question kept a request open for 40–80 s waiting on Ollama: during that wait the uvicorn
worker is blocked on a socket, not computing. So the HPA measures JSON parsing
and Pydantic validation, not load. It scales late or never, exactly when the queue is
longest. The right signal is **concurrency or queue depth**: in-flight requests per pod,
or p95 of `rahbar_request_seconds` / `rahbar_generation_seconds`. To use it we would need
Prometheus scraping `/metrics`, an in-flight gauge (a middleware that increments and
decrements `rahbar_inflight_requests`), and the Prometheus Adapter or KEDA to expose it as
a `type: Pods` metric with a target such as 2 in-flight requests per pod. Scaling the api
also does nothing if the bottleneck is the single Ollama on the laptop, so the real
capacity limit is the model server.

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong
Measured with `kubectl top pods -n rahbar --containers` sampled every 5 s during a full
re-index (`docs/evidence/kubectl-top.txt`): **peak 549 Mi** for the indexer container.
Limit: **1536 Mi** (`indexer-job.yaml:68`), request 512 Mi. Margin = 1536 − 549 = **987 Mi
(64 %)**. We keep it larger than the usual 30–50 % because the peaks are short and a 5 s
sample can miss them: OCR holds a 300 dpi page bitmap (~25 MB each in Pillow), and before
the store all 1,038 chunk texts and 768-dimension vectors are in memory together. If we
set the limit below the real peak, the kernel OOM-kills the container: **exit code 137**
(128 + SIGKILL 9), `Reason: OOMKilled` in `kubectl describe pod`, and the Job retries up
to `backoffLimit: 3` before failing. The old index keeps serving throughout, because a new
one is only switched on after a complete store. Padding it far higher is a different
waste: the request is reserved on the node for the whole run.

## 5. recall@5 by question type: where is it worst, and why?
From `eval/results.json`: ordinary **0.773** (11 questions), version **0.667** (3),
injection **1.0** (1), **exact 0.25 (4)**: worst by far. Exact questions turn on a token
whose identity matters ("clause 1.21", "B-", "PKR per day", "Rs 12,000"). A dense
embedding captures what a passage is *about*, not which number it contains, so
"clause 1.21" sits close to every attendance or grading passage, and the handbook,
prospectus and Student Handbook all discuss the same topics with different numbering (the
prospectus calls attendance clause 1). The lecture's remedy is **hybrid retrieval**: a
lexical BM25 score fused with the dense score (reciprocal-rank fusion), optionally
followed by a cross-encoder reranker. We implemented only a light version: 4 × k dense
candidates reranked with an exact-term boost (`api/app/retrieval/chroma.py:50`,
`exact_boost`; weight 0.15 at line 209). Measured honestly, it made **no difference**:
exact recall@5 is 0.25 with the boost and 0.25 without it (`docs/evidence/hybrid-off.txt`),
most likely because the right chunk is not among the 20 dense candidates in the first place.
Reranking cannot promote what was never retrieved. Real BM25 over the whole index, as a
second candidate source, is the fix.

## 6. EMBED_MODEL changed on the API only: what does your system do?
At boot the api reads the active collection's metadata (`api/app/main.py:48-50` →
`retrieval/chroma.py:105`, `pin_conflict`) and logs `"embedding pin check failed"` with
the index model, its dimension and the api's model. `/readyz` returns 503 with
`"failed": ["embed_model"]` (`routes/health.py:26-29`), so the pod stays 0/1 READY and is
never added to the Service; a query would also be refused (`_check_pin`, and the dimension
check at `chroma.py:184`). During a rollout the old pods keep serving because
`maxUnavailable: 0`. We ran exactly this (`docs/evidence/pin-guard.txt`): the new pod sat
at 0/1 while three pods served; the log said *"embedding model mismatch: index
'rahbar-v1794388406' was built with 'nomic-embed-text' (dimension 768), but this API has
EMBED_MODEL='all-minilm'"*; the kubelet reported *"Readiness probe failed … statuscode:
503"*; and `kubectl rollout undo` restored it. **Without the guardrail**, the api would
embed questions with the new model and compare them against vectors from the old one. If
the dimensions matched, every query would return arbitrary chunks with plausible scores
and the model would answer confidently from them; if not, Chroma would raise on every
query. Nothing in the manifests would look wrong.

## 7. The failure that cost us more than an hour
<!-- Read this, check it matches what you remember, and rewrite it in your own words. -->
Our re-index at 200/30 was marked **Failed** after **65 minutes**
(`DeadlineExceeded: Job was active longer than specified deadline`), although the first
index had taken about 6½ minutes. We first believed the indexer was stuck or starved:
that the tuning runs we had started on the same laptop were competing with it for Ollama
and slowing embedding to a crawl, or that the init container was hanging on the
archive.org download of the 2025 handbook. We checked the tuning processes
(`pgrep -af tune_chunking`: none running) and the init container's log (all 9 documents
fetched, *"9 of 9 match the manifest"*), so neither was it. The truth was in the
indexer's own log, `kubectl logs -n rahbar -l job-name=reindex-1794384464 -c indexer`:
`"chunking done"` at **08:10:36**, then `"embedded" … "seconds": 728.8` logged at
**09:13:24**. Embedding had taken 12 minutes by the process's clock, but 63 minutes had
passed on the wall clock. The ~50-minute gap was the laptop going to sleep: WSL and
minikube were suspended, while `activeDeadlineSeconds: 3600` is measured in wall time.
The work had actually finished (`"indexer finished", "chunks": 1038`) and the new index
was live (`/readyz`: `rahbar-v1794388406, 1038 chunks`). The fix was operational: disable
sleep while the cluster runs. The lesson is to read the stage timestamps before
theorising, and that a Job deadline measures wall time, not work.
