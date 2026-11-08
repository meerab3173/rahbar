# Engineering notes

Line numbers refer to the files as committed.

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?
Three pods start together (`k8s/base/api-deployment.yaml:8`, `replicas: 3`) and each would
delete and rebuild the collection on the single RWO volume of `chroma-0`
(`chroma-statefulset.yaml:64`, `volumeClaimTemplates`). Their `add` calls interleave, so
the collection ends up with chunks from three half-finished runs, or one pod deletes the
collection while another is writing: an index that no run produced, served without an
error. Every rolling update (`maxSurge: 1`, line 15) would also add a fourth writer, and
every restart would pay the full OCR and embedding cost again. The object that prevents
it is the **Job** (`indexer-job.yaml:5`, `kind: Job`, `restartPolicy: OnFailure` at
line 20): the only writer, run once, with `concurrencyPolicy: Forbid` on the CronJob
(`reindex-cronjob.yaml:11`) so two indexers never overlap. The API's code has no write
call at all; `indexer/pipeline/store.py:100` is the only `add`.

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?
Liveness would then call `/readyz`, which waits on Chroma. With `periodSeconds: 20`,
`failureThreshold: 3` (`api-deployment.yaml:55-58`) the kubelet kills each container
after ~60 s of a slow Chroma. All three pods restart at roughly the same time; the
startup probe gives each 60 s (`failureThreshold: 30 × periodSeconds: 2`, lines 51-54),
during which Chroma is still slow, so they are killed again: a restart storm, and what
was a 90 s slowdown becomes a full outage with CrashLoopBackOff back-off on top. Readiness
on `/healthz` would meanwhile keep sending traffic to pods that cannot answer. As built,
`/healthz` touches nothing (`api/app/routes/health.py:20`), so a slow Chroma only marks
pods unready (removed from the Service, not restarted) and they return the moment Chroma
recovers.

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?
`hpa.yaml` scales on CPU utilisation = usage / `requests.cpu` (100m,
`api-deployment.yaml:46`). While a request waits for Ollama (seconds), the uvicorn worker
is blocked on a socket, not computing: CPU stays near idle even when every worker is busy
and requests are queuing. So the HPA measures JSON parsing and Pydantic validation, not
load; it scales late or never, exactly when the queue is longest. The right signal is
**concurrency / queue depth**: in-flight requests per pod (or p95 latency of
`rahbar_request_seconds`, or generation tokens per second if the model ran in-cluster).
To use it we would need Prometheus scraping `/metrics`, an in-flight gauge
(`rahbar_inflight_requests`, a middleware inc/dec), and the Prometheus Adapter (or KEDA)
to expose it as a `type: Pods` custom metric with a target like 4 in-flight per pod.

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong
Limit: **1536 Mi** (`indexer-job.yaml:68`), request 512 Mi. **[measure]** Peak from
`kubectl top pods -n rahbar --containers` sampled every 5 s during the run
(`watch -n5 kubectl top pods -n rahbar --containers | tee -a docs/evidence/kubectl-top.txt`):
peak ___ Mi → margin = 1536 − peak = ___ Mi (aim for 30–50 %). The peak is in OCR
(a 300 dpi page bitmap is ~25 MB in Pillow) and holding all chunk texts and vectors before
the store. Set it below the peak and the kernel OOM-kills the container: **exit code 137**
(128 + SIGKILL), `Reason: OOMKilled` in `kubectl describe pod`, and the Job retries up to
`backoffLimit: 3` before failing. Too high is a different waste: a padded request is
reserved on the node for the whole run.

## 5. recall@5 by question type: where is it worst, and why?
**[measure]** from `eval/results.json` → `by_type`: ordinary ___, version ___, exact ___,
injection ___. We expect **exact** to be worst: "What does clause 1.21 require?" embeds
the number as a couple of sub-word tokens whose vector barely differs from "clause 1.12"
or "1.2"; dense embeddings capture topic, not identity. The handbook and the prospectus
number the same rule differently (1.21 vs 1), which makes it worse. The lecture's remedy is
**hybrid retrieval**: a lexical/BM25 score (exact token match) fused with the dense score,
optionally followed by a cross-encoder reranker. We implemented the light version: 4 × k
dense candidates reranked with an exact-term boost (`api/app/retrieval/chroma.py`,
`exact_boost`, `HYBRID_WEIGHT=0.15`); compare `--hybrid-weight 0` vs `0.15` in
`make tune` to show its effect on the exact type.

## 6. EMBED_MODEL changed on the API only: what does your system do?
At boot the API reads the active collection's metadata (`api/app/main.py:47-49` →
`retrieval/chroma.py:98` `pin_conflict`) and logs `"embedding pin check failed"` with the
index model, its dimension and the API's model. `/readyz` returns 503 with
`"failed": ["embed_model"]` (`routes/health.py:26`), so the pod stays 0/1 READY and is
never added to the Service; `/api/ask` would also refuse the query (`_check_pin` and the
dimension check at `chroma.py:177`). During a rollout the old pods keep serving because
`maxUnavailable: 0`. **Without the guardrail**, the API would embed questions with the
new model and compare them with vectors from the old one: if the dimensions matched, every
query would return arbitrary chunks with plausible-looking scores and the model would
answer confidently from them; if not, Chroma would raise on every query. Nothing in the
manifests would look wrong. Evidence: `docs/evidence/pin-guard.txt` (`make demo-pin`).

## 7. The failure that cost us more than an hour
**[write in your own words: this must be your real experience.]** A likely candidate on
this stack, if it happened to you: *Every service name stopped resolving after we applied
the NetworkPolicies.* Symptoms: api pods 0/1 READY, `/readyz` saying `chroma:
unreachable: ConnectError`, the indexer failing on `chroma:8000`. What we wrongly believed
first: Chroma was crashing or the Service selector was wrong, so we restarted chroma-0 and
re-checked labels. What told us the truth: `kubectl exec deploy/api -n rahbar -- python -c
"import socket; print(socket.gethostbyname('chroma'))"` failed with "Temporary failure in
name resolution" while the chroma pod was Ready, which pointed at DNS, and default-deny
egress blocks port 53 to kube-dns. Fix: the `allow-dns` policy in `networkpolicy.yaml`.
Replace this paragraph with what actually happened to you, with the exact command.
