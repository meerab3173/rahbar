# Running RAHBAR on a fresh cluster

Everything RAHBAR runs is in this folder: `base/` holds every object, and the overlays
change only images and a few values. A fresh minikube reaches a working pipeline with the
commands below and nothing else.

## 1. Prerequisites (once per laptop)

Docker, minikube, kubectl, make, Python 3.12, and [Ollama](https://ollama.com) on the
laptop. On Windows, everything runs inside WSL 2 (Ubuntu), with Docker Engine installed in
Ubuntu (not Docker Desktop), so the cluster can reach Ollama on the same machine.

```bash
# Ollama must listen on all interfaces, not only 127.0.0.1, so the cluster can reach it
# at host.minikube.internal:11434. Keep the model loaded between questions.
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\nEnvironment="OLLAMA_KEEP_ALIVE=1h"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama pull llama3.2:3b && ollama pull nomic-embed-text && ollama pull all-minilm
```

## 2. Cluster, images, objects

| Laptop | Cluster | Overlay |
| --- | --- | --- |
| 16 GB RAM | `make cluster` (6 GB, 4 CPUs, Calico, metrics-server, ingress) | `k8s/overlays/dev` |
| 8 GB RAM | `minikube start --driver=docker --cni=calico --memory=3584 --cpus=4`, then `minikube addons enable metrics-server && minikube addons enable ingress` | `k8s/overlays/lite` (1 api pod, 600 s LLM timeout) |

```bash
make images                                  # build api, ui, indexer; load them into minikube
kubectl apply -k k8s/overlays/dev            # or k8s/overlays/lite
kubectl -n rahbar wait --for=condition=complete job/indexer --timeout=60m
kubectl -n rahbar rollout status deployment/api
make ask Q="What CGPA puts a student on academic warning?"
```

Calico matters: the default CNI ignores NetworkPolicy, and the policies are what keep
every component except the ingestion Job away from the vector store.

**Re-applying after a code change.** A Job's pod template cannot be edited, so delete the
finished Job before `kubectl apply`: `kubectl -n rahbar delete job indexer`. Rebuild only
what changed, for example `docker build -t rahbar-api:dev api && minikube image load
rahbar-api:dev && kubectl -n rahbar rollout restart deployment/api`.

## 3. Which object carries which RAGOps responsibility

| Object | Kind | Responsibility (paper, Figure 3) |
| --- | --- | --- |
| `indexer` | Job | ingest, verify, update the data lake, update the retrieval sources (Part A); the **only writer** to the vector store |
| `reindex` | CronJob | re-runs the same Job every Sunday 03:00 (`concurrencyPolicy: Forbid`: never two writers at once) |
| `datalake` | PersistentVolumeClaim | the data lake: fetched sources, inbox, verified documents, archive, quarantine, catalog, reports |
| `datalake-shell` | Deployment | a person's door into the lake (new batches, restore and delete requests); no network egress, so it cannot write to Chroma |
| `chroma` | StatefulSet + PVC | the vector store; one collection per (embedding model, chunking) |
| `chroma-ingress` | NetworkPolicy | admits only the api and the indexer: enforces the single writer together with the code |
| `api` | Deployment + Service | the query pipeline; rolling updates with `maxUnavailable: 0` |
| `traces` | PersistentVolumeClaim | one trace record per query and each exact prompt (Part D, F4) |
| `rahbar-config` | ConfigMap | every threshold and setting, read at start-up: changing one needs no rebuild |
| `api` | HorizontalPodAutoscaler | 3 to 8 api pods at 70% CPU (1 to 8 in the lite overlay) |

## 4. The Part A to D commands on the cluster

| Part | Command |
| --- | --- |
| A: run ingestion (verify + incremental update) | `make ingest` |
| A: ingest a new batch of documents | `make batch DIR=<folder>`, then `make ingest` |
| A: status, reports, rollback, delete | `make lake-status`, `make lake-report KIND=verify`, `make restore DOC=<doc_id>@<version>`, `make delete-doc DOC=...` |
| D: traces, one trace with its exact prompt, summary | `make traces`, `make trace ID=<id>`, `make trace-summary SINCE=2h` |
| F4: trace records on stdout, metrics | `make trace-logs`, `make metrics` (port-forward) |
| F5: measure resources during a command | `python scripts/measure_resources.py --label ingest -- make ingest` |
| copy results into the repository | `make lake-pull`, `make traces-pull` |

## 5. Demo: ingest documents we have not seen

```bash
mkdir -p /tmp/new-batch && cp <the files> /tmp/new-batch/
# PDFs carry their title and date in their own metadata; for files that do not, write
# /tmp/new-batch/_meta.yaml:   <file name>: {title: ..., doc_id: ..., doc_version: "2026-11"}
make batch DIR=/tmp/new-batch NAME=demo && make ingest
make lake-report KIND=verify
```
