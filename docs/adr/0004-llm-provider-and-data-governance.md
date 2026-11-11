# ADR 0004: LLM provider behind one interface, and what data leaves the cluster

## Status
Accepted · 2026-11-06

## Context
The grader's laptop must run it (low memory), CI must run it with no model at all, and
a student's question may contain personal data (a name, a roll number, a grade).

## Decision
- `LLM_PROVIDER` in the ConfigMap selects the provider (`api/app/providers/llm/__init__.py`).
  `ChatJSONProvider` (`base.py`) holds the shared part: prompt, enforce-then-validate with
  Pydantic, one retry, `GenerationError`. A provider is one file implementing one HTTP
  call: `ollama.py` (`format=` the Answer JSON schema), `hosted.py` (OpenAI-compatible
  `/chat/completions`, `response_format: json_object`), `stub.py` (CI, no network).
- **Default: `ollama-host`** with `llama3.2:3b` on the laptop at
  `host.minikube.internal:11434`; embeddings through the same Ollama with
  `nomic-embed-text` (768-d), pinned in the ConfigMap and checked at runtime.
- Temperature 0.1 (ConfigMap, with its reason). Timeout 60 s per call.

## Data governance
- With `ollama-host`, no question or document leaves the laptop. With `hosted`, the
  question and the retrieved public passages go to the provider; nothing else does. The
  key lives only in the `rahbar-secret` Secret, created from the command line; the
  committed `secret.placeholder.yaml` holds `REPLACE_ME` and is not in kustomization.
- Logs carry the request id, chunk ids, scores and outcome, never the question text or the
  key. The stored turn (`rahbar:turns`, last 1000, 7-day TTL) keeps a SHA-256 prefix of
  the question, not the question.
- The cache stores question text (needed to debug a false hit) for 24 h in-cluster only.
- Model output is untrusted: schema-validated, citations re-checked against retrieved
  chunks, never passed to `eval`, never used to build a query.

## Options considered
In-cluster Ollama (bonus mode, needs ≥ 8 GB for minikube and a startupProbe of 30+
attempts); a hosted model only (fast, but sends data out and needs a key in CI).

## Consequences
A provider swap is one file and one ConfigMap value. `llama3.2:3b` is small: it follows
the schema (enforced) but can still mis-read a clause, which the golden set's `expect`
strings surface.
