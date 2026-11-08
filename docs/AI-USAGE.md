# AI usage

We used AI assistance throughout; this is a specific account of what it did.

| Tool | What it wrote or shaped | What we changed afterwards, and why |
| --- | --- | --- |
| Claude (Anthropic, claude.ai, Opus 5.5) | First drafts of almost all of the code that was not in the starter kit: the indexer stages (`extract`, `clean`, `chunk`, `embed`, `store` with the versioned-collection pointer flip), the API query pipeline (`services/answer.py`, `guards.py`, `cache.py`, `prompt.py`), the Chroma retriever with the embedding-pin check, the Ollama and hosted providers, `eval/run_eval.py`'s four functions, `eval/tune_chunking.py`, `eval/cache_threshold.py`, the Dockerfiles, every Kubernetes manifest, `ci.yml` and `cd.yml`, the tests, and drafts of these docs and ADRs. It also read the FAST-NUCES handbooks on the website to propose golden-set questions with clause numbers. | **[fill in honestly]** e.g. we ran every command ourselves on minikube and replaced every **[measure]** placeholder with our own output; fixed what broke on our machine (list it: NetworkPolicy host IP, Chroma volume permissions, etc.); checked each golden question against the PDFs and corrected section names with `make sections`; rewrote Engineering Note 7 from our own experience. |
| | Proposed the exact-term guard on the cache after the offline measurement showed "B-" and "B+" scoring 1.000 | We kept it and verified it with `make cache-threshold`. |

Both partners can explain every file: the ADRs, the probes, the pin check, the cache
invalidation and the gate are the parts we expect to be asked about in a viva.
