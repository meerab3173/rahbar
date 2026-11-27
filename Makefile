# RAHBAR starter Makefile. `make` or `make help` lists the targets.

SHELL := /bin/bash
NS ?= rahbar
TAG ?= dev
PY ?= python3
VENV ?= .venv
BIN := $(abspath $(VENV))/bin
# The Ingress address. On Linux this is the minikube IP. On macOS or Windows with the
# docker driver, run `minikube tunnel` in another terminal and use INGRESS_IP=127.0.0.1.
INGRESS_IP ?= $(shell minikube ip 2>/dev/null)
HOST ?= rahbar.local

.DEFAULT_GOAL := help

.PHONY: rollout-demo ingest batch lake-status lake-report restore release delete-doc lake-pull trace traces trace-summary trace-logs traces-pull metrics
.PHONY: help install test lint dev cluster images apply secret index ask fetch eval eval-fixtures check tune cache-threshold sections evidence demo-pin

help: ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

install: ## create .venv and install the api, indexer and tool dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r api/requirements-dev.txt -r indexer/requirements-dev.txt

test: ## run the api and indexer tests (stubs only: no cluster, no model, no network)
	cd api && $(BIN)/pytest -q --cov=app --cov-fail-under=60
	cd indexer && $(BIN)/pytest -q --cov=pipeline

lint: ## ruff and mypy over everything in Python
	$(BIN)/ruff check api indexer eval corpus
	cd api && $(BIN)/mypy app
	cd indexer && $(BIN)/mypy pipeline

dev: ## run the API with stubs on http://localhost:8000, serving the chat UI too
	cd api && SERVE_UI_DIR=../ui RETRIEVER=stub LLM_PROVIDER=stub $(BIN)/uvicorn app.main:app --reload --port 8000

cluster: ## start minikube with Calico, then enable metrics-server and ingress
	minikube start --driver=docker --cni=calico --memory=6144 --cpus=4
	minikube addons enable metrics-server
	minikube addons enable ingress

images: ## build the three images and load them into minikube
	docker build -t rahbar-api:$(TAG) api
	docker build -t rahbar-indexer:$(TAG) -f indexer/Dockerfile .
	docker build -t rahbar-ui:$(TAG) ui
	minikube image load rahbar-api:$(TAG)
	minikube image load rahbar-indexer:$(TAG)
	minikube image load rahbar-ui:$(TAG)

apply: ## apply the dev overlay 
	kubectl apply -k k8s/overlays/dev

secret: ## create the LLM key Secret from the LLM_API_KEY variable; never from a file
	@test -n "$$LLM_API_KEY" || { echo "export LLM_API_KEY=... first"; exit 1; }
	kubectl create secret generic rahbar-secret -n $(NS) \
	  --from-literal=LLM_API_KEY="$$LLM_API_KEY" --dry-run=client -o yaml | kubectl apply -f -

index: ## run the indexer once, from the CronJob's template (make ingest also waits and prints)
	kubectl create job --from=cronjob/reindex reindex-$$(date +%s) -n $(NS)
	@echo "watch it with: kubectl get jobs -n $(NS) -w"

ask: ## ask through the Ingress: make ask Q="What CGPA puts a student on academic warning?"
	@curl -s -X POST "http://$(INGRESS_IP)/api/ask" -H "Host: $(HOST)" \
	  -H 'Content-Type: application/json' -d "$$(Q="$(Q)" $(PY) -c 'import json,os; print(json.dumps({"question": os.environ["Q"]}))')" \
	  | $(PY) -m json.tool

fetch: ## download the documents listed in corpus/sources.yaml
	$(BIN)/python corpus/fetch.py

eval: ## score eval/golden.yaml through the Ingress and apply the CI gate
	$(BIN)/python eval/run_eval.py --api "http://$(INGRESS_IP)" --host $(HOST)

eval-fixtures: ## score the CI fixture golden set against a local `make dev`
	$(BIN)/python eval/run_eval.py --api http://localhost:8000 --golden eval/fixtures/golden.fixture.yaml --baseline eval/fixtures/baseline.fixture.json --out /tmp/rahbar-fixture-results.json

tune: ## recall@3/5 for three chunk settings, offline, with Ollama on the laptop (PIPELINE.md)
	$(BIN)/python eval/tune_chunking.py --corpus corpus/raw --golden eval/golden.yaml --configs 200:30,400:50,800:100 --floor-report --json docs/evidence/chunking.json | tee docs/evidence/chunking.txt

cache-threshold: ## choose CACHE_THRESHOLD from eval/cache_pairs.yaml (PIPELINE.md)
	$(BIN)/python eval/cache_threshold.py | tee docs/evidence/cache-threshold.txt

sections: ## list the section labels the indexer produces for the corpus
	$(BIN)/python eval/list_sections.py --corpus corpus/raw

evidence: ## capture kubectl evidence into docs/evidence/ (run while the stack is up)
	kubectl get pods -n $(NS) -o wide > docs/evidence/get-pods.txt
	kubectl top pods -n $(NS) --containers > docs/evidence/kubectl-top.txt
	kubectl get endpoints api -n $(NS) -o wide > docs/evidence/get-endpoints.txt
	kubectl get hpa -n $(NS) > docs/evidence/get-hpa.txt
	kubectl get all,pvc,ingress,networkpolicy -n $(NS) > docs/evidence/get-all.txt
	@echo "now add the screenshots (branch protection, blocked merge, green merge) by hand"

demo-pin: ## section 4.4: change EMBED_MODEL on the api only and watch it refuse readiness
	kubectl -n $(NS) set env deployment/api EMBED_MODEL=all-minilm
	kubectl -n $(NS) rollout status deployment/api --timeout=60s || true
	kubectl -n $(NS) get pods -l app.kubernetes.io/name=api | tee docs/evidence/pin-guard.txt
	kubectl -n $(NS) logs $$(kubectl -n $(NS) get pods -l app.kubernetes.io/name=api --sort-by=.metadata.creationTimestamp -o name | tail -1) | grep "embedding pin" | tee -a docs/evidence/pin-guard.txt
	kubectl -n $(NS) rollout undo deployment/api

check: ## run the submission checker
	$(PY) scripts/check_submission.py

rollout-demo: ## section 4.6: ship v2 with kubectl set image, watch it, then roll back
	docker build --build-arg APP_VERSION=v2 -t rahbar-api:v2 api
	minikube image load rahbar-api:v2
	kubectl -n $(NS) set image deployment/api api=rahbar-api:v2
	kubectl -n $(NS) rollout status deployment/api
	curl -s -o /dev/null -D - -H "Host: $(HOST)" http://$(INGRESS_IP)/api/ask -X POST -H 'Content-Type: application/json' -d '{"question":"version check please"}' | grep -i x-app-version
	kubectl -n $(NS) rollout undo deployment/api
	kubectl -n $(NS) rollout status deployment/api
	curl -s -o /dev/null -D - -H "Host: $(HOST)" http://$(INGRESS_IP)/api/ask -X POST -H 'Content-Type: application/json' -d '{"question":"version check please"}' | grep -i x-app-version
	kubectl -n $(NS) rollout history deployment/api

# --- Assignment 03, Part A: the data lake (docs/DATA-LAKE.md) -----------------------------------
SHELL_POD = $$(kubectl -n $(NS) get pod -l app.kubernetes.io/name=datalake-shell -o jsonpath='{.items[0].metadata.name}')
LAKE = kubectl -n $(NS) exec deploy/datalake-shell -- python -m pipeline.lake

ingest: ## run the ingestion Job once (verify + incremental update), wait, print its reports
	@name=ingest-$$(date +%s); \
	kubectl create job --from=cronjob/reindex $$name -n $(NS) && \
	kubectl -n $(NS) wait --for=condition=complete job/$$name --timeout=60m && \
	kubectl -n $(NS) logs job/$$name -c indexer | grep -v '^{'

batch: ## copy a folder of new documents into the lake's inbox: make batch DIR=corpus/batch-b
	@test -n "$(DIR)" || { echo "usage: make batch DIR=<folder> [NAME=<batch name>]"; exit 1; }
	kubectl -n $(NS) cp "$(DIR)" "$(SHELL_POD):/datalake/inbox/$${NAME:-$$(basename $(DIR))}"
	$(LAKE) status | tail -3
	@echo "now run: make ingest"

lake-status: ## every document version in the data lake, and where it is
	$(LAKE) status

lake-report: ## print the latest report: make lake-report KIND=verify (or KIND=ingest)
	$(LAKE) report $${KIND:-verify} $(RUN)

restore: ## roll back to an archived version on the next run: make restore DOC=<doc_id>@<version>
	$(LAKE) request restore "$(DOC)"

release: ## undo a restore, so recency decides again: make release DOC_ID=<doc_id>
	$(LAKE) request release "$(DOC_ID)"

delete-doc: ## delete a document version and all its chunks on the next run: make delete-doc DOC=...
	$(LAKE) request delete "$(DOC)"

lake-pull: ## copy the data lake's reports and catalog into eval/results/datalake/ (commit them)
	mkdir -p eval/results/datalake
	kubectl -n $(NS) cp "$(SHELL_POD):/datalake/reports" eval/results/datalake/reports
	kubectl -n $(NS) cp "$(SHELL_POD):/datalake/state/catalog.json" eval/results/datalake/catalog.json
	ls eval/results/datalake/reports | tail -6

# --- Assignment 03, Part D: traces (docs/OBSERVABILITY.md) ---------------------------------------
API_POD = $$(kubectl -n $(NS) get pod -l app.kubernetes.io/name=api -o jsonpath='{.items[0].metadata.name}')
TRACETOOL = kubectl -n $(NS) exec deploy/api -- python -m app.tracetool

traces: ## the latest traces, one line each: make traces N=20
	$(TRACETOOL) list -n $${N:-20}

trace: ## D2: the full trace and the exact prompt for one id: make trace ID=<trace id>
	@test -n "$(ID)" || { echo "usage: make trace ID=<trace id (or its first characters)>"; exit 1; }
	$(TRACETOOL) show "$(ID)"

trace-summary: ## D3: metrics over a time window from the trace log: make trace-summary SINCE=2h
	$(TRACETOOL) summary $${SINCE:+--since $$SINCE}

trace-logs: ## F4: trace records on standard output, read with kubectl logs
	kubectl -n $(NS) logs deploy/api --tail=$${N:-200} | grep '"rahbar.trace"'

traces-pull: ## copy the trace volume into traces/ (commit samples for the report)
	mkdir -p traces
	kubectl -n $(NS) cp "$(API_POD):/traces/." traces/
	ls traces | head

metrics: ## F4: the metrics endpoint through kubectl port-forward (Ctrl+C to stop)
	@echo "port-forwarding svc/api to localhost:9000; in another window: curl -s localhost:9000/metrics | grep rahbar_"
	kubectl -n $(NS) port-forward svc/api 9000:8000
