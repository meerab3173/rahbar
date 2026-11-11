# Evidence

Real output from our cluster and repository, captured with `make evidence` and by hand.

| File | How it was captured |
| --- | --- |
| `kubectl-top.txt` | `watch -n5 kubectl top pods -n rahbar --containers` during the indexer Job and the eval run |
| `get-endpoints.txt` | `kubectl get endpoints api -n rahbar -o wide` (3+ IPs, Service untouched) |
| `get-hpa.txt` | `kubectl get hpa -n rahbar` (a real percentage, not `<unknown>`) |
| `pin-guard.txt` | `make demo-pin` (0/1 READY + the conflict log line) |
| `rollout.txt` | `kubectl set image`, `rollout status`, `rollout undo` session |
| `chunking.txt`, `cache-threshold.txt` | `make tune`, `make cache-threshold` |
| `branch-protection.png` | GitHub → Settings → Branches → rule for `main` |
| `blocked-merge.png`, `green-merge.png` | the PR that failed the eval gate, then the fixed one |
| `screenshot-ui.png` | the chat UI answering with citations |
