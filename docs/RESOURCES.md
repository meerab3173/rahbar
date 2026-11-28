# Resources (F5): requests and limits from measurements

Measured with `scripts/measure_resources.py`, which samples `kubectl top` every 5 s while a
command runs. Raw samples are in `eval/results/resources/`.

## Measurements

Ingestion Job (`make ingest`, 209 samples)

| container | cpu median | cpu peak | mem median | mem peak |
|---|---|---|---|---|
| api | 7m | 54m | 95Mi | 106Mi |
| chroma | 2m | 8m | 113Mi | 119Mi |
| indexer | 1133m | 1300m | 392Mi | 703Mi |
| redis | 17m | 18m | 15Mi | 16Mi |

One query (`make ask`, 135 samples)

| container | cpu median | cpu peak | mem median | mem peak |
|---|---|---|---|---|
| api | 17m | 18m | 94Mi | 94Mi |
| chroma | 11m | 12m | 117Mi | 117Mi |
| redis | 17m | 34m | 13Mi | 13Mi |

Idle for 60 s (65 samples)

| container | cpu median | cpu peak | mem median | mem peak |
|---|---|---|---|---|
| api | 18m | 18m | 93Mi | 93Mi |
| chroma | 18m | 18m | 117Mi | 117Mi |
| redis | 23m | 24m | 17Mi | 18Mi |

## Rule

Memory request = median under load, rounded up. Memory limit = 1.5 x peak, rounded up, and
never below 256 Mi for a long-running service. CPU limits are left as they were: going over a
CPU limit only throttles a pod, while going over a memory limit kills it (OOMKilled).

## Result

| object | memory request -> limit | before (A2) | from |
|---|---|---|---|
| api Deployment | 128Mi -> 256Mi | 192Mi -> 384Mi | median 95Mi, peak 106Mi |
| chroma StatefulSet | 128Mi -> 256Mi | 256Mi -> 1Gi | median 117Mi, peak 119Mi |
| ingestion Job and CronJob (indexer) | 448Mi -> 1088Mi | 512Mi -> 1536Mi | median 392Mi, peak 703Mi |

The indexer is the heavy component: it reads PDFs and runs OCR, at 1.3 CPU and 703 Mi at its
peak. The api and Chroma stay around 95 to 120 Mi whether idle or answering.
