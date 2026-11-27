# Batch B: eleven test documents with planted problems (Assignment 03, Part A)

Synthetic documents written for this assignment. None is an official FAST-NUCES
publication, and each says so in its text. They go through the same ingestion Job as the
real corpus: `make batch DIR=corpus/batch-b` copies them into the data lake's inbox, and
`make ingest` verifies and indexes them.

| File | Planted problem | Expected verdict |
| --- | --- | --- |
| `b01-exam-hall-conduct.md` | none | active |
| `b02-exam-hall-conduct-copy.md` | byte-for-byte copy of b01 | quarantine: uniqueness (exact duplicate) |
| `b03-library-borrowing-rules.md` | none | active |
| `b04-library-borrowing-rules-copy.md` | byte-for-byte copy of b03 | quarantine: uniqueness (exact duplicate) |
| `b05-scholarship-policy.md` | none | active |
| `b06-scholarship-policy-reissue.md` | b05 re-titled and lightly reworded, under another doc_id | quarantine: uniqueness (near duplicate) |
| `b07-lab-safety-rules.pdf` | a real PDF cut to 55% of its bytes | quarantine: quality (failed extraction) and completeness (no `%%EOF`) |
| `b08-grievance-procedure.md` | no front matter: no title, no date or version | quarantine: completeness (missing metadata) |
| `b09-admissions-faq-2027.md` | newer version (2027-06) of the corpus's Admissions FAQ (2026-06) | active; the 2026 FAQ fails recency and is archived |
| `b10-counselling-services.md` | an injected instruction in section 2 | active: verification does not look for it; the query-time guard does (Part D4) |
| `b11-transport-route-notice.md` | mojibake: the text is garbled | quarantine: quality (garbled) |

`_meta.yaml` carries the metadata of the PDF, which has no front matter.
