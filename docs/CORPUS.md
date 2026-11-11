# Corpus

Source: **FAST-NUCES public website (nu.edu.pk)**, public pages only (no SLATE, Flex or
NU Mail). · Documents: **9** · Pages: **310** (counted by the indexer, which logs `pages`
per document; see the table below).

We do not commit the documents (they are not ours). `corpus/sources.yaml` lists them,
`python corpus/fetch.py` downloads them politely (one at a time, 2 s apart) and writes
`corpus/MANIFEST.md` with every URL, size and SHA-256; `--verify` re-hashes local files
against it. On the cluster, the indexer Job's init container runs the same script and
verified **9 of 9 match the manifest**.

| Document | Pages | OCR pages | Chunks (200/30) | doc_version | superseded |
| --- | --- | --- | --- | --- | --- |
| Academic Rules Handbook (Jun 2025) | 55 | 0 | 174 | 2025-06 | **true** |
| Academic Rules Handbook (Jun 2026) | 58 | 5 | 151 | 2026-06 | false |
| Prospectus 2026: Academic Rules and Regulations | 5 | 0 | 46 | 2026 | false |
| Student Handbook (Jun 2026) | 64 | 4 | 181 | 2026-06 | false |
| Annual Report 2019-20 | 33 | 0 | 94 | 2019-20 | false |
| Annual Report 2023-24 | 85 | 0 | 322 | 2023-24 | false |
| Fee Structure page (HTML) | 1 | 0 | 10 | 2026-27 | false |
| Admissions FAQ page (HTML) | 1 | 0 | 34 | 2026-06 | false |
| Faculty Entrepreneurship and Technical Start-ups Policy | 8 | **8** | 26 | 2022-05 | false |
| **Total** | **310** | **17** | **1,038** | | |

| # | Required property | Document(s) that provide it | Evidence or notes |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | 9 documents, 310 pages | `corpus/MANIFEST.md`; the indexer's `chunked` log lines (table above) |
| 2 | Two versions of one policy | **Academic Rules Handbook Jun 2025** (`superseded: true`) and **Jun 2026** (`superseded: false`) | The 2026 edition replaced 2025. Real differences are in the summer-semester rules: 2025 clause 6.4 bars new courses except listed exceptions and has no summer attendance rule; 2026 clauses 6.1–6.3 admit four categories and allow repeat courses only, and 6.7 applies the 80 % attendance rule to summer. **The university removed the 2025 PDF from nu.edu.pk on or about 9 Nov 2026** (HTTP 404); we fetch the identical public file from the Internet Archive's Wayback Machine (`web.archive.org/web/2025id_/…`), recorded with its SHA-256 in the manifest. The `superseded` metadata excludes its chunks unless the user asks about an older version (`api/app/services/guards.py`, `wants_old_version`). |
| 3 | Scanned, image-only PDF | **Faculty Entrepreneurship and Technical Start-ups Policy** (ORIC 27.pdf) | `pdffonts` lists no fonts (no text layer). The indexer OCR'd **8 of 8 pages** (`"ocr_pages": 8`), giving 26 chunks; without OCR the file contributes 0 chunks (FAILURE-ANALYSIS stage A). |
| 4 | A table with real information | Grade-points table (clause 2.3, both handbooks); the library loan/fine table in the Student Handbook (pp. 32–33: patron, max loan, renewals, period, fine in PKR); Annual Reports (faculty by campus, graduates); Fee Structure page (HTML tables) | Tables are emitted as one row per line (`chunk.py`, `_ROWLIKE`; HTML `<tr>` → `| … |`). Before/after in PIPELINE.md. |
| 5 | Long definitions | **Student Handbook Jun 2026**, section XII "Policy on Generative AI Based Tools" (twelve numbered sub-sections), and the same policy as section 10 of the 2026 Academic Rules Handbook | Fixed 100-token windows drop recall@5 to 0.105 (FAILURE-ANALYSIS stage B). |
| 6 | A deliberately absent topic | **Hostel and transport facilities**: the prospectus's *Transport and Hostel Facilities* PDF is left out on purpose | The Student Handbook only says hostels and transport exist at some campuses, with no fees or procedure. Golden q19 (hostel rent, mess, applying) and q20 (bus routes, transport fee) must abstain; both did (abstention 1.0), stopped by the relevance floor before the model was called (top similarity 0.640 and 0.686 < floor 0.69). |
| 7 | Exact-term content | Clause numbers (1.21 attendance, 5.4 warning threshold, 6.7 summer attendance), thresholds (CGPA 2.00), grade points (B- = 2.67), library fine (PKR per item per day), tuition (Rs 12,000 per credit hour from Fall 2026, Admissions FAQ) | The prospectus numbers attendance as clause 1 while the handbook says 1.21: the exact-term trap. Golden q15–q18; recall@5 for this type is the weakest (0.25, ENGINEERING-NOTES Q5). |

## Other real traps we noticed on the site
- The 2025 handbook disappearing mid-assignment: source documents change underneath an
  index, which is exactly why the manifest records hashes and the CronJob re-fetches.
- "Prospectus 2026" pages next to "Prospectus 2025" on the Downloads page.
- Two versions of the EE programme learning outcomes (12 PLOs before 2024, 11 from 2024).
