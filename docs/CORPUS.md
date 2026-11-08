# Corpus

Source: **FAST-NUCES public website (nu.edu.pk)**, public pages only (no SLATE, Flex or
NU Mail). · Documents: **9** · Pages: **≈ 240** (two 53-page Academic Rules handbooks,
the ~60-page Student Handbook, two annual reports, the 5-page prospectus rules, the scanned
ORIC policy, and two HTML pages). Exact page counts per file are printed by the indexer
(`"msg": "chunked", "pages": …`) and by `make sections`.

We do not commit the documents (they are not ours). `corpus/sources.yaml` lists them,
`python corpus/fetch.py` downloads them politely (one at a time, 2 s apart) and writes
`corpus/MANIFEST.md` with every URL, size and SHA-256, and `--verify` re-hashes local
files against it. On the cluster, the indexer Job's init container runs the same script.

| # | Required property | Document(s) that provide it | Evidence or notes |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | 9 documents, ≈ 240 pages | `corpus/MANIFEST.md`; per-file page counts in the indexer's `chunked` log lines |
| 2 | Two versions of one policy | **Academic Rules Handbook Jun 2025** (`superseded: true`, `doc_version: 2025-06`) and **Jun 2026** (`superseded: false`, `2026-06`) | The 2026 edition replaced 2025. Real differences: summer semester rules. 2025 clause 6.4 bars new courses except listed exceptions and has no summer attendance rule; 2026 clauses 6.1–6.3 admit four categories and allow repeat courses only, and 6.7 applies the 80 % attendance rule to summer. Withdrawal from a summer course is clause 6.7 in 2025 and 6.8 in 2026. The metadata `superseded` excludes the 2025 chunks unless the user asks about an older version (`api/app/services/guards.py`, `wants_old_version`). |
| 3 | Scanned, image-only PDF | **Faculty Entrepreneurship and Technical Start-ups Policy** (ORIC 27.pdf) | No text came back from the site's own text layer. Confirm with `pdffonts corpus/raw/faculty-entrepreneurship-and-technical-start-ups-policy.pdf`: an empty font list (header row only) means image-only. The indexer OCRs every page with fewer than 25 characters of text and logs `ocr_pages` (`indexer/pipeline/extract.py`). |
| 4 | A table with real information | Grade-points table (clause 2.3, both handbooks); Annual Reports 2019-20 and 2023-24 (faculty by campus, graduates, publications); Fee Structure page (HTML tables) | PDF tables are re-emitted as `| cell | cell |` rows by pdfplumber's table finder; HTML `<tr>` become pipe rows. Before/after example in PIPELINE.md. |
| 5 | Long definitions | **Student Handbook Jun 2026**, section XII "Policy on Generative AI Based Tools" (pp. 40–54, twelve numbered sub-sections), and the same policy as section 10 of the 2026 Academic Rules Handbook | A 100-word fixed window cuts "Unacceptable Uses of Generative AI Tools" mid-list; FAILURE-ANALYSIS stage B. |
| 6 | A deliberately absent topic | **Hostel and transport facilities**: the prospectus's *Transport and Hostel Facilities* PDF is left out on purpose | Students ask about hostel rent, mess charges, applying for a room, bus routes and transport fees all the time. The Student Handbook only says hostels exist at two campuses and transport at some, with no fees or procedure, so the honest answer is "the documents do not cover this". Golden q19 and q20 must abstain (gate: 100 %). |
| 7 | Exact-term content | Clause numbers (1.21 attendance, 5.4 warning threshold, 6.7 summer attendance), CGPA thresholds (2.00, 3.00), grade points (B- = 2.67), library fine (PKR 10/item/day), tuition (Rs 12,000 per credit hour from Fall 2026, Admissions FAQ) | The prospectus numbers attendance as clause 1 while the handbook says 1.21: the exact-term trap. Golden q15–q18. |

## Other real traps on the site we noticed
- The 2026 prospectus page vs. "Prospectus 2025" on the Downloads page.
- Two versions of the EE learning outcomes (12 PLOs before 2024, 11 from 2024) on the IQAE
  page: a second version pair we did not need.
- The Aug 2023 undergraduate rules PDF is still linked but did not download when checked;
  the Jun 2025 handbook is a cleaner, downloadable superseded version.
