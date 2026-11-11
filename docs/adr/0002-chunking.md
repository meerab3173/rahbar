# ADR 0002: Structure-aware chunking with section labels

## Status
Accepted · 2026-11-06

## Context
The handbooks are numbered clauses ("1.21 Students are required to maintain 100%
attendance…") under headings ("1. Medium of Instruction" › "Attendance Requirements").
Answers must cite `[source / section]`, and a citation is only valid if it names a chunk
that was retrieved. The Generative AI policy has long definitions with numbered
sub-sections that a fixed window cuts in half (corpus property 5).

## Decision
`indexer/pipeline/chunk.py`: sections first (a heading stack; the label is the chapter
plus the deepest heading), then paragraphs rebuilt from wrapped PDF lines (a clause
number, list item or table row always starts a new one), then sentences, then word
windows only for a sentence longer than the budget. Whole paragraphs are packed up to
`CHUNK_SIZE` words; the next chunk in the same section starts with up to `CHUNK_OVERLAP`
words of whole trailing sentences. Overlap never crosses a heading. Each chunk's text
begins with its section label (a contextual header), which is why storage amplification
is above 1. The shipped size and overlap are the measured winner in docs/PIPELINE.md.

## Options considered
- **Fixed 100-word windows**: measured in FAILURE-ANALYSIS.md stage B: the chunks lose
  their section, so recall by section collapses and no citation can name a clause.
- **One chunk per clause**: precise, but most clauses are one sentence; retrieval then
  returns five fragments with no surrounding rule, and the chunk count triples.
- **Semantic (embedding-similarity) splitting**: an extra model pass at index time for a
  corpus that already carries explicit structure.

## Consequences
Citations are human-readable and checkable. Heading detection is heuristic (Title Case
short lines), so a mis-detected heading produces an odd label but never loses text;
`eval/list_sections.py` shows every label so the golden set can be checked against it.
