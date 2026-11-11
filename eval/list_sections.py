#!/usr/bin/env python3
"""Print every section label the indexer produces, per document, with its first page.

    python eval/list_sections.py --corpus corpus/raw            # your corpus
    python eval/list_sections.py --corpus corpus/raw --grep warning

Use it to check that each `section:` in eval/golden.yaml appears in a real label
(recall@k matches when the golden text is contained in the label). Write the golden
questions first; only adjust a section's spelling here, never which passage is relevant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))

from pipeline.chunk import sections  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.run import discover  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="corpus/raw")
    ap.add_argument("--grep", default="")
    args = ap.parse_args()
    for path, version, superseded in discover(args.corpus):
        if not path.exists():
            print(f"\n## {path.name}  (MISSING)")
            continue
        doc = clean(extract(str(path), version, superseded))
        ocr = sum(1 for p in doc.pages if p.ocr)
        print(f"\n## {path.name}  version={version} superseded={superseded} "
              f"pages={len(doc.pages)} ocr_pages={ocr}")
        for sec in sections(doc):
            if args.grep.lower() in sec.label.lower():
                print(f"  p{sec.paras[0].page:<4} {sec.label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
