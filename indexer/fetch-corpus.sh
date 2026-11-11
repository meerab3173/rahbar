#!/bin/sh
# The indexer Job's init container. CORPUS_SOURCE (ConfigMap) decides where documents
# come from: "fetch" downloads corpus/sources.yaml into $CORPUS_DIR (an emptyDir) and
# compares the bytes with the committed MANIFEST.md; "fixtures" uses the CI fixture
# corpus baked into the image, so nothing is downloaded.
set -eu
if [ "${CORPUS_SOURCE:-fetch}" = "fetch" ]; then
  python corpus/fetch.py --out "${CORPUS_DIR}"
  python corpus/fetch.py --out "${CORPUS_DIR}" --verify --warn-only
else
  echo "CORPUS_SOURCE=${CORPUS_SOURCE}: using the corpus baked into the image"
fi
