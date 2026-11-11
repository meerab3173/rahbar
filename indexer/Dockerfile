# RAHBAR indexer: the Job's image, and the only writer to Chroma.
# Build context: the REPOSITORY ROOT, so the image can carry corpus/fetch.py, the
# committed corpus/sources.yaml and MANIFEST.md, and the CI fixture corpus:
#   docker build -f indexer/Dockerfile -t rahbar-indexer:dev .
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OCR needs system packages, not only pip packages (section 3.1).
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin rahbar

WORKDIR /app
COPY indexer/requirements.txt .
RUN pip install --upgrade pip "setuptools>=78.1.1" wheel \
    && rm -f /usr/local/lib/python3.12/ensurepip/_bundled/setuptools-*.whl \
    && pip install -r requirements.txt
COPY indexer/pipeline ./pipeline
COPY corpus/ ./corpus/
COPY eval/fixtures ./fixtures
COPY indexer/fetch-corpus.sh ./fetch-corpus.sh

USER 10001
CMD ["python", "-m", "pipeline.run"]
