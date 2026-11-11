# RAHBAR api: reads the index, never writes it. Build context: api/
FROM python:3.12.8-slim-bookworm

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pick up Debian security fixes so Trivy's "HIGH with a fix" gate passes.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin rahbar

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools>=78.1.1" wheel \
    && rm -f /usr/local/lib/python3.12/ensurepip/_bundled/setuptools-*.whl \
    && pip install -r requirements.txt
COPY app ./app

USER 10001
EXPOSE 8000
# exec form: SIGTERM reaches uvicorn as PID 1, so in-flight requests finish (preStop + grace)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "20"]
