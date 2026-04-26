# =============================================================================
# NON-REPRODUCIBLE SCAFFOLDING — base image is not digest-pinned.
#
# For a reproducible production build, pin python:3.12-slim to a specific
# digest, e.g.:
#   FROM python:3.12-slim@sha256:<digest>
# Obtain the current digest with:
#   docker pull python:3.12-slim && docker inspect python:3.12-slim --format '{{index .RepoDigests 0}}'
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Copy packaging metadata before source so pip can resolve the package name.
COPY README.md .
COPY pyproject.toml .
COPY vetinari/ vetinari/
COPY config/ config/

# Include the observability extra so OTEL instrumentation works when the
# docker-compose.otel.yaml sidecar is present.
RUN pip install --no-cache-dir -e ".[local,ml,observability]"

EXPOSE 8080
# Canonical bind env names — VETINARI_WEB_HOST / VETINARI_WEB_PORT
ENV VETINARI_WEB_HOST=0.0.0.0
ENV VETINARI_WEB_PORT=8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import json, os, sys, urllib.request; url='http://localhost:' + os.environ.get('VETINARI_WEB_PORT', '8080') + '/ready'; data=json.load(urllib.request.urlopen(url, timeout=5)); sys.exit(0 if data.get('status') == 'healthy' else 1)"

CMD ["python", "-m", "vetinari", "serve"]
