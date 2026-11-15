# syntax=docker/dockerfile:1.7
# CPU-only SearchForge image (API, CLI, evaluation). No GPU, no CUDA, no model downloads.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-install-project --no-editable
COPY src ./src
RUN uv sync --frozen --no-editable

FROM python:3.12-slim AS runtime
RUN groupadd --gid 10001 searchforge \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin searchforge
COPY --from=build /app/.venv /app/.venv
COPY configs /app/configs
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /work
RUN chown searchforge:searchforge /work
USER searchforge
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"
ENTRYPOINT ["searchforge"]
# Default: build (or reuse) the fixture index in /work/.searchforge and serve it.
CMD ["serve-fixture"]
