FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY q_lab ./q_lab

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 && \
    python -m pip install .

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/reports /app/artifacts && \
    chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["q-lab"]
CMD ["--help"]
