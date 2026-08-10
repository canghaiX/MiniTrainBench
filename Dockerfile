ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE} AS base
ARG PYTHON_BIN=python3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src

RUN ${PYTHON_BIN} -m pip install --no-build-isolation -e .

FROM base AS gpu-deepspeed
ARG PYTHON_BIN=python3
ARG DEEPSPEED_VERSION=0.19.4
ENV DS_BUILD_OPS=0

RUN --mount=type=secret,id=https_proxy,required=false \
    if [ -f /run/secrets/https_proxy ]; then \
      export HTTPS_PROXY="$(cat /run/secrets/https_proxy)"; \
      export HTTP_PROXY="${HTTPS_PROXY}"; \
      export https_proxy="${HTTPS_PROXY}"; \
      export http_proxy="${HTTPS_PROXY}"; \
    fi; \
    ${PYTHON_BIN} -m pip install "deepspeed==${DEEPSPEED_VERSION}"

FROM base AS gpu

CMD ["python3", "-m", "minitrainbench", "--help"]
