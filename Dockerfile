ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b

FROM ${BASE_IMAGE} AS base
ARG BASE_IMAGE
ARG PYTHON_BIN=python3
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG SOURCE_URL=https://github.com/canghaiX/MiniTrainBench

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TOKENIZERS_PARALLELISM=false \
    MINITRAINBENCH_BASE_IMAGE="${BASE_IMAGE}" \
    MINITRAINBENCH_BUILD_REVISION="${VCS_REF}"

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src

RUN ${PYTHON_BIN} -m pip install --no-build-isolation -e .

FROM base AS gpu-deepspeed
ARG PYTHON_BIN=python3
ARG DEEPSPEED_VERSION=0.19.4
ENV DS_BUILD_OPS=0
LABEL io.minitrainbench.deepspeed.version="${DEEPSPEED_VERSION}"

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
