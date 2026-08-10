ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG PYTHON_BIN=python3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src

RUN ${PYTHON_BIN} -m pip install --no-build-isolation -e .

CMD ["python3", "-m", "minitrainbench", "--help"]
