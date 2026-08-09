FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ARG TORCH_VERSION=2.11.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/app/code

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /uvx /bin/
COPY requirements.txt .
RUN uv pip install --system --no-cache "torch==${TORCH_VERSION}" --torch-backend=cpu
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip check \
    && python -c "import torch; assert torch.version.cuda is None; print(f'torch={torch.__version__} cpu_only=true')"

COPY code/jobpilot/retrieval/model_contract.py /usr/local/bin/jobpilot_model_contract.py
COPY scripts/prepare_cloud_runtime.py /usr/local/bin/prepare_cloud_runtime.py
RUN python /usr/local/bin/prepare_cloud_runtime.py download-model \
    --model-dir /opt/models/all-MiniLM-L6-v2

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV JOBPILOT_EMBEDDING_BACKEND=sentence-transformers
ENV JOBPILOT_SENTENCE_MODEL=/opt/models/all-MiniLM-L6-v2
ENV JOBPILOT_REQUIRE_MODEL_MANIFEST=1
ENV JOBPILOT_STARTUP_WARM_RANKER=1
ENV JOBPILOT_REQUIRE_RANKER_WARMUP=1

COPY app app
COPY code code
COPY data/processed data/processed
COPY README.md brief.md brief.pdf THIRD_PARTY_NOTICES.md ./

RUN python /usr/local/bin/prepare_cloud_runtime.py build-cache \
    --project-root /app \
    --model-dir /opt/models/all-MiniLM-L6-v2

ENV JOBPILOT_REQUIRE_PREBUILT_EMBEDDINGS=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
