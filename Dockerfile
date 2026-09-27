# RepoRecall API + UI. Built and started by `docker compose up --build`.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# CPU-only PyTorch first: otherwise sentence-transformers pulls the CUDA
# build, several GB larger, which this app never uses.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so startup needs no download.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

# The model is already in the image: don't contact Hugging Face at runtime,
# and don't send Gradio usage analytics.
ENV HF_HUB_OFFLINE=1 \
    GRADIO_ANALYTICS_ENABLED=False

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "repo_recall.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
