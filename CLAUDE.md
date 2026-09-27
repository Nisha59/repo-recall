# RepoRecall

## What this is

RepoRecall is a RAG (retrieval-augmented generation) assistant that answers developer questions using a GitHub project's documentation and closed issues. First demo: Kubernetes (docs cloned from `kubernetes/website`, issues fetched from `kubernetes/kubernetes` via the GitHub API). The pipeline is built to be repo-agnostic — the target repo/docs source is a config value, not a hardcoded assumption.

Goals: production-style engineering, measurable retrieval/answer evaluation, and code clean enough to walk through in an interview.

## Virtual environment

**This project already has a virtual environment at `.venv`. Always reuse it — never create a new one.** Activate it before running anything:

```
.venv\Scripts\Activate.ps1        # PowerShell
source .venv/Scripts/activate      # Git Bash
```

Install/update dependencies with `pip install -r requirements.txt`. When adding a new dependency, install it into `.venv` and add it to `requirements.txt` pinned to the version actually installed (check with `pip show <package>`), not a guessed version number.

## Tech stack

- Python 3.11+
- Vector storage behind a common interface (`indexing/`), with two interchangeable backends:
  - PostgreSQL + pgvector (via Docker Compose, not yet added)
  - LanceDB (local file-based)
- Embeddings: `sentence-transformers`, starting with `BAAI/bge-small-en-v1.5`
- API: FastAPI
- UI: Gradio
- Tests: pytest
- Config: `.env` file + a `pydantic-settings`-based config module (`src/repo_recall/config.py`)

## Folder structure

```
repo-recall/
├── .env                 # secrets (gitignored) — GITHUB_TOKEN, DSNs, etc.
├── .env.example          # documents required env vars, no secrets
├── requirements.txt
├── data/
│   ├── raw/               # cloned docs, raw fetched issues (gitignored)
│   └── processed/         # cleaned/chunked data ready for embedding (gitignored)
├── src/repo_recall/
│   ├── config.py           # Settings loaded from .env
│   ├── collection/         # clone target repo's docs, fetch its GitHub issues+comments
│   ├── processing/         # clean and chunk raw docs/issues
│   ├── indexing/           # embed chunks; common vector store interface
│   │   └── backends/       # postgres+pgvector and lancedb implementations
│   ├── retrieval/          # embed a query, run top-k search against the index
│   ├── generation/         # build the RAG prompt, call the LLM, attach citations
│   ├── evaluation/         # test-question harness, retrieval + answer scoring
│   └── api/                # FastAPI app + Gradio UI
└── tests/                  # pytest, mirrors src/repo_recall structure
```

Each subpackage stays scoped to its one job above — e.g. `retrieval/` never talks to the LLM, `generation/` never does similarity search.

## Coding conventions

- Type hints on all function signatures, including return types.
- Docstrings on every public function/class: what it does, not a restatement of the code.
- Keep functions small and single-purpose; prefer composing several small functions over one that branches a lot.
- Use the `logging` module, never `print`, for anything other than final CLI output.
- New code that talks to an external system (GitHub API, Postgres, an LLM) should sit behind a small interface so it can be swapped or mocked in tests.
- Tests live under `tests/`, mirroring the `src/repo_recall/` package layout (e.g. `tests/retrieval/test_retriever.py`).
