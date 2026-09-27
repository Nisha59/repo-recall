# RepoRecall

A RAG assistant that answers developer questions using a project's documentation and GitHub issues. First demo target is Kubernetes (docs from [kubernetes/website](https://github.com/kubernetes/website), issues from [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)), but the pipeline is configurable to work against any GitHub repository.

Every answer cites the docs pages and issues it was built from, and reports how long search and generation took and what the LLM call cost.

## Run it

Needs [Docker](https://docs.docker.com/get-docker/) and a `.env` file with your LLM settings (see [Configuration](#configuration)). Then, from the project root:

```
docker compose up --build
```

This starts Postgres (the search index) and the app together. When the log says `Application startup complete`:

- **UI:** <http://localhost:8000> — type a question, get an answer with clickable source links.
- **API docs:** <http://localhost:8000/docs> — interactive docs for every endpoint.

Stop with `Ctrl+C`, or `docker compose down`. The index is kept in the `pgdata` Docker volume, so it survives restarts (`docker compose down -v` deletes it).

The first build downloads CPU-only PyTorch and the embedding model (the image is about 3 GB), so it takes a few minutes. On startup the app runs one warm-up search (~10 s) so the first real question is fast. Measured memory use: about 450 MB for the app and under 150 MB for Postgres.

> The index has to exist before questions can be answered. If this is a fresh clone, build it once first: see [Building the index](#building-the-index).

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/ask` | Answer a question. Body: `{"question": "...", "top_k": 5, "source_type": "doc" \| "issue"}` (only `question` is required). |
| `GET` | `/health` | `200 {"status": "ok", ...}` when the search index is reachable, `503` when it isn't. |
| `GET` | `/ui` | The Gradio UI (`/` redirects here). |

```
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I roll back a deployment to a previous revision?"}'
```

```json
{
  "question": "How do I roll back a deployment to a previous revision?",
  "answer": "Roll back to the previous revision with:\n\n```shell\nkubectl rollout undo deployment/nginx-deployment\n```\n[1][2]\n\nTo roll back to a specific revision, use `--to-revision`: ... [2]",
  "answered": true,
  "sources": [
    {"number": 1, "title": "Update a Deployment Without Downtime", "url": "https://kubernetes.io/docs/tasks/run-application/update-deployment-rolling/", "source_type": "doc"},
    {"number": 2, "title": "Deployments", "url": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/", "source_type": "doc"}
  ],
  "retrieved_count": 3,
  "timing": {"retrieval_seconds": 0.084, "generation_seconds": 1.787, "total_seconds": 1.871},
  "usage": {"model": "gpt-6-luna", "prompt_tokens": 1450, "completion_tokens": 125, "estimated_cost_usd": 0.0002075}
}
```

`sources` lists only what the answer cites; each `number` matches a `[n]` marker in `answer`. `answered` is `false` when the sources didn't contain the answer. Errors: `422` for an invalid request (e.g. an empty question), `502` if the LLM call fails, `503` if the index is unreachable.

## Configuration

Copy `.env.example` to `.env` and fill it in. The settings that matter for running the app:

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `openai` (any OpenAI-compatible API) or `ollama` (local) |
| `LLM_MODEL`, `LLM_API_KEY` | model name and API key |
| `LLM_INPUT_COST_PER_MTOK`, `LLM_OUTPUT_COST_PER_MTOK` | USD per million tokens, for the cost estimate |
| `RETRIEVAL_MODE` | `vector`, `keyword` or `hybrid` (see [evaluation results](#evaluation)) |
| `GITHUB_TOKEN` | only needed to collect issues when building the index |

Docker Compose passes `.env` to the app and points `POSTGRES_DSN` at the `db` container itself. If you use Ollama running on your machine, set `LLM_BASE_URL=http://host.docker.internal:11434`, since `localhost` inside the container is the container.

## Building the index

A one-time step (re-run to refresh). It uses the project's virtual environment and the database container:

```
.venv\Scripts\Activate.ps1                          # PowerShell (Git Bash: source .venv/Scripts/activate)
pip install -r requirements.txt
docker compose up -d db                             # just the database

python -m repo_recall.collection.docs               # clone the docs repo, extract pages
python -m repo_recall.collection.issues             # fetch closed issues + comments (needs GITHUB_TOKEN; --limit N)
python -m repo_recall.processing.chunk_all          # clean and chunk docs and issues
python -m repo_recall.indexing.index_chunks         # embed chunks into Postgres (incremental, resumable)
```

## Development

Run the app from the virtual environment instead of Docker (the database still runs in Docker):

```
docker compose up -d db
python -m repo_recall.api                           # http://127.0.0.1:8000
```

Terminal versions of search and answering, handy for debugging:

```
python -m repo_recall.retrieval.cli "why does my pod keep restarting"
python -m repo_recall.generation.cli "why does my pod keep restarting"
```

Tests (Postgres integration tests are skipped when no database is running):

```
pytest
```

## Evaluation

`python -m repo_recall.evaluation.run` scores retrieval (Recall@k, MRR), answers (correctness and faithfulness, graded by a separate LLM judge), latency and cost for each retrieval mode, on 200 real questions from closed `kind/support` issues. It calls the paid LLM API, so pass `--max-cost` (and try `--sample 30` first); `--resume` continues an interrupted run. Baseline:

| Mode | Recall@5 | MRR | Correctness (1–5) | Faithfulness (1–5) | Latency p50 |
|---|---|---|---|---|---|
| vector | 0.389 | 0.295 | 2.73 | 4.43 | 3.9s |
| keyword | 0.217 | 0.159 | 2.10 | 4.22 | 6.9s |
| hybrid | 0.391 | 0.277 | 2.56 | 4.46 | 7.6s |

## Project structure

See [CLAUDE.md](./CLAUDE.md) for the folder layout, tech stack, and coding conventions.

## Attribution

Kubernetes documentation is sourced from [kubernetes/website](https://github.com/kubernetes/website), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
