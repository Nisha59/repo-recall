"""Application configuration, loaded from a `.env` file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for RepoRecall.

    Values are loaded from environment variables / a `.env` file, falling
    back to Kubernetes-focused defaults so the project runs out of the box.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str | None = None
    github_repo: str = "kubernetes/kubernetes"
    # Kubernetes' prow bots are regular User accounts (not `type: "Bot"`),
    # so they have to be listed explicitly to be filtered out.
    github_bot_logins: list[str] = [
        "k8s-ci-robot",
        "k8s-triage-robot",
        "fejta-bot",
        "k8s-github-robot",
        "k8s-merge-robot",
    ]

    docs_repo_url: str = "https://github.com/kubernetes/website"
    docs_content_path: str = "content/en/docs"
    docs_public_base_url: str = "https://kubernetes.io/docs"

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")

    vector_backend: str = "lancedb"
    lancedb_path: Path = Path("data/lancedb")
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/repo_recall"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 64

    # Chunk sizes are measured in the embedding model's tokens. bge-small
    # truncates input past 512 tokens, so the max leaves room for its
    # special tokens and covers the whole embedded text (heading prefix too).
    chunk_max_tokens: int = 480
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 40
    issue_max_comments: int = 5

    # Retrieval. The query instruction is bge's recommended prefix for
    # queries (documents are embedded without it).
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    retrieval_top_k: int = 5
    retrieval_candidates: int = 50
    # Long pages/issues are split into many chunks that rank together;
    # cap how many one source can contribute to a result list (0 = no cap).
    retrieval_max_per_source: int = 2
    rrf_k: int = 60
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Answer generation. "openai" means any OpenAI-compatible API (OpenAI,
    # vLLM, LM Studio, OpenRouter, ...); llm_base_url picks the server and
    # defaults to OpenAI's or local Ollama's. Prices are USD per million
    # tokens, used only to estimate cost (0 for a local model).
    llm_provider: Literal["openai", "ollama"] = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_timeout: float = 120.0
    llm_input_cost_per_mtok: float = 0.0
    llm_output_cost_per_mtok: float = 0.0
    # Ollama's default context window can silently cut off a prompt with
    # several ~400-token sources; 8192 fits the default top-k comfortably.
    ollama_num_ctx: int = 8192

    # Evaluation set. The candidate query picks question-like issues; in
    # kubernetes/kubernetes maintainers label user questions kind/support.
    # Eval issues are excluded from the index by URL (see index_chunks).
    eval_candidate_query: str = "label:kind/support"
    eval_set_path: Path = Path("data/eval/eval_set.jsonl")
    eval_min_examples: int = 150
    eval_max_examples: int = 200

    # Evaluation run. The judge uses the same provider/key as generation but
    # its own model, so the generator doesn't grade itself. Reasoning models
    # spend hidden tokens before replying, hence the generous token limit.
    eval_judge_model: str = "gpt-5-mini"
    eval_judge_max_tokens: int = 4000
    eval_judge_input_cost_per_mtok: float = 0.25
    eval_judge_output_cost_per_mtok: float = 2.00
    eval_results_dir: Path = Path("data/eval/results")

    log_level: str = "INFO"
