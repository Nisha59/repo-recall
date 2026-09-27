"""Chat-completion clients behind one interface: OpenAI-compatible APIs and Ollama."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from repo_recall.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434",
}
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The LLM request failed or returned something unusable."""


class LLMRequestError(LLMError):
    """The server answered with a non-retryable HTTP error."""

    def __init__(self, url: str, status: int, body: str) -> None:
        super().__init__(f"{url} returned HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class Completion:
    """The model's reply plus token counts (None when the server doesn't report them)."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class LLMClient(Protocol):
    """Anything that can answer a system + user prompt."""

    model: str

    def complete(self, system: str, user: str) -> Completion:
        """Return the model's reply to one system/user message pair."""
        ...


class OpenAICompatibleClient:
    """Calls `POST {base_url}/chat/completions`.

    Works with OpenAI itself and with servers that copy its API (vLLM,
    LM Studio, OpenRouter, Groq, ...), so `base_url` picks the server.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
        session: requests.Session | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._temperature: float | None = temperature
        self._max_tokens = max_tokens
        self._token_param = "max_tokens"
        self._timeout = timeout
        self._retry = _RetryPolicy(max_retries, sleep)
        self._session = session or requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def complete(self, system: str, user: str) -> Completion:
        """Send one chat request and return the first choice."""
        data = self._post_adapting(_messages(system, user))
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from {self._url}: {data!r:.300}") from exc
        if not text and choice.get("finish_reason") == "length":
            logger.warning(
                "%s returned no text: the %d-token limit ran out (reasoning models spend "
                "tokens before answering; raise the limit)",
                self.model,
                self._max_tokens,
            )
        usage = data.get("usage") or {}
        return Completion(
            text=text or "",
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """The request body in the form this model currently accepts."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._token_param: self._max_tokens,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        return body

    def _post_adapting(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """POST, adjusting the parameters once per kind if the model rejects them.

        Newer OpenAI models reject `max_tokens` (they want
        `max_completion_tokens`) and any non-default `temperature`. The
        working form is kept for later calls.
        """
        while True:
            body = self._body(messages)
            try:
                return _post_json(self._session, self._url, body, self._timeout, self._retry)
            except LLMRequestError as exc:
                if not self._adapt_to(exc, body):
                    raise

    def _adapt_to(self, error: LLMRequestError, sent: dict[str, Any]) -> bool:
        """Switch to the parameter form the error asks for; False if the request can't be fixed.

        Decides from what *this* request sent, not from shared state, so
        concurrent requests that fail the same way all retry correctly even
        after another thread has already switched.
        """
        if error.status != 400:
            return False
        if "max_tokens" in error.body and "max_tokens" in sent:
            if self._token_param == "max_tokens":
                logger.info("%s rejects max_tokens; using max_completion_tokens", self.model)
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in error.body and "temperature" in sent:
            if self._temperature is not None:
                logger.info("%s only supports the default temperature; not sending one", self.model)
            self._temperature = None
            return True
        return False


class OllamaClient:
    """Calls a local Ollama server's `POST {base_url}/api/chat` (non-streaming).

    `num_ctx` is set explicitly: Ollama's default context window is small
    enough that a prompt with several sources can be silently truncated.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
        self._timeout = timeout
        self._session = session or requests.Session()

    def complete(self, system: str, user: str) -> Completion:
        """Send one chat request and return the reply."""
        body = {
            "model": self.model,
            "messages": _messages(system, user),
            "stream": False,
            "options": self._options,
        }
        data = _post_json(self._session, self._url, body, self._timeout)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from {self._url}: {data!r:.300}") from exc
        return Completion(
            text=text or "",
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )


def make_llm_client(settings: Settings) -> LLMClient:
    """Build the client selected by `settings.llm_provider`."""
    base_url = settings.llm_base_url or DEFAULT_BASE_URLS[settings.llm_provider]
    if settings.llm_provider == "ollama":
        return OllamaClient(
            settings.llm_model,
            base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.llm_timeout,
        )
    return OpenAICompatibleClient(
        settings.llm_model,
        base_url,
        settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )


def _messages(system: str, user: str) -> list[dict[str, str]]:
    """The chat message list both APIs expect."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class _RetryPolicy:
    """How often and how patiently to retry transient failures."""

    max_retries: int = 3
    sleep: Callable[[float], None] = time.sleep
    backoff_base: float = 2.0

    def delay(self, attempt: int, response: requests.Response | None = None) -> float:
        """Seconds to wait before retry `attempt` (0-based), honoring Retry-After."""
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.backoff_base ** (attempt + 1)


def _post_json(
    session: requests.Session,
    url: str,
    body: dict[str, Any],
    timeout: float,
    retry: _RetryPolicy | None = None,
) -> dict[str, Any]:
    """POST `body` as JSON and return the decoded reply, wrapping failures in LLMError.

    Rate limits (429), server errors (5xx) and network errors are retried
    with backoff; other HTTP errors raise LLMRequestError at once.
    """
    retry = retry or _RetryPolicy()
    for attempt in range(retry.max_retries + 1):
        last_try = attempt == retry.max_retries
        try:
            response = session.post(url, json=body, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if last_try:
                raise LLMError(f"Could not reach {url}: {exc}") from exc
            logger.warning("Network error calling %s; retrying: %s", url, exc)
            retry.sleep(retry.delay(attempt))
            continue
        if response.status_code in TRANSIENT_STATUS_CODES and not last_try:
            logger.warning("HTTP %d from %s; retrying", response.status_code, url)
            retry.sleep(retry.delay(attempt, response))
            continue
        if not response.ok:
            raise LLMRequestError(url, response.status_code, response.text)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{url} returned invalid JSON") from exc
    raise AssertionError("unreachable")  # the loop always returns or raises
