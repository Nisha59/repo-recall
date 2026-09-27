"""A small GitHub REST API client with rate-limit handling and retries."""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.github.com"
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
RATE_LIMIT_STATUS_CODES = {403, 429}


class GitHubClient:
    """Authenticated GET access to the GitHub REST API.

    Waits out primary and secondary rate limits and retries transient
    server/network errors with exponential backoff, so callers can treat
    `get` as "eventually returns a successful response or raises".
    """

    def __init__(
        self,
        token: str,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        min_remaining: int = 1,
        timeout: float = 30.0,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self._sleep = sleep
        self._clock = clock
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._min_remaining = min_remaining
        self._timeout = timeout
        self.last_remaining: int | None = None
        """Core quota left according to the most recent response's headers."""

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """GET `url` (absolute, or a path relative to the API root).

        Raises `requests.HTTPError` for non-retryable errors, or once
        transient errors have exhausted `max_retries`.
        """
        if not url.startswith("http"):
            url = f"{API_BASE_URL}/{url.lstrip('/')}"

        attempt = 0
        while True:
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                attempt = self._backoff_or_raise(attempt, exc, f"network error: {exc}")
                continue

            if response.status_code in RATE_LIMIT_STATUS_CODES and self._is_rate_limited(response):
                self._wait_for_rate_limit(response)
                continue
            if response.status_code in TRANSIENT_STATUS_CODES:
                error = requests.HTTPError(f"{response.status_code} for {url}", response=response)
                attempt = self._backoff_or_raise(attempt, error, f"HTTP {response.status_code}")
                continue

            response.raise_for_status()
            self._record_remaining(response)
            self._throttle_if_low(response)
            return response

    def paginate(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Iterator[tuple[list[dict], str | None]]:
        """Yield `(items, next_url)` for each page, following `Link: rel="next"`.

        `next_url` is `None` on the last page. Params only apply to the first
        request; GitHub's `next` links already carry them.
        """
        next_url: str | None = url
        while next_url:
            response = self.get(next_url, params)
            params = None
            next_url = response.links.get("next", {}).get("url")
            yield response.json(), next_url

    def _backoff_or_raise(self, attempt: int, error: Exception, reason: str) -> int:
        """Sleep with exponential backoff, or raise `error` if out of retries."""
        if attempt >= self._max_retries:
            raise error
        delay = self._backoff_base ** (attempt + 1)
        logger.warning(
            "Transient %s; retry %d/%d in %.0fs", reason, attempt + 1, self._max_retries, delay
        )
        self._sleep(delay)
        return attempt + 1

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        """True if a 403/429 is a rate limit rather than a permissions error."""
        return (
            "Retry-After" in response.headers
            or response.headers.get("X-RateLimit-Remaining") == "0"
        )

    def _wait_for_rate_limit(self, response: requests.Response) -> None:
        """Sleep as long as GitHub asks after a rate-limited response."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            delay = float(retry_after)
            logger.warning("Secondary rate limit hit; waiting %.0fs", delay)
        else:
            delay = self._seconds_until_reset(response)
            logger.warning("Primary rate limit exhausted; waiting %.0fs for reset", delay)
        self._sleep(delay)

    def _record_remaining(self, response: requests.Response) -> None:
        """Remember the quota left, from the response's rate-limit header."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            self.last_remaining = int(remaining)

    def _throttle_if_low(self, response: requests.Response) -> None:
        """Pre-emptively wait for the reset when the remaining quota is nearly gone."""
        remaining = self.last_remaining
        if remaining is None or remaining > self._min_remaining:
            return
        delay = self._seconds_until_reset(response)
        logger.warning("Only %s requests left; waiting %.0fs for reset", remaining, delay)
        self._sleep(delay)

    def _seconds_until_reset(self, response: requests.Response, buffer: float = 5.0) -> float:
        """Seconds until `X-RateLimit-Reset`, plus a small safety buffer."""
        reset = response.headers.get("X-RateLimit-Reset")
        if reset is None:
            return 60.0
        return max(float(reset) - self._clock(), 0.0) + buffer
