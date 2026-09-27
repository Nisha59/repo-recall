from unittest.mock import MagicMock

import pytest
import requests

from repo_recall.collection.github_client import GitHubClient

NOW = 1_000_000.0


def make_response(status: int, headers: dict | None = None, json_data: object = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {"X-RateLimit-Remaining": "4999"}
    response.links = {}
    response.json.return_value = json_data
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(str(status), response=response)
    return response


def make_client(*responses: object) -> tuple[GitHubClient, MagicMock, list[float]]:
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = list(responses)
    sleeps: list[float] = []
    client = GitHubClient("token", session=session, sleep=sleeps.append, clock=lambda: NOW)
    return client, session, sleeps


def test_sets_auth_header() -> None:
    _, session, _ = make_client()
    assert session.headers["Authorization"] == "Bearer token"


def test_waits_for_reset_when_primary_limit_exhausted() -> None:
    limited = make_response(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW) + 100)})
    client, session, sleeps = make_client(limited, make_response(200))

    assert client.get("repos/o/r").status_code == 200
    assert sleeps == [105.0]  # 100s until reset + 5s buffer
    assert session.get.call_count == 2


def test_honors_retry_after_on_secondary_limit() -> None:
    client, _, sleeps = make_client(make_response(429, {"Retry-After": "30"}), make_response(200))

    client.get("repos/o/r")

    assert sleeps == [30.0]


def test_throttles_proactively_when_quota_nearly_gone() -> None:
    low = make_response(200, {"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": str(int(NOW) + 10)})
    client, _, sleeps = make_client(low)

    client.get("repos/o/r")

    assert sleeps == [15.0]


def test_retries_transient_errors_with_backoff() -> None:
    client, _, sleeps = make_client(
        make_response(502), requests.ConnectionError("reset"), make_response(200)
    )

    assert client.get("repos/o/r").status_code == 200
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_max_retries() -> None:
    client, _, _ = make_client(*[make_response(503) for _ in range(6)])

    with pytest.raises(requests.HTTPError):
        client.get("repos/o/r")


def test_raises_immediately_on_non_retryable_error() -> None:
    client, session, sleeps = make_client(make_response(404))

    with pytest.raises(requests.HTTPError):
        client.get("repos/o/r")
    assert session.get.call_count == 1
    assert sleeps == []


def test_forbidden_without_rate_limit_headers_is_not_retried() -> None:
    client, session, _ = make_client(make_response(403, {"X-RateLimit-Remaining": "4000"}))

    with pytest.raises(requests.HTTPError):
        client.get("repos/o/r")
    assert session.get.call_count == 1


def test_paginate_follows_next_links() -> None:
    first = make_response(200, json_data=[{"n": 1}])
    first.links = {"next": {"url": "https://api.github.com/page2"}}
    second = make_response(200, json_data=[{"n": 2}])
    client, session, _ = make_client(first, second)

    pages = list(client.paginate("repos/o/r/issues", {"per_page": 100}))

    assert pages == [([{"n": 1}], "https://api.github.com/page2"), ([{"n": 2}], None)]
    assert session.get.call_args_list[1].kwargs["params"] is None
