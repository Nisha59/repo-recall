from unittest.mock import MagicMock

import pytest
import requests

from repo_recall.config import Settings
from repo_recall.generation.llm import (
    LLMError,
    LLMRequestError,
    OllamaClient,
    OpenAICompatibleClient,
    make_llm_client,
)


def session_returning(json_data: object, status: int = 200) -> MagicMock:
    return session_with(response(json_data, status))


def session_with(*responses: object) -> MagicMock:
    session = MagicMock()
    session.headers = {}
    session.post.side_effect = list(responses)
    return session


def openai_client(session: MagicMock) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        "gpt-x",
        "https://llm.example/v1/",
        "key",
        temperature=0.0,
        max_tokens=100,
        timeout=5.0,
        session=session,
        sleep=lambda seconds: None,
    )


def response(json_data: object, status: int = 200, text: str = "error body") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.ok = status < 400
    r.status_code = status
    r.text = text
    r.headers = {}
    r.json.return_value = json_data
    return r


OK_REPLY = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}


def test_openai_client_sends_chat_request_and_reads_usage():
    session = session_returning(
        {
            "model": "gpt-x-2026",
            "choices": [{"message": {"content": "answer [1]"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
    )
    completion = openai_client(session).complete("sys", "user")

    body = session.post.call_args.kwargs["json"]
    assert session.post.call_args.args[0] == "https://llm.example/v1/chat/completions"
    assert session.headers["Authorization"] == "Bearer key"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert body["max_tokens"] == 100
    assert (completion.text, completion.model) == ("answer [1]", "gpt-x-2026")
    assert (completion.prompt_tokens, completion.completion_tokens) == (120, 30)


def test_openai_client_tolerates_missing_usage():
    session = session_returning({"choices": [{"message": {"content": "hi"}}]})
    completion = openai_client(session).complete("s", "u")
    assert completion.prompt_tokens is None
    assert completion.model == "gpt-x"


def test_ollama_client_is_non_streaming_sets_context_and_reads_counts():
    session = session_returning(
        {
            "model": "llama3.1:8b",
            "message": {"content": "ok"},
            "prompt_eval_count": 900,
            "eval_count": 50,
        }
    )
    client = OllamaClient(
        "llama3.1:8b",
        "http://localhost:11434",
        temperature=0.0,
        max_tokens=200,
        num_ctx=8192,
        timeout=5.0,
        session=session,
    )
    completion = client.complete("s", "u")

    body = session.post.call_args.kwargs["json"]
    assert session.post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 200, "num_ctx": 8192}
    assert (completion.text, completion.prompt_tokens, completion.completion_tokens) == (
        "ok",
        900,
        50,
    )


def test_http_error_becomes_llm_error():
    with pytest.raises(LLMError, match="HTTP 401"):
        openai_client(session_returning({}, status=401)).complete("s", "u")


def test_connection_failure_becomes_llm_error():
    session = session_returning({})
    session.post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(LLMError, match="Could not reach"):
        openai_client(session).complete("s", "u")


def test_unexpected_response_shape_becomes_llm_error():
    with pytest.raises(LLMError, match="Unexpected response"):
        openai_client(session_returning({"choices": []})).complete("s", "u")


def test_factory_picks_client_and_default_base_url_from_settings():
    ollama = make_llm_client(Settings(_env_file=None, llm_provider="ollama"))
    openai = make_llm_client(Settings(_env_file=None, llm_provider="openai"))
    assert isinstance(ollama, OllamaClient)
    assert ollama._url == "http://localhost:11434/api/chat"
    assert isinstance(openai, OpenAICompatibleClient)
    assert openai._url == "https://api.openai.com/v1/chat/completions"


def test_factory_uses_configured_base_url():
    client = make_llm_client(
        Settings(_env_file=None, llm_provider="openai", llm_base_url="http://localhost:8000/v1")
    )
    assert client._url == "http://localhost:8000/v1/chat/completions"


def sent_bodies(session: MagicMock) -> list[dict]:
    return [call.kwargs["json"] for call in session.post.call_args_list]


def test_switches_to_max_completion_tokens_and_drops_temperature_when_rejected():
    session = session_with(
        response({}, 400, "Unsupported parameter: 'max_tokens' is not supported with this model."),
        response({}, 400, "Unsupported value: 'temperature' does not support 0.0 with this model."),
        response(OK_REPLY),
        response(OK_REPLY),
    )
    client = openai_client(session)

    assert client.complete("s", "u").text == "fine"
    client.complete("s", "u")

    first, second, third, later = sent_bodies(session)
    assert "max_tokens" in first and "temperature" in first
    assert "max_completion_tokens" in second and "temperature" in second
    assert third["max_completion_tokens"] == 100 and "temperature" not in third
    assert later == third  # the working form is remembered


def test_other_400_errors_are_not_retried():
    session = session_with(response({}, 400, "Invalid 'messages': empty"))
    with pytest.raises(LLMError, match="HTTP 400"):
        openai_client(session).complete("s", "u")
    assert session.post.call_count == 1


def test_rate_limits_and_server_errors_are_retried():
    session = session_with(response({}, 429), response({}, 503), response(OK_REPLY))
    assert openai_client(session).complete("s", "u").text == "fine"
    assert session.post.call_count == 3


def test_gives_up_after_max_retries():
    session = session_with(*[response({}, 503) for _ in range(4)])
    with pytest.raises(LLMError, match="HTTP 503"):
        openai_client(session).complete("s", "u")
    assert session.post.call_count == 4


def test_warns_when_token_limit_leaves_no_text(caplog):
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    completion = openai_client(session_returning(empty)).complete("s", "u")
    assert completion.text == ""
    assert "token limit ran out" in caplog.text


def test_request_rejected_after_another_thread_already_adapted_still_retries():
    session = session_with(
        response({}, 400, "Unsupported parameter: 'max_tokens' is not supported with this model."),
        response(OK_REPLY),
    )
    client = openai_client(session)
    stale_body = client._body([{"role": "user", "content": "u"}])  # built before the switch
    client._token_param = "max_completion_tokens"  # another thread switched meanwhile

    error = LLMRequestError("url", 400, "'max_tokens' is not supported")
    assert client._adapt_to(error, stale_body) is True
