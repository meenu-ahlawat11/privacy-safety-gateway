"""Tests for LLM clients. No real network calls. Synthetic data only."""

import json

import httpx
import pytest

from app.llm_client import (
    GeminiClient,
    GroqClient,
    LLMError,
    MockLLMClient,
    get_client,
)

FAKE_KEY = "fake-test-key-12345"


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_client.load_dotenv", lambda: None)


def _groq_transport(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "groq says hi"}}]},
    )


def test_mock_client_echoes_tokens() -> None:
    client = MockLLMClient()
    reply = client.complete("email <EMAIL_1> phone <PHONE_2> end")
    assert "<EMAIL_1>" in reply
    assert "<PHONE_2>" in reply
    assert client.complete("no tokens here") == "Mock reply."


def test_groq_request_headers_body_and_parse() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _groq_transport(request)

    client = GroqClient(FAKE_KEY, transport=httpx.MockTransport(handler))
    reply = client.complete("hello there")
    assert reply == "groq says hi"
    assert captured["auth"] == f"Bearer {FAKE_KEY}"
    body = captured["body"]
    assert body["messages"] == [{"role": "user", "content": "hello there"}]
    assert isinstance(body["model"], str) and body["model"]


def test_gemini_key_in_header_not_in_url() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key_header"] = request.headers.get("x-goog-api-key")
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}}]},
        )

    client = GeminiClient(FAKE_KEY, transport=httpx.MockTransport(handler))
    reply = client.complete("hello")
    assert reply == "gemini says hi"
    assert captured["key_header"] == FAKE_KEY
    assert FAKE_KEY not in captured["url"]


@pytest.mark.parametrize("status", [401, 500])
def test_http_errors_become_llmerror(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    for client in (
        GroqClient(FAKE_KEY, transport=httpx.MockTransport(handler)),
        GeminiClient(FAKE_KEY, transport=httpx.MockTransport(handler)),
    ):
        with pytest.raises(LLMError) as exc_info:
            client.complete("hi")
        assert FAKE_KEY not in str(exc_info.value)


def test_timeout_becomes_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = GroqClient(FAKE_KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert FAKE_KEY not in str(exc_info.value)


def test_malformed_json_becomes_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    client = GeminiClient(FAKE_KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert FAKE_KEY not in str(exc_info.value)


def test_missing_response_fields_become_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = GroqClient(FAKE_KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        client.complete("hi")


def test_factory_picks_right_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert isinstance(get_client(), GroqClient)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert isinstance(get_client(), GeminiClient)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert isinstance(get_client(), MockLLMClient)
    monkeypatch.delenv("LLM_PROVIDER")
    assert isinstance(get_client(), MockLLMClient)


def test_factory_unknown_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mystery")
    with pytest.raises(LLMError) as exc_info:
        get_client()
    assert FAKE_KEY not in str(exc_info.value)


def test_factory_missing_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError) as exc_info:
        get_client("groq")
    assert FAKE_KEY not in str(exc_info.value)
