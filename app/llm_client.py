"""LLM client abstractions for the gateway.

Security rules: API keys come ONLY from environment variables. Keys and
full prompts are NEVER logged, printed, or included in exception
messages.
"""

import json
import os
import re
from abc import ABC, abstractmethod

import httpx
from dotenv import load_dotenv

_TIMEOUT_SECONDS = 30.0
_TOKEN_RE = re.compile(r"<[A-Z_]+_\d+>")

# Gemini model names are retired often; set GEMINI_MODEL in .env.
_DEFAULT_GEMINI_MODEL = "your-gemini-model-name"


class LLMError(Exception):
    """Safe error raised for any client failure."""


class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class MockLLMClient(LLMClient):
    """Deterministic offline client that echoes <TYPE_N> tokens."""

    def complete(self, prompt: str) -> str:
        tokens = _TOKEN_RE.findall(prompt)
        if tokens:
            return "Mock reply referencing tokens: " + " ".join(tokens)
        return "Mock reply."


class GroqClient(LLMClient):
    """Groq chat completions client (OpenAI-style)."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self._transport = transport

    def complete(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError("groq request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError("groq request failed") from exc
        if response.status_code != 200:
            raise LLMError(f"groq request failed: HTTP {response.status_code}")
        try:
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("groq response malformed") from exc


class GeminiClient(LLMClient):
    """Google Gemini generateContent client."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
        self._transport = transport

    def complete(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        headers = {"x-goog-api-key": self._api_key}
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError("gemini request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError("gemini request failed") from exc
        if response.status_code != 200:
            raise LLMError(f"gemini request failed: HTTP {response.status_code}")
        try:
            data = response.json()
            return str(data["candidates"][0]["content"]["parts"][0]["text"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("gemini response malformed") from exc


def get_client(provider: str | None = None) -> LLMClient:
    """Factory: build a client from LLM_PROVIDER env (groq|gemini|mock)."""
    load_dotenv()
    name = (provider or os.environ.get("LLM_PROVIDER", "mock")).strip().lower()
    if name == "mock":
        return MockLLMClient()
    if name == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("missing GROQ_API_KEY environment variable")
        return GroqClient(key)
    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("missing GEMINI_API_KEY environment variable")
        return GeminiClient(key)
    raise LLMError(f"unknown LLM provider: {name!r}")
