"""Provider abstraction for the Ollama backend used by the evaluation harness."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when the Ollama provider cannot be initialized or used."""


class ProviderClient:
    """Simple wrapper around the local Ollama provider."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = resolve_provider(provider)
        self.model = model or self._default_model()
        self.api_key = api_key
        self._error: Optional[str] = None
        self._base_url: Optional[str] = None
        self._init_ollama()

    def _default_model(self) -> str:
        return os.getenv("OLLAMA_MODEL", os.getenv("TARGET_MODEL", "llama3.2:3b"))

    def _init_ollama(self) -> None:
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        if not self._base_url:
            self._error = "OLLAMA_BASE_URL is not set."
            logger.warning(self._error)

    def is_available(self) -> bool:
        return self._error is None

    def get_error(self) -> Optional[str]:
        return self._error

    def generate_text(self, prompt: str, *, max_tokens: int = 512) -> str:
        if not self.is_available():
            raise LLMProviderError(self._error or "Provider is not available")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        request = Request(
            f"{self._base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
                return str(body.get("response", "")).strip()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise LLMProviderError(f"Ollama request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise LLMProviderError(f"Ollama server not reachable at {self._base_url}: {exc.reason}") from exc


def resolve_provider(provider: Optional[str] = None) -> str:
    """Resolve the selected provider from environment variables and CLI input."""
    if provider is not None:
        value = provider.strip().lower()
        if value in {"", "ollama", "local"}:
            return "ollama"
        return "ollama"

    env_value = (os.getenv("LLM_PROVIDER") or os.getenv("PROVIDER") or "").strip().lower()
    if env_value in {"", "ollama", "local"}:
        return "ollama"

    return "ollama"
