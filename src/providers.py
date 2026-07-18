"""Provider abstraction for LLM backends used by the evaluation harness."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when the selected provider cannot be initialized or used."""


class ProviderClient:
    """Simple wrapper around Anthropic and Google Gemini providers."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = resolve_provider(provider)
        self.model = model or self._default_model()
        self.api_key = api_key
        self._client = None
        self._error: Optional[str] = None

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise LLMProviderError(f"Unsupported provider: {provider}")

    def _default_model(self) -> str:
        if resolve_provider() == "gemini":
            return os.getenv("TARGET_MODEL", "gemini-2.0-flash")
        return os.getenv("TARGET_MODEL", "claude-haiku-4-5-20251001")

    def _init_anthropic(self) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError:
            self._error = "Anthropic package is not installed. Install it with `pip install anthropic`."
            logger.warning(self._error)
            return

        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self._error = "ANTHROPIC_API_KEY is not set."
            logger.warning(self._error)
            return

        self._client = Anthropic(api_key=key)

    def _init_gemini(self) -> None:
        key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            self._error = "GOOGLE_API_KEY is not set."
            logger.warning(self._error)
            return

        try:
            from google import genai as google_genai  # type: ignore
        except ImportError:
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError:
                self._error = "Google Generative AI package is not installed. Install it with `pip install google-generativeai` or `pip install google-genai`."
                logger.warning(self._error)
                return
            genai.configure(api_key=key)
            self._client = genai
            self._client_mode = "legacy"
            return

        self._client = google_genai.Client(api_key=key)
        self._client_mode = "modern"

    def is_available(self) -> bool:
        return self._error is None and self._client is not None

    def get_error(self) -> Optional[str]:
        return self._error

    def generate_text(self, prompt: str, *, max_tokens: int = 512) -> str:
        if not self.is_available():
            raise LLMProviderError(self._error or "Provider is not available")

        if self.provider == "anthropic":
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()

        if self.provider == "gemini":
            max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
            base_delay = float(os.getenv("GEMINI_RETRY_DELAY", "5"))
            last_error: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    if getattr(self, "_client_mode", "") == "modern":
                        response = self._client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config={"max_output_tokens": max_tokens},
                        )
                    else:
                        model = self._client.GenerativeModel(self.model)
                        response = model.generate_content(prompt)

                    text = getattr(response, "text", None)
                    if text:
                        return str(text).strip()
                    return str(response).strip()
                except Exception as exc:  # pragma: no cover - exercised at runtime
                    last_error = exc
                    message = str(exc)
                    if "429" in message or "quota" in message.lower() or "rate limit" in message.lower():
                        if attempt < max_retries:
                            delay = base_delay * (attempt + 1)
                            logger.warning("Gemini quota exceeded, retrying in %ss (%s)", delay, message)
                            time.sleep(delay)
                            continue
                    raise LLMProviderError(message) from exc

            if last_error:
                raise LLMProviderError(str(last_error))

        raise LLMProviderError(f"Unsupported provider: {self.provider}")


def resolve_provider(provider: Optional[str] = None) -> str:
    """Resolve the selected provider from environment variables and CLI input."""
    if provider:
        value = provider.strip().lower()
        if value in {"gemini", "google"}:
            return "gemini"
        if value in {"anthropic", "claude"}:
            return "anthropic"

    env_value = (os.getenv("LLM_PROVIDER") or os.getenv("PROVIDER") or "").strip().lower()
    if env_value in {"gemini", "google"}:
        return "gemini"
    if env_value in {"anthropic", "claude"}:
        return "anthropic"

    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "anthropic"
