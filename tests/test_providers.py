import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import ProviderClient, resolve_provider


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_resolve_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PROVIDER", raising=False)
    assert resolve_provider() == "ollama"
    assert resolve_provider("ollama") == "ollama"
    assert resolve_provider("OLLAMA") == "ollama"


def test_generate_text_uses_ollama_endpoint(monkeypatch):
    import src.providers as providers

    captured = {}

    def fake_urlopen(request, *args, **kwargs):
        captured["url"] = request.full_url
        captured["data"] = request.data.decode("utf-8")
        return DummyResponse({"response": "hello from ollama"})

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(providers, "urlopen", fake_urlopen)

    client = ProviderClient(provider="ollama", model="llama3.2:3b")
    result = client.generate_text("hello", max_tokens=64)

    assert result == "hello from ollama"
    assert captured["url"].endswith("/api/generate")
    assert '"model": "llama3.2:3b"' in captured["data"]
