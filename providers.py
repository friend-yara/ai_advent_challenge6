#!/usr/bin/env python3
"""
providers.py — LLM provider abstraction.

Two providers with identical interface:
- OpenAIProvider: OpenAI Responses API (existing behaviour)
- OllamaProvider: Ollama via Chat Completions API (/v1/chat/completions)

Both return data in Responses API shape so the rest of the codebase
sees a uniform format regardless of backend.
"""

import json
import sys
import time

import requests


# ---------------- OpenAI Provider ----------------

class OpenAIProvider:
    """OpenAI Responses API provider (original _post logic from agent.py)."""

    URL = "https://api.openai.com/v1/responses"

    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.name = "openai"
        self.supports_tools = True
        self.api_key = api_key

    def post(self, payload: dict, timeout: int) -> tuple[dict, float]:
        """Send request to OpenAI Responses API with retry on timeout."""
        retries = 3
        delay = 2
        start = time.monotonic()
        for i in range(retries):
            try:
                r = requests.post(
                    self.URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                elapsed = time.monotonic() - start
                return r.json(), elapsed
            except requests.exceptions.ReadTimeout:
                if i == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return {}, 0.0

    def resolve_model(self, model: str) -> str:
        """Return the model name that will actually be used."""
        return model

    def summary(self) -> str:
        """One-line status for /provider and /show."""
        return f"openai ({self.URL})"


# ---------------- Ollama Provider ----------------

class OllamaProvider:
    """Ollama provider via native /api/chat endpoint."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 default_model: str = "qwen3.5:9b",
                 num_threads: int | None = None,
                 num_predict: int | None = None):
        """Initialize with Ollama server URL and default model."""
        self.name = "ollama"
        self.supports_tools = False
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.num_threads = num_threads
        self.num_predict = num_predict

    def _map_model(self, model: str) -> str:
        """Map model name: names with ':' pass through, others use default."""
        if ":" in model:
            return model
        return self.default_model

    def _convert_payload(self, payload: dict) -> dict:
        """Convert Responses API payload → Ollama /api/chat payload."""
        inp = payload.get("input", "")
        messages: list[dict] = []

        if isinstance(inp, str):
            messages = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict):
                    continue
                if "role" in item and "content" in item:
                    messages.append({
                        "role": item["role"],
                        "content": item["content"],
                    })

        ollama_payload: dict = {
            "model": self._map_model(payload.get("model", self.default_model)),
            "messages": messages,
            "stream": True,
            "think": False,
        }
        options: dict = {}
        if payload.get("temperature") is not None:
            options["temperature"] = payload["temperature"]
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        elif payload.get("max_output_tokens") is not None:
            options["num_predict"] = payload["max_output_tokens"]
        else:
            options["num_predict"] = 1024
        if self.num_threads is not None:
            options["num_thread"] = self.num_threads
        if payload.get("stop"):
            ollama_payload["stop"] = payload["stop"]
        if options:
            ollama_payload["options"] = options

        return ollama_payload

    def _normalize_response(self, data: dict) -> dict:
        """Convert Ollama /api/chat response → Responses API shape."""
        output_items: list[dict] = []

        msg = data.get("message", {})
        content = msg.get("content", "")
        if content:
            output_items.append({
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            })

        usage_out = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }

        return {"output": output_items, "usage": usage_out}

    def post(self, payload: dict, timeout: int) -> tuple[dict, float]:
        """Convert payload, POST to Ollama /api/chat (streaming), normalize response."""
        url = f"{self.base_url}/api/chat"
        ollama_payload = self._convert_payload(payload)
        # Local models are slower; enforce a minimum timeout
        effective_timeout = max(timeout, 180)

        print("Ollama: генерация", end="", file=sys.stderr, flush=True)
        start = time.monotonic()
        try:
            r = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=ollama_payload,
                timeout=effective_timeout,
                stream=True,
            )
            collected = ""
            data: dict = {}
            dot_interval = 2.0  # print dot every N seconds
            last_dot = start
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    elapsed = time.monotonic() - start
                    print(f" ошибка {elapsed:.1f}s", file=sys.stderr, flush=True)
                    return {"error": {"message": chunk["error"]}}, elapsed
                token = chunk.get("message", {}).get("content", "")
                collected += token
                now = time.monotonic()
                if now - last_dot >= dot_interval:
                    print(".", end="", file=sys.stderr, flush=True)
                    last_dot = now
                if chunk.get("done"):
                    data = chunk
                    break
            elapsed = time.monotonic() - start
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)
            if eval_count and eval_duration:
                tok_per_sec = eval_count / (eval_duration / 1e9)
                print(f" {elapsed:.1f}s, {tok_per_sec:.1f} tok/s", file=sys.stderr, flush=True)
            else:
                print(f" {elapsed:.1f}s", file=sys.stderr, flush=True)
        except requests.exceptions.ConnectionError:
            print(" ошибка", file=sys.stderr, flush=True)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}"
            )
        except requests.exceptions.ReadTimeout:
            print(" таймаут", file=sys.stderr, flush=True)
            raise RuntimeError(
                f"Ollama request timed out after {effective_timeout}s. "
                f"Try increasing --timeout."
            )

        # Build response from streamed content
        if not data:
            data = {}
        # Override message content with fully collected text
        data["message"] = {"role": "assistant", "content": collected}

        return self._normalize_response(data), elapsed

    def list_models(self) -> list[str]:
        """Fetch available model names from Ollama /api/tags."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def resolve_model(self, model: str) -> str:
        """Return the model name that will actually be used."""
        return self._map_model(model)

    def summary(self) -> str:
        """One-line status for /provider and /show."""
        parts = [f"ollama {self.default_model}"]
        np = self.num_predict if self.num_predict is not None else 1024
        parts.append(f"predict={np}")
        if self.num_threads is not None:
            parts.append(f"threads={self.num_threads}")
        parts.append(self.base_url)
        return " | ".join(parts)
