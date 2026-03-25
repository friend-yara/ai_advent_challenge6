"""Text embeddings via OpenAI API or Ollama /api/embed."""
from __future__ import annotations

import os
import sys

import requests


class Embedder:
    """Batch-embeds texts using OpenAI text-embedding-3-small."""

    MODEL = "text-embedding-3-small"
    BATCH = 100

    def __init__(self):
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._url = "https://api.openai.com/v1/embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches; return list of float vectors."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            payload = {"model": self.MODEL, "input": batch}
            resp = requests.post(
                self._url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            data = resp.json()
            if data.get("error"):
                raise RuntimeError(f"Embeddings API error: {data['error']}")
            batch_vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            results.extend(batch_vecs)
        return results

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]


# ---------------- Ollama embedder ----------------

class OllamaEmbedder:
    """Embed texts using Ollama /api/embed endpoint."""

    BATCH = 100
    MAX_CHARS = 1500  # mxbai-embed-large context ≈ 512 tokens

    def __init__(self, model: str = "mxbai-embed-large",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches; return list of float vectors."""
        results: list[list[float]] = []
        truncated = 0
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            for j, t in enumerate(batch):
                if len(t) > self.MAX_CHARS:
                    batch[j] = t[: self.MAX_CHARS]
                    truncated += 1
            resp = requests.post(
                self._url,
                json={"model": self.model, "input": batch},
                timeout=120,
            )
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Ollama embed error: {data['error']}")
            results.extend(data["embeddings"])
        if truncated:
            print(f"[WARN] {truncated} chunk(s) truncated to {self.MAX_CHARS} chars for embedding", file=sys.stderr)
        return results

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]


# ---------------- Factory ----------------

def get_embedder(provider: str = "openai", **kwargs) -> Embedder | OllamaEmbedder:
    """Return the right embedder for the given provider name."""
    if provider == "ollama":
        return OllamaEmbedder(**kwargs)
    return Embedder()
