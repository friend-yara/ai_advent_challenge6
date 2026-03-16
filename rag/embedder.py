"""OpenAI text embeddings via raw HTTP requests."""
from __future__ import annotations

import os
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
