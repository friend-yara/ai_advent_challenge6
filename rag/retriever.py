"""Cosine similarity search over persisted FAISS indexes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss

from rag.chunkers import Chunk
from rag.embedder import Embedder

_cache: dict[str, tuple[faiss.Index, list[Chunk]]] = {}

_DEFAULT_INDEX_DIR = Path(__file__).parent / "index"


def _load(chunker: str, index_dir: Path) -> tuple[faiss.Index, list[Chunk]]:
    """Load index and chunks from disk into module-level cache."""
    if chunker in _cache:
        return _cache[chunker]

    index = faiss.read_index(str(index_dir / f"index_{chunker}.faiss"))
    chunks_raw = json.loads((index_dir / f"chunks_{chunker}.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in chunks_raw]
    _cache[chunker] = (index, chunks)
    return index, chunks


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def search(
    query: str,
    top_k: int = 5,
    chunker: str | None = None,
    index_dir: Path | None = None,
) -> list[dict]:
    """Search index for top_k chunks most similar to query."""
    if index_dir is None:
        index_dir = _DEFAULT_INDEX_DIR

    if chunker is None:
        config_path = index_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError("RAG index not found. Run /index first.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        chunker = config.get("active", "fixed")

    index, chunks = _load(chunker, index_dir)
    embedder = Embedder()
    qvec = embedder.embed_one(query)
    q = np.array([qvec], dtype=np.float32)
    q = _l2_normalize(q[0])
    q = q.reshape(1, -1)

    scores, indices = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        c = chunks[idx]
        results.append({
            "id": c.id,
            "text": c.text,
            "source": c.source,
            "filename": c.filename,
            "section": c.section,
            "score": float(score),
        })
    return results


def filter_results(
    results: list[dict],
    threshold: float,
    query: str,
    use_keyword: bool,
) -> tuple[list[dict], list[dict]]:
    """Filter results by similarity threshold and optional keyword overlap.

    Returns (kept, dropped). Each dropped item gets a 'filtered_reason' key added.
    """
    significant_words = [w.lower() for w in query.split() if len(w) >= 4]
    kept: list[dict] = []
    dropped: list[dict] = []
    for r in results:
        if r["score"] < threshold:
            r["filtered_reason"] = "score"
            dropped.append(r)
            continue
        if use_keyword and significant_words:
            text_lower = r["text"].lower()
            if not any(w in text_lower for w in significant_words):
                r["filtered_reason"] = "keyword"
                dropped.append(r)
                continue
        kept.append(r)
    return kept, dropped


def extract_quote(text: str, max_chars: int = 180) -> str:
    """Extract a readable quote from chunk text, truncated at word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


def search_improved(
    query: str,
    initial_top_k: int = 15,
    final_top_k: int = 5,
    threshold: float = 0.3,
    use_keyword: bool = False,
    chunker: str | None = None,
    index_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Improved search: initial broad retrieval → threshold filter → keyword filter → top N.

    Returns (final_results[:final_top_k], dropped).
    """
    results = search(query, top_k=initial_top_k, chunker=chunker, index_dir=index_dir)
    kept, dropped = filter_results(results, threshold, query, use_keyword)
    return kept[:final_top_k], dropped
