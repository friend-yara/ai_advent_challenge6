"""Build and persist FAISS indexes from a markdown corpus."""
from __future__ import annotations

import html as _html
import json
import re
import time
from pathlib import Path

import mistune as _mistune
import numpy as np
import faiss

from rag.chunkers import FixedSizeChunker, StructuredChunker, chunker_metrics, Chunk
from rag.embedder import get_embedder

_md = _mistune.create_markdown()  # module-level singleton


def _md_to_plain(text: str) -> str:
    """Convert markdown to plain text via HTML intermediate."""
    raw_html = _md(text)
    plain = re.sub(r"<[^>]+>", " ", raw_html)
    return _html.unescape(plain)


def _strip_frontmatter(text: str) -> str:
    """Strip only YAML front-matter, preserving markdown structure."""
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def _clean(text: str) -> str:
    """Strip YAML front-matter, markdown, and normalize whitespace."""
    # Strip YAML front-matter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Convert markdown to plain text
    text = _md_to_plain(text)
    # Normalize whitespace (preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D array of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def build_index(
    corpus_dir: Path | list[Path],
    chunker: str,
    index_dir: Path,
    embedder_name: str = "openai",
    embed_model: str | None = None,
    ollama_url: str | None = None,
) -> dict:
    """Build FAISS index from corpus and persist to disk. Returns metrics."""
    # Normalise to list
    corpus_dirs = corpus_dir if isinstance(corpus_dir, list) else [corpus_dir]

    # Select chunker
    if chunker == "fixed":
        ch = FixedSizeChunker()
    elif chunker == "structured":
        ch = StructuredChunker()
    else:
        raise ValueError(f"Unknown chunker: {chunker!r}")

    # Walk corpus directories
    use_structured = isinstance(ch, StructuredChunker)
    all_chunks: list[Chunk] = []
    for cdir in corpus_dirs:
        for md_file in sorted(cdir.glob("**/*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text = _strip_frontmatter(text) if use_structured else _clean(text)
            if not text.strip():
                continue
            rel = str(md_file.relative_to(cdir))
            fname = md_file.name
            chunks = ch.chunk(text, rel, fname)
            if use_structured:
                for c in chunks:
                    c.text = _md_to_plain(c.text).strip()
                chunks = [c for c in chunks if c.text]
            all_chunks.extend(chunks)

        for txt_file in sorted(cdir.glob("**/*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            text = _clean(text)
            if not text.strip():
                continue
            rel = str(txt_file.relative_to(cdir))
            fname = txt_file.name
            all_chunks.extend(ch.chunk(text, rel, fname))

    if not all_chunks:
        raise RuntimeError(f"No chunks produced from {corpus_dir}")

    # Embed
    emb_kwargs: dict = {}
    if embedder_name == "ollama":
        if embed_model:
            emb_kwargs["model"] = embed_model
        if ollama_url:
            emb_kwargs["base_url"] = ollama_url
    embedder = get_embedder(embedder_name, **emb_kwargs)
    texts = [c.text for c in all_chunks]
    t0 = time.time()
    vectors = embedder.embed(texts)
    embed_seconds = time.time() - t0

    # Build FAISS index
    mat = np.array(vectors, dtype=np.float32)
    mat = _l2_normalize(mat)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    # Persist
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / f"index_{chunker}.faiss"))

    chunks_data = [
        {
            "id": c.id,
            "text": c.text,
            "source": c.source,
            "filename": c.filename,
            "section": c.section,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in all_chunks
    ]
    (index_dir / f"chunks_{chunker}.json").write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Update config
    config_path = index_dir / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    config["active"] = chunker
    config["embedder"] = embedder_name
    if embed_model:
        config["embed_model"] = embed_model
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = chunker_metrics(all_chunks)
    metrics["embed_seconds"] = embed_seconds
    return metrics
