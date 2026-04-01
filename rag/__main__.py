"""CLI entry point: python -m rag <path> [<path>...] --chunker fixed|structured"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index from markdown corpus")
    parser.add_argument("paths", nargs="+", type=Path, help="Corpus directories to index")
    parser.add_argument("--chunker", choices=["fixed", "structured"], default="fixed")
    parser.add_argument("--index-dir", type=Path, default=Path("rag/index"))
    parser.add_argument("--embedder", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--ollama-url", default=None)
    args = parser.parse_args()

    for p in args.paths:
        if not p.is_dir():
            print(f"Error: {p} is not a directory", file=sys.stderr)
            sys.exit(1)

    metrics = build_index(
        corpus_dir=args.paths,
        chunker=args.chunker,
        index_dir=args.index_dir,
        embedder_name=args.embedder,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
    )
    print(f"Indexed {metrics.get('total_chunks', '?')} chunks ({args.chunker})")
    print(f"Embedding: {metrics.get('embed_seconds', 0):.1f}s")


if __name__ == "__main__":
    main()
