"""Chunking strategies for the RAG corpus."""
from __future__ import annotations

import re
import math
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str          # "{filename}#{index}"
    text: str
    source: str      # relative path from corpus root
    filename: str    # basename
    section: str     # heading or "" for fixed
    char_start: int
    char_end: int


class FixedSizeChunker:
    """Splits text into fixed-size overlapping chunks at whitespace boundaries."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str, filename: str) -> list[Chunk]:
        """Split text into overlapping fixed-size chunks."""
        chunks = []
        start = 0
        idx = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            # extend to next whitespace boundary
            if end < length:
                boundary = text.rfind(" ", start, end + 40)
                if boundary > start:
                    end = boundary
            fragment = text[start:end].strip()
            if fragment:
                chunks.append(Chunk(
                    id=f"{filename}#{idx}",
                    text=fragment,
                    source=source,
                    filename=filename,
                    section="",
                    char_start=start,
                    char_end=end,
                ))
                idx += 1
            # next start with overlap
            next_start = end - self.overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return chunks


class StructuredChunker:
    """Splits text on Markdown headings; further splits large sections at paragraphs."""

    def __init__(self, max_section_chars: int = 1200):
        self.max_section_chars = max_section_chars

    def chunk(self, text: str, source: str, filename: str) -> list[Chunk]:
        """Split text by Markdown headings then by paragraphs for large sections."""
        # Split on headings (# ## ###)
        heading_re = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
        parts = heading_re.split(text)
        # parts alternates: [pre-text, heading, content, heading, content, ...]
        sections: list[tuple[str, str]] = []
        i = 0
        if parts[0].strip():
            sections.append(("", parts[0]))
        i = 1
        while i < len(parts) - 1:
            heading = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((heading, content))
            i += 2

        chunks: list[Chunk] = []
        idx = 0
        char_pos = 0

        for heading, content in sections:
            section_text = (f"{heading}\n{content}" if heading else content).strip()
            if not section_text:
                char_pos += len(heading) + len(content) + 1
                continue

            if len(section_text) <= self.max_section_chars:
                chunks.append(Chunk(
                    id=f"{filename}#{idx}",
                    text=section_text,
                    source=source,
                    filename=filename,
                    section=heading.lstrip("# "),
                    char_start=char_pos,
                    char_end=char_pos + len(section_text),
                ))
                idx += 1
            else:
                # Split at paragraph boundaries
                paragraphs = re.split(r"\n\n+", section_text)
                buf = ""
                buf_start = char_pos
                for para in paragraphs:
                    if len(buf) + len(para) + 2 > self.max_section_chars and buf:
                        chunks.append(Chunk(
                            id=f"{filename}#{idx}",
                            text=buf.strip(),
                            source=source,
                            filename=filename,
                            section=heading.lstrip("# "),
                            char_start=buf_start,
                            char_end=buf_start + len(buf),
                        ))
                        idx += 1
                        buf_start = buf_start + len(buf)
                        buf = para
                    else:
                        buf = f"{buf}\n\n{para}" if buf else para
                if buf.strip():
                    chunks.append(Chunk(
                        id=f"{filename}#{idx}",
                        text=buf.strip(),
                        source=source,
                        filename=filename,
                        section=heading.lstrip("# "),
                        char_start=buf_start,
                        char_end=buf_start + len(buf),
                    ))
                    idx += 1

            char_pos += len(heading) + len(content) + 1

        return chunks


def chunker_metrics(chunks: list[Chunk]) -> dict:
    """Compute corpus statistics for a list of chunks."""
    if not chunks:
        return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0, "std_chars": 0, "unique_sources": 0}
    lengths = [len(c.text) for c in chunks]
    n = len(lengths)
    avg = sum(lengths) / n
    variance = sum((x - avg) ** 2 for x in lengths) / n
    return {
        "count": n,
        "avg_chars": avg,
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "std_chars": math.sqrt(variance),
        "unique_sources": len({c.source for c in chunks}),
    }
