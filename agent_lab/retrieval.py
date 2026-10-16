"""Retrieval for RAG: heading-aware chunking + TF-IDF vectors + cosine search.

Pure Python on purpose, so the retrieval step is deterministic, inspectable and
testable offline. Each chunk keeps its source file and heading for citations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its of on or "
    "our should that the their this to was we what when where which who why will with "
    "you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def chunk_markdown(source: str, text: str, max_chars: int = 900) -> list[Chunk]:
    """Split on headings, then on paragraphs when a section is too long."""
    chunks: list[Chunk] = []
    heading = source
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) > max_chars:
                chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, current))
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, heading, current))

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            buffer = []
            heading = line.lstrip("#").strip()
        else:
            buffer.append(line)
    flush()
    return chunks


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = chunks
        docs = [tokenize(f"{c.heading} {c.heading} {c.text}") for c in chunks]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n = len(docs)
        self.idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in df.items()}
        self.vectors = [self._vectorize(tokens) for tokens in docs]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[Hit]:
        qvec = self._vectorize(tokenize(query))
        if not qvec:
            return []
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in qvec.items())
            if score >= min_score:
                scored.append(Hit(chunk, round(score, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    @classmethod
    def from_directory(cls, directory: Path) -> "TfidfIndex":
        chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
        return cls(chunks)
