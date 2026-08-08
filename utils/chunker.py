from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str

def simple_text_cleanup(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())

def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    clean_text = simple_text_cleanup(text)
    if not clean_text:
        return []

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end].strip())
        if end == len(clean_text):
            break
        start = max(end - overlap, 0)
    return [chunk for chunk in chunks if chunk]


def load_corpus_documents(corpus_dir: str | Path) -> List[Dict[str, str]]:
    corpus_dir = Path(corpus_dir)
    docs_dir = corpus_dir / "docs"
    if not docs_dir.exists():
        raise FileNotFoundError(f"Corpus docs directory not found: {docs_dir}")

    documents: List[Dict[str, str]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            continue

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        title = path.stem.replace("_", " ").title()
        source = "Local curated corpus"
        content_start = 0

        if lines and lines[0].lower().startswith("title:"):
            title = lines[0].split(":", 1)[1].strip() or title
            content_start = 1
        if len(lines) > content_start and lines[content_start].lower().startswith("source:"):
            source = lines[content_start].split(":", 1)[1].strip() or source
            content_start += 1

        content = "\n".join(lines[content_start:]).strip()
        documents.append(
            {
                "doc_id": path.stem,
                "title": title,
                "source": source,
                "text": content,
            }
        )
    return documents


def build_chunk_records(corpus_dir: str | Path, chunk_size: int = 450, overlap: int = 80) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for document in load_corpus_documents(corpus_dir):
        pieces = chunk_text(document["text"], chunk_size=chunk_size, overlap=overlap)
        for index, piece in enumerate(pieces):
            records.append(
                ChunkRecord(
                    chunk_id=f"{document['doc_id']}_chunk_{index}",
                    doc_id=document["doc_id"],
                    title=document["title"],
                    source=document["source"],
                    text=piece,
                )
            )
    return records
