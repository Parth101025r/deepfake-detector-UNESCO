from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.chunker import ChunkRecord, build_chunk_records


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedIndexer:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


def save_metadata(metadata_path: str | Path, records: List[ChunkRecord]) -> None:
    metadata_path = Path(metadata_path)
    serializable = [
        {
            "chunk_id": record.chunk_id,
            "doc_id": record.doc_id,
            "title": record.title,
            "source": record.source,
            "text": record.text,
        }
        for record in records
    ]
    metadata_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_metadata(metadata_path: str | Path) -> List[dict]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def build_or_load_faiss_index(
    corpus_dir: str | Path,
    index_dir: str | Path,
    model_name: str = DEFAULT_EMBED_MODEL,
    force_rebuild: bool = False,
) -> tuple[faiss.IndexFlatIP, List[dict], EmbedIndexer]:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "corpus.index"
    metadata_path = index_dir / "corpus_chunks.json"

    embedder = EmbedIndexer(model_name=model_name)

    if index_path.exists() and metadata_path.exists() and not force_rebuild:
        index = faiss.read_index(str(index_path))
        metadata = load_metadata(metadata_path)
        return index, metadata, embedder

    records = build_chunk_records(corpus_dir)
    if not records:
        raise ValueError("No corpus chunks were created. Add .txt files to corpus/docs.")

    embeddings = embedder.encode([record.text for record in records])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    save_metadata(metadata_path, records)
    return index, load_metadata(metadata_path), embedder
