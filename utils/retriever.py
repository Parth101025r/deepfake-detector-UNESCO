from __future__ import annotations

from typing import List

import numpy as np
# Enhanced with LLM-based reasoning support

from utils.embed_index import EmbedIndexer


class FaissRetriever:
    def __init__(self, index, metadata: List[dict], embedder: EmbedIndexer):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        query_vec = self.embedder.encode([query])
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        hits: List[dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["rank"] = rank
            item["score"] = float(score)
            hits.append(item)
        return hits
