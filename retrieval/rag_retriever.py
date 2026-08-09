import json
from pathlib import Path


class RealRAGRetriever:
    """
    FAISS + SentenceTransformer retriever used by the final multimodal demo pipeline.
    """

    def __init__(self, index_path="retrieval/index.faiss", metadata_path="retrieval/metadata.json"):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.ready = False
        self.metadata = []

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.faiss = faiss
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                raw_metadata = json.load(handle)
            self.metadata = [self._normalize_metadata_item(item, idx) for idx, item in enumerate(raw_metadata)]
            self.ready = True
            print("Loaded FAISS index and evidence metadata.")
        except ImportError:
            print("Missing dependencies. Please install faiss-cpu and sentence-transformers.")
        except Exception as exc:
            print(f"Failed to load FAISS index: {exc}. RAG retrieval will be empty.")

    @staticmethod
    def _normalize_metadata_item(item, idx):
        if isinstance(item, str):
            return {
                "doc_id": f"legacy::{idx}",
                "text": item,
                "label": None,
                "gt_answers": None,
                "fake_cls": None,
                "text_source": None,
                "image_source": None,
                "image_path": None,
                "source_file": None,
            }

        normalized = dict(item)
        normalized.setdefault("doc_id", f"doc::{idx}")
        normalized.setdefault("text", "")
        normalized.setdefault("label", None)
        normalized.setdefault("gt_answers", None)
        normalized.setdefault("fake_cls", None)
        normalized.setdefault("text_source", None)
        normalized.setdefault("image_source", None)
        normalized.setdefault("image_path", None)
        normalized.setdefault("source_file", None)
        return normalized

    def retrieve(self, query, top_k=3):
        if not self.ready or not str(query or "").strip():
            return []

        actual_k = max(1, min(int(top_k), len(self.metadata)))
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self.faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, actual_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["rank"] = rank
            item["score"] = float(score)
            results.append(item)
        return results
