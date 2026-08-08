from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.embed_index import DEFAULT_EMBED_MODEL, build_or_load_faiss_index
from utils.llm_infer import generate_verdict
from utils.ner_kg import build_basic_kg
from utils.retriever import FaissRetriever
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rag_scaffold import MultimediaRAGRetriever
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"


class ClaimVerificationPipeline:
    def __init__(self, embed_model_name: str = DEFAULT_EMBED_MODEL):
        index, metadata, embedder = build_or_load_faiss_index(
            corpus_dir=CORPUS_DIR,
            index_dir=INDEX_DIR,
            model_name=embed_model_name,
            force_rebuild=False,
        )
        self.retriever = FaissRetriever(index=index, metadata=metadata, embedder=embedder)

    def verify_claim(self, claim: str, top_k: int = 3, has_image: bool = False) -> Dict:
        # Step 1: Text-based evidence retrieval (RAG foundation)
        # TODO: Enhance to retrieve evidence based on image+text query using CLIP embeddings
        retrieved = self.retriever.search(claim, top_k=top_k)
        llm_ready_evidence: List[dict] = []
        for hit in retrieved:
            llm_ready_evidence.append(
                {
                    "rank": hit["rank"],
                    "doc_id": hit["doc_id"],
                    "title": hit["title"],
                    "source": hit["source"],
                    "chunk_id": hit["chunk_id"],
                    "text": hit["text"],
                    "snippet": hit["text"],
                    "score": round(hit["score"], 4),
                }
            )

        # Step 2: Extract attributes for fusion (entities, kg)
        # For full multimodal, this step will be injected into MultimodalFakeNewsClassifier

        entities, kg_triples, kg_summary = build_basic_kg(
            claim=claim,
            evidence_texts=[item["text"] for item in llm_ready_evidence],
        )
        verdict = generate_verdict(
            claim=claim,
            evidence=llm_ready_evidence,
            entities=entities,
            kg_triples=kg_triples,
        )

        return {
            "claim": claim,
            "has_image": has_image,
            "predicted_label": verdict["predicted_label"],
            "explanation": verdict["explanation"] + (" [Note: Multimodal Fusion Scaffolded]" if has_image else ""),
            "confidence": max(0.0, min(float(verdict["confidence"]), 1.0)),
            "evidence": llm_ready_evidence,
            "entities": entities,
            "kg_triples": kg_triples,
            "retrieval_summary": (
                f"Retrieved {len(llm_ready_evidence)} evidence chunks. "
                f"KG nodes: {len(kg_summary['nodes'])}, edges: {kg_summary['edges']}."
            ),
            "model_used": "MultimodalBaseline + " + verdict["model_used"],
        }
