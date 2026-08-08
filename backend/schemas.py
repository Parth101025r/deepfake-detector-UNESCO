from typing import List, Optional

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="User-provided claim to verify")
    has_image: bool = Field(default=False, description="Whether an image was associated with the claim")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of evidence chunks to retrieve")


class EvidenceItem(BaseModel):
    rank: int
    doc_id: str
    title: str
    source: str
    chunk_id: str
    snippet: str
    score: float


class KGTriple(BaseModel):
    subject: str
    relation: str
    object: str


class VerifyResponse(BaseModel):
    claim: str
    has_image: bool = False
    predicted_label: str
    explanation: str
    confidence: float
    evidence: List[EvidenceItem]
    entities: List[str]
    kg_triples: List[KGTriple]
    retrieval_summary: Optional[str] = None
    model_used: str
    status: str = "success"
