from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    rank: int
    doc_id: str
    text: str
    score: float
    label: Optional[int] = None
    gt_answers: Optional[str] = None
    fake_cls: Optional[str] = None
    text_source: Optional[str] = None
    image_source: Optional[str] = None
    image_path: Optional[str] = None
    source_file: Optional[str] = None
    claimant: Optional[str] = None
    claim_text: Optional[str] = None
    claim_date: Optional[str] = None
    publisher: Optional[str] = None
    review_date: Optional[str] = None
    review_url: Optional[str] = None
    textual_rating: Optional[str] = None
    evidence_type: Optional[str] = None
    entity_id: Optional[str] = None
    entity_label: Optional[str] = None
    entity_description: Optional[str] = None
    current_positions: Optional[List[str]] = None
    all_positions: Optional[List[str]] = None
    death_date: Optional[str] = None
    article_title: Optional[str] = None
    article_domain: Optional[str] = None
    article_language: Optional[str] = None
    article_country: Optional[str] = None
    knowledge_label: Optional[str] = None
    knowledge_reason: Optional[str] = None
    uri: Optional[str] = None
    title: Optional[str] = None


class VerifyResponse(BaseModel):
    claim: str
    predicted_label: str
    prediction_index: int = Field(..., ge=0, le=2)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: dict
    explanation: str
    evidence: List[EvidenceItem]
    image_status: str
    resolved_image_path: Optional[str] = None
    retrieval_ready: bool
    fact_check_ready: bool = False
    gemini_ready: bool = False
    fact_check_error: Optional[str] = None
    checkpoint_loaded: bool
    classifier_predicted_label: str
    classifier_prediction_index: int = Field(..., ge=0, le=1)
    classifier_confidence: float = Field(..., ge=0.0, le=1.0)
    classifier_explanation: str
    llm_available: bool
    llm_model_used: str
    generation_mode: str
    model_used: str
    status: str = "success"
