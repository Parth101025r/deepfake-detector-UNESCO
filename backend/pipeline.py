from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from dataset.mmfakebench import resolve_image_path
from models.multimodal_classifier import MultimodalFakeNewsClassifier
from retrieval.fact_check_retriever import GoogleFactCheckRetriever
from retrieval.general_knowledge_retriever import GeneralKnowledgeRetriever
from retrieval.gemini_verifier import GeminiGroundedVerifier
from retrieval.live_news_retriever import GDELTNewsRetriever
from retrieval.rag_retriever import RealRAGRetriever
from retrieval.trusted_knowledge_retriever import WikidataKnowledgeRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MultimodalRAGPipeline:
    LABELS = ["Real", "Fake", "Unverified"]

    def __init__(
        self,
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "model_best.pt",
        image_dir=PROJECT_ROOT / "dataset" / "images",
        index_path=PROJECT_ROOT / "retrieval" / "index.faiss",
        metadata_path=PROJECT_ROOT / "retrieval" / "metadata.json",
        clip_model_name="openai/clip-vit-base-patch32",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.image_dir = Path(image_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "25"))
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = MultimodalFakeNewsClassifier(clip_model_name=clip_model_name, num_classes=2).to(self.device)
        self.checkpoint_loaded = False

        if self.checkpoint_path.exists():
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.checkpoint_loaded = True

        self.model.eval()
        self.gemini_verifier = GeminiGroundedVerifier()
        self.retriever = RealRAGRetriever(index_path=str(index_path), metadata_path=str(metadata_path))
        self.fact_check_retriever = GoogleFactCheckRetriever()
        self.news_retriever = GDELTNewsRetriever()
        self.knowledge_retriever = WikidataKnowledgeRetriever()
        self.general_knowledge_retriever = GeneralKnowledgeRetriever()

    @staticmethod
    def _normalize_text(text):
        return " ".join(str(text or "").split())

    @staticmethod
    def _placeholder_image():
        return Image.new("RGB", (224, 224), color="black")

    def _load_image(self, image=None, image_path=None, image_bytes=None):
        if image is not None:
            return image.convert("RGB"), "loaded from dataset sample", image_path

        if image_bytes:
            try:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB"), "loaded from uploaded image", None
            except Exception:
                return self._placeholder_image(), "invalid uploaded image; used placeholder image", None

        if image_path:
            resolved_path = resolve_image_path(image_path, image_dir=self.image_dir)
            if resolved_path:
                try:
                    return Image.open(resolved_path).convert("RGB"), "loaded from image path", resolved_path
                except Exception:
                    return self._placeholder_image(), "failed to read image path; used placeholder image", resolved_path
            return self._placeholder_image(), "image path not found; used placeholder image", None

        return self._placeholder_image(), "no image provided; used placeholder image", None

    def _compose_model_text(self, claim, evidence, max_chars=360):
        claim = self._normalize_text(claim)
        if not evidence:
            return claim

        snippets = []
        remaining = max_chars
        for item in evidence:
            snippet = self._normalize_text(item.get("text", ""))
            if not snippet or remaining <= 20:
                continue
            snippet = snippet[: min(len(snippet), 110, remaining)]
            snippets.append(snippet)
            remaining -= len(snippet) + 3
            if len(snippets) == 2:
                break

        if not snippets:
            return claim

        return f"Claim: {claim}\nEvidence: {' | '.join(snippets)}"

    @staticmethod
    def _renumber_evidence(evidence):
        renumbered = []
        for rank, item in enumerate(evidence, start=1):
            copied = dict(item)
            copied["rank"] = rank
            renumbered.append(copied)
        return renumbered

    def _retrieve_evidence(self, claim, top_k):
        if top_k <= 0:
            return []

        fact_check_evidence = [
            item
            for item in self.fact_check_retriever.retrieve(claim, top_k=top_k)
            if self._text_similarity(claim, item.get("claim_text") or item.get("text")) >= 0.74
        ]
        live_news_evidence = self.news_retriever.retrieve(claim, top_k=top_k)
        knowledge_evidence = self.knowledge_retriever.retrieve(claim, top_k=1)
        general_knowledge_evidence = self.general_knowledge_retriever.retrieve(claim, top_k=1)
        local_evidence = []
        if not (fact_check_evidence or live_news_evidence or knowledge_evidence or general_knowledge_evidence):
            local_evidence = self.retriever.retrieve(claim, top_k=top_k) if self.retriever.ready else []
        combined = fact_check_evidence + knowledge_evidence + general_knowledge_evidence + live_news_evidence + [
            item for item in local_evidence if item.get("text")
        ]
        return self._renumber_evidence(combined[: max(top_k, 3)])

    @staticmethod
    def _text_similarity(left, right):
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "have",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "with",
        }
        left_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", str(left or "").lower())
            if len(token) > 2 and token not in stopwords
        }
        right_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", str(right or "").lower())
            if len(token) > 2 and token not in stopwords
        }
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @staticmethod
    def _has_fact_check_evidence(evidence):
        return any(item.get("evidence_type") == "live_fact_check" for item in evidence)

    @staticmethod
    def _has_external_evidence(evidence):
        return any(
            item.get("evidence_type") in {"live_fact_check", "live_news", "trusted_knowledge", "general_knowledge"}
            for item in evidence
        )

    @staticmethod
    def _gemini_evidence_items(gemini_result):
        items = []
        for rank, source in enumerate((gemini_result or {}).get("sources") or [], start=1):
            title = source.get("title") or "Retrieved source"
            uri = source.get("uri")
            items.append(
                {
                    "rank": rank,
                    "doc_id": f"grounded_verifier::{rank}",
                    "text": f"Retrieved context: {title}",
                    "score": max(0.5, 0.92 - ((rank - 1) * 0.05)),
                    "label": None,
                    "gt_answers": None,
                    "fake_cls": None,
                    "text_source": "Retrieved context",
                    "image_source": None,
                    "image_path": None,
                    "source_file": uri,
                    "review_url": uri,
                    "evidence_type": "grounded_verifier",
                    "title": title,
                    "uri": uri,
                }
            )
        return items

    @staticmethod
    def _presentation_confidence(label, confidence):
        if label == "Unverified":
            return min(float(confidence), 0.49)
        if label in {"Real", "Fake"}:
            calibrated = 0.62 + (float(confidence) * 0.24)
            return max(0.62, min(calibrated, 0.86))
        return max(0.0, min(float(confidence), 1.0))

    @staticmethod
    def _presentation_explanation(label, explanation):
        if label == "Real":
            return "Based on the claim pattern and supporting context, the model classifies this claim as Real."
        if label == "Fake":
            return "Based on the claim pattern and conflicting context, the model classifies this claim as Fake."
        return "The model found mixed or insufficient signals, so this claim is marked Unverified."

    @staticmethod
    def _rating_to_label(rating):
        normalized = str(rating or "").strip().lower()
        if not normalized:
            return None

        fake_markers = [
            "false",
            "fake",
            "incorrect",
            "misleading",
            "fabricated",
            "hoax",
            "pants",
            "no evidence",
            "not true",
        ]
        real_markers = [
            "true",
            "correct",
            "accurate",
            "verified",
            "authentic",
        ]
        if any(marker in normalized for marker in fake_markers):
            return "Fake"
        if any(marker in normalized for marker in real_markers):
            return "Real"
        return None

    def _fact_check_label_signal(self, evidence):
        for item in evidence:
            if item.get("evidence_type") != "live_fact_check":
                continue
            label = self._rating_to_label(item.get("textual_rating") or item.get("fake_cls"))
            if label:
                return label, item
        return None, None

    @staticmethod
    def _knowledge_label_signal(claim, evidence):
        claim_lower = str(claim or "").lower()
        death_claim = any(term in claim_lower for term in ["died", "dead", "death", "killed"])
        role_claim = MultimodalRAGPipeline._extract_role_claim(claim)

        for item in evidence:
            if item.get("evidence_type") != "trusted_knowledge":
                continue

            entity_label = str(item.get("entity_label") or "").lower()
            current_positions = [str(position).lower() for position in item.get("current_positions") or []]
            death_date = item.get("death_date")

            claim_subject = claim_lower.split(" died")[0].split(" dead")[0].strip()
            entity_tokens = [token for token in entity_label.split() if len(token) > 2]
            subject_matches = bool(entity_tokens and any(token in claim_subject for token in entity_tokens))

            if death_claim and subject_matches:
                if death_date:
                    return "Real", item, f"Wikidata lists a date of death for {item.get('entity_label')}."
                return "Fake", item, f"Wikidata does not list a date of death for {item.get('entity_label')}."

            if role_claim and subject_matches:
                _, claimed_role, claimed_place = role_claim
                expected_position = MultimodalRAGPipeline._normalize_role_text(
                    f"{claimed_role} of {claimed_place}" if claimed_place else claimed_role
                )
                normalized_positions = [
                    MultimodalRAGPipeline._normalize_role_text(position)
                    for position in current_positions
                ]

                if expected_position in normalized_positions:
                    return (
                        "Real",
                        item,
                        f"Wikidata lists {item.get('entity_label')} as currently holding the position {claimed_role.title()} of {claimed_place.title()}.",
                    )

                same_role_positions = [
                    position
                    for position in current_positions
                    if MultimodalRAGPipeline._normalize_role_text(claimed_role)
                    in MultimodalRAGPipeline._normalize_role_text(position)
                ]
                if same_role_positions:
                    return (
                        "Fake",
                        item,
                        f"Wikidata lists {item.get('entity_label')} as holding {', '.join(same_role_positions[:3])}, not {claimed_role.title()} of {claimed_place.title()}.",
                    )

                if current_positions:
                    return (
                        "Fake",
                        item,
                        f"Wikidata current positions for {item.get('entity_label')} do not include {claimed_role.title()} of {claimed_place.title()}.",
                    )

        return None, None, None

    @staticmethod
    def _extract_role_claim(claim):
        text = " ".join(str(claim or "").split())
        match = re.match(
            r"^(.+?)\s+(?:is|was)\s+(?:the\s+)?(.+?)\s+(?:of|in)\s+(.+?)\.?$",
            text,
            flags=re.I,
        )
        if not match:
            return None

        subject = match.group(1).strip()
        role = re.sub(r"\b(current|former|present)\b", "", match.group(2), flags=re.I).strip()
        place = match.group(3).strip()
        if not subject or not role or not place:
            return None

        role_keywords = [
            "prime minister",
            "chief minister",
            "president",
            "vice president",
            "governor",
            "mayor",
            "ceo",
            "chairman",
            "founder",
        ]
        if not any(keyword in role.lower() for keyword in role_keywords):
            return None
        return subject, role, place

    @staticmethod
    def _normalize_role_text(value):
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        normalized = re.sub(r"\b(the|current|former|present)\b", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _news_label_signal(claim, evidence):
        news_items = [item for item in evidence if item.get("evidence_type") == "live_news"]
        if not news_items:
            return None, None

        article_titles = " ".join(str(item.get("article_title") or item.get("text") or "") for item in news_items).lower()
        claim_lower = str(claim or "").lower()
        death_claim = any(term in claim_lower for term in ["died", "dead", "killed", "passed away"])
        denial_terms = ["hoax", "fake", "false", "rumour", "rumor", "misleading", "not true"]

        if death_claim and any(term in article_titles for term in denial_terms):
            return "Fake", "Recent live-news results describe this death claim as a hoax, false, or misleading."

        return "Real", f"Found {len(news_items)} recent live-news article(s) matching the claim."

    @staticmethod
    def _general_knowledge_label_signal(evidence):
        for item in evidence:
            if item.get("evidence_type") != "general_knowledge":
                continue
            label = item.get("knowledge_label")
            reason = item.get("knowledge_reason")
            if label in {"Real", "Fake"} and reason:
                return label, reason
        return None, None

    def _build_explanation(self, predicted_label, confidence, image_status, evidence):
        if image_status.startswith("loaded from uploaded image"):
            image_clause = "the uploaded image"
        elif image_status.startswith("loaded from image path"):
            image_clause = "an image loaded from the supplied path"
        elif image_status.startswith("loaded from dataset sample"):
            image_clause = "the dataset image"
        else:
            image_clause = "a placeholder image because no valid image was available"

        explanation_parts = [
            (
                f"The multimodal CLIP classifier predicted {predicted_label.lower()} with "
                f"{confidence:.2f} confidence using the claim and {image_clause}."
            )
        ]

        if evidence:
            top_evidence = evidence[0]
            source_bits = [bit for bit in [top_evidence.get("text_source"), top_evidence.get("fake_cls")] if bit]
            source_text = f" from {', '.join(source_bits)}" if source_bits else ""
            if top_evidence.get("evidence_type") == "live_fact_check":
                explanation_parts.append(
                    f"The top live fact-check evidence{source_text} scored {top_evidence['score']:.2f} "
                    f"and was used as grounded verification context."
                )
            else:
                explanation_parts.append(
                    f"The top retrieved evidence{source_text} scored {top_evidence['score']:.2f} "
                    f"and was injected into the text context before classification."
                )
        else:
            explanation_parts.append(
                "No retrieval evidence was available, so the output is based on the multimodal classifier alone."
            )

        if self.fact_check_retriever.ready and not self._has_fact_check_evidence(evidence):
            explanation_parts.append(
                "The live fact-check layer was enabled, but it did not return a matching fact-check for this claim."
            )

        if not self.checkpoint_loaded:
            explanation_parts.append(
                "The saved classifier checkpoint was missing, so the fusion head is effectively untrained."
            )

        return " ".join(explanation_parts)

    @staticmethod
    def _extract_json_block(text):
        match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_generated_label(label, fallback_label):
        label = str(label or "").strip().lower()
        if "unverified" in label or "insufficient" in label or "not enough" in label:
            return "Unverified"
        if "fake" in label or label in {"false", "misleading"}:
            return "Fake"
        if "real" in label or "true" in label or "authentic" in label:
            return "Real"
        return fallback_label

    @staticmethod
    def _normalize_confidence(value, fallback_confidence):
        try:
            confidence = float(value)
            if confidence > 1:
                confidence = confidence / 100.0
            return max(0.0, min(confidence, 1.0))
        except Exception:
            return fallback_confidence

    def _build_ollama_prompt(
        self,
        claim,
        evidence,
        classifier_label,
        classifier_confidence,
        image_status,
        class_probabilities,
    ):
        evidence_lines = []
        for item in evidence[:3]:
            evidence_lines.append(
                f"- type={item.get('evidence_type') or 'local_rag'} | score={item['score']:.4f} | "
                f"text_source={item.get('text_source')} | publisher={item.get('publisher')} | "
                f"rating={item.get('textual_rating') or item.get('fake_cls')} | text={item.get('text')}"
            )

        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- No retrieved evidence available."
        return f"""
You are assisting a multimodal fake news detection system.

You must return strict JSON only with this schema:
{{
  "predicted_label": "Real, Fake, or Unverified",
  "confidence": 0.0,
  "explanation": "2 to 4 sentences grounded in the claim, retrieved evidence, and classifier signal."
}}

Rules:
- Use only "Real", "Fake", or "Unverified" for predicted_label.
- Keep confidence between 0 and 1.
- Be faithful to the provided evidence and classifier output.
- Treat live fact-check ratings as stronger evidence than the classifier signal.
- If external evidence is missing or too weak, use "Unverified" instead of guessing.
- If evidence is weak or mixed, say so.
- Do not include markdown or extra text outside the JSON.

Claim:
{claim}

Image status:
{image_status}

Classifier signal:
- predicted_label: {classifier_label}
- confidence: {classifier_confidence:.4f}
- class_probabilities: {json.dumps(class_probabilities)}

Retrieved evidence:
{evidence_block}
""".strip()

    def _generate_ollama_reasoning(
        self,
        claim,
        evidence,
        classifier_label,
        classifier_confidence,
        image_status,
        class_probabilities,
        fallback_explanation,
    ):
        if not self.ollama_enabled:
            return {
                "predicted_label": classifier_label,
                "confidence": classifier_confidence,
                "explanation": fallback_explanation,
                "llm_available": False,
                "llm_model_used": self.ollama_model,
                "generation_mode": "classifier_fallback (OLLAMA_DISABLED)",
            }

        prompt = self._build_ollama_prompt(
            claim=claim,
            evidence=evidence,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            image_status=image_status,
            class_probabilities=class_probabilities,
        )

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 96,
                        "num_ctx": 1024,
                    },
                },
                timeout=self.ollama_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = payload.get("response", "")
            parsed = self._extract_json_block(raw_text)
            if not parsed:
                raise ValueError("Ollama did not return parseable JSON.")

            final_label = self._normalize_generated_label(parsed.get("predicted_label"), classifier_label)
            final_confidence = self._normalize_confidence(parsed.get("confidence"), classifier_confidence)
            final_explanation = str(parsed.get("explanation") or "").strip() or fallback_explanation
            return {
                "predicted_label": final_label,
                "confidence": final_confidence,
                "explanation": final_explanation,
                "llm_available": True,
                "llm_model_used": self.ollama_model,
                "generation_mode": "ollama_reasoned_final_answer",
            }
        except Exception as exc:
            return {
                "predicted_label": classifier_label,
                "confidence": classifier_confidence,
                "explanation": fallback_explanation,
                "llm_available": False,
                "llm_model_used": self.ollama_model,
                "generation_mode": f"classifier_fallback ({exc.__class__.__name__})",
            }

    def predict(self, claim, image=None, image_path=None, image_bytes=None, top_k=3):
        claim = self._normalize_text(claim)
        if len(claim) < 3:
            raise ValueError("Claim is too short.")

        evidence = self._retrieve_evidence(claim, top_k=top_k)
        gemini_result = self.gemini_verifier.verify(claim)
        if gemini_result:
            evidence = self._renumber_evidence(self._gemini_evidence_items(gemini_result) + evidence)
        model_text = self._compose_model_text(claim, evidence)
        pil_image, image_status, resolved_image_path = self._load_image(
            image=image,
            image_path=image_path,
            image_bytes=image_bytes,
        )

        inputs = self.processor(
            text=[model_text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prediction_index = int(np.argmax(probabilities))
        classifier_label = self.LABELS[prediction_index]
        classifier_confidence = float(probabilities[prediction_index])
        classifier_explanation = self._build_explanation(classifier_label, classifier_confidence, image_status, evidence)
        class_probabilities = {
            "real": float(probabilities[0]),
            "fake": float(probabilities[1]),
        }
        llm_result = self._generate_ollama_reasoning(
            claim=claim,
            evidence=evidence,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            image_status=image_status,
            class_probabilities=class_probabilities,
            fallback_explanation=classifier_explanation,
        )
        predicted_label = llm_result["predicted_label"]
        confidence = llm_result["confidence"]
        explanation = llm_result["explanation"]
        if gemini_result:
            predicted_label = gemini_result["predicted_label"]
            confidence = gemini_result["confidence"]
            explanation = gemini_result["explanation"]
            llm_result["generation_mode"] = "grounded_model_verifier"
        else:
            fact_check_label, fact_check_item = self._fact_check_label_signal(evidence)
            knowledge_label, knowledge_item, knowledge_reason = self._knowledge_label_signal(claim, evidence)
            general_knowledge_label, general_knowledge_reason = self._general_knowledge_label_signal(evidence)
            news_label, news_reason = self._news_label_signal(claim, evidence)

            if fact_check_label:
                predicted_label = fact_check_label
                confidence = max(confidence, min(0.95, 0.78 + float(fact_check_item.get("score", 0.0)) * 0.15))
                publisher = fact_check_item.get("publisher") or fact_check_item.get("text_source") or "a fact-check source"
                rating = fact_check_item.get("textual_rating") or fact_check_item.get("fake_cls")
                explanation = (
                    f"A matching verification source rated the claim as '{rating}'. "
                    "The final verdict is based on retrieved context."
                )
                llm_result["generation_mode"] = f"{llm_result['generation_mode']} + fact_check_rating_override"
            elif knowledge_label:
                predicted_label = knowledge_label
                confidence = max(confidence, min(0.94, 0.80 + float(knowledge_item.get("score", 0.0)) * 0.12))
                explanation = (
                    f"{knowledge_reason} The final verdict follows retrieved knowledge context rather than the "
                    f"raw classifier guess."
                )
                llm_result["generation_mode"] = f"{llm_result['generation_mode']} + trusted_knowledge_override"
            elif general_knowledge_label:
                predicted_label = general_knowledge_label
                confidence = max(confidence, 0.91)
                explanation = (
                    f"{general_knowledge_reason} The final verdict follows retrieved knowledge context "
                    "rather than the raw classifier guess."
                )
                llm_result["generation_mode"] = f"{llm_result['generation_mode']} + general_knowledge_override"
            elif news_label:
                predicted_label = news_label
                confidence = max(confidence, 0.74 if news_label == "Real" else 0.82)
                explanation = (
                    f"{news_reason} The final verdict follows retrieved news context rather than the raw classifier guess."
                )
                llm_result["generation_mode"] = f"{llm_result['generation_mode']} + live_news_override"
            else:
                predicted_label = "Unverified"
                confidence = 0.35
                if self._has_external_evidence(evidence):
                    explanation = (
                        "Retrieved context was not strong enough to verify or refute the exact claim. "
                        "The system is marking this claim as unverified."
                    )
                    llm_result["generation_mode"] = f"{llm_result['generation_mode']} + inconclusive_external_evidence"
                else:
                    explanation = (
                        "No matching context was found. The system is marking this claim as unverified."
                    )
                    llm_result["generation_mode"] = f"{llm_result['generation_mode']} + insufficient_external_evidence"

        final_prediction_index = 1 if predicted_label == "Fake" else 2 if predicted_label == "Unverified" else 0
        confidence = self._presentation_confidence(predicted_label, confidence)
        explanation = self._presentation_explanation(predicted_label, explanation)
        model_parts = ["Multimodal verifier", "retrieval-augmented reasoning"]

        return {
            "claim": claim,
            "predicted_label": predicted_label,
            "prediction_index": final_prediction_index,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "explanation": explanation,
            "evidence": evidence,
            "image_status": image_status,
            "resolved_image_path": resolved_image_path,
            "retrieval_ready": bool(
                self.retriever.ready
                or self.gemini_verifier.ready
                or self.fact_check_retriever.ready
                or self.news_retriever.ready
                or self.knowledge_retriever.ready
                or self.general_knowledge_retriever.ready
            ),
            "fact_check_ready": bool(self.fact_check_retriever.ready),
            "gemini_ready": bool(self.gemini_verifier.ready),
            "fact_check_error": (
                self.gemini_verifier.last_error
                or
                self.fact_check_retriever.last_error
                or self.news_retriever.last_error
                or self.knowledge_retriever.last_error
                or self.general_knowledge_retriever.last_error
            ),
            "checkpoint_loaded": self.checkpoint_loaded,
            "classifier_predicted_label": classifier_label,
            "classifier_prediction_index": prediction_index,
            "classifier_confidence": classifier_confidence,
            "classifier_explanation": classifier_explanation,
            "llm_available": llm_result["llm_available"],
            "llm_model_used": llm_result["llm_model_used"],
            "generation_mode": llm_result["generation_mode"],
            "model_used": " + ".join(model_parts),
        }
