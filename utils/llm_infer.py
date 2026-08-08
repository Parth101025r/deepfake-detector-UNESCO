"""
LLM Verdict Module

This module uses an LLM (Ollama - llama3.1) to classify claims into six truth categories
based on evidence, entities, and knowledge graph triples. It generates a predicted label,
explanation, and confidence score. If the LLM is unavailable, a heuristic fallback is used.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

import requests


LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]


def _extract_json_block(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    if label in LABELS:
        return label
    if "pants" in label:
        return "pants-fire"
    if "mostly" in label:
        return "mostly-true"
    if "barely" in label:
        return "barely-true"
    if "half" in label:
        return "half-true"
    if "false" in label:
        return "false"
    if "true" in label:
        return "true"
    return "half-true"


def _heuristic_fallback(claim: str, evidence: List[dict]) -> dict:
    joined = " ".join(item["text"] for item in evidence).lower()
    claim_lower = claim.lower()

    if any(token in claim_lower for token in ["never", "hoax", "fake", "not real"]) and any(
        token in joined for token in ["evidence", "study", "confirmed", "data", "records"]
    ):
        label = "false"
    elif any(token in joined for token in ["mixed", "partial", "some", "while"]) or len(evidence) >= 2:
        label = "half-true"
    else:
        label = "mostly-true"

    return {
        "predicted_label": label,
        "explanation": (
            "Fallback verdict used because Ollama was unavailable. The system relied on retrieved evidence overlap "
            "and simple heuristics, so treat this as a demo-only approximation."
        ),
        "confidence": 0.42,
        "model_used": "heuristic-fallback",
    }


def generate_verdict(
    claim: str,
    evidence: List[dict],
    entities: List[str],
    kg_triples: List[dict],
    ollama_model: str | None = None,
    timeout: int = 120,
) -> dict:
    ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    prompt = f"""
You are a fake news classifier for the LIAR-style six-class setup.

Allowed labels:
- pants-fire
- false
- barely-true
- half-true
- mostly-true
- true

Given the claim, retrieved evidence, and simple knowledge graph triples, respond with strict JSON only:
{{
  "predicted_label": "<one allowed label>",
  "explanation": "<2-3 sentence explanation>",
  "confidence": <number between 0 and 1>
}}

Claim:
{claim}

Entities:
{entities}

Knowledge Graph Triples:
{kg_triples}

Evidence:
{json.dumps(evidence, indent=2)}
"""

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        parsed = _extract_json_block(payload.get("response", "")) or json.loads(payload.get("response", "{}"))
        return {
            "predicted_label": _normalize_label(parsed.get("predicted_label", "")),
            "explanation": parsed.get("explanation", "No explanation returned by the model."),
            "confidence": float(parsed.get("confidence", 0.5)),
            "model_used": ollama_model,
        }
    except Exception:
        return _heuristic_fallback(claim, evidence)
