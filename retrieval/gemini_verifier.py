from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests


class GeminiGroundedVerifier:
    """
    Gemini verifier with Google Search grounding.

    This is the highest-priority verifier because it can search the web and
    reason over current/general evidence in one call.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SECRET_KEY_FILE = PROJECT_ROOT / "secrets" / "gemini_api_key.txt"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or self._read_key_file()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = int(timeout if timeout is not None else os.getenv("GEMINI_TIMEOUT", "35"))
        self.ready = bool(self.api_key)
        self.last_error: str | None = None

    @classmethod
    def _read_key_file(cls) -> str | None:
        try:
            key = cls.SECRET_KEY_FILE.read_text(encoding="utf-8").strip()
            return key or None
        except OSError:
            return None

    @property
    def endpoint(self) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_label(label: Any) -> str:
        normalized = str(label or "").strip().lower()
        if "fake" in normalized or "false" in normalized or "refuted" in normalized:
            return "Fake"
        if "real" in normalized or "true" in normalized or "supported" in normalized:
            return "Real"
        return "Unverified"

    @staticmethod
    def _normalize_confidence(value: Any, fallback: float = 0.65) -> float:
        try:
            confidence = float(value)
            if confidence > 1:
                confidence /= 100.0
            return max(0.0, min(confidence, 1.0))
        except Exception:
            return fallback

    @staticmethod
    def _grounding_chunks(payload: dict[str, Any]) -> list[dict[str, str]]:
        candidates = payload.get("candidates") or []
        if not candidates:
            return []
        metadata = candidates[0].get("groundingMetadata") or {}
        chunks = []
        for chunk in metadata.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            uri = web.get("uri")
            title = web.get("title")
            if uri or title:
                chunks.append({"uri": uri, "title": title})
        return chunks

    def verify(self, claim: str) -> dict[str, Any] | None:
        if not self.ready or not str(claim or "").strip():
            return None

        prompt = f"""
You are a strict claim verifier.

Use Google Search grounding when useful. Return only valid JSON with this exact schema:
{{
  "label": "Real or Fake or Unverified",
  "confidence": 0.0,
  "explanation": "one short, plain, slightly general sentence"
}}

Rules:
- The label must be correct even if the explanation is brief.
- Use Real only when reliable sources support the claim.
- Use Fake only when reliable sources refute the claim or prove a contradiction.
- Use Unverified when sources are weak, unclear, missing, or mixed.
- Keep the explanation generic and model-like; do not include exact dates, source names, article names, or detailed citations.
- Prefer wording such as "The retrieved context mostly supports this claim" or "The retrieved context does not support this claim".
- Avoid sounding overly certain in the explanation.
- Do not include markdown.

Claim: {claim}
""".strip()

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 256,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            raw_payload = response.json()
            text = ""
            candidates = raw_payload.get("candidates") or []
            if candidates:
                parts = ((candidates[0].get("content") or {}).get("parts") or [])
                text = " ".join(str(part.get("text") or "") for part in parts)

            parsed = self._extract_json(text)
            if not parsed:
                raise ValueError("Gemini did not return parseable JSON.")

            chunks = self._grounding_chunks(raw_payload)
            self.last_error = None
            return {
                "predicted_label": self._normalize_label(parsed.get("label")),
                "confidence": self._normalize_confidence(parsed.get("confidence")),
                "explanation": str(parsed.get("explanation") or "").strip() or "The claim was checked against retrieved context.",
                "sources": chunks,
                "raw_text": text,
            }
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return None
