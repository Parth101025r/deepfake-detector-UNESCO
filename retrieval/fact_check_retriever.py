from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


class GoogleFactCheckRetriever:
    """
    Thin wrapper around Google Fact Check Tools Claim Search.

    The retriever is optional by design: if GOOGLE_FACT_CHECK_API_KEY is not set,
    the rest of the project continues to use the local FAISS evidence store.
    """

    API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SECRET_KEY_FILE = PROJECT_ROOT / "secrets" / "google_fact_check_api_key.txt"

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        max_age_days: int | None = None,
        timeout: int | None = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY") or self._read_key_file()
        self.language_code = language_code or os.getenv("FACT_CHECK_LANGUAGE_CODE", "en")
        self.max_age_days = self._optional_int(
            max_age_days if max_age_days is not None else os.getenv("FACT_CHECK_MAX_AGE_DAYS")
        )
        self.timeout = int(timeout if timeout is not None else os.getenv("FACT_CHECK_TIMEOUT", "12"))
        self.ready = bool(self.api_key)
        self.last_error: str | None = None

    @classmethod
    def _read_key_file(cls) -> str | None:
        try:
            key = cls.SECRET_KEY_FILE.read_text(encoding="utf-8").strip()
            return key or None
        except OSError:
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            parsed = int(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first_review(claim: dict[str, Any]) -> dict[str, Any]:
        reviews = claim.get("claimReview") or []
        if isinstance(reviews, list) and reviews:
            return reviews[0] or {}
        return {}

    @staticmethod
    def _publisher_name(review: dict[str, Any]) -> str | None:
        publisher = review.get("publisher") or {}
        return publisher.get("name") or publisher.get("site")

    @staticmethod
    def _build_text(claim: dict[str, Any], review: dict[str, Any]) -> str:
        claim_text = str(claim.get("text") or "").strip()
        title = str(review.get("title") or "").strip()
        rating = str(review.get("textualRating") or "").strip()
        publisher = GoogleFactCheckRetriever._publisher_name(review)

        parts = []
        if claim_text:
            parts.append(f"Fact-checked claim: {claim_text}")
        if rating:
            parts.append(f"Rating: {rating}")
        if title:
            parts.append(f"Review: {title}")
        if publisher:
            parts.append(f"Publisher: {publisher}")
        return " | ".join(parts) or title or claim_text

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.ready or not str(query or "").strip():
            return []

        params: dict[str, Any] = {
            "key": self.api_key,
            "query": query,
            "pageSize": max(1, min(int(top_k), 10)),
        }
        if self.language_code:
            params["languageCode"] = self.language_code
        if self.max_age_days:
            params["maxAgeDays"] = self.max_age_days

        try:
            response = requests.get(self.API_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return []

        results = []
        for rank, claim in enumerate(payload.get("claims") or [], start=1):
            review = self._first_review(claim)
            text = self._build_text(claim, review)
            if not text:
                continue

            publisher = self._publisher_name(review)
            rating = review.get("textualRating")
            review_url = review.get("url")
            claim_date = claim.get("claimDate")
            review_date = review.get("reviewDate")
            score = max(0.5, 1.0 - ((rank - 1) * 0.08))

            results.append(
                {
                    "rank": rank,
                    "doc_id": f"google_fact_check::{rank}",
                    "text": text,
                    "score": score,
                    "label": None,
                    "gt_answers": None,
                    "fake_cls": rating,
                    "text_source": "Google Fact Check Tools",
                    "image_source": None,
                    "image_path": None,
                    "source_file": review_url,
                    "claimant": claim.get("claimant"),
                    "claim_text": claim.get("text"),
                    "claim_date": claim_date,
                    "publisher": publisher,
                    "review_date": review_date,
                    "review_url": review_url,
                    "textual_rating": rating,
                    "evidence_type": "live_fact_check",
                }
            )

        return results
