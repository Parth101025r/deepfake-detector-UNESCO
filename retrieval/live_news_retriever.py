from __future__ import annotations

import os
from typing import Any

import requests


class GDELTNewsRetriever:
    """
    Live news retriever using the public GDELT DOC 2.0 API.

    This is used for general news claims where Google Fact Check has no matching
    reviewed claim. It returns article-level evidence, not final truth by itself.
    """

    API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, timeout: int | None = None, timespan: str | None = None):
        self.timeout = int(timeout if timeout is not None else os.getenv("GDELT_TIMEOUT", "20"))
        self.timespan = timespan or os.getenv("GDELT_TIMESPAN", "3months")
        self.ready = True
        self.last_error: str | None = None

    @staticmethod
    def _article_text(article: dict[str, Any]) -> str:
        title = str(article.get("title") or "").strip()
        domain = str(article.get("domain") or article.get("sourceCountry") or "").strip()
        language = str(article.get("language") or "").strip()
        seen_date = str(article.get("seendate") or "").strip()

        parts = []
        if title:
            parts.append(f"News article: {title}")
        if domain:
            parts.append(f"Source: {domain}")
        if language:
            parts.append(f"Language: {language}")
        if seen_date:
            parts.append(f"Seen date: {seen_date}")
        return " | ".join(parts)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0 or not str(query or "").strip():
            return []

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max(1, min(int(top_k), 20)),
            "timespan": self.timespan,
            "sort": "HybridRel",
        }

        try:
            response = requests.get(self.API_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return []

        articles = payload.get("articles") or []
        results = []
        for rank, article in enumerate(articles, start=1):
            text = self._article_text(article)
            if not text:
                continue

            url = article.get("url")
            domain = article.get("domain")
            score = max(0.45, 0.82 - ((rank - 1) * 0.06))
            results.append(
                {
                    "rank": rank,
                    "doc_id": f"gdelt::{rank}",
                    "text": text,
                    "score": score,
                    "label": None,
                    "gt_answers": None,
                    "fake_cls": None,
                    "text_source": "GDELT live news",
                    "image_source": article.get("socialimage"),
                    "image_path": None,
                    "source_file": url,
                    "publisher": domain,
                    "review_url": url,
                    "review_date": article.get("seendate"),
                    "evidence_type": "live_news",
                    "article_title": article.get("title"),
                    "article_domain": domain,
                    "article_language": article.get("language"),
                    "article_country": article.get("sourceCountry"),
                }
            )
        return results
