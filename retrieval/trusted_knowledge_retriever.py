from __future__ import annotations

import re
from typing import Any

import requests


class WikidataKnowledgeRetriever:
    """
    Lightweight trusted-knowledge retriever for common public-entity claims.

    Google Fact Check is best for viral/disputed claims. Wikidata is useful for
    direct factual claims such as current public roles and known death dates.
    """

    SEARCH_URL = "https://www.wikidata.org/w/api.php"
    ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.ready = True
        self.last_error: str | None = None
        self.headers = {
            "User-Agent": "FakeNewsDetectionRAGPBL/1.0 (student project; local demo)"
        }

    @staticmethod
    def _clean_query(query: str) -> str:
        query_lower = str(query or "").lower()
        quoted_name = re.search(r'"([^"]{3,80})"', str(query or ""))
        if quoted_name:
            return quoted_name.group(1)

        death_match = re.search(
            r"^(.+?)\s+(died|dead|death|killed|murdered|passed away)\b",
            str(query or ""),
            flags=re.I,
        )
        if death_match:
            return death_match.group(1).strip()

        role_match = re.search(
            r"^(.+?)\s+(is|was)\s+(the\s+)?(prime minister|president|chief minister|ceo|founder)\b",
            str(query or ""),
            flags=re.I,
        )
        if role_match:
            return role_match.group(1).strip()

        cleaned = re.sub(
            r"\b(is|was|the|a|an|of|in|has|have|died|dead|death|killed|prime minister|president|chief minister|ceo)\b",
            " ",
            query,
            flags=re.I,
        )
        cleaned = " ".join(cleaned.split())
        return cleaned or query

    def _search_entity(self, query: str) -> dict[str, Any] | None:
        params = {
            "action": "wbsearchentities",
            "search": self._clean_query(query),
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        response = requests.get(self.SEARCH_URL, params=params, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        search_results = response.json().get("search") or []
        return search_results[0] if search_results else None

    def _load_entity(self, entity_id: str) -> dict[str, Any] | None:
        response = requests.get(self.ENTITY_URL.format(entity_id=entity_id), headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        entities = response.json().get("entities") or {}
        return entities.get(entity_id)

    def _label_entities(self, entity_ids: list[str]) -> dict[str, str]:
        if not entity_ids:
            return {}
        params = {
            "action": "wbgetentities",
            "ids": "|".join(sorted(set(entity_ids))),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }
        response = requests.get(self.SEARCH_URL, params=params, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        entities = response.json().get("entities") or {}
        labels = {}
        for entity_id, entity in entities.items():
            labels[entity_id] = ((entity.get("labels") or {}).get("en") or {}).get("value", entity_id)
        return labels

    @staticmethod
    def _extract_entity_ids(entity: dict[str, Any], property_id: str) -> list[tuple[str, bool]]:
        extracted = []
        for claim in (entity.get("claims") or {}).get(property_id, []):
            mainsnak = claim.get("mainsnak") or {}
            datavalue = mainsnak.get("datavalue") or {}
            value = datavalue.get("value") or {}
            entity_id = value.get("id")
            if not entity_id:
                continue
            has_end_time = "P582" in (claim.get("qualifiers") or {})
            extracted.append((entity_id, not has_end_time))
        return extracted

    @staticmethod
    def _extract_death_date(entity: dict[str, Any]) -> str | None:
        death_claims = (entity.get("claims") or {}).get("P570", [])
        if not death_claims:
            return None
        mainsnak = death_claims[0].get("mainsnak") or {}
        datavalue = mainsnak.get("datavalue") or {}
        value = datavalue.get("value") or {}
        return value.get("time")

    def retrieve(self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        if top_k <= 0 or not str(query or "").strip():
            return []

        try:
            search_result = self._search_entity(query)
            if not search_result:
                return []
            entity_id = search_result["id"]
            entity = self._load_entity(entity_id)
            if not entity:
                return []

            label = ((entity.get("labels") or {}).get("en") or {}).get("value", search_result.get("label", entity_id))
            description = ((entity.get("descriptions") or {}).get("en") or {}).get(
                "value",
                search_result.get("description", ""),
            )
            position_ids = self._extract_entity_ids(entity, "P39")
            position_labels = self._label_entities([entity_id for entity_id, _ in position_ids])
            all_positions = [position_labels.get(entity_id, entity_id) for entity_id, _ in position_ids]
            current_positions = [position_labels.get(entity_id, entity_id) for entity_id, is_current in position_ids if is_current]
            death_date = self._extract_death_date(entity)

            text_parts = [f"Trusted knowledge entity: {label}"]
            if description:
                text_parts.append(f"Description: {description}")
            if current_positions:
                text_parts.append(f"Current positions: {', '.join(current_positions[:5])}")
            if death_date:
                text_parts.append(f"Date of death: {death_date}")
            else:
                text_parts.append("Date of death: not listed")

            self.last_error = None
            return [
                {
                    "rank": 1,
                    "doc_id": f"wikidata::{entity_id}",
                    "text": " | ".join(text_parts),
                    "score": 0.88,
                    "label": None,
                    "gt_answers": None,
                    "fake_cls": None,
                    "text_source": "Wikidata trusted knowledge",
                    "image_source": None,
                    "image_path": None,
                    "source_file": f"https://www.wikidata.org/wiki/{entity_id}",
                    "review_url": f"https://www.wikidata.org/wiki/{entity_id}",
                    "entity_id": entity_id,
                    "entity_label": label,
                    "entity_description": description,
                    "current_positions": current_positions,
                    "all_positions": all_positions,
                    "death_date": death_date,
                    "evidence_type": "trusted_knowledge",
                }
            ]
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return []
