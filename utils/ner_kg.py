from __future__ import annotations

import re
from typing import List, Tuple

import networkx as nx
import spacy


def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")


NLP = _load_spacy_model()


def extract_entities(text: str) -> List[str]:
    doc = NLP(text)
    entities = [ent.text.strip() for ent in getattr(doc, "ents", []) if ent.text.strip()]
    if entities:
        return list(dict.fromkeys(entities))

    fallback = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    return list(dict.fromkeys([item.strip() for item in fallback if item.strip()]))


def build_basic_kg(claim: str, evidence_texts: List[str]) -> Tuple[List[str], List[dict], dict]:
    claim_entities = extract_entities(claim)
    evidence_entities: List[str] = []
    for text in evidence_texts:
        evidence_entities.extend(extract_entities(text))

    all_entities = list(dict.fromkeys(claim_entities + evidence_entities))[:12]
    graph = nx.Graph()
    triples: List[dict] = []

    for entity in claim_entities:
        graph.add_edge("CLAIM", entity, relation="mentions")
        triples.append({"subject": "CLAIM", "relation": "mentions", "object": entity})

    for entity in all_entities:
        if entity not in claim_entities:
            graph.add_edge("EVIDENCE", entity, relation="mentions")
            triples.append({"subject": "EVIDENCE", "relation": "mentions", "object": entity})

    for left in claim_entities:
        for right in all_entities:
            if left != right:
                graph.add_edge(left, right, relation="related_to")
                triples.append({"subject": left, "relation": "related_to", "object": right})

    summary = {
        "nodes": list(graph.nodes())[:15],
        "edges": min(graph.number_of_edges(), 20),
    }
    return all_entities, triples[:20], summary
