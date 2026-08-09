from tavily import TavilyClient
import os
import re
import numpy as np

API_KEY = os.getenv("TAVILY_API_KEY", "")

client = None
if API_KEY:
    try:
        client = TavilyClient(api_key=API_KEY)
    except Exception as err:
        print(f"TavilyClient init warning: {err}")

TRUSTED_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "cnn.com",
    "abcnews.go.com",
    "abcnews4.com",
    "aljazeera.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "npr.org",
    "cbsnews.com",
    "nbcnews.com",
    "foxnews.com",
]

BLOCKED_DOMAINS = [
    "facebook.com",
    "youtube.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
]

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "from", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "discover", "discovers",
    "discovered", "evidence", "claim", "according", "report", "reports",
    "reported", "new", "news", "break", "breaking", "invented", "by", "that",
    "this", "these", "those"
}

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"SentenceTransformer load error: {e}")
            _embed_model = False
    return _embed_model if _embed_model is not False else None


def generate_search_query(claim: str) -> str:
    """
    Generates a concise keyword-based search query from a long text claim.
    Example: 'Scientists discover evidence of past liquid water on Mars' -> 'Mars past liquid water'
    """
    clean_text = re.sub(r'[^\w\s]', ' ', claim)
    words = clean_text.split()
    keywords = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
    
    if len(keywords) >= 2:
        return " ".join(keywords[:6])
    return claim.strip()


def is_trusted(url):
    url = url.lower()
    for blocked in BLOCKED_DOMAINS:
        if blocked in url:
            return False

    for trusted in TRUSTED_DOMAINS:
        if trusted in url:
            return True

    return False


def search_news(query, max_results=6):
    if not client:
        print("Tavily client not initialized.")
        return []
    try:
        response = client.search(
            query=query,
            topic="news",
            max_results=max_results * 2
        )

        results = response.get("results", [])
        filtered = []
        for item in results:
            url = item.get("url", "")
            if not is_trusted(url):
                continue
            filtered.append({
                "title": item.get("title", ""),
                "url": url,
                "content": item.get("content", ""),
                "api_score": item.get("score", 0)
            })

        return filtered[:max_results]

    except Exception as e:
        print(f"Search Error: {e}")
        return []


def compute_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


_stance_model = None

def get_stance_model():
    global _stance_model
    if _stance_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _stance_model = CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')
        except Exception as e:
            print(f"CrossEncoder stance model load warning: {e}")
            _stance_model = False
    return _stance_model if _stance_model is not False else None


def heuristic_stance_analysis(claim: str, title: str, snippet: str, sim_score: float) -> dict:
    """
    Fallback heuristic stance analysis when NLI neural model is unavailable or during network offline.
    """
    combined_text = f"{title} {snippet}".lower()
    claim_lower = claim.lower()

    if sim_score < 0.22:
        return {
            "evidence_label": "UNRELATED",
            "stance_score": 0.85,
            "reason": "Low semantic similarity score."
        }

    # Contradiction & Debunking signals
    contradiction_cues = [
        "fake", "false", "hoax", "debunk", "debunked", "debunks", "refutes",
        "refuted", "denies", "denied", "incorrect", "untrue", "no evidence", "myth",
        "baseless", "fabricated", "misleading", "fact check", "fact-check",
        "died" if "alive" in claim_lower else None,
        "alive" if ("dead" in claim_lower or "died" in claim_lower) else None
    ]
    contradiction_cues = [c for c in contradiction_cues if c is not None]

    # Affirmation & Supporting signals
    support_cues = [
        "confirms", "confirmed", "proves", "proven", "discovers", "discovered",
        "found", "official", "verifies", "verified", "announces", "announced",
        "shows", "showed", "reports", "reported"
    ]

    has_contradiction_cue = any(cue in combined_text for cue in contradiction_cues)
    has_support_cue = any(cue in combined_text for cue in support_cues)

    if has_contradiction_cue:
        return {
            "evidence_label": "CONTRADICTS",
            "stance_score": round(max(0.75, sim_score), 4),
            "reason": "Debunking / contradiction keywords detected in article."
        }

    if has_support_cue:
        return {
            "evidence_label": "SUPPORTS",
            "stance_score": round(max(0.70, sim_score), 4),
            "reason": "Affirmation / support keywords detected in article."
        }

    return {
        "evidence_label": "UNRELATED",
        "stance_score": round(0.60, 4),
        "reason": "Insufficient stance alignment."
    }


def extract_sentences(text: str) -> list:
    if not text:
        return []
    # Clean up whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Simple regex for sentence splitting: split by punctuation (. ! ?) followed by space
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) > 10 and len(s_clean.split()) >= 3:
            sentences.append(s_clean)
    return sentences


def get_most_relevant_sentence(claim: str, title: str, content: str, snippet: str) -> str:
    embedder = get_embed_model()
    
    text_to_split = content if content else snippet
    sentences = []
    if text_to_split:
        sentences = extract_sentences(text_to_split)
    
    candidates = []
    if title:
        candidates.append(title.strip())
    for s in sentences:
        if s not in candidates:
            candidates.append(s)
            
    if not candidates:
        return claim
        
    if embedder is None or len(candidates) == 1:
        return candidates[0]
        
    try:
        claim_emb = embedder.encode(claim, convert_to_numpy=True)
        cand_embs = embedder.encode(candidates, convert_to_numpy=True)
        
        best_sim = -1.0
        best_cand = candidates[0]
        
        for cand, emb in zip(candidates, cand_embs):
            sim = compute_cosine_similarity(claim_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_cand = cand
        return best_cand
    except Exception as e:
        print(f"Error extracting relevant sentence: {e}")
        return candidates[0]


def verify_article_evidence(claim: str, title: str, snippet: str, full_content: str = "", sim_score: float = 1.0) -> dict:
    """
    Stage 2: Evidence Verification Stage
    Determines whether a retrieved article SUPPORTS, CONTRADICTS, or is UNRELATED to the claim.
    """
    # Extract the most relevant evidence sentence
    evidence_sentence = get_most_relevant_sentence(claim, title, full_content, snippet)
    
    stance_model = get_stance_model()

    if stance_model is not None:
        try:
            # Run NLI on (evidence_sentence, claim)
            scores = stance_model.predict([(evidence_sentence, claim)], apply_softmax=True)[0]
            prob_contradict = float(scores[0])
            prob_support = float(scores[1])
            prob_unrelated = float(scores[2])

            max_idx = int(scores.argmax())
            if max_idx == 0 and prob_contradict >= 0.70:
                return {
                    "evidence_label": "CONTRADICTS",
                    "stance_score": round(prob_contradict, 4),
                    "reason": "NLI CrossEncoder detected contradiction.",
                    "evidence_sentence": evidence_sentence
                }
            elif max_idx == 1 and prob_support >= 0.70:
                return {
                    "evidence_label": "SUPPORTS",
                    "stance_score": round(prob_support, 4),
                    "reason": "NLI CrossEncoder detected entailment.",
                    "evidence_sentence": evidence_sentence
                }
            else:
                confidence = float(scores[max_idx])
                return {
                    "evidence_label": "UNRELATED",
                    "stance_score": round(confidence, 4),
                    "reason": "NLI CrossEncoder labeled article as neutral/unrelated or confidence was low.",
                    "evidence_sentence": evidence_sentence
                }
        except Exception as err:
            print(f"NLI model inference exception: {err}")

    # Fallback to heuristic stance analysis (uses similarity) if stance model is None
    fallback = heuristic_stance_analysis(claim, title, snippet, sim_score)
    fallback["evidence_sentence"] = evidence_sentence
    return fallback


def search_and_rank_news(claim: str, top_k: int = 3):
    """
    Two-Stage Pipeline:
    Stage 1: Retrieval & Ranking via SentenceTransformers cosine similarity.
    Stage 2: Evidence Verification (SUPPORTS, CONTRADICTS, UNRELATED) per article.
    """
    keyword_query = generate_search_query(claim)
    print(f"Generated Keyword Query: '{keyword_query}' (Original Claim: '{claim[:40]}...')")
    
    raw_articles = search_news(query=keyword_query, max_results=top_k * 3)
    if not raw_articles and keyword_query != claim:
        raw_articles = search_news(query=claim, max_results=top_k * 3)

    if not raw_articles:
        return []

    embedder = get_embed_model()
    
    candidates = []
    if embedder is not None:
        claim_emb = embedder.encode(claim, convert_to_numpy=True)
        article_texts = [f"{art['title']}. {art['content'][:250]}" for art in raw_articles]
        article_embs = embedder.encode(article_texts, convert_to_numpy=True)

        for art, art_emb in zip(raw_articles, article_embs):
            sim_score = compute_cosine_similarity(claim_emb, art_emb)
            candidates.append({
                "title": art.get("title", "Untitled News Article"),
                "url": art.get("url", ""),
                "snippet": art.get("content", "")[:350],
                "full_content": art.get("content", ""),
                "score": round(float(sim_score), 4),
                "api_score": art.get("api_score", 0)
            })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
    else:
        for art in raw_articles:
            candidates.append({
                "title": art.get("title", "Untitled News Article"),
                "url": art.get("url", ""),
                "snippet": art.get("content", "")[:350],
                "full_content": art.get("content", ""),
                "score": round(float(art.get("api_score", 0.0)), 4),
                "api_score": art.get("api_score", 0)
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)

    ranked = []
    for idx, art in enumerate(candidates[:top_k], start=1):
        art["rank"] = idx
        # Stage 2: Evidence Verification Stage
        verification = verify_article_evidence(claim, art["title"], art["snippet"], art.get("full_content", ""), art["score"])
        art["evidence_label"] = verification["evidence_label"]
        art["stance_score"] = verification["stance_score"]
        art["stance_reason"] = verification["reason"]
        art["evidence_sentence"] = verification.get("evidence_sentence", "")
        ranked.append(art)

    return ranked


def build_evidence_summary(articles, generated_query=""):
    """
    Synthesizes Stage 2 verification labels into a structured evidence payload.
    """
    if not articles:
        return {
            "evidence_found": False,
            "count": 0,
            "supports_count": 0,
            "contradicts_count": 0,
            "unrelated_count": 0,
            "dominant_stance": "NONE",
            "articles": [],
            "generated_query": generated_query,
            "top_similarity": 0.0,
            "summary_text": "No live web evidence retrieved from trusted sources.",
            "top_headline": "None"
        }

    supports_count = sum(1 for a in articles if a.get("evidence_label") == "SUPPORTS")
    contradicts_count = sum(1 for a in articles if a.get("evidence_label") == "CONTRADICTS")
    unrelated_count = sum(1 for a in articles if a.get("evidence_label") == "UNRELATED")

    if contradicts_count > supports_count:
        dominant_stance = "CONTRADICTS"
    elif supports_count > contradicts_count:
        dominant_stance = "SUPPORTS"
    elif supports_count > 0 and contradicts_count > 0:
        dominant_stance = "MIXED"
    else:
        dominant_stance = "UNRELATED"

    headlines_summary = []
    for art in articles:
        label = art.get("evidence_label", "UNRELATED")
        headlines_summary.append(f"Rank {art['rank']}: [{label}] {art['title']} (Score: {art['score']:.4f})")

    top_sim = articles[0]["score"] if articles else 0.0

    return {
        "evidence_found": True,
        "count": len(articles),
        "supports_count": supports_count,
        "contradicts_count": contradicts_count,
        "unrelated_count": unrelated_count,
        "dominant_stance": dominant_stance,
        "articles": articles,
        "generated_query": generated_query,
        "top_similarity": top_sim,
        "summary_text": " | ".join(headlines_summary),
        "top_headline": articles[0]["title"] if articles else "None"
    }


def print_results(results):
    if not results:
        print("No trusted results found.")
        return
    print("=" * 80)
    for i, article in enumerate(results, 1):
        print(f"\nResult {i} [{article.get('evidence_label', 'UNRELATED')}]")
        print("-" * 80)
        print("Title :", article.get("title", ""))
        print("URL   :", article.get("url", ""))
        print("Similarity Score :", article.get("score", 0))
        print("Stance Score     :", article.get("stance_score", 0))
        print("Stance Reason    :", article.get("stance_reason", ""))
        print("\nContent:\n")
        print(article.get("snippet", article.get("content", ""))[:700])
        print()


if __name__ == "__main__":
    claim = input("Enter a claim: ")
    ranked_articles = search_and_rank_news(claim, top_k=3)
    evidence = build_evidence_summary(ranked_articles)
    print("Evidence Payload:", evidence)
    print_results(ranked_articles)


