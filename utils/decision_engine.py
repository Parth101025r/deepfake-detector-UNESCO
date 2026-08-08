class FinalDecisionEngine:
    """
    Final Decision Engine for fusing the outputs of the parallel pipeline:
    Branch A: Stage 1 Retrieval + Stage 2 Evidence Verification (SUPPORTS, CONTRADICTS, UNRELATED)
    Branch B: CLIP Processor + Multimodal Fake News Classifier
    """
    def evaluate(self, multimodal_result: dict, evidence_payload: dict) -> dict:
        mm_label = multimodal_result.get("predicted_label", "Real")
        mm_conf = float(multimodal_result.get("confidence", 0.5))

        articles = evidence_payload.get("articles", [])
        evidence_found = evidence_payload.get("evidence_found", False)
        supports_count = evidence_payload.get("supports_count", 0)
        contradicts_count = evidence_payload.get("contradicts_count", 0)
        unrelated_count = evidence_payload.get("unrelated_count", 0)
        dominant_stance = evidence_payload.get("dominant_stance", "NONE")
        top_sim = float(evidence_payload.get("top_similarity", 0.0))

        # Decision Logic based on Stage 2 Evidence Verification + Multimodal CLIP Branch
        if evidence_found and contradicts_count > 0 and contradicts_count >= supports_count:
            final_label = "Fake"
            combined_confidence = round(min(0.98, max(0.80, 0.70 + 0.10 * contradicts_count)), 4)
            if mm_label == "Real":
                explanation = (
                    f"Verdict 'Fake': Multimodal CLIP model predicted 'Real' ({mm_conf:.2%}), "
                    f"but trusted live web evidence explicitly CONTRADICTS the claim "
                    f"({contradicts_count} contradicting article(s) found). Evidence overrules model."
                )
            else:
                explanation = (
                    f"Verdict 'Fake': Trusted live web evidence explicitly CONTRADICTS the claim "
                    f"({contradicts_count} contradicting article(s)), reinforcing the Multimodal prediction ({mm_conf:.2%})."
                )

        elif evidence_found and supports_count > 0 and supports_count > contradicts_count:
            final_label = "Real"
            combined_confidence = round(min(0.98, max(0.80, mm_conf + 0.05 * supports_count)), 4)
            if mm_label == "Fake":
                explanation = (
                    f"Verdict 'Real': Multimodal CLIP model predicted 'Fake' ({mm_conf:.2%}), "
                    f"but trusted live web evidence SUPPORTS the claim "
                    f"({supports_count} supporting article(s) found). Evidence overrules model."
                )
            else:
                explanation = (
                    f"Verdict 'Real': Trusted live web evidence SUPPORTS the claim "
                    f"({supports_count} supporting article(s)), reinforcing the Multimodal prediction ({mm_conf:.2%})."
                )

        elif evidence_found and dominant_stance == "MIXED":
            final_label = "Uncertain"
            combined_confidence = round(0.50, 4)
            explanation = (
                f"Verdict 'Uncertain': Live web evidence returned conflicting stance signals "
                f"({supports_count} supporting vs {contradicts_count} contradicting article(s)). "
                f"Manual fact-checking recommended."
            )

        else:
            if mm_conf >= 0.60:
                final_label = mm_label
                combined_confidence = round(mm_conf, 4)
                explanation = (
                    f"Verdict '{final_label}' based on Multimodal CLIP prediction (Confidence: {mm_conf:.2%}). "
                    f"No direct evidence stance found in live web search (retrieved articles were topically unrelated or unavailable)."
                )
            else:
                final_label = "Uncertain"
                combined_confidence = round(mm_conf, 4)
                explanation = (
                    f"Verdict 'Uncertain': Multimodal model confidence is low ({mm_conf:.2%}) "
                    f"and no decisive live web evidence stance was retrieved."
                )

        return {
            "final_label": final_label,
            "final_confidence": combined_confidence,
            "multimodal_prediction": {
                "label": mm_label,
                "confidence": round(float(mm_conf), 4)
            },
            "evidence_summary": {
                "count": len(articles),
                "evidence_found": evidence_found,
                "supports_count": supports_count,
                "contradicts_count": contradicts_count,
                "unrelated_count": unrelated_count,
                "dominant_stance": dominant_stance,
                "top_similarity": top_sim,
                "top_headline": evidence_payload.get("top_headline", "None"),
                "articles": articles
            },
            "explanation": explanation
        }


