import argparse
import sys
import os
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import CLIPProcessor
    from multimodal import MultimodalFakeNewsClassifier
    from web_search import search_and_rank_news, build_evidence_summary, generate_search_query
    from utils.decision_engine import FinalDecisionEngine
except ImportError as e:
    print(f"Error importing modules: {e}. Please ensure requirements are installed.")
    sys.exit(1)

def infer_claim(text: str, image_path: str = "", top_k: int = 3, checkpoint_path: str = "checkpoints/model_best.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    decision_engine = FinalDecisionEngine()
    
    # Branch A: Keyword Search -> SentenceTransformers Semantic Ranking -> Evidence Builder
    gen_query = generate_search_query(text)
    print(f"\n[Branch A] Extracted Keyword Query: '{gen_query}' (Original: '{text}')")
    ranked_articles = search_and_rank_news(text, top_k=top_k)
    evidence_payload = build_evidence_summary(ranked_articles, generated_query=gen_query)

    # Branch B: CLIP Processor -> Multimodal Classifier
    print("[Branch B] Processing Image + Claim via CLIP Processor & Multimodal Classifier")
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
    else:
        image = Image.new('RGB', (224, 224), color='black')
        
    inputs = processor(
        text=[text], 
        images=[image], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=77
    ).to(device)
    
    with torch.no_grad():
        logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    pred = prob.argmax()
    mm_label = "Fake" if pred == 1 else "Real"
    multimodal_result = {
        "predicted_label": mm_label,
        "confidence": float(prob[pred])
    }

    # Final Decision Engine Fusion
    evaluation = decision_engine.evaluate(multimodal_result, evidence_payload)
    
    print(f"\n==========================================")
    print(f"       FINAL DECISION ENGINE VERDICT       ")
    print(f"==========================================")
    print(f"Claim             : {text}")
    print(f"Generated Query   : {gen_query}")
    print(f"Final Label       : {evaluation['final_label']}")
    print(f"Combined Conf     : {evaluation['final_confidence']:.4f}")
    print(f"Multimodal Branch : {mm_label} (Conf: {prob[pred]:.4f})")
    print(f"Evidence Stances  : Supports={evidence_payload.get('supports_count', 0)}, Contradicts={evidence_payload.get('contradicts_count', 0)}, Unrelated={evidence_payload.get('unrelated_count', 0)}")
    print(f"Explanation       : {evaluation['explanation']}")
    
    if ranked_articles:
        print(f"\n--- Two-Stage Evidence Verification Articles ({len(ranked_articles)}) ---")
        for art in ranked_articles:
            label = art.get('evidence_label', 'UNRELATED')
            print(f"[{art['rank']}] [{label}] Title : {art['title']}")
            print(f"    URL          : {art['url']}")
            print(f"    NLI Label    : {label}")
            print(f"    Confidence   : {art.get('stance_score', 0):.4f}")
            print(f"    Topical Sim  : {art['score']:.4f}")
            print(f"    Evidence Sent: {art.get('evidence_sentence', 'N/A')}")
            print(f"    Snippet      : {art['snippet'][:150]}...")
    else:
        print("\nNo trusted live web search evidence found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Parallel Multimodal Inference")
    parser.add_argument("--text", type=str, required=False, default="", help="Text claim to verify")
    parser.add_argument("--image_path", type=str, required=False, default="", help="Path to associated image")
    parser.add_argument("--top_k", type=int, default=3, help="Top K evidence articles")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    
    args = parser.parse_args()

    if args.text.strip():
        infer_claim(args.text, args.image_path, args.top_k, args.checkpoint_path)
    else:
        print("==================================================")
        print(" Interactive Parallel Multimodal Fake News Detector ")
        print("==================================================")
        while True:
            try:
                user_claim = input("\nEnter a claim to verify (or type 'exit' / 'q' to quit): ").strip()
                if not user_claim or user_claim.lower() in ["exit", "q", "quit"]:
                    print("Exiting interactive session.")
                    break
                user_image = input("Enter path to image (optional, press Enter to skip): ").strip()
                infer_claim(user_claim, user_image, args.top_k, args.checkpoint_path)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

