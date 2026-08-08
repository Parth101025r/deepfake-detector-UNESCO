import os
import sys
import io
import torch
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from multimodal import MultimodalFakeNewsClassifier
    from transformers import CLIPProcessor
    from web_search import search_and_rank_news, build_evidence_summary, generate_search_query
    from utils.decision_engine import FinalDecisionEngine
except ImportError as e:
    print(f"Error importing modules: {e}")

app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="Parallel Web Search & CLIP Multimodal Classifier fused by Final Decision Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Multimodal model on {device}...")

try:
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    checkpoint_path = "checkpoints/model_best.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model checkpoint loaded.")
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    decision_engine = FinalDecisionEngine()
except Exception as e:
    print(f"Warning: Model initialization failed. {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify")
async def verify(
    claim: str = Form(..., description="The text claim to verify"),
    top_k: int = Form(3, description="Web search top-k evidence to fetch"),
    image: UploadFile = File(None, description="Optional image payload")
):
    if len(claim.strip()) < 3:
        raise HTTPException(status_code=400, detail="Claim is too short.")

    try:
        # Branch A: Keyword Query Extraction -> Web Search -> SentenceTransformers Semantic Ranking
        gen_query = generate_search_query(claim.strip())
        ranked_articles = search_and_rank_news(claim.strip(), top_k=top_k)
        evidence_payload = build_evidence_summary(ranked_articles, generated_query=gen_query)

        # Branch B: CLIP Processor -> Multimodal Classifier
        img_obj = Image.new('RGB', (224, 224), color='black')
        if image and image.filename:
            content = await image.read()
            img_obj = Image.open(io.BytesIO(content)).convert('RGB')

        inputs = processor(
            text=[claim.strip()], 
            images=[img_obj], 
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

        return {
            "predicted_label": evaluation["final_label"],
            "confidence": evaluation["final_confidence"],
            "multimodal_prediction": evaluation["multimodal_prediction"],
            "evidence": evaluation["evidence_summary"]["articles"],
            "evidence_summary": evaluation["evidence_summary"],
            "explanation": evaluation["explanation"],
            "model_used": "CLIP Multimodal Classifier + SentenceTransformers Semantic Search"
        }
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


