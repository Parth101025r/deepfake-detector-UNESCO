import argparse
import sys
import os
import torch
import json
from transformers import CLIPProcessor
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from multimodal import MultimodalFakeNewsClassifier
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_inference(args):
    print("Initializing Offline Multimodal Inference Flow...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Model checkpoint loaded.")
    else:
        print("Warning: Running with untrained baseline model (scaffold phase).")
        
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, text in enumerate(args.texts):
            image_path = args.image_paths[i] if i < len(args.image_paths) else None
            
            # Load image
            image = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                
            inputs = processor(
                text=[text], 
                images=[image] if image else None, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # If no image was provided, clip processor might fail or we need dummy inputs.
            # Assuming processor handles missing images based on how we passed them.
            if image is None:
                print(f"Skipping prediction for '{text[:20]}...' (No valid image provided).")
                continue
                
            logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values
            )
            
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(prob)
            label = "Fake" if pred == 1 else "Real"
            
            res = {
                "text": text,
                "image": image_path,
                "predicted_label": label,
                "confidence": float(prob[pred])
            }
            results.append(res)
            print(f"Claim: {text[:50]}... => {label} ({prob[pred]:.2f})")
            
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved {len(results)} predictions to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multimodal Fake News Inference")
    parser.add_argument("--texts", nargs='+', required=True, help="List of claims to verify")
    parser.add_argument("--image_paths", nargs='+', required=True, help="List of image paths corresponding to texts")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--output", type=str, default="outputs/predictions.json")
    
    args = parser.parse_args()
    import numpy as np # deferred import
    run_inference(args)
