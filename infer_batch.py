import argparse
import sys
import os
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor
    from multimodal import MultimodalFakeNewsClassifier
    from dataset.mmfakebench import MMFakeBenchDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def collate_fn(batch):
    return {
        "text": [b["text"] for b in batch],
        "image": [b["image"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch]) if "label" in batch[0] else None
    }

def infer_batch(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Model checkpoint loaded.")
    else:
        print("Warning: Checkpoint not found. Using untrained multimodal fusion head.")
        
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # MMFakeBench Dataset fallback val split / test setup
    dataset = MMFakeBenchDataset(args.annotation_file, args.image_dir, split_mode=args.split)
    from PIL import Image
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = []
    print(f"Running batch inference on split '{args.split}'...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            texts = batch["text"]
            images = batch["image"]
            
            # Replace missing images with a dummy black image to prevent CLIP crash
            valid_images = [img if img is not None else Image.new('RGB', (224, 224), color='black') for img in images]
            
            inputs = processor(
                text=texts, images=valid_images, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)
            
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for t, p in zip(texts, preds):
                results.append({
                    "text": t, 
                    "predicted_label": "Fake" if p == 1 else "Real"
                })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1} batches...")
                
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Batch inference complete. Saved {len(results)} predictions to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Multimodal Inference")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--annotation_file", type=str, default="dataset/MMFakeBench_test.json")
    parser.add_argument("--image_dir", type=str, default="dataset/images/")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--output_file", type=str, default="outputs/batch_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    infer_batch(args)