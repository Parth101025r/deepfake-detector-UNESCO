import argparse
import sys
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dataset.mmfakebench import MMFakeBenchDataset, derive_mmfakebench_label
    from models.multimodal_classifier import MultimodalFakeNewsClassifier
    from sklearn.metrics import f1_score
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def collate_fn(batch):
    return {
        "text": [b["text"] for b in batch],
        "image": [b["image"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch]) if "label" in batch[0] else None
    }

def eval_epoch(model, dataloader, processor, device):
    model.eval()
    all_preds_eval = []
    all_labels_eval = []
    from PIL import Image
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch["text"]
            labels = batch["label"].to(device)
            images = batch["image"]
            
            valid_images = [img if img is not None else Image.new('RGB', (224, 224), color='black') for img in images]

            inputs = processor(
                text=texts, images=valid_images, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)

            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values)
            preds = torch.argmax(logits, dim=1)
            
            all_preds_eval.extend(preds.cpu().numpy())
            all_labels_eval.extend(labels.cpu().numpy())
            
    if len(all_labels_eval) == 0:
        return 0.0
    return f1_score(all_labels_eval, all_preds_eval, average='macro')

def train(args):
    print("Initializing MultiModal Fake News Training Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = MultimodalFakeNewsClassifier(clip_model_name=args.clip_model, num_classes=2).to(device)
    
    print(f"Loading datasets using fallback val-split strategy from: {args.annotation_file}")
    train_dataset = MMFakeBenchDataset(
        args.annotation_file,
        args.image_dir,
        split_mode="train",
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    val_dataset = MMFakeBenchDataset(
        args.annotation_file,
        args.image_dir,
        split_mode="val",
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    train_label_counts = Counter(derive_mmfakebench_label(item) for item in train_dataset.data)
    val_label_counts = Counter(derive_mmfakebench_label(item) for item in val_dataset.data)
    print(f"Train label counts: real={train_label_counts.get(0, 0)}, fake={train_label_counts.get(1, 0)}")
    print(f"Val label counts: real={val_label_counts.get(0, 0)}, fake={val_label_counts.get(1, 0)}")

    total_train = max(1, sum(train_label_counts.values()))
    class_weights = torch.tensor(
        [
            total_train / max(1, 2 * train_label_counts.get(0, 0)),
            total_train / max(1, 2 * train_label_counts.get(1, 0)),
        ],
        dtype=torch.float32,
        device=device,
    )
    print(f"Using class weights: real={class_weights[0].item():.4f}, fake={class_weights[1].item():.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    best_f1 = 0.0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)
    from PIL import Image
    
    print(f"Starting actual training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            texts = batch["text"]
            labels = batch["label"].to(device)
            images = batch["image"]
            
            valid_images = [img if img is not None else Image.new('RGB', (224, 224), color='black') for img in images]

            inputs = processor(
                text=texts, images=valid_images, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / max(1, batches)
        print(f"--- Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f} ---")
        
        # Validation evaluation
        val_f1 = eval_epoch(model, val_loader, processor, device)
        print(f"Validation F1-Score (Macro): {val_f1:.4f}")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_macro_f1": val_f1,
            }
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            out_path = os.path.join(args.output_dir, "model_best.pt")
            torch.save(model.state_dict(), out_path)
            print(f"New Best Model Saved to {out_path}!")

    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(
            {
                "annotation_file": args.annotation_file,
                "split_ratio": args.split_ratio,
                "seed": args.seed,
                "best_val_macro_f1": best_f1,
                "history": history,
            },
            handle,
            indent=2,
        )
    print(f"Saved training metrics to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Assuming MMFakeBench_val.json is mapped via argument
    parser.add_argument("--annotation_file", type=str, default="dataset/MMFakeBench_val.json", help="Use val.json as fallback for train/val split")
    parser.add_argument("--image_dir", type=str, default="dataset/images/")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)
