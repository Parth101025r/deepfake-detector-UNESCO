import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dataset.mmfakebench import MMFakeBenchDataset
    from multimodal import MultimodalFakeNewsClassifier
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def collate_fn(batch):
    return {
        "text": [b["text"] for b in batch],
        "image": [b["image"] for b in batch],
        "image_path": [b.get("image_path", "") for b in batch],
        "label": torch.tensor([b["label"] for b in batch]) if "label" in batch[0] else None
    }

def evaluate(args):
    print("Evaluating Multimodal Fake News Classifier on MMFakeBench...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model and Processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    
    if os.path.exists(args.checkpoint_path):
        print(f"Loading weights from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}. Running inference with untrained fusion head.")

    model.eval()
    from PIL import Image

    # Load Dataset
    print(f"Testing against annotations at: {args.test_annotation}")
    test_dataset = MMFakeBenchDataset(args.test_annotation, args.image_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_labels = []
    all_texts = []
    all_img_paths = []

    print("Starting inference loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            texts = batch["text"]
            labels = batch["label"].to(device)
            images = batch["image"]
            img_paths = batch["image_path"]
            
            valid_images = [img if img is not None else Image.new('RGB', (224, 224), color='black') for img in images]

            inputs = processor(
                text=texts, 
                images=valid_images, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=77 # CLIP max length
            ).to(device)

            logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values
            )
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
            all_img_paths.extend(img_paths)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    print("\n--- Model Evaluation Results ---")
    if len(all_labels) == 0:
        print("No valid samples evaluated. Check your test dataset and image paths.")
        return

    label_order = [0, 1]
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=label_order,
        average='macro',
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=label_order)
    
    print(f"Total evaluated samples: {len(all_labels)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (Macro)")
    print(f"Recall:    {recall:.4f} (Macro)")
    print(f"F1 Score:  {f1:.4f} (Macro)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=label_order,
            target_names=["Real", "Fake"],
            zero_division=0,
        )
    )
    
    # Save predictions
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_predictions.csv")
    df = pd.DataFrame({
        "Text": all_texts,
        "Image_Path": all_img_paths,
        "True_Label": all_labels,
        "Predicted_Label": all_preds
    })
    df.to_csv(out_file, index=False)
    print(f"\nSaved detailed predictions to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_annotation", type=str, default="dataset/MMFakeBench_test.json")
    parser.add_argument("--image_dir", type=str, default="dataset/images/")
    parser.add_argument("--checkpoint_path", type=str, required=False, default="checkpoints/model_best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    evaluate(args)
