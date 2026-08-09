import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline import MultimodalRAGPipeline
from dataset.mmfakebench import MMFakeBenchDataset


def evaluate(args):
    dataset = MMFakeBenchDataset(
        args.annotation_file,
        args.image_dir,
        split_mode=args.split_mode,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    pipeline = MultimodalRAGPipeline(
        checkpoint_path=args.checkpoint_path,
        image_dir=args.image_dir,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
    )

    limit = min(len(dataset), args.limit) if args.limit else len(dataset)
    rows = []
    y_true = []
    y_pred = []

    for idx in tqdm(range(limit), desc="Evaluation"):
        sample = dataset[idx]
        prediction = pipeline.predict(
            claim=sample["text"],
            image=sample["image"],
            image_path=sample["resolved_image_path"],
            top_k=args.top_k,
        )
        y_true.append(sample["label"])
        y_pred.append(prediction["prediction_index"])
        rows.append(
            {
                "text": sample["text"],
                "image_path": sample["image_path"],
                "resolved_image_path": sample["resolved_image_path"],
                "true_label": sample["label"],
                "predicted_label": prediction["prediction_index"],
                "predicted_label_name": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "image_status": prediction["image_status"],
                "explanation": prediction["explanation"],
                "evidence_summary": " || ".join(item["text"] for item in prediction["evidence"][:2]),
            }
        )

    if not y_true:
        raise ValueError("No samples were evaluated.")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], zero_division=0)

    os.makedirs("outputs", exist_ok=True)
    predictions_path = "outputs/eval_predictions.csv"
    metrics_path = "outputs/eval_metrics.json"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    metrics = {
        "samples_evaluated": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n--- Evaluation Results ---")
    print(f"Samples:    {len(y_true)}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"Macro-F1:   {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the unified multimodal + RAG pipeline")
    parser.add_argument("--annotation_file", type=str, default="dataset/MMFakeBench_test.json")
    parser.add_argument("--image_dir", type=str, default="dataset/images")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--index_path", type=str, default="retrieval/index.faiss")
    parser.add_argument("--metadata_path", type=str, default="retrieval/metadata.json")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of samples to run; 0 means full file.")
    parser.add_argument("--split_mode", type=str, choices=["all", "train", "val"], default="all")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    evaluate(parser.parse_args())
