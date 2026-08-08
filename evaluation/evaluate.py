from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from backend.pipeline import ClaimVerificationPipeline
from utils.load_data import load_claim_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 fake news prototype")
    parser.add_argument("--dataset-file", type=str, default=None, help="Specific dataset CSV inside dataset/")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()

    df = load_claim_dataset(PROJECT_ROOT / "dataset", preferred_file=args.dataset_file)
    df = df.head(args.limit)
    pipeline = ClaimVerificationPipeline()

    predictions = []
    y_true = []
    y_pred = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating"):
        result = pipeline.verify_claim(row.statement, top_k=3)
        predictions.append(
            {
                "statement": row.statement,
                "true_label": row.label,
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "explanation": result["explanation"],
                "model_used": result["model_used"],
            }
        )
        y_true.append(row.label)
        y_pred.append(result["predicted_label"])

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "phase1_eval_predictions.json"
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"Evaluated samples: {len(predictions)}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Classification report:")
    print(report)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
