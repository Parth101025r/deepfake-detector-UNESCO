import argparse
import os

import pandas as pd
from tqdm import tqdm

from backend.pipeline import MultimodalRAGPipeline
from dataset.mmfakebench import MMFakeBenchDataset


def _evidence_summary(evidence):
    return " || ".join(item["text"] for item in evidence[:2])


def infer_batch(args):
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
    results = []

    for idx in tqdm(range(limit), desc="Batch inference"):
        sample = dataset[idx]
        prediction = pipeline.predict(
            claim=sample["text"],
            image=sample["image"],
            image_path=sample["resolved_image_path"],
            top_k=args.top_k,
        )
        results.append(
            {
                "text": sample["text"],
                "image_path": sample["image_path"],
                "resolved_image_path": sample["resolved_image_path"],
                "true_label": sample["label"],
                "predicted_label": prediction["predicted_label"],
                "prediction_index": prediction["prediction_index"],
                "confidence": prediction["confidence"],
                "image_status": prediction["image_status"],
                "explanation": prediction["explanation"],
                "evidence_summary": _evidence_summary(prediction["evidence"]),
            }
        )

    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_file, index=False)
    print(f"Saved {len(results)} predictions to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch multimodal fake news inference")
    parser.add_argument("--annotation_file", type=str, default="dataset/MMFakeBench_test.json")
    parser.add_argument("--image_dir", type=str, default="dataset/images")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--index_path", type=str, default="retrieval/index.faiss")
    parser.add_argument("--metadata_path", type=str, default="retrieval/metadata.json")
    parser.add_argument("--output_file", type=str, default="outputs/batch_predictions.csv")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of samples to run; 0 means full file.")
    parser.add_argument("--split_mode", type=str, choices=["all", "train", "val"], default="all")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    infer_batch(parser.parse_args())
