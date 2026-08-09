import argparse
import json

from backend.pipeline import MultimodalRAGPipeline


def infer_single(args):
    pipeline = MultimodalRAGPipeline(
        checkpoint_path=args.checkpoint_path,
        image_dir=args.image_dir,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
    )
    result = pipeline.predict(
        claim=args.text,
        image_path=args.image_path or None,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample multimodal fake news inference")
    parser.add_argument("--text", type=str, required=True, help="Text claim to verify")
    parser.add_argument("--image_path", type=str, default="", help="Optional path to the associated image")
    parser.add_argument("--image_dir", type=str, default="dataset/images")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--index_path", type=str, default="retrieval/index.faiss")
    parser.add_argument("--metadata_path", type=str, default="retrieval/metadata.json")
    parser.add_argument("--top_k", type=int, default=3)

    infer_single(parser.parse_args())
