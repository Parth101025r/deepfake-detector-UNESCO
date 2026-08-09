import argparse
import json
import os
import sys

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install faiss-cpu and sentence-transformers: pip install faiss-cpu sentence-transformers")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.mmfakebench import derive_mmfakebench_label, load_mmfakebench_records


def _build_metadata_records(corpus_paths):
    metadata = []
    seen_texts = set()

    for corpus_path in corpus_paths:
        print(f"Loading corpus from {corpus_path}")
        records = load_mmfakebench_records(corpus_path)
        for idx, item in enumerate(records):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            dedupe_key = text.lower()
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)
            metadata.append(
                {
                    "doc_id": f"{os.path.basename(corpus_path)}::{idx}",
                    "text": text,
                    "label": derive_mmfakebench_label(item),
                    "gt_answers": item.get("gt_answers"),
                    "fake_cls": item.get("fake_cls"),
                    "text_source": item.get("text_source"),
                    "image_source": item.get("image_source"),
                    "image_path": item.get("image_path"),
                    "source_file": corpus_path,
                }
            )

    return metadata


def build_faiss_index(corpus_paths, index_path, metadata_path):
    metadata = _build_metadata_records(corpus_paths)
    print(f"Prepared {len(metadata)} unique evidence snippets for the store.")

    if not metadata:
        raise ValueError("No text records were found to index.")

    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding evidence corpus...")
    texts = [item["text"] for item in metadata]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"FAISS index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a local FAISS evidence index for MMFakeBench.")
    parser.add_argument(
        "--corpus_json",
        nargs="+",
        default=["dataset/MMFakeBench_val.json"],
        help="One or more MMFakeBench JSON files to use as the evidence corpus.",
    )
    parser.add_argument("--index_out", type=str, default="retrieval/index.faiss")
    parser.add_argument("--metadata_out", type=str, default="retrieval/metadata.json")

    args = parser.parse_args()
    build_faiss_index(args.corpus_json, args.index_out, args.metadata_out)
