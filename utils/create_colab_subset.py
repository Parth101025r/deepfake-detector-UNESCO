import argparse
import json
import os
import random
import shutil
from typing import Iterable, List, Tuple


def load_json_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
            if isinstance(data, list):
                return data
            raise ValueError(f"Expected a list of records in {path}")
        except json.JSONDecodeError:
            handle.seek(0)
            return [json.loads(line) for line in handle if line.strip()]


def normalize_image_path(record: dict) -> str:
    image_path = record.get("image_path", "") or ""
    return image_path.lstrip("/\\")


def derive_label(record: dict) -> int:
    image_path = (record.get("image_path", "") or "").lower().lstrip("/\\")
    if image_path.startswith("fake/"):
        return 1
    if image_path.startswith("real/"):
        return 0

    fake_cls = str(record.get("fake_cls", "") or "").lower()
    if fake_cls in {"original", "real", "authentic"}:
        return 0
    if fake_cls:
        return 1

    gt_answer = str(record.get("gt_answers", "") or "").lower()
    if gt_answer in {"true", "real", "original", "0"}:
        return 0
    if gt_answer in {"false", "fake", "1"}:
        return 1

    return 0


def filter_existing_records(records: Iterable[dict], images_root: str) -> Tuple[List[dict], int]:
    kept = []
    missing = 0
    for record in records:
        image_path = normalize_image_path(record)
        source_image = os.path.join(images_root, image_path)
        if image_path and os.path.exists(source_image):
            record = dict(record)
            record["image_path"] = image_path
            kept.append(record)
        else:
            missing += 1
    return kept, missing


def sample_balanced(records: List[dict], target_count: int, seed: int) -> List[dict]:
    if target_count <= 0 or not records:
        return []

    randomizer = random.Random(seed)
    fake_records = [record for record in records if derive_label(record) == 1]
    real_records = [record for record in records if derive_label(record) == 0]
    randomizer.shuffle(fake_records)
    randomizer.shuffle(real_records)

    half = target_count // 2
    fake_take = min(len(fake_records), half)
    real_take = min(len(real_records), half)

    selected = fake_records[:fake_take] + real_records[:real_take]
    remaining_slots = min(target_count, len(records)) - len(selected)

    leftovers = fake_records[fake_take:] + real_records[real_take:]
    randomizer.shuffle(leftovers)
    selected.extend(leftovers[:remaining_slots])
    randomizer.shuffle(selected)
    return selected


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_images(records: Iterable[dict], source_images_root: str, target_images_root: str) -> int:
    copied = 0
    seen = set()
    for record in records:
        rel_path = record["image_path"]
        if rel_path in seen:
            continue
        seen.add(rel_path)
        source = os.path.join(source_images_root, rel_path)
        target = os.path.join(target_images_root, rel_path)
        ensure_parent(target)
        shutil.copy2(source, target)
        copied += 1
    return copied


def save_json(records: List[dict], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def summarize(name: str, records: List[dict]) -> str:
    fake_count = sum(1 for record in records if derive_label(record) == 1)
    real_count = len(records) - fake_count
    return f"{name}: {len(records)} samples ({real_count} real, {fake_count} fake)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a smaller MMFakeBench subset for Colab uploads."
    )
    parser.add_argument("--source-dir", default="dataset", help="Folder containing JSON files and images/")
    parser.add_argument("--output-dir", default="mini_dataset", help="Where the smaller subset will be written")
    parser.add_argument("--val-count", type=int, default=200, help="Number of validation samples to keep")
    parser.add_argument("--test-count", type=int, default=100, help="Number of test samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    images_root = os.path.join(source_dir, "images")
    val_json = os.path.join(source_dir, "MMFakeBench_val.json")
    test_json = os.path.join(source_dir, "MMFakeBench_test.json")

    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images folder not found: {images_root}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation JSON not found: {val_json}")
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"Test JSON not found: {test_json}")

    val_records, missing_val = filter_existing_records(load_json_records(val_json), images_root)
    test_records, missing_test = filter_existing_records(load_json_records(test_json), images_root)

    val_subset = sample_balanced(val_records, args.val_count, args.seed)
    test_subset = sample_balanced(test_records, args.test_count, args.seed + 1)

    output_images_root = os.path.join(output_dir, "images")
    copied_val = copy_images(val_subset, images_root, output_images_root)
    copied_test = copy_images(test_subset, images_root, output_images_root)

    save_json(val_subset, os.path.join(output_dir, "MMFakeBench_val.json"))
    save_json(test_subset, os.path.join(output_dir, "MMFakeBench_test.json"))

    print("Subset created successfully.")
    print(f"Source dataset: {source_dir}")
    print(f"Output dataset: {output_dir}")
    print(summarize("Validation subset", val_subset))
    print(summarize("Test subset", test_subset))
    print(f"Copied image files: {copied_val + copied_test}")
    print(f"Skipped validation records with missing images: {missing_val}")
    print(f"Skipped test records with missing images: {missing_test}")
    print("")
    print("Upload this smaller folder to Colab or zip it first:")
    print(output_dir)


if __name__ == "__main__":
    main()
