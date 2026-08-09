import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def normalize_image_path(image_path):
    return str(image_path or "").replace("\\", "/").lstrip("/").strip()


def derive_mmfakebench_label(item):
    image_path = normalize_image_path(item.get("image_path", "")).lower()
    if image_path.startswith("fake/"):
        return 1
    if image_path.startswith("real/"):
        return 0

    fake_cls = str(item.get("fake_cls", "") or "").lower()
    if fake_cls in {"original", "real", "authentic"}:
        return 0
    if fake_cls:
        return 1

    gt_answer = str(item.get("gt_answers", "") or "").lower()
    if gt_answer in {"true", "real", "original", "0"}:
        return 0
    if gt_answer in {"false", "fake", "1"}:
        return 1

    return 0


def load_mmfakebench_records(annotation_file):
    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        print(f"Warning: Annotation file '{annotation_file}' not found. Using an empty dataset structure.")
        return []

    with annotation_path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            data = [json.loads(line) for line in handle if line.strip()]

    for item in data:
        item["image_path"] = normalize_image_path(item.get("image_path", ""))
        item["text"] = str(item.get("text", "") or "").strip()

    return data


def resolve_image_path(image_path, image_dir=None, annotation_file=None):
    normalized = normalize_image_path(image_path)
    if not normalized:
        return None

    raw_path = Path(image_path)
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path.resolve())

    repo_root = Path(__file__).resolve().parents[1]
    annotation_parent = Path(annotation_file).resolve().parent if annotation_file else repo_root / "dataset"
    roots = []
    if image_dir:
        roots.append(Path(image_dir))
    roots.extend(
        [
            annotation_parent / "images",
            annotation_parent / "MMFakeBench_val",
            annotation_parent / "MMFakeBench_test",
            annotation_parent,
            repo_root / "dataset" / "images",
            repo_root / "dataset" / "MMFakeBench_val",
            repo_root / "dataset" / "MMFakeBench_test",
            repo_root,
        ]
    )

    seen = set()
    for root in roots:
        root = Path(root)
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        candidate = root / normalized
        if candidate.exists():
            return str(candidate.resolve())

    return None


def load_image_from_path(image_path):
    if not image_path:
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Error loading image {image_path}: {exc}")
        return None


class MMFakeBenchDataset(Dataset):
    """
    MMFakeBench dataset loader.
    Expects entries with:
    - text
    - image_path
    - gt_answers
    - fake_cls
    """

    def __init__(self, annotation_file, image_dir, transform=None, split_mode="all", split_ratio=0.8, seed=42):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.seed = seed
        self.data = self._load_data()

    def _load_data(self):
        data = load_mmfakebench_records(self.annotation_file)

        if self.split_mode not in {"all", "train", "val"}:
            raise ValueError("split_mode must be one of: all, train, val")

        if self.split_mode == "all":
            return data

        random.seed(self.seed)
        real_items = [item for item in data if derive_mmfakebench_label(item) == 0]
        fake_items = [item for item in data if derive_mmfakebench_label(item) == 1]
        random.shuffle(real_items)
        random.shuffle(fake_items)

        real_split_idx = int(len(real_items) * self.split_ratio)
        fake_split_idx = int(len(fake_items) * self.split_ratio)

        if self.split_mode == "train":
            print(
                f"Fallback assumption: using {self.split_ratio * 100:.0f}% of "
                f"{Path(self.annotation_file).name} for training."
            )
            data = real_items[:real_split_idx] + fake_items[:fake_split_idx]
        else:
            print(
                f"Fallback assumption: using {(1 - self.split_ratio) * 100:.0f}% of "
                f"{Path(self.annotation_file).name} for validation."
            )
            data = real_items[real_split_idx:] + fake_items[fake_split_idx:]

        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = derive_mmfakebench_label(item)
        resolved_image_path = resolve_image_path(
            item.get("image_path", ""),
            image_dir=self.image_dir,
            annotation_file=self.annotation_file,
        )
        image = load_image_from_path(resolved_image_path)
        image_status = "loaded" if image is not None else "missing"

        if image is not None and self.transform:
            image = self.transform(image)

        return {
            "text": item.get("text", ""),
            "image": image,
            "label": label,
            "image_path": item.get("image_path", ""),
            "resolved_image_path": resolved_image_path,
            "image_available": image is not None,
            "image_status": image_status,
            "gt_answers": item.get("gt_answers"),
            "fake_cls": item.get("fake_cls"),
            "text_source": item.get("text_source"),
            "image_source": item.get("image_source"),
        }
