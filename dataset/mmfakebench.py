import json
import os
from PIL import Image
from torch.utils.data import Dataset


def derive_mmfakebench_label(item):
    image_path = (item.get("image_path", "") or "").lower().lstrip("/\\")
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

class MMFakeBenchDataset(Dataset):
    """
    MMFakeBench dataset loader.
    Expects a JSON/JSONL file where each entry has:
    - text: str
    - image_path: str
    - gt_answers / fake_cls: labels indicating fake or real.
    """
    def __init__(self, annotation_file, image_dir, transform=None, split_mode="all", split_ratio=0.8, seed=42):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        self.split_mode = split_mode  # 'all', 'train', 'val'
        self.split_ratio = split_ratio
        self.seed = seed
        self.data = self._load_data()

    def _load_data(self):
        """
        Load structured data. Implements 80/20 fallback split logic.
        """
        import random
        data = []
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    data = [json.loads(line) for line in f]
            
            # Implementation of the fallback train/val split strategy. We keep
            # the split stratified so class imbalance does not get amplified by
            # a random slice.
            if self.split_mode != "all":
                random.seed(self.seed)
                real_items = [item for item in data if derive_mmfakebench_label(item) == 0]
                fake_items = [item for item in data if derive_mmfakebench_label(item) == 1]
                random.shuffle(real_items)
                random.shuffle(fake_items)

                real_split_idx = int(len(real_items) * self.split_ratio)
                fake_split_idx = int(len(fake_items) * self.split_ratio)

                if self.split_mode == "train":
                    print(f"Fallback assumption: using {self.split_ratio*100}% of {os.path.basename(self.annotation_file)} for training.")
                    data = real_items[:real_split_idx] + fake_items[:fake_split_idx]
                elif self.split_mode == "val":
                    print(f"Fallback assumption: using {(1-self.split_ratio)*100}% of {os.path.basename(self.annotation_file)} for validation.")
                    data = real_items[real_split_idx:] + fake_items[fake_split_idx:]

                random.shuffle(data)
        else:
            print(f"Warning: Annotation file '{self.annotation_file}' not found. Using an empty dataset structure.")
        
        # Ensure image_path doesn't start with leading slash to avoid path join issues
        for item in data:
            if 'image_path' in item and item['image_path'].startswith('/'):
                item['image_path'] = item['image_path'][1:]
                
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        img_filename = item.get("image_path", "")
        
        label = derive_mmfakebench_label(item)

        img_path = os.path.join(self.image_dir, img_filename)
        image = None
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        else:
            # Handle missing images gracefully
            pass
            
        return {
            "text": text,
            "image": image,
            "label": label,
            "image_path": img_path
        }
