import os
import json
import shutil

base_dir = r"d:\Enginner Yatharth\Fake News Detection using Rag PBL\dataset"
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

for split in ["test", "val"]:
    json_name = f"MMFakeBench_{split}.json"
    json_path = os.path.join(base_dir, json_name)
    split_dir = os.path.join(base_dir, f"MMFakeBench_{split}")
    
    # 1. Move images
    if os.path.exists(split_dir):
        for sub in ["fake", "real", "source"]:
            sub_path = os.path.join(split_dir, sub)
            if os.path.exists(sub_path):
                dest_sub_path = os.path.join(images_dir, sub)
                os.makedirs(dest_sub_path, exist_ok=True)
                for root, _, files in os.walk(sub_path):
                    for file in files:
                        src_file = os.path.join(root, file)
                        # Construct relative structure
                        rel_path = os.path.relpath(root, split_dir)
                        dest_folder = os.path.join(images_dir, rel_path)
                        os.makedirs(dest_folder, exist_ok=True)
                        dest_file = os.path.join(dest_folder, file)
                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
    
    # 2. Fix JSON paths (remove leading slash)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'image_path' in item and item['image_path'].startswith('/'):
                item['image_path'] = item['image_path'][1:]
        
        # Save renamed JSON
        target_name = "train.json" if split == "val" else "test.json"
        with open(os.path.join(base_dir, target_name), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

print("Dataset organized.")

