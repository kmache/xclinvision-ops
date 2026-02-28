import os
import shutil
import re
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ---------------------------------------------------------------------------
# Dynamic Path Resolution
# ---------------------------------------------------------------------------
# Path to the script (e.g., /.../project/scripts/organize_data.py)
SCRIPT_PATH = Path(__file__).resolve()
# Project root (one level up from scripts/)
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# Resolve data paths relative to the project root
SOURCE_CHEST_XRAY = PROJECT_ROOT / "new_data" / "chest_xray"
SOURCE_TB = PROJECT_ROOT / "new_data" / "Tuberculosis_Chest_Xray"
DEST_RAW = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Seed for reproducible random splitting
random.seed(42)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_patient_id(filename):
    match = re.search(r'^([^_ ]+)', filename)
    return match.group(1) if match else filename

def count_images_in_raw(raw_path):
    counts = defaultdict(lambda: defaultdict(int))
    if not raw_path.exists():
        return counts
    
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'pneumonia', 'tuberculosis']:
            path = raw_path / split / cls
            if path.exists():
                images = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                counts[split][cls] = len(images)
    return counts

def print_summary(before, after):
    print("\n" + "="*60)
    print(f"{'SPLIT':<10} | {'CLASS':<15} | {'BEFORE':<8} | {'AFTER':<8}")
    print("-"*60)
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'pneumonia', 'tuberculosis']:
            b = before[split][cls]
            a = after[split][cls]
            print(f"{split:<10} | {cls:<15} | {b:<8} | {a:<8}")
    print("="*60 + "\n")

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

def create_dirs():
    print(f"Creating directory structure in: {DEST_RAW}")
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'pneumonia', 'tuberculosis']:
            (DEST_RAW / split / cls).mkdir(parents=True, exist_ok=True)

def handle_chest_xray():
    print(f"Scanning Chest_Xray in: {SOURCE_CHEST_XRAY}")
    patient_to_split = {} 

    for split in ['train', 'val', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            src_dir = SOURCE_CHEST_XRAY / split / cls
            if not src_dir.exists(): 
                print(f"  Warning: {src_dir} not found.")
                continue
            
            target_cls = cls.lower()
            images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            
            for img in images:
                p_id = get_patient_id(img.name)
                assigned_split = patient_to_split.get(p_id, split)
                patient_to_split[p_id] = assigned_split
                
                dest = DEST_RAW / assigned_split / target_cls / img.name
                shutil.copy2(img, dest)

def handle_tb_dataset():
    print(f"Scanning TB Dataset in: {SOURCE_TB}")
    
    class_mapping = {
        'Normal Chest X-rays': 'normal',
        'TB Chest X-rays': 'tuberculosis'
    }

    for src_folder_name, target_cls in class_mapping.items():
        src_dir = SOURCE_TB / src_folder_name
        if not src_dir.exists():
            print(f"  Warning: Folder '{src_dir}' not found.")
            continue
            
        images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        random.shuffle(images)

        n = len(images)
        train_idx = int(n * 0.70)
        val_idx = train_idx + int(n * 0.15)

        splits = {'train': images[:train_idx], 'val': images[train_idx:val_idx], 'test': images[val_idx:]}

        print(f"  - Found {n} images in '{src_folder_name}'. Splitting...")
        for split_name, img_list in splits.items():
            target_dir = DEST_RAW / split_name / target_cls
            for img in img_list:
                shutil.copy2(img, target_dir / img.name)

if __name__ == "__main__":
    # 1. Initial State
    initial_counts = count_images_in_raw(DEST_RAW)
    
    # 2. Organize
    create_dirs()
    handle_chest_xray()
    handle_tb_dataset()
    
    # 3. Final State
    final_counts = count_images_in_raw(DEST_RAW)
    
    # 4. Report
    print_summary(initial_counts, final_counts)