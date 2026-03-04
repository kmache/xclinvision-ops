import os
import shutil
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import csv

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

# M-4 fix: track the source dataset for each copied file so downstream
# analysis can distinguish Normal images from different hospital distributions.
# Entries are written to data/raw/source_metadata.csv.
_source_records: list = []

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_patient_id(filename):
    """C-4 fix: extract a reliable patient identifier from common X-ray filename conventions.

    Supported patterns
    ------------------
    * Mooney chest-xray PNEUMONIA:  ``person1_bacteria_1.jpeg``  -> ``person1``
    * Mooney chest-xray NORMAL:     ``IM-0001-0001.jpeg``        -> ``IM-0001``
    * Generic fallback:             use the full stem

    Prior implementation (``re.search(r'^([^_ ]+)', name)``) grabbed everything
    before the first ``_`` or space.  For NORMAL images named ``IM-0001-0001``
    that yields ``IM-0001-0001`` (the full stem, including the sequence number),
    so each file looked like a unique patient — preventing effective patient-wise
    split enforcement.
    """
    stem = Path(filename).stem
    # Pattern 1: Mooney Pneumonia — person<N>_bacteria/virus_<k>
    m = re.match(r'^(person\d+)', stem, re.IGNORECASE)
    if m:
        return m.group(1)
    # Pattern 2: Mooney Normal — IM-<4-digit-patient>-<frame>
    m = re.match(r'^(IM-\d{4})', stem, re.IGNORECASE)
    if m:
        return m.group(1)
    # Generic fallback: first token before any separator
    m = re.match(r'^([^_\-. ]+)', stem)
    return m.group(1) if m else stem

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
                # M-4: record source dataset for downstream analysis
                _source_records.append({
                    "filename": img.name,
                    "split": assigned_split,
                    "class": target_cls,
                    "source_dataset": "chest_xray_pneumonia",
                })

def handle_tb_dataset():
    """C-3 fix: group TB dataset images by inferred patient ID before splitting.

    The original code shuffled *files* randomly, which allows the same patient
    to appear in both train and validation when a patient has multiple X-rays
    — a primary cause of data leakage and inflated validation metrics.

    True patient-wise splitting requires an explicit patient metadata CSV.
    In its absence we use ``get_patient_id()`` on the filename stem as a
    best-effort patient surrogate and assign entire patient groups to one
    split using deterministic hash-bucketing (no random-seed dependency,
    fully reproducible).

    WARNING: If the TB dataset does not embed a patient identifier in filenames
    this approach degrades to file-level splitting.  For rigorous leakage
    prevention, obtain and use an official patient metadata file.
    """
    print(f"Scanning TB Dataset in: {SOURCE_TB}")

    class_mapping = {
        'Normal Chest X-rays': 'normal',
        'TB Chest X-rays': 'tuberculosis',
    }

    for src_folder_name, target_cls in class_mapping.items():
        src_dir = SOURCE_TB / src_folder_name
        if not src_dir.exists():
            print(f"  Warning: Folder '{src_dir}' not found.")
            continue

        images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        # Group all files by inferred patient ID
        patient_to_files = defaultdict(list)
        for img in images:
            pid = get_patient_id(img.name)
            patient_to_files[pid].append(img)

        patient_ids = sorted(patient_to_files.keys())
        n_patients = len(patient_ids)

        print(
            f"  - Found {len(images)} images across {n_patients} inferred patients "
            f"in '{src_folder_name}'. Splitting by patient group..."
        )
        if n_patients < 10:
            print(
                f"  WARNING: Only {n_patients} unique patient IDs detected in "
                f"'{src_folder_name}'. Filenames may not carry patient identifiers — "
                "true patient-wise splitting is not possible without a metadata file."
            )

        # Assign each patient deterministically to train/val/test via MD5 hash bucket.
        # Buckets 0-69 → train (70 %), 70-84 → val (15 %), 85-99 → test (15 %).
        # This is reproducible without an external random seed.
        assigned: dict = {'train': [], 'val': [], 'test': []}
        for pid in patient_ids:
            bucket = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
            if bucket < 70:
                split_name = 'train'
            elif bucket < 85:
                split_name = 'val'
            else:
                split_name = 'test'
            assigned[split_name].extend(patient_to_files[pid])

        for split_name, img_list in assigned.items():
            target_dir = DEST_RAW / split_name / target_cls
            for img in img_list:
                shutil.copy2(img, target_dir / img.name)
                # M-4: record source dataset for downstream analysis
                _source_records.append({
                    "filename": img.name,
                    "split": split_name,
                    "class": target_cls,
                    "source_dataset": "tuberculosis_chest_xray",
                })

        counts = {k: len(v) for k, v in assigned.items()}
        print(f"    Image counts after patient-wise split: {counts}")

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

    # 5. M-4: save source metadata so the manifest can distinguish images from
    #    different hospital/scanner distributions (critical for understanding
    #    why Normal activation patterns differ across the dataset).
    if _source_records:
        meta_path = DEST_RAW / "source_metadata.csv"
        with open(meta_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename", "split", "class", "source_dataset"])
            writer.writeheader()
            writer.writerows(_source_records)
        print(f"\nSource metadata written to: {meta_path} ({len(_source_records)} entries)")