"""organize_data.py - Unified dataset organisation script.

Two independent pipelines, selectable via CLI flags:

  A) Legacy pipeline  (Mooney Chest X-ray + Cardiomegaly datasets)
     target: data/raw/{train,val,test}/{normal,pneumonia,cardiomegaly}/

  B) NIH ChestX-ray14 subset pipeline       [--nih, default when no flag given]
     Selects Cardiomegaly, Pneumonia and sampled No-Finding images, honours the
     official train_val_list.txt / test_list.txt patient-level boundaries, then
     performs a patient-wise train/val split inside the train-val pool.
     target: data/NIH-ChestX-ray3/{train,val,test}/<class>/

     Add more diseases with --diseases:
       python scripts/organize_data.py --nih --diseases Cardiomegaly Pneumonia Effusion

The train/val/test split helpers (create_stratified_split, save_split_metadata)
that previously lived in src/xclinvision/processing.py are now co-located here
so that all data-organisation logic lives in one place.

Usage
-----
  python scripts/organize_data.py              # NIH subset (default)
  python scripts/organize_data.py --nih        # NIH subset (explicit)
  python scripts/organize_data.py --legacy     # legacy pipeline only
  python scripts/organize_data.py --nih --legacy  # both
  python scripts/organize_data.py --nih --diseases Cardiomegaly Pneumonia Edema
  python scripts/organize_data.py --nih --normal-ratio 1.5 --copy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root relative paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

NIH_ROOT = PROJECT_ROOT / "NIH_ChestX-ray14"
NIH_DEST = PROJECT_ROOT / "data" / "NIH-ChestX-ray3"

# Legacy pipeline paths
SOURCE_CHEST_XRAY = PROJECT_ROOT / "new_data" / "chest_xray"
SOURCE_TB         = PROJECT_ROOT / "new_data" / "Tuberculosis_Chest_Xray"
DEST_RAW          = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _patient_hash_bucket(patient_id: str) -> int:
    """Deterministic 0-99 bucket for a patient ID (MD5-based).

    Fix #27: normalise patient_id to a canonical integer string before hashing
    so that equivalent IDs stored as different types (e.g. '1234', '1234.0')
    always land in the same bucket.
    """
    try:
        canonical = str(int(float(patient_id)))
    except (ValueError, OverflowError):
        canonical = str(patient_id)
    return int(hashlib.md5(canonical.encode()).hexdigest(), 16) % 100


def _assign_trainval_split(patient_id: str, val_ratio: float = 0.18) -> str:
    """Return 'train' or 'val' for a patient based on hash bucket.

    val_ratio=0.18 gives ~18% val / ~82% train from the trainval pool.
    Combined with the official NIH test split (~38% disease images in test)
    the effective split per disease class is ~50% train / ~12% val / ~38% test.
    """
    threshold = int(round((1.0 - val_ratio) * 100))
    return "val" if _patient_hash_bucket(patient_id) >= threshold else "train"


# ---------------------------------------------------------------------------
# NIH ChestX-ray14 subset pipeline
# ---------------------------------------------------------------------------

def _build_nih_image_index(nih_root: Path) -> Dict[str, Path]:
    """Scan images_00X/images/ sub-folders; return filename -> absolute path."""
    index: Dict[str, Path] = {}
    image_dirs = sorted(nih_root.glob("images_*/images"))
    if not image_dirs:
        raise FileNotFoundError(
            f"No 'images_*/images/' folders found under {nih_root}. "
            "Ensure the NIH ChestX-ray14 dataset is present."
        )
    for img_dir in image_dirs:
        for p in img_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                index[p.name] = p
    logger.info("NIH image index built: %d files across %d folders.",
                len(index), len(image_dirs))
    return index


def _load_nih_metadata(nih_root: Path) -> pd.DataFrame:
    """Load Data_Entry_2017.csv and normalise column names."""
    csv_path = nih_root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data_Entry_2017.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data_Entry_2017.csv is missing columns: {missing}")
    df["Patient ID"] = df["Patient ID"].astype(str)
    return df


def _select_disease_images(
    df: pd.DataFrame,
    diseases: List[str],
) -> Dict[str, pd.DataFrame]:
    """Return per-class DataFrames with unambiguous single-label assignment.

    An image is assigned to class <d> if:
      - Finding Labels contains <d>, AND
      - none of the other selected diseases appear in its labels.
    Images containing two or more selected diseases are excluded.
    'normal' class: Finding Labels == 'No Finding' exactly.
    """
    class_dfs: Dict[str, pd.DataFrame] = {}
    for disease in diseases:
        mask = df["Finding Labels"].str.contains(disease, regex=False)
        for other in diseases:
            if other == disease:
                continue
            mask = mask & ~df["Finding Labels"].str.contains(other, regex=False)
        subset = df[mask].copy()
        cls_key = disease.lower().replace(" ", "_")
        class_dfs[cls_key] = subset
        logger.info("  %-22s %d images,  %d unique patients",
                    disease, len(subset), subset["Patient ID"].nunique())

    nf = df[df["Finding Labels"] == "No Finding"].copy()
    class_dfs["normal"] = nf
    logger.info("  %-22s %d images,  %d unique patients",
                "No Finding (all)", len(nf), nf["Patient ID"].nunique())
    return class_dfs


def _load_official_split_sets(nih_root: Path) -> Tuple[set, set]:
    """Return (test_images, trainval_images) filename sets from official lists."""
    def _read(path: Path) -> set:
        return set(path.read_text().splitlines())

    test_set     = _read(nih_root / "test_list.txt")
    trainval_set = _read(nih_root / "train_val_list.txt")
    logger.info("Official split lists: %d test,  %d train-val.",
                len(test_set), len(trainval_set))
    return test_set, trainval_set


def _sample_no_finding(
    nf_df: pd.DataFrame,
    disease_dfs: Dict[str, pd.DataFrame],
    normal_ratio: float,
    test_set: set,
    trainval_set: set,
    val_ratio: float,
) -> pd.DataFrame:
    """Sample No-Finding patients to limit class imbalance.

    Target count = normal_ratio x (size of the largest disease class).
    Sampling is patient-level so no patient spans multiple splits.
    Test / trainval membership follows the official NIH lists.
    """
    disease_totals = {cls: len(sub) for cls, sub in disease_dfs.items()
                      if cls != "normal"}
    if not disease_totals:
        return nf_df

    max_disease   = max(disease_totals.values())
    target_total  = int(round(normal_ratio * max_disease))
    logger.info("No Finding sampling target: %d  (%.1f x largest disease=%d)",
                target_total, normal_ratio, max_disease)

    nf_test     = nf_df[nf_df["Image Index"].isin(test_set)]
    nf_trainval = nf_df[nf_df["Image Index"].isin(trainval_set)]
    total_avail = len(nf_test) + len(nf_trainval)
    if total_avail == 0:
        return nf_df

    test_frac       = len(nf_test) / total_avail
    target_test     = min(int(round(target_total * test_frac)), len(nf_test))
    target_trainval = min(target_total - target_test, len(nf_trainval))

    def _greedy_sample(subset: pd.DataFrame, target: int) -> List[str]:
        collected: List[str] = []
        for pid in subset["Patient ID"].unique():
            batch = subset[subset["Patient ID"] == pid]["Image Index"].tolist()
            if len(collected) + len(batch) <= target:
                collected.extend(batch)
            if len(collected) >= target:
                break
        return collected

    test_imgs = _greedy_sample(nf_test,     target_test)
    tv_imgs   = _greedy_sample(nf_trainval, target_trainval)

    sampled = set(test_imgs) | set(tv_imgs)
    result  = nf_df[nf_df["Image Index"].isin(sampled)].copy()
    logger.info("  No Finding sampled: %d test + %d trainval = %d total.",
                len(test_imgs), len(tv_imgs), len(result))
    return result


def _assign_splits(
    class_dfs: Dict[str, pd.DataFrame],
    test_set: set,
    trainval_set: set,
    val_ratio: float,
) -> List[dict]:
    """Assign train / val / test to each selected image.

    - Official test_list.txt  => test
    - Official train_val_list => patient hash bucket decides train vs val
    - Each patient lands in exactly ONE split (no data leakage).
    """
    records: List[dict] = []
    tv_patient_cache: Dict[str, str] = {}

    for cls_name, sub_df in class_dfs.items():
        for _, row in sub_df.iterrows():
            img = row["Image Index"]
            pid = str(row["Patient ID"])

            if img in test_set:
                split = "test"
            elif img in trainval_set:
                if pid not in tv_patient_cache:
                    tv_patient_cache[pid] = _assign_trainval_split(pid, val_ratio)
                split = tv_patient_cache[pid]
            else:
                continue  # not in either official list

            records.append({
                "image_index": img,
                "class":       cls_name,
                "split":       split,
                "patient_id":  pid,
            })

    return records


def _transfer_files(
    records: List[dict],
    image_index: Dict[str, Path],
    dest_root: Path,
    use_copy: bool,
) -> Tuple[int, int]:
    """Create dirs and copy / symlink image files. Returns (succeeded, skipped)."""
    succeeded = skipped = 0
    for rec in tqdm(records, desc="Transferring files"):
        fname = rec["image_index"]
        src   = image_index.get(fname)
        if src is None:
            logger.debug("Not in index, skipping: %s", fname)
            skipped += 1
            continue

        dest_dir = dest_root / rec["split"] / rec["class"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname

        if dest.exists() or dest.is_symlink():
            succeeded += 1
            continue

        try:
            if use_copy:
                shutil.copy2(src, dest)
            else:
                os.symlink(src.resolve(), dest)
            succeeded += 1
        except Exception as exc:
            logger.warning("Transfer failed %s -> %s: %s", fname, dest, exc)
            skipped += 1

    return succeeded, skipped


def _write_manifest(records: List[dict], dest_root: Path) -> None:
    path = dest_root / "manifest.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_index", "class", "split", "patient_id"]
        )
        writer.writeheader()
        writer.writerows(records)
    logger.info("Manifest written: %s  (%d entries)", path, len(records))


def _print_nih_summary(records: List[dict], classes: List[str]) -> None:
    print()
    print("=" * 74)
    print(f"{'CLASS':<24} | {'TRAIN':>7} | {'VAL':>7} | {'TEST':>7} | {'TOTAL':>7}")
    print("-" * 74)
    for cls in classes:
        counts = {"train": 0, "val": 0, "test": 0}
        for r in records:
            if r["class"] == cls:
                counts[r["split"]] += 1
        total = sum(counts.values())
        print(f"{cls:<24} | {counts['train']:>7} | {counts['val']:>7} | "
              f"{counts['test']:>7} | {total:>7}")
    print("-" * 74)
    overall = {"train": 0, "val": 0, "test": 0}
    for r in records:
        overall[r["split"]] += 1
    grand = sum(overall.values())
    print(f"{'TOTAL':<24} | {overall['train']:>7} | {overall['val']:>7} | "
          f"{overall['test']:>7} | {grand:>7}")
    if grand:
        pct = {s: 100 * overall[s] / grand for s in ("train", "val", "test")}
        print(f"{'%':<24} | {pct['train']:>6.1f}% | {pct['val']:>6.1f}% | "
              f"{pct['test']:>6.1f}% |")
    print("=" * 74)
    print()


def handle_nih_subset(
    diseases: Optional[List[str]] = None,
    normal_ratio: float = 2.0,
    val_ratio: float = 0.18,
    use_copy: bool = False,
    nih_root: Path = NIH_ROOT,
    dest_root: Path = NIH_DEST,
) -> None:
    """Build the NIH ChestX-ray14 N-class subset dataset.

    Parameters
    ----------
    diseases : list[str]
        Disease labels exactly as in Data_Entry_2017.csv.
        Default: ['Cardiomegaly', 'Pneumonia'].
    normal_ratio : float
        No Finding count ~= normal_ratio x (largest disease class count). Default 2.0.
    val_ratio : float
        Fraction of train-val patients assigned to val. Default 0.18.
    use_copy : bool
        True -> shutil.copy2.  False (default) -> symlinks (saves disk).
    """
    if diseases is None:
        diseases = ["Cardiomegaly", "Pneumonia"]

    logger.info("=== NIH ChestX-ray14 subset pipeline ===")
    logger.info("Diseases        : %s", diseases)
    logger.info("Normal ratio    : %.1f", normal_ratio)
    logger.info("Val ratio       : %.2f (from train-val pool)", val_ratio)
    logger.info("Transfer mode   : %s", "copy" if use_copy else "symlink")
    logger.info("Output root     : %s", dest_root)

    image_index              = _build_nih_image_index(nih_root)
    df                       = _load_nih_metadata(nih_root)
    test_set, trainval_set   = _load_official_split_sets(nih_root)

    logger.info("Selecting disease images ...")
    class_dfs = _select_disease_images(df, diseases)

    class_dfs["normal"] = _sample_no_finding(
        class_dfs["normal"], class_dfs,
        normal_ratio, test_set, trainval_set, val_ratio,
    )

    logger.info("Assigning splits (patient-level, no leakage) ...")
    records = _assign_splits(class_dfs, test_set, trainval_set, val_ratio)
    logger.info("Total records: %d", len(records))

    succeeded, skipped = _transfer_files(records, image_index, dest_root, use_copy)
    if skipped:
        logger.warning("%d images skipped (not found in image index).", skipped)
    logger.info("Transfer complete: %d succeeded, %d skipped.", succeeded, skipped)

    _write_manifest(records, dest_root)

    cls_keys = [d.lower().replace(" ", "_") for d in diseases] + ["normal"]
    _print_nih_summary(records, cls_keys)


# ---------------------------------------------------------------------------
# Split helpers (moved from src/xclinvision/processing.py)
# ---------------------------------------------------------------------------

def create_stratified_split(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Create stratified train / val / test splits from a flat class-folder layout.

    Useful when the dataset has no pre-defined split (e.g. data_dir contains
    sub-folders 'pneumonia/', 'normal/' without train/val/test sub-structure).

    Parameters
    ----------
    data_dir : str
        Root directory containing one sub-folder per class.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5,         "Split ratios must sum to 1.0"

    from sklearn.preprocessing import LabelEncoder

    data_dir   = Path(data_dir)
    all_images: List[str] = []
    all_labels: List[str] = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name.lower()
        for ext in IMAGE_EXTS:
            for img_path in class_dir.glob(f"*{ext}"):
                all_images.append(str(img_path))
                all_labels.append(cls)

    if not all_images:
        raise ValueError(f"No images found in {data_dir}")

    le = LabelEncoder()
    label_indices = le.fit_transform(all_labels)

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        all_images, all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=label_indices,
        random_state=seed,
    )

    temp_indices  = le.transform(temp_lbls)
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls,
        test_size=(1 - val_ratio_adj),
        stratify=temp_indices,
        random_state=seed,
    )

    logger.info("Stratified split: %d train  %d val  %d test",
                len(train_imgs), len(val_imgs), len(test_imgs))
    return train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls


def save_split_metadata(
    train_paths: List[str],
    val_paths:   List[str],
    test_paths:  List[str],
    train_labels: List[str],
    val_labels:   List[str],
    test_labels:  List[str],
    output_dir: str,
) -> None:
    """Persist split information as CSVs for downstream reproducibility.

    Writes train_split.csv, val_split.csv, test_split.csv and a unified
    manifest_splits.csv to output_dir.  Columns: filepath_processed, class,
    split, image_path, label  (compatible with ChestXrayDataModule).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    for split_name, paths, labels in [
        ("train", train_paths, train_labels),
        ("val",   val_paths,   val_labels),
        ("test",  test_paths,  test_labels),
    ]:
        frame = pd.DataFrame({
            "filepath_processed": paths,
            "class":              labels,
            "split":              [split_name] * len(paths),
            "image_path":         paths,
            "label":              labels,
        })
        frame.to_csv(output_dir / f"{split_name}_split.csv", index=False)
        all_frames.append(frame)

    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_csv(
            output_dir / "manifest_splits.csv", index=False
        )
    logger.info("Split metadata saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Legacy pipeline (Mooney Chest X-ray + Cardiomegaly)
# ---------------------------------------------------------------------------

def _get_legacy_patient_id(filename: str) -> str:
    """Extract patient ID from common X-ray filename patterns."""
    stem = Path(filename).stem
    m = re.match(r"^(person\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^(IM-\d{4})", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^([^_\-. ]+)", stem)
    return m.group(1) if m else stem


_legacy_source_records: List[dict] = []


def _legacy_count_images(raw_path: Path) -> Dict:
    counts: Dict = defaultdict(lambda: defaultdict(int))
    if not raw_path.exists():
        return counts
    for split in ("train", "val", "test"):
        for cls in ("normal", "pneumonia", "cardiomegaly"):
            p = raw_path / split / cls
            if p.exists():
                counts[split][cls] = sum(
                    1 for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
    return counts


def _legacy_print_summary(before: Dict, after: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'SPLIT':<10} | {'CLASS':<15} | {'BEFORE':<8} | {'AFTER':<8}")
    print("-" * 60)
    for split in ("train", "val", "test"):
        for cls in ("normal", "pneumonia", "cardiomegaly"):
            b = before[split][cls]
            a = after[split][cls]
            print(f"{split:<10} | {cls:<15} | {b:<8} | {a:<8}")
    print("=" * 60 + "\n")


def handle_chest_xray() -> None:
    """Copy Mooney Chest X-ray images honouring the existing patient-level split."""
    logger.info("Scanning Chest_Xray in: %s", SOURCE_CHEST_XRAY)
    patient_to_split: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            src_dir = SOURCE_CHEST_XRAY / split / cls
            if not src_dir.exists():
                logger.warning("  Not found: %s", src_dir)
                continue
            target_cls = cls.lower()
            images = [
                f for f in src_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ]
            for img in images:
                p_id     = _get_legacy_patient_id(img.name)
                assigned = patient_to_split.get(p_id, split)
                patient_to_split[p_id] = assigned
                dest = DEST_RAW / assigned / target_cls / img.name
                shutil.copy2(img, dest)
                _legacy_source_records.append({
                    "filename":       img.name,
                    "split":          assigned,
                    "class":          target_cls,
                    "source_dataset": "chest_xray_pneumonia",
                })


def handle_tb_dataset() -> None:
    """Copy Cardiomegaly dataset with patient-wise 70/15/15 hash split."""
    logger.info("Scanning Cardiomegaly dataset in: %s", SOURCE_TB)

    class_mapping = {
        "Normal Chest X-rays": "normal",
        "TB Chest X-rays":     "cardiomegaly",
    }

    for src_folder, target_cls in class_mapping.items():
        src_dir = SOURCE_TB / src_folder
        if not src_dir.exists():
            logger.warning("  Not found: %s", src_dir)
            continue

        images = [
            f for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        patient_to_files: Dict[str, List[Path]] = defaultdict(list)
        for img in images:
            patient_to_files[_get_legacy_patient_id(img.name)].append(img)

        patient_ids = sorted(patient_to_files.keys())
        logger.info("  %s: %d images / %d inferred patients",
                    src_folder, len(images), len(patient_ids))

        assigned: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        for pid in patient_ids:
            bucket = _patient_hash_bucket(pid)
            sp = "train" if bucket < 70 else ("val" if bucket < 85 else "test")
            assigned[sp].extend(patient_to_files[pid])

        for sp, img_list in assigned.items():
            for img in img_list:
                shutil.copy2(img, DEST_RAW / sp / target_cls / img.name)
                _legacy_source_records.append({
                    "filename":       img.name,
                    "split":          sp,
                    "class":          target_cls,
                    "source_dataset": "cardiomegaly_chest_xray",
                })
        logger.info("    Split counts: %s",
                    {k: len(v) for k, v in assigned.items()})


def run_legacy_pipeline() -> None:
    """Create dirs, run both legacy sources, print summary, write metadata."""
    initial = _legacy_count_images(DEST_RAW)

    for split in ("train", "val", "test"):
        for cls in ("normal", "pneumonia", "cardiomegaly"):
            (DEST_RAW / split / cls).mkdir(parents=True, exist_ok=True)

    handle_chest_xray()
    handle_tb_dataset()

    _legacy_print_summary(initial, _legacy_count_images(DEST_RAW))

    if _legacy_source_records:
        meta_path = DEST_RAW / "source_metadata.csv"
        with open(meta_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["filename", "split", "class", "source_dataset"]
            )
            writer.writeheader()
            writer.writerows(_legacy_source_records)
        logger.info("Source metadata: %s  (%d entries)",
                    meta_path, len(_legacy_source_records))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nih", action="store_true", default=False,
        help="Run the NIH ChestX-ray14 subset pipeline.",
    )
    parser.add_argument(
        "--legacy", action="store_true", default=False,
        help="Run the legacy Mooney+TB pipeline (data/raw/).",
    )
    parser.add_argument(
        "--diseases", nargs="+",
        default=["Cardiomegaly", "Pneumonia"],
        metavar="DISEASE",
        help=(
            "Disease labels as in Data_Entry_2017.csv (space-separated). "
            "Default: Cardiomegaly Pneumonia. "
            "Example: --diseases Cardiomegaly Pneumonia Effusion"
        ),
    )
    parser.add_argument(
        "--normal-ratio", type=float, default=2.0, metavar="R",
        help="Sample No Finding at R x (largest disease class). Default: 2.0.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.18, metavar="V",
        help="Fraction of train-val patients assigned to val (0-1). Default: 0.18.",
    )
    parser.add_argument(
        "--copy", action="store_true", default=False,
        help="Copy files instead of symlinks (slower, uses more disk).",
    )
    parser.add_argument(
        "--nih-root", type=Path, default=NIH_ROOT, metavar="DIR",
        help=f"Path to NIH ChestX-ray14 root. Default: {NIH_ROOT}",
    )
    parser.add_argument(
        "--dest", type=Path, default=NIH_DEST, metavar="DIR",
        help=f"Output directory for NIH subset. Default: {NIH_DEST}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Default: run NIH pipeline when no flag given
    if not args.nih and not args.legacy:
        args.nih = True

    if args.nih:
        handle_nih_subset(
            diseases=args.diseases,
            normal_ratio=args.normal_ratio,
            val_ratio=args.val_ratio,
            use_copy=args.copy,
            nih_root=args.nih_root,
            dest_root=args.dest,
        )

    if args.legacy:
        run_legacy_pipeline()
