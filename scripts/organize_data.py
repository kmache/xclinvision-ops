"""organize_data.py -- Config-driven multi-label dataset organisation.

Takes data from a source directory and organises it into ``data/raw/``
with the following structure::

    data/raw/
        images/           # all image files
        labels.csv        # multi-hot encoded labels (one row per image)
        splits/
            train.txt     # one filename per line
            val.txt
            test.txt

Two source formats are supported via ``--source-format``:

  csv   (default) Source contains a CSV annotation file and an image
        directory.  Designed for VinBigData-style annotation CSVs with
        columns ``image_id, class_name, ...`` where each row is one
        annotation (possibly multiple rows per image).

  folder  (legacy) Source is organised as ``class/image`` or
          ``class/split/image``.  Single-label only; kept for backward
          compatibility.

Configuration
-------------
Class names and classification mode are read from ``configs/system.yaml``::

    model:
      class_names: [Normal, Cardiomegaly, Aortic enlargement, ...]
      classification_mode: multilabel

All samples (disease and normal) are kept at their natural proportions.
Class imbalance is handled downstream via weighted loss (pos_weight).

Usage
-----
  # VinBigData CSV source (default)
  python scripts/organize_data.py \\
      --source /path/to/vinbigdata \\
      --label-csv train.csv --image-dir train

  # Legacy folder source
  python scripts/organize_data.py \\
      --source /path/to/my_data --source-format folder
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to python path for local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xclinvision.config import PipelineConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPLIT_NAMES = ("train", "val", "test")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom",
}

# Known aliases for the normal / no-disease class (lowercased)
NORMAL_ALIASES = frozenset({"no finding", "normal", "no_finding", "healthy"})

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

DEST_RAW = PROJECT_ROOT / "data" / "raw"
SYSTEM_CONFIG = PROJECT_ROOT / "configs" / "system.yaml"


# ===================================================================
# Configuration helpers
# ===================================================================

def _load_system_config() -> dict:
    """Load and return the full system.yaml config dict."""
    if not SYSTEM_CONFIG.exists():
        raise FileNotFoundError(
            f"System config not found: {SYSTEM_CONFIG}. "
            "Create configs/system.yaml with model.class_names."
        )
    with open(SYSTEM_CONFIG, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _get_class_names_from_config(config: PipelineConfig) -> List[str]:
    """Extract class names from PipelineConfig; validate."""
    names = config.class_names
    if not isinstance(names, list) or len(names) < 2:
        raise ValueError(
            "model.class_names in system.yaml must be a list with >= 2 entries. "
            f"Got: {names}"
        )
    return names


def _identify_normal_class(class_names: List[str]) -> Optional[str]:
    """Identify which class name is the 'normal' / 'no finding' class."""
    for name in class_names:
        if name.lower() in NORMAL_ALIASES:
            return name
    logger.warning(
        "Could not identify a normal class in %s. "
        "No normal-class sampling will be applied.",
        class_names,
    )
    return None


# ===================================================================
# CSV-based multi-label pipeline
# ===================================================================

def _build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Build ``{image_id_stem: filepath}`` from all images in *image_dir*."""
    index: Dict[str, Path] = {}
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    for f in image_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            index[f.stem] = f
    if not index:
        raise ValueError(f"No images found in {image_dir}")
    logger.info("Image index: %d files in %s", len(index), image_dir)
    return index


def _build_multilabel_dataframe(
    label_csv_path: Path,
    class_names: List[str],
    normal_class: Optional[str],
    image_index: Dict[str, Path],
    min_rads: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read source annotation CSV and return a multi-hot-encoded DataFrame.

    Source CSV format (VinBigData-style)::

        image_id, class_name, class_id, rad_id, ...

    Each row is one annotation; an image may appear in multiple rows.

    When ``min_rads > 1``, images are categorised as:

    * **consensus disease** – at least one of the configured diseases has
      ≥ *min_rads* radiologists annotating it.  Kept with disease labels.
    * **truly normal** – NO radiologist annotated any configured disease.
      Kept as all-zero (normal).
    * **ambiguous** – at least one radiologist annotated a configured
      disease, but none reached the *min_rads* threshold.  **Dropped**
      and returned in the second element of the tuple so the caller can
      save them to an ``ambiguous_img/`` directory.

    Returns
    -------
    (DataFrame, ambiguous_image_ids)
        DataFrame with columns ``[filename, image_id, <class_name_1>, ...]``
        where each class column is 0 or 1.
        ambiguous_image_ids is a list of image_id strings that were dropped.
    """
    df_raw = pd.read_csv(label_csv_path)

    for col in ("image_id", "class_name"):
        if col not in df_raw.columns:
            raise ValueError(
                f"Source CSV missing required column '{col}'. "
                f"Found: {df_raw.columns.tolist()}"
            )
    df_raw = df_raw.dropna(subset=["image_id", "class_name"])

    # Build case-insensitive source-name -> config-name mapping
    disease_names = [n for n in class_names if n.lower() not in NORMAL_ALIASES]
    source_to_config: Dict[str, str] = {}
    for name in class_names:
        source_to_config[name.lower()] = name
    # Map all normal aliases to the configured normal class
    if normal_class:
        for alias in NORMAL_ALIASES:
            source_to_config[alias] = normal_class

    df_raw["class_mapped"] = (
        df_raw["class_name"].str.strip().str.lower().map(source_to_config)
    )

    # Keep only annotations for configured classes
    df_filtered = df_raw[df_raw["class_mapped"].notna()].copy()
    n_dropped = len(df_raw) - len(df_filtered)
    if n_dropped > 0:
        dropped = df_raw.loc[df_raw["class_mapped"].isna(), "class_name"].unique()
        logger.info(
            "Filtered out %d annotations not in class_names. Dropped: %s",
            n_dropped, sorted(set(dropped)),
        )

    ambiguous_ids: List[str] = []

    # ---- Aggregate per-radiologist annotations into multi-hot labels ----
    # When min_rads > 1, require at least that many radiologists to agree
    # for a positive label (majority vote). This reduces label noise from
    # single-radiologist disagreements.
    if min_rads > 1 and "rad_id" in df_filtered.columns:
        logger.info(
            "Majority-vote aggregation: requiring >= %d radiologists for positive label.",
            min_rads,
        )
        disease_names_set = {n for n in class_names if n.lower() not in NORMAL_ALIASES}

        # Disease annotations only (exclude "No finding")
        disease_annotations = df_filtered[
            df_filtered["class_mapped"].isin(disease_names_set)
        ]

        # For each (image_id, class_mapped), count distinct radiologists
        votes = (
            disease_annotations
            .groupby(["image_id", "class_mapped"])["rad_id"]
            .nunique()
            .reset_index(name="n_rads")
        )

        # Images with ANY disease annotation (even from 1 rad)
        images_with_any_disease = set(disease_annotations["image_id"].unique())

        # Keep only labels with >= min_rads agreement → consensus disease
        consensus_votes = votes[votes["n_rads"] >= min_rads]

        # Pivot to multi-hot
        if len(consensus_votes) > 0:
            consensus_votes = consensus_votes.copy()
            consensus_votes["value"] = 1
            df_multi = consensus_votes.pivot_table(
                index="image_id", columns="class_mapped", values="value",
                fill_value=0, aggfunc="max",
            ).reset_index()
        else:
            df_multi = pd.DataFrame(columns=["image_id"])

        consensus_ids = set(df_multi["image_id"]) if len(df_multi) > 0 else set()

        # Ambiguous = has disease annotation(s) but NONE reached consensus.
        # These images are unreliable — drop them.
        ambiguous_set = images_with_any_disease - consensus_ids
        ambiguous_ids = sorted(ambiguous_set)

        # Truly normal = in the CSV but has NO disease annotation at all
        all_image_ids = set(df_filtered["image_id"].unique())
        truly_normal_ids = all_image_ids - images_with_any_disease

        # Add truly normal images with all-zero disease columns
        if truly_normal_ids:
            df_normal = pd.DataFrame({"image_id": list(truly_normal_ids)})
            df_multi = pd.concat([df_multi, df_normal], ignore_index=True)

        logger.info(
            "Majority-vote result: %d consensus disease, %d truly normal, "
            "%d AMBIGUOUS (dropped).",
            len(consensus_ids), len(truly_normal_ids), len(ambiguous_ids),
        )
    else:
        if min_rads > 1:
            logger.warning(
                "min_rads=%d requested but 'rad_id' column not found; "
                "falling back to OR aggregation.", min_rads,
            )
        # Original OR/MAX aggregation (any 1 radiologist -> positive)
        df_dummies = pd.get_dummies(df_filtered["class_mapped"]).astype(int)
        df_encoded = pd.concat(
            [df_filtered[["image_id"]].reset_index(drop=True), df_dummies], axis=1,
        )
        df_multi = df_encoded.groupby("image_id").max().reset_index()

    # Ensure every configured class column exists (as int)
    for name in class_names:
        if name not in df_multi.columns:
            df_multi[name] = 0
        else:
            df_multi[name] = df_multi[name].fillna(0).astype(int)

    # Recompute normal class: 1 iff ALL disease columns are 0
    if normal_class:
        df_multi[normal_class] = (
            df_multi[disease_names].sum(axis=1) == 0
        ).astype(int)

    # Resolve image_id -> actual filename on disk
    filenames: List[Optional[str]] = []
    found_mask: List[bool] = []
    for img_id in df_multi["image_id"]:
        path = image_index.get(img_id)
        if path is not None:
            filenames.append(path.name)
            found_mask.append(True)
        else:
            filenames.append(None)
            found_mask.append(False)

    n_missing = sum(not m for m in found_mask)
    if n_missing > 0:
        logger.warning(
            "%d images in CSV but not found on disk (skipped).", n_missing,
        )

    df_multi["filename"] = filenames
    df_multi = df_multi[pd.Series(found_mask).values].reset_index(drop=True)

    # Final column order
    df_multi = df_multi[["filename", "image_id"] + class_names]

    logger.info(
        "Multi-label DataFrame: %d images, %d classes.\n%s",
        len(df_multi), len(class_names),
        df_multi[class_names].sum().to_string(),
    )
    return df_multi, ambiguous_ids

# ---------------------------------------------------------------------------
# Multi-label stratified split
# ---------------------------------------------------------------------------

def _iterative_stratification(
    y: np.ndarray,
    ratios: List[float],
    seed: int,
) -> List[np.ndarray]:
    """Iterative stratification for multi-label data (Sechidis et al., 2011).

    Distributes samples across *k* folds so that each label's proportion is
    as close as possible to *ratios*.  Uses only numpy — no extra packages.

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_labels), binary indicator matrix.
    ratios : list of k floats summing to 1.0 (e.g. [0.7, 0.15, 0.15]).
    seed : random seed for tie-breaking.

    Returns
    -------
    List of k arrays, each containing the sample indices for that fold.
    """
    rng = np.random.RandomState(seed)
    n_samples, n_labels = y.shape
    k = len(ratios)
    ratios = np.asarray(ratios, dtype=np.float64)

    # Desired number of +ve per label per fold
    label_totals = y.sum(axis=0)  # (n_labels,)
    desired = np.outer(ratios, label_totals)  # (k, n_labels)

    folds: List[List[int]] = [[] for _ in range(k)]
    fold_label_counts = np.zeros((k, n_labels), dtype=np.float64)
    remaining = np.ones(n_samples, dtype=bool)

    # Process labels from rarest to most common
    label_order = np.argsort(label_totals)

    for label_idx in label_order:
        # Indices of remaining samples that are +ve for this label
        positives = np.where(remaining & (y[:, label_idx] == 1))[0]
        rng.shuffle(positives)

        for sample_idx in positives:
            # Assign to the fold with the greatest remaining need for this label
            needs = desired[:, label_idx] - fold_label_counts[:, label_idx]
            best_fold = int(np.argmax(needs))
            folds[best_fold].append(sample_idx)
            fold_label_counts[best_fold] += y[sample_idx]
            remaining[sample_idx] = False

    # Distribute any remaining samples (all-zero rows) proportionally
    leftover = np.where(remaining)[0]
    if len(leftover) > 0:
        rng.shuffle(leftover)
        targets = (ratios * len(leftover)).astype(int)
        # Fix rounding so totals match
        diff = len(leftover) - targets.sum()
        for i in range(abs(diff)):
            targets[i % k] += 1 if diff > 0 else -1
        ptr = 0
        for f in range(k):
            folds[f].extend(leftover[ptr : ptr + targets[f]].tolist())
            ptr += targets[f]

    return [np.array(fold, dtype=int) for fold in folds]


def _multilabel_stratified_split(
    df: pd.DataFrame,
    label_cols: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """Split *df* into train/val/test preserving multi-label distribution.

    Uses an iterative stratification algorithm (Sechidis et al., 2011)
    implemented with numpy only — no external dependencies required.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    y = df[label_cols].values.astype(int)
    ratios = [train_ratio, val_ratio, test_ratio]
    fold_indices = _iterative_stratification(y, ratios, seed)

    logger.info(
        "Multi-label iterative stratification: %d train, %d val, %d test.",
        len(fold_indices[0]), len(fold_indices[1]), len(fold_indices[2]),
    )

    return {
        "train": df.iloc[fold_indices[0]].reset_index(drop=True),
        "val": df.iloc[fold_indices[1]].reset_index(drop=True),
        "test": df.iloc[fold_indices[2]].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# File transfer (CSV mode)
# ---------------------------------------------------------------------------

def _transfer_images(
    df: pd.DataFrame,
    image_index: Dict[str, Path],
    dest_images_dir: Path,
    use_copy: bool,
) -> Tuple[int, int]:
    """Copy or symlink images listed in *df* into *dest_images_dir*."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    succeeded = skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transferring images"):
        src = image_index.get(row["image_id"])
        if src is None:
            skipped += 1
            continue

        dest = dest_images_dir / src.name
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
            logger.warning("Transfer failed %s -> %s: %s", src, dest, exc)
            skipped += 1

    return succeeded, skipped


# ---------------------------------------------------------------------------
# Output writers (CSV mode)
# ---------------------------------------------------------------------------

def _write_outputs(
    splits: Dict[str, pd.DataFrame],
    class_names: List[str],
    dest_dir: Path,
) -> None:
    """Write ``labels.csv`` and ``splits/{train,val,test}.txt``."""
    all_dfs = []
    for split_name in SPLIT_NAMES:
        sdf = splits[split_name].copy()
        sdf["split"] = split_name
        all_dfs.append(sdf)
    df_all = pd.concat(all_dfs, ignore_index=True)

    # labels.csv: filename + one column per class (multi-hot)
    labels_path = dest_dir / "labels.csv"
    df_all[["filename"] + class_names].to_csv(labels_path, index=False)
    logger.info("Wrote %s (%d rows)", labels_path, len(df_all))

    # splits/*.txt: one filename per line
    splits_dir = dest_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        fnames = splits[split_name]["filename"].tolist()
        (splits_dir / f"{split_name}.txt").write_text(
            "\n".join(fnames) + "\n"
        )
        logger.info("Wrote splits/%s.txt (%d files)", split_name, len(fnames))


def _print_multilabel_summary(
    splits: Dict[str, pd.DataFrame],
    class_names: List[str],
) -> None:
    """Print a class x split summary table for multi-label data."""
    print()
    print("=" * 80)
    header = f"{'CLASS':<28}"
    for s in SPLIT_NAMES:
        header += f" | {s.upper():>7}"
    header += f" | {'TOTAL':>7}"
    print(header)
    print("-" * 80)

    for cls in class_names:
        row = f"{cls:<28}"
        total = 0
        for s in SPLIT_NAMES:
            n = int(splits[s][cls].sum())
            total += n
            row += f" | {n:>7}"
        row += f" | {total:>7}"
        print(row)

    print("-" * 80)
    row = f"{'IMAGES':<28}"
    grand = 0
    for s in SPLIT_NAMES:
        n = len(splits[s])
        grand += n
        row += f" | {n:>7}"
    row += f" | {grand:>7}"
    print(row)

    if grand:
        row = f"{'%':<28}"
        for s in SPLIT_NAMES:
            pct = 100 * len(splits[s]) / grand
            row += f" | {pct:>6.1f}%"
        row += " |"
        print(row)
    print("=" * 80)

    # Multi-label statistics
    all_df = pd.concat(splits.values(), ignore_index=True)
    n_labels = all_df[class_names].sum(axis=1)
    print("\nLabels per image:")
    for k in sorted(n_labels.unique()):
        cnt = int((n_labels == k).sum())
        print(f"  {int(k)} label(s): {cnt} images ({cnt / len(all_df) * 100:.1f}%)")

    disease_cols = [c for c in class_names if c.lower() not in NORMAL_ALIASES]
    if len(disease_cols) > 1:
        n_multi = int((all_df[disease_cols].sum(axis=1) > 1).sum())
        print(f"\nMulti-disease images: {n_multi} ({n_multi / len(all_df) * 100:.1f}%)")
    print()


# ===================================================================
# Legacy folder-based pipeline (single-label, backward compat)
# ===================================================================

def _discover_classes(source_dir: Path) -> List[str]:
    """Auto-discover class names from immediate subdirectories."""
    classes = sorted(
        d.name
        for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise ValueError(
            f"No class subdirectories found in {source_dir}. "
            "Expected one folder per class (e.g. normal/, pneumonia/, ...)."
        )
    return classes


def _detect_layout(source_dir: Path, classes: List[str]) -> str:
    """Return 'pre-split' if every class folder contains train/val/test."""
    for cls in classes:
        child_names = {d.name for d in (source_dir / cls).iterdir() if d.is_dir()}
        if not set(SPLIT_NAMES).issubset(child_names):
            return "flat"
    return "pre-split"


def _collect_images_flat(
    source_dir: Path, classes: List[str],
) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    labels: List[str] = []
    for cls in classes:
        for f in sorted((source_dir / cls).iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                paths.append(f)
                labels.append(cls)
    if not paths:
        raise ValueError(f"No images found in {source_dir}")
    return paths, labels


def _collect_images_presplit(
    source_dir: Path, classes: List[str],
) -> Dict[str, List[Tuple[Path, str]]]:
    splits: Dict[str, List[Tuple[Path, str]]] = {s: [] for s in SPLIT_NAMES}
    for cls in classes:
        for split_name in SPLIT_NAMES:
            split_dir = source_dir / cls / split_name
            if not split_dir.exists():
                continue
            for f in sorted(split_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    splits[split_name].append((f, cls))
    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise ValueError(f"No images found in {source_dir}")
    return splits


def _stratified_split_folder(
    paths: List[Path],
    labels: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, str]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels,
        test_size=val_ratio + test_ratio, stratify=labels, random_state=seed,
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l,
        test_size=1 - val_frac, stratify=temp_l, random_state=seed,
    )
    return {
        "train": list(zip(train_p, train_l)),
        "val": list(zip(val_p, val_l)),
        "test": list(zip(test_p, test_l)),
    }


def _transfer_files_folder(
    splits: Dict[str, List[Tuple[Path, str]]],
    dest_root: Path,
    use_copy: bool,
) -> Tuple[int, int]:
    succeeded = skipped = 0
    for split_name, items in splits.items():
        for src, cls in tqdm(items, desc=f"Transferring {split_name}"):
            dest_dir = dest_root / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
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
                logger.warning("Transfer failed %s -> %s: %s", src, dest, exc)
                skipped += 1
    return succeeded, skipped


def _print_folder_summary(
    splits: Dict[str, List[Tuple[Path, str]]],
    classes: List[str],
) -> None:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, items in splits.items():
        for _, cls in items:
            counts[cls][split_name] += 1
    print()
    print("=" * 74)
    print(f"{'CLASS':<24} | {'TRAIN':>7} | {'VAL':>7} | {'TEST':>7} | {'TOTAL':>7}")
    print("-" * 74)
    for cls in classes:
        c = counts[cls]
        total = c["train"] + c["val"] + c["test"]
        print(
            f"{cls:<24} | {c['train']:>7} | {c['val']:>7} | "
            f"{c['test']:>7} | {total:>7}"
        )
    print("=" * 74)


# ===================================================================
# CLI entry point
# ===================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", type=Path, required=True, metavar="DIR",
        help="Path to source data directory.",
    )
    p.add_argument(
        "--dest", type=Path, default=DEST_RAW, metavar="DIR",
        help=f"Output directory (default: {DEST_RAW}).",
    )
    p.add_argument(
        "--source-format", choices=["csv", "folder"], default="csv",
        help="Source data format: 'csv' (multi-label, default) or 'folder' (legacy single-label).",
    )

    # CSV-specific arguments
    csv_grp = p.add_argument_group("CSV source options")
    csv_grp.add_argument(
        "--label-csv", type=str, default="train.csv", metavar="FILE",
        help="Annotation CSV filename inside --source (default: train.csv).",
    )
    csv_grp.add_argument(
        "--image-dir", type=str, default="train", metavar="DIR",
        help="Image subdirectory inside --source (default: train).",
    )
    csv_grp.add_argument(
        "--min-rads", type=int, default=1, metavar="N",
        help="Minimum radiologists that must agree for a positive label "
             "(default: 1 = OR/any; 2 = majority vote for 3-rater datasets).",
    )

    # Split ratios
    split_grp = p.add_argument_group("Split ratios")
    split_grp.add_argument("--train-ratio", type=float, default=0.7, metavar="R")
    split_grp.add_argument("--val-ratio", type=float, default=0.15, metavar="R")
    split_grp.add_argument("--test-ratio", type=float, default=0.15, metavar="R")
    split_grp.add_argument("--seed", type=int, default=42, metavar="S",
                           help="Random seed (default: 42).")

    # File transfer
    p.add_argument(
        "--copy", action="store_true", default=False,
        help="Copy files instead of symlinks (slower, more disk).",
    )
    p.add_argument(
        "--mode", choices=["append", "replace"], default="append",
        help="append (default): skip existing files. replace: wipe dest first.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Aborting: source directory does not exist: {args.source}")

    if args.mode == "replace" and args.dest.exists():
        logger.warning("--mode replace: removing %s ...", args.dest)
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # CSV-based multi-label pipeline
    # ---------------------------------------------------------------
    if args.source_format == "csv":
        pipeline_config = PipelineConfig.from_yaml(SYSTEM_CONFIG)
        cfg = _load_system_config()
        class_names_raw = cfg.get("model", {}).get("class_names", [])
        
        # Use the RAW class names from system.yaml (including "No finding")
        # rather than PipelineConfig which filters "No finding" for multilabel.
        # organize_data.py needs "No finding" for normal-class subsampling and
        # split outputs.
        class_names = class_names_raw if class_names_raw else _get_class_names_from_config(pipeline_config)
        normal_class = _identify_normal_class(class_names_raw)

        logger.info("Class names (from config): %s", class_names)
        logger.info("Normal class:              %s", normal_class)

        # Build image index
        image_dir = args.source / args.image_dir
        image_index = _build_image_index(image_dir)

        # Build multi-label dataframe from source CSV
        label_csv_path = args.source / args.label_csv
        df, ambiguous_ids = _build_multilabel_dataframe(
            label_csv_path, class_names, normal_class, image_index,
            min_rads=args.min_rads,
        )

        # Save ambiguous images to a separate directory for review
        if ambiguous_ids:
            ambiguous_dir = args.dest.parent / "ambiguous_img"
            ambiguous_dir.mkdir(parents=True, exist_ok=True)
            n_copied = 0
            for img_id in ambiguous_ids:
                src = image_index.get(img_id)
                if src is not None:
                    dest_file = ambiguous_dir / src.name
                    if not dest_file.exists():
                        try:
                            shutil.copy2(src, dest_file) if args.copy else os.symlink(src.resolve(), dest_file)
                            n_copied += 1
                        except Exception as exc:
                            logger.warning("Failed to save ambiguous image %s: %s", src.name, exc)
            logger.info(
                "Saved %d ambiguous images (out of %d) to %s",
                n_copied, len(ambiguous_ids), ambiguous_dir,
            )

        logger.info("Total images (no downsampling): %d", len(df))

        # Guard against duplicate filenames
        dups = df["filename"].duplicated()
        if dups.any():
            logger.warning("%d duplicate filenames removed.", dups.sum())
            df = df[~dups].reset_index(drop=True)

        # Multi-label stratified split
        splits = _multilabel_stratified_split(
            df, class_names,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
        )
        for s in SPLIT_NAMES:
            logger.info("Split %s: %d images", s, len(splits[s]))

        # Transfer images to data/raw/images/
        dest_images = args.dest / "images"
        succeeded, skipped = _transfer_images(df, image_index, dest_images, args.copy)
        if skipped:
            logger.warning("%d images skipped (transfer errors).", skipped)
        logger.info("Transfer complete: %d succeeded, %d skipped.", succeeded, skipped)

        # Write labels.csv and splits/*.txt
        _write_outputs(splits, class_names, args.dest)

        # Summary
        _print_multilabel_summary(splits, class_names)

    # ---------------------------------------------------------------
    # Legacy folder-based pipeline (single-label)
    # ---------------------------------------------------------------
    else:
        classes = _discover_classes(args.source)
        logger.info("Discovered %d classes: %s", len(classes), classes)

        layout = _detect_layout(args.source, classes)
        logger.info("Detected source layout: %s", layout)

        if layout == "pre-split":
            splits = _collect_images_presplit(args.source, classes)
        else:
            paths, labels = _collect_images_flat(args.source, classes)
            logger.info(
                "Collected %d images across %d classes.", len(paths), len(classes),
            )
            splits = _stratified_split_folder(
                paths, labels,
                args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
            )

        for s in SPLIT_NAMES:
            logger.info("Split %s: %d items", s, len(splits[s]))

        succeeded, skipped = _transfer_files_folder(splits, args.dest, args.copy)
        if skipped:
            logger.warning("%d images skipped.", skipped)
        logger.info("Transfer complete: %d succeeded, %d skipped.", succeeded, skipped)

        _print_folder_summary(splits, classes)

        # Post-hoc config check
        try:
            cfg = _load_system_config()
            pipeline_config = PipelineConfig.from_yaml(SYSTEM_CONFIG)
            cfg_classes = {c.lower() for c in pipeline_config.class_names}
            disc_classes = {c.lower() for c in classes}
            if cfg_classes != disc_classes:
                logger.warning(
                    "Discovered classes %s differ from system.yaml %s.",
                    sorted(disc_classes), sorted(cfg_classes),
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()

