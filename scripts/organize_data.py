"""organize_data.py - Standalone data organisation utility.

NOT part of the main training pipeline.  Run this script *before* training
to populate ``data/raw/{train,val,test}/<class>/`` from any external source
directory organised by class.

Two source layouts are auto-detected:

  A) Flat — images directly inside each class folder (split performed
     automatically via stratified sampling)::

        /path/to/my_data/
            normal/
                img001.png
            pneumonia/
                img002.png

  B) Pre-split — each class folder already contains train/val/test
     sub-folders (files are transferred as-is, no re-splitting)::

        /path/to/my_data/
            normal/
                train/
                    img001.png
                val/
                    img002.png
                test/
                    img003.png
            pneumonia/
                train/
                    img004.png
                ...

Classes are auto-discovered from sub-folder names — nothing is hardcoded.

Modes
-----
  --mode append   (default) Add images to data/raw/, skip files that already
                  exist.  Use this to add more data from a new source
                  while keeping what is already there.
  --mode replace  Wipe data/raw/ first, then populate from scratch.

Usage
-----
  python scripts/organize_data.py --source /path/to/my_data
  python scripts/organize_data.py --source /path/to/my_data --mode replace
  python scripts/organize_data.py --source /path/to/my_data --copy
  python scripts/organize_data.py --source /path/to/my_data --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
  python scripts/organize_data.py --source /path/to/my_data --seed 123
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

SPLIT_NAMES = {"train", "val", "test"}

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dcm"}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

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
            "Expected one folder per class (e.g. normal/, pneumonia/, …)."
        )
    return classes



def _detect_layout(source_dir: Path, classes: List[str]) -> str:
    """Return 'pre-split' if every class folder contains train/val/test, else 'flat'."""
    for cls in classes:
        cls_dir = source_dir / cls
        child_names = {d.name for d in cls_dir.iterdir() if d.is_dir()}
        if not SPLIT_NAMES.issubset(child_names):
            return "flat"
    return "pre-split"


def _collect_images_flat(
    source_dir: Path,
    classes: List[str],
) -> Tuple[List[Path], List[str]]:
    """Collect images from a flat layout (class/image)."""
    paths: List[Path] = []
    labels: List[str] = []
    for cls in classes:
        cls_dir = source_dir / cls
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                paths.append(f)
                labels.append(cls)
    if not paths:
        raise ValueError(f"No images found in {source_dir}")
    return paths, labels


def _collect_images_presplit(
    source_dir: Path,
    classes: List[str],
) -> Dict[str, List[Tuple[Path, str]]]:
    """Collect images from a pre-split layout (class/split/image)."""
    splits: Dict[str, List[Tuple[Path, str]]] = {
        s: [] for s in SPLIT_NAMES
    }
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

def _stratified_split(
    paths: List[Path],
    labels: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, str]]]:
    """Stratified train / val / test split.

    Returns a dict mapping split name -> list of (path, class_label).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    val_frac = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_frac),
        stratify=temp_labels,
        random_state=seed,
    )

    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }

# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------

def _transfer_files(
    splits: Dict[str, List[Tuple[Path, str]]],
    dest_root: Path,
    use_copy: bool,
) -> Tuple[int, int]:
    """Copy or symlink files into dest_root/{split}/{class}/."""
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    splits: Dict[str, List[Tuple[Path, str]]],
    classes: List[str],
) -> None:
    """Print a class x split summary table."""
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
    print("-" * 74)

    overall = {"train": 0, "val": 0, "test": 0}
    for split_name, items in splits.items():
        overall[split_name] = len(items)
    grand = sum(overall.values())
    print(
        f"{'TOTAL':<24} | {overall['train']:>7} | {overall['val']:>7} | "
        f"{overall['test']:>7} | {grand:>7}"
    )
    if grand:
        pct = {s: 100 * n / grand for s, n in overall.items()}
        print(
            f"{'%':<24} | {pct['train']:>6.1f}% | {pct['val']:>6.1f}% | "
            f"{pct['test']:>6.1f}% |"
        )
    print("=" * 74)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Path to source data directory. Must contain one sub-folder per "
            "class, each holding image files (e.g. source/normal/, "
            "source/pneumonia/)."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEST_RAW,
        metavar="DIR",
        help=f"Output directory. Default: {DEST_RAW}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        metavar="R",
        help="Fraction of data for training. Default: 0.7.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        metavar="R",
        help="Fraction of data for validation. Default: 0.15.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        metavar="R",
        help="Fraction of data for testing. Default: 0.15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for reproducible splits. Default: 42.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy files instead of symlinks (slower, uses more disk).",
    )
    parser.add_argument(
        "--mode",
        choices=["append", "replace"],
        default="append",
        help=(
            "append: add images to existing data/raw/, skip duplicates. "
            "replace: wipe data/raw/ first, then populate from scratch. "
            "Default: append."
        ),
    )
    return parser.parse_args()


def _validate_source(source: Path) -> None:
    """Fail fast if the source directory is missing or empty."""
    if not source.exists():
        raise SystemExit(f"Aborting: source directory does not exist: {source}")
    if not source.is_dir():
        raise SystemExit(f"Aborting: source path is not a directory: {source}")
    subdirs = [d for d in source.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        raise SystemExit(
            f"Aborting: no class sub-folders found in {source}. "
            "Expected one folder per class (e.g. normal/, pneumonia/, …)."
        )


def _apply_replace_mode(dest: Path) -> None:
    """Wipe the destination directory so it can be rebuilt from scratch."""
    if dest.exists():
        logger.warning("--mode replace: removing %s …", dest)
        shutil.rmtree(dest)
        logger.info("Removed %s.", dest)
    dest.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = _parse_args()

    _validate_source(args.source)

    if args.mode == "replace": 
        _apply_replace_mode(args.dest)

    classes = _discover_classes(args.source)
    logger.info("Discovered %d classes: %s", len(classes), classes)

    layout = _detect_layout(args.source, classes)
    logger.info("Detected source layout: %s", layout)

    if layout == "pre-split":
        splits = _collect_images_presplit(args.source, classes)
        logger.info(
            "Pre-split data: %d train, %d val, %d test.",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
    else:
        paths, labels = _collect_images_flat(args.source, classes)
        logger.info("Collected %d images across %d classes.", len(paths), len(classes))
        splits = _stratified_split(
            paths,
            labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        logger.info(
            "Split: %d train, %d val, %d test.",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

    # Transfer files
    succeeded, skipped = _transfer_files(splits, args.dest, args.copy)
    if skipped: 
        logger.warning("%d images skipped (transfer errors).", skipped)
    logger.info("Transfer complete: %d succeeded, %d skipped.", succeeded, skipped)

    _print_summary(splits, classes)

    # Cross-check discovered class names against system.yaml so mismatches are
    # caught here rather than silently failing during train.py manifest loading.
    try:
        import yaml as _yaml
        _sys_cfg = PROJECT_ROOT / "configs" / "system.yaml"
        if _sys_cfg.exists():
            _cfg_classes = {
                c.lower()
                for c in (_yaml.safe_load(_sys_cfg.open()) or {})
                .get("model", {})
                .get("class_names", [])
            }
            _disc_classes = {c.lower() for c in classes}
            if _cfg_classes and _cfg_classes != _disc_classes:
                logger.warning(
                    "Discovered classes %s differ from system.yaml class_names %s. "
                    "Update configs/system.yaml before running train.py, otherwise "
                    "ChestXrayDataset will raise a label-mapping error.",
                    sorted(_disc_classes),
                    sorted(_cfg_classes),
                )
    except Exception:
        pass  # cross-check is best-effort; never block data organisation
