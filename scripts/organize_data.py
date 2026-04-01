"""organize_data.py -- Config-driven dataset organisation for classification.

Takes data from a source directory and organises it into ``data/raw/``
with the following structure depending on classification mode:

For multi-label with CSV annotations::

    data/raw/
        images/           # all image files
        labels.csv        # multi-hot encoded labels (one row per image)
        splits/
            train.txt     # one filename per line
            val.txt
            test.txt

For multi-class with folder structure::

    data/raw/
        train/
            class1/
            class2/
        val/
            class1/
            class2/
        test/
            class1/
            class2/
    or 
    data/raw/
        images/
        labels.csv        # with 'split' column indicating train/val/test
        splits/
            train.txt     # one filename per line
            val.txt
            test.txt

Two source formats are supported via ``--source-format``:

  csv   (default)  Source contains a CSV annotation file and an image
        directory.  Column names are fully configurable via CLI.

  folder  Source is organised as ``class/image`` (flat) or ``train/class/image`` (pre-split).

Architecture
------------
``BaseOrganizer`` → ``CSVOrganizer`` / ``FolderOrganizer``

Patient-level splitting is enforced whenever a patient identifier is
available (``--col-patient-id`` for CSV, ``--patient-id-regex`` for
folder mode).  This **prevents data leakage** by ensuring that all
images from the same patient appear in exactly one split.

Configuration
-------------
Class names and classification mode are read from ``configs/system.yaml``::

    model:
      class_names: [No finding, Cardiomegaly, Aortic enlargement, ...]
      classification_mode: multilabel   # or multiclass

All normal aliases (Normal, healthy, no_finding, …) are strictly
normalised to the string ``'no finding'`` in every output artefact.

Usage
-----
  # CSV source with patient-level splitting
  python scripts/organize_data.py \\
      --source /path/to/dataset \\
      --label-csv train.csv --image-dir train \\
      --col-image-id image_id --col-class-name class_name \\
      --col-patient-id patient_id --col-rad-id rad_id \\
      --min-rads 2 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

  # Folder source with patient-ID regex
  python scripts/organize_data.py \\
      --source /path/to/data --source-format folder \\
      --patient-id-regex "^(patient\\d+)" --seed 42
"""

from __future__ import annotations

import abc
import argparse
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")

IMAGE_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom",
}

NORMAL_ALIASES: frozenset = frozenset({
    "no finding", "normal", "no_finding", "healthy", "no-finding",
    "nofinding", "no findings", "none", "no abnormality", "no abnormality detected",
})

BBOX_COLUMNS: List[str] = [
    "x_min", "y_min", "x_max", "y_max",
    "xmin", "ymin", "xmax", "ymax",
    "x", "y", "w", "h", "width", "height",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]

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
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DEST_RAW = PROJECT_ROOT / "data" / "raw"
SYSTEM_CONFIG = PROJECT_ROOT / "configs" / "system.yaml"

# ===================================================================
# Base organizer
# ===================================================================
class BaseOrganizer(abc.ABC):
    """Common initialisation shared by CSV and folder organisers.

    Handles argument parsing, config loading, output directory creation,
    seed management, and iterative-stratification.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seed: int = args.seed
        self.source_dir: Path = args.source
        self.dest_dir: Path = args.dest
        self.train_ratio: float = args.train_ratio
        self.val_ratio: float = args.val_ratio
        self.test_ratio: float = args.test_ratio
        self.use_copy: bool = args.copy

        # Load system config ------------------------------------------------
        self.config: Dict[str, Any] = self._load_system_config()
        self.classification_mode: str = self.config.get("model", {}).get(
            "classification_mode", "multilabel"
        )
        if self.classification_mode not in ("multilabel", "multiclass"):
            raise ValueError(
                f"Invalid classification_mode '{self.classification_mode}' in config. "
                "Must be 'multilabel' or 'multiclass'."
            )

        # Normalise class names from config ---------------------------------
        self.class_names: List[str] = self._get_class_names()
        self.disease_names: List[str] = [
            c for c in self.class_names if c != "no finding"
        ]
        self.has_normal_class: bool = "no finding" in self.class_names

        logger.info("Classification mode : %s", self.classification_mode)
        logger.info("Class names         : %s", self.class_names)
        logger.info("Normal class present: %s", self.has_normal_class)

        # Prepare destination ------------------------------------------------
        if args.mode == "replace" and self.dest_dir.exists():
            logger.warning("--mode replace: removing %s …", self.dest_dir)
            shutil.rmtree(self.dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        self._validate_ratios()

    # ------------------------------------------------------------------ config
    def _load_system_config(self) -> Dict[str, Any]:
        """Load and return the full ``system.yaml`` config dict."""
        if not SYSTEM_CONFIG.exists():
            raise FileNotFoundError(
                f"System config not found: {SYSTEM_CONFIG}. "
                "Create configs/system.yaml with model.class_names."
            )
        with open(SYSTEM_CONFIG, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg

    def _get_class_names(self) -> List[str]:
        """Extract class names from config, normalising normal aliases.

        Any class name matching ``NORMAL_ALIASES`` is replaced with the
        exact string ``'no finding'``.  Duplicates are collapsed.
        """
        raw_names: list = self.config.get("model", {}).get("class_names", [])
        if not isinstance(raw_names, list) or len(raw_names) < 2:
            raise ValueError(
                "model.class_names in system.yaml must be a list with >= 2 entries. "
                f"Got: {raw_names}"
            )
        normalised: List[str] = []
        seen_normal = False
        for name in raw_names:
            if name.strip().lower() in NORMAL_ALIASES:
                if not seen_normal:
                    normalised.append("no finding")
                    seen_normal = True
                # skip duplicate normal aliases
            else:
                normalised.append(name)
        return normalised

    @staticmethod
    def normalize_class_label(name: str) -> str:
        """Normalise a single class label: any normal alias → ``'no finding'``."""
        if name.strip().lower() in NORMAL_ALIASES:
            return "no finding"
        return name.strip()

    # -------------------------------------------------------------- validation
    def _validate_ratios(self) -> None:
        """Ensure that train/val/test ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    # -------------------------------------------------------- file transfer
    def _transfer_file(self, src: Path, dest: Path) -> bool:
        """Returns True if transferred/exists, False on failure."""
        if dest.exists() or dest.is_symlink():
            return True
        try:
            if self.use_copy:
                shutil.copy2(src, dest)
            else:
                os.symlink(os.path.relpath(src, dest.parent), dest)
            return True
        except Exception as exc:
            logger.warning("Transfer failed %s → %s: %s", src, dest, exc)
            return False

    # -------------------------------------------------------- stratification
    @staticmethod
    def iterative_stratification(
        y: np.ndarray,
        ratios: List[float],
        seed: int,
    ) -> List[np.ndarray]:
        """Iterative stratification for multi-label data (Sechidis et al., 2011).

        Distributes samples across *k* folds so that each label's proportion
        is as close as possible to *ratios*.  Uses only numpy.

        Parameters
        ----------
        y : ndarray of shape ``(n_samples, n_labels)``, binary.
        ratios : list of *k* floats that sum to 1.0.
        seed : random seed for tie-breaking.

        Returns
        -------
        List of *k* arrays with sample indices for each fold.
        """
        rng = np.random.RandomState(seed)
        n_samples, n_labels = y.shape
        k = len(ratios)
        ratios_arr = np.asarray(ratios, dtype=np.float64)

        label_totals = y.sum(axis=0)
        desired = np.outer(ratios_arr, label_totals)  # (k, n_labels)

        folds: List[List[int]] = [[] for _ in range(k)]
        fold_label_counts = np.zeros((k, n_labels), dtype=np.float64)
        remaining = np.ones(n_samples, dtype=bool)

        # Process labels from rarest to most common
        label_order = np.argsort(label_totals)

        for label_idx in label_order:
            positives = np.where(remaining & (y[:, label_idx] == 1))[0]
            rng.shuffle(positives)
            for sample_idx in positives:
                needs = desired[:, label_idx] - fold_label_counts[:, label_idx]
                best_fold = int(np.argmax(needs))
                folds[best_fold].append(sample_idx)
                fold_label_counts[best_fold] += y[sample_idx]
                remaining[sample_idx] = False

        # Distribute remaining samples (all-zero rows) proportionally
        leftover = np.where(remaining)[0]
        if len(leftover) > 0:
            rng.shuffle(leftover)
            targets = (ratios_arr * len(leftover)).astype(int)
            diff = len(leftover) - targets.sum()
            for i in range(abs(diff)):
                targets[i % k] += 1 if diff > 0 else -1
            ptr = 0
            for f in range(k):
                folds[f].extend(leftover[ptr : ptr + targets[f]].tolist())
                ptr += targets[f]

        return [np.array(fold, dtype=int) for fold in folds]

    # --------------------------------------------------------- abstract
    @abc.abstractmethod
    def run(self) -> None:
        """Execute the data-organisation pipeline."""


# ===================================================================
# CSV-based organizer (multi-label or multiclass from CSV + images)
# ===================================================================
class CSVOrganizer(BaseOrganizer):
    """Organise a CSV-annotated dataset.

    Supports both **multilabel** and **multiclass** classification modes.
    Patient-level splitting is enforced when ``--col-patient-id`` is
    provided and the column exists in the CSV.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.col_image_id: str = args.col_image_id
        self.col_class_name: str = args.col_class_name
        self.col_patient_id: Optional[str] = args.col_patient_id
        self.col_rad_id: Optional[str] = args.col_rad_id
        self.min_rads: int = args.min_rads
        self.label_csv_name: str = args.label_csv
        self.image_dir_name: str = args.image_dir
        self.image_index: Dict[str, Path] = {}

    # ---------------------------------------------------------------- public
    def run(self) -> None:
        """Full CSV pipeline: load → encode → split → write."""
        # 1. Build image index
        image_dir = self.source_dir / self.image_dir_name
        self.image_index = self._build_image_index(image_dir)

        # 2. Load & clean CSV
        csv_path = self.source_dir / self.label_csv_name
        df_raw = self._load_csv(csv_path)

        # 3. Build multi-hot DataFrame
        df_multi, ambiguous_ids = self._build_multilabel_df(df_raw)

        # 4. Save ambiguous images for manual review
        if ambiguous_ids:
            self._save_ambiguous(ambiguous_ids)

        logger.info("Total images after processing: %d", len(df_multi))

        # 5. Remove duplicate filenames
        dups = df_multi["filename"].duplicated()
        if dups.any():
            logger.warning("%d duplicate filenames removed.", dups.sum())
            df_multi = df_multi[~dups].reset_index(drop=True)

        # 6. Classification-mode validation
        if self.classification_mode == "multiclass":
            self._validate_multiclass(df_multi)

        # 7. Determine stratification columns (use diseases only)
        strat_cols = self.disease_names if self.disease_names else self.class_names

        # 8. Split — patient-level or image-level
        has_patient_col = (
            self.col_patient_id is not None
            and self.col_patient_id in df_multi.columns
        )
        if has_patient_col:
            logger.info(
                "Patient-level splitting on column '%s'.", self.col_patient_id
            )
            splits = self._patient_level_split(df_multi, strat_cols)
        else:
            if self.col_patient_id:
                logger.warning(
                    "Patient-ID column '%s' not found — "
                    "falling back to image-level splitting.",
                    self.col_patient_id,
                )
            splits = self._image_level_split(df_multi, strat_cols)

        for s in SPLIT_NAMES:
            logger.info("Split %s: %d images", s, len(splits[s]))

        # 9. Write outputs
        if self.classification_mode == "multilabel":
            self._transfer_images(df_multi)
            self._write_multilabel_output(splits)
            self._print_multilabel_summary(splits)
        else:
            self._write_multiclass_output(splits)
            self._print_multiclass_summary(splits)

    # ---------------------------------------------------------- image index
    def _build_image_index(self, image_dir: Path) -> Dict[str, Path]:
        """Build ``{key: filepath}`` from all images in *image_dir*.

        Each file is indexed both by its stem (``abc``) and its full
        name (``abc.jpg``) so that CSV image-IDs with or without
        extensions are resolved.
        """
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        index: Dict[str, Path] = {}
        for f in image_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                index[f.stem] = f
                index[f.name] = f  # also by full filename

        n_unique = len({id(v) for v in index.values()})
        if n_unique == 0:
            raise ValueError(f"No images found in {image_dir}")
        logger.info("Image index: %d files in %s", n_unique, image_dir)
        return index

    # ---------------------------------------------------------- CSV loading
    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load the annotation CSV, validate columns, drop bbox columns."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(
            "Loaded CSV: %d rows, columns: %s", len(df), df.columns.tolist()
        )

        # Validate required columns
        for label, col in [
            ("image-ID", self.col_image_id),
            ("class-name", self.col_class_name),
        ]:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' ({label}) not found in CSV. "
                    f"Available columns: {df.columns.tolist()}"
                )

        # Drop bounding-box columns early to save memory
        bbox_present = [c for c in BBOX_COLUMNS if c in df.columns]
        if bbox_present:
            df.drop(columns=bbox_present, inplace=True)
            logger.info("Dropped bounding-box columns: %s", bbox_present)

        # Drop rows missing required fields
        df = df.dropna(subset=[self.col_image_id, self.col_class_name])

        return df

    # ------------------------------------------------ multi-hot encoding
    def _build_multilabel_df(
        self, df_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build a multi-hot-encoded DataFrame from raw CSV annotations.

        Returns
        -------
        (df_multi, ambiguous_ids)
            ``df_multi`` has columns
            ``[filename, image_id, <patient_id?>, <class1>, <class2>, …]``.
            ``ambiguous_ids`` lists image IDs dropped by majority-vote.
        """
        col_img = self.col_image_id
        col_rad = self.col_rad_id

        # --- class-name mapping (source → normalised config name) ----------
        source_to_config: Dict[str, str] = {}
        for name in self.class_names:
            source_to_config[name.lower()] = name
        # Map every normal alias as well
        for alias in NORMAL_ALIASES:
            source_to_config[alias] = "no finding"

        df_raw = df_raw.copy()
        df_raw["class_mapped"] = (
            df_raw[self.col_class_name]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(source_to_config)
        )

        # Keep only annotations that map to a configured class
        df_filtered = df_raw[df_raw["class_mapped"].notna()].copy()
        n_dropped = len(df_raw) - len(df_filtered)
        if n_dropped > 0:
            dropped_names = (
                df_raw.loc[df_raw["class_mapped"].isna(), self.col_class_name]
                .unique()
            )
            logger.info(
                "Filtered %d annotations not in class_names. Dropped: %s",
                n_dropped, sorted(set(str(d) for d in dropped_names)),
            )

        ambiguous_ids: List[str] = []
        disease_set = set(self.disease_names)

        # ---- majority-vote aggregation ------------------------------------
        if self.min_rads > 1 and col_rad and col_rad in df_filtered.columns:
            logger.info(
                "Majority-vote: requiring >= %d radiologists for positive.",
                self.min_rads,
            )
            disease_annots = df_filtered[
                df_filtered["class_mapped"].isin(disease_set)
            ]

            votes = (
                disease_annots
                .groupby([col_img, "class_mapped"])[col_rad]
                .nunique()
                .reset_index(name="n_rads")
            )

            images_with_any_disease = set(disease_annots[col_img].unique())
            consensus = votes[votes["n_rads"] >= self.min_rads]

            if len(consensus) > 0:
                consensus = consensus.copy()
                consensus["value"] = 1
                df_multi = consensus.pivot_table(
                    index=col_img,
                    columns="class_mapped",
                    values="value",
                    fill_value=0,
                    aggfunc="max",
                ).reset_index()
            else:
                df_multi = pd.DataFrame(columns=[col_img])

            consensus_ids = set(df_multi[col_img]) if len(df_multi) else set()
            ambiguous_ids = sorted(images_with_any_disease - consensus_ids)

            all_ids = set(df_filtered[col_img].unique())
            truly_normal = all_ids - images_with_any_disease
            if truly_normal:
                df_normal = pd.DataFrame({col_img: sorted(truly_normal)})
                df_multi = pd.concat([df_multi, df_normal], ignore_index=True)

            logger.info(
                "Majority-vote result: %d consensus, %d normal, %d ambiguous (dropped).",
                len(consensus_ids), len(truly_normal), len(ambiguous_ids),
            )
        # ---- OR aggregation (default) -------------------------------------
        else:
            if self.min_rads > 1:
                logger.warning(
                    "min_rads=%d but rad_id column '%s' not found; "
                    "using OR aggregation.",
                    self.min_rads, col_rad,
                )
            df_dummies = pd.get_dummies(df_filtered["class_mapped"]).astype(int)
            df_encoded = pd.concat(
                [df_filtered[[col_img]].reset_index(drop=True), df_dummies], axis=1,
            )
            df_multi = df_encoded.groupby(col_img).max().reset_index()

        # ---- ensure all class columns exist --------------------------------
        for name in self.class_names:
            if name not in df_multi.columns:
                df_multi[name] = 0
            else:
                df_multi[name] = df_multi[name].fillna(0).astype(int)

        # Compute 'no finding' as complement of disease columns
        if self.has_normal_class:
            df_multi["no finding"] = (
                df_multi[self.disease_names].sum(axis=1) == 0
            ).astype(int)

        # ---- resolve filenames from image index ----------------------------
        filenames: List[Optional[str]] = []
        found_mask: List[bool] = []
        for img_id in df_multi[col_img]:
            key = str(img_id)
            path = self.image_index.get(key)
            if path is None:
                # try without extension
                path = self.image_index.get(key.rsplit(".", 1)[0])
            if path is not None:
                filenames.append(path.name)
                found_mask.append(True)
            else:
                filenames.append(None)
                found_mask.append(False)

        n_missing = sum(not m for m in found_mask)
        if n_missing > 0:
            logger.warning(
                "%d images in CSV not found on disk (skipped).", n_missing,
            )

        df_multi["filename"] = filenames
        df_multi = df_multi[pd.Series(found_mask).values].reset_index(drop=True)

        # ---- carry over patient_id from raw data ---------------------------
        if self.col_patient_id and self.col_patient_id in df_raw.columns:
            patient_map = (
                df_raw
                .groupby(col_img)[self.col_patient_id]
                .first()
                .to_dict()
            )
            df_multi[self.col_patient_id] = (
                df_multi[col_img].map(patient_map)
            )

        # ---- standardise image-ID column name ------------------------------
        if col_img != "image_id":
            df_multi = df_multi.rename(columns={col_img: "image_id"})

        # ---- final column order --------------------------------------------
        base_cols = ["filename", "image_id"]
        if self.col_patient_id and self.col_patient_id in df_multi.columns:
            base_cols.append(self.col_patient_id)
        df_multi = df_multi[base_cols + self.class_names]

        logger.info(
            "Multi-hot DataFrame: %d images, %d classes.\n%s",
            len(df_multi),
            len(self.class_names),
            df_multi[self.class_names].sum().to_string(),
        )
        return df_multi, ambiguous_ids

    # ------------------------------------------------------------ validation
    def _validate_multiclass(self, df: pd.DataFrame) -> None:
        """Ensure every image has at most one active disease (multiclass)."""
        labels_per_row = df[self.disease_names].sum(axis=1)
        multi = labels_per_row > 1
        if multi.any():
            n_bad = int(multi.sum())
            examples = df.loc[multi, ["image_id"] + self.class_names].head(5)
            raise ValueError(
                f"Multiclass mode but {n_bad} images have >1 disease label.\n"
                f"First examples:\n{examples}\n"
                "Switch to multilabel mode or fix the annotations."
            )

    # --------------------------------------------------------- splitting
    def _patient_level_split(
        self, df: pd.DataFrame, strat_cols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Stratified split at the **patient** level to prevent data leakage.

        1. Group by ``patient_id`` and aggregate labels with ``.max()``.
        2. Run iterative stratification on the patient-level matrix.
        3. Map patient-level fold assignments back to image-level rows.
        """
        patient_col: str = self.col_patient_id  # type: ignore[assignment]

        # Step 1 — aggregate
        patient_df = (
            df.groupby(patient_col)[strat_cols].max().reset_index()
        )

        n_patients = len(patient_df)
        n_images = len(df)
        logger.info(
            "Patient-level split: %d patients, %d images (%.1f img/patient avg).",
            n_patients, n_images,
            n_images / n_patients if n_patients else 0,
        )

        # Multiclass guard: each patient must have <= 1 disease after agg
        if self.classification_mode == "multiclass":
            per_patient = patient_df[strat_cols].sum(axis=1)
            multi = per_patient > 1
            if multi.any():
                examples = patient_df.loc[multi].head(5)
                raise ValueError(
                    f"Multiclass: {int(multi.sum())} patients have >1 disease "
                    f"after aggregation.\nExamples:\n{examples}"
                )

        # Step 2 — stratify patients
        y_pat = patient_df[strat_cols].values.astype(int)
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        fold_indices = self.iterative_stratification(y_pat, ratios, self.seed)

        # Step 3 — map back to images
        splits: Dict[str, pd.DataFrame] = {}
        for i, split_name in enumerate(SPLIT_NAMES):
            pids = set(patient_df.iloc[fold_indices[i]][patient_col].values)
            mask = df[patient_col].isin(pids)
            splits[split_name] = df[mask].reset_index(drop=True)

        # Verify no leakage
        self._verify_no_leakage(splits, patient_col)
        return splits

    def _image_level_split(
        self, df: pd.DataFrame, strat_cols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Fallback: stratified split at the image level."""
        y = df[strat_cols].values.astype(int)
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        fold_indices = self.iterative_stratification(y, ratios, self.seed)

        logger.info(
            "Image-level split: %d train, %d val, %d test.",
            len(fold_indices[0]), len(fold_indices[1]), len(fold_indices[2]),
        )
        return {
            name: df.iloc[fold_indices[i]].reset_index(drop=True)
            for i, name in enumerate(SPLIT_NAMES)
        }

    @staticmethod
    def _verify_no_leakage(
        splits: Dict[str, pd.DataFrame], patient_col: str,
    ) -> None:
        """Assert that no patient appears in more than one split."""
        patient_sets: Dict[str, Set] = {}
        for name in SPLIT_NAMES:
            if patient_col in splits[name].columns:
                patient_sets[name] = set(splits[name][patient_col].dropna().unique())
            else:
                patient_sets[name] = set()

        for i, s1 in enumerate(SPLIT_NAMES):
            for s2 in SPLIT_NAMES[i + 1 :]:
                overlap = patient_sets[s1] & patient_sets[s2]
                if overlap:
                    raise RuntimeError(
                        f"DATA LEAKAGE DETECTED: {len(overlap)} patients in both "
                        f"'{s1}' and '{s2}'. First 5: {sorted(overlap)[:5]}"
                    )
        logger.info(
            "Leakage check PASSED — patients per split: %s",
            {n: len(ps) for n, ps in patient_sets.items()},
        )

    # -------------------------------------------------------- file transfer
    def _transfer_images(self, df: pd.DataFrame) -> None:
        """Copy / symlink images into ``dest/images/`` (multilabel mode)."""
        dest_images = self.dest_dir / "images"
        dest_images.mkdir(parents=True, exist_ok=True)
        succeeded = skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Transferring images"):
            src = self.image_index.get(str(row["image_id"]))
            if src is None:
                src = self.image_index.get(str(row["image_id"]).rsplit(".", 1)[0])
            if src is None:
                skipped += 1
                continue

            if self._transfer_file(src, dest_images / src.name):
                succeeded += 1
            else:
                skipped += 1

        if skipped:
            logger.warning("%d images skipped (transfer errors).", skipped)
        logger.info("Transfer: %d OK, %d skipped.", succeeded, skipped)

    # ------------------------------------------------------- output writers
    def _write_multilabel_output(
        self, splits: Dict[str, pd.DataFrame],
    ) -> None:
        """Write ``labels.csv`` and ``splits/{train,val,test}.txt``."""
        all_dfs = []
        for split_name in SPLIT_NAMES:
            sdf = splits[split_name].copy()
            sdf["split"] = split_name
            all_dfs.append(sdf)
        df_all = pd.concat(all_dfs, ignore_index=True)

        # labels.csv
        labels_path = self.dest_dir / "labels.csv"
        df_all[["filename"] + self.class_names].to_csv(labels_path, index=False)
        logger.info("Wrote %s (%d rows)", labels_path, len(df_all))

        # splits/*.txt
        splits_dir = self.dest_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name in SPLIT_NAMES:
            fnames = splits[split_name]["filename"].tolist()
            (splits_dir / f"{split_name}.txt").write_text(
                "\n".join(fnames) + "\n"
            )
            logger.info("Wrote splits/%s.txt (%d)", split_name, len(fnames))

    def _write_multiclass_output(
        self, splits: Dict[str, pd.DataFrame],
    ) -> None:
        """Write folder structure ``dest/{split}/{class}/img`` (multiclass)."""
        succeeded = skipped = 0

        for split_name in SPLIT_NAMES:
            df_split = splits[split_name]
            for _, row in tqdm(
                df_split.iterrows(),
                total=len(df_split),
                desc=f"Transferring {split_name}",
            ):
                # Determine the single active class
                active = [c for c in self.class_names if row.get(c, 0) == 1]
                if not active:
                    skipped += 1
                    continue
                cls_name = active[0]

                src = self.image_index.get(str(row["image_id"]))
                if src is None:
                    src = self.image_index.get(
                        str(row["image_id"]).rsplit(".", 1)[0]
                    )
                if src is None:
                    skipped += 1
                    continue

                dest_folder = self.dest_dir / split_name / cls_name
                dest_folder.mkdir(parents=True, exist_ok=True)

                if self._transfer_file(src, dest_folder / src.name):
                    succeeded += 1
                else:
                    skipped += 1

        if skipped:
            logger.warning("%d images skipped.", skipped)
        logger.info("Transfer: %d OK, %d skipped.", succeeded, skipped)

    # ----------------------------------------------------------- ambiguous
    def _save_ambiguous(self, ambiguous_ids: List[str]) -> None:
        """Copy / symlink ambiguous images to ``data/ambiguous_img/``."""
        ambiguous_dir = self.dest_dir.parent / "ambiguous_img"
        ambiguous_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for img_id in ambiguous_ids:
            src = self.image_index.get(img_id)
            if src is None:
                continue
            if self._transfer_file(src, ambiguous_dir / src.name):
                n_saved += 1
        logger.info(
            "Saved %d / %d ambiguous images to %s",
            n_saved, len(ambiguous_ids), ambiguous_dir,
        )

    # ------------------------------------------------------------ summaries
    def _print_multilabel_summary(
        self, splits: Dict[str, pd.DataFrame],
    ) -> None:
        """Print a class × split summary table (multilabel)."""
        print()
        print("=" * 80)
        header = f"{'CLASS':<28}"
        for s in SPLIT_NAMES:
            header += f" | {s.upper():>7}"
        header += f" | {'TOTAL':>7}"
        print(header)
        print("-" * 80)

        for cls in self.class_names:
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

        # Patient counts
        if self.col_patient_id:
            pcol = self.col_patient_id
            has_col = all(pcol in splits[s].columns for s in SPLIT_NAMES)
            if has_col:
                print("\nPatients per split:")
                for name in SPLIT_NAMES:
                    n = splits[name][pcol].nunique()
                    print(f"  {name}: {n} unique patients")

        # Labels-per-image distribution
        all_df = pd.concat(splits.values(), ignore_index=True)
        n_labels = all_df[self.class_names].sum(axis=1)
        print("\nLabels per image:")
        for k in sorted(n_labels.unique()):
            cnt = int((n_labels == k).sum())
            print(
                f"  {int(k)} label(s): {cnt} images "
                f"({cnt / len(all_df) * 100:.1f}%)"
            )

        if len(self.disease_names) > 1:
            n_multi = int((all_df[self.disease_names].sum(axis=1) > 1).sum())
            print(
                f"\nMulti-disease images: {n_multi} "
                f"({n_multi / len(all_df) * 100:.1f}%)"
            )
        print()

    def _print_multiclass_summary(
        self, splits: Dict[str, pd.DataFrame],
    ) -> None:
        """Print a class × split summary table (multiclass from CSV)."""
        print()
        print("=" * 80)
        print(
            f"{'CLASS':<28} | {'TRAIN':>7} | {'VAL':>7} "
            f"| {'TEST':>7} | {'TOTAL':>7}"
        )
        print("-" * 80)

        for cls in self.class_names:
            counts = {}
            for s in SPLIT_NAMES:
                counts[s] = (
                    int(splits[s][cls].sum()) if cls in splits[s].columns else 0
                )
            total = sum(counts.values())
            print(
                f"{cls:<28} | {counts['train']:>7} | {counts['val']:>7} | "
                f"{counts['test']:>7} | {total:>7}"
            )

        print("-" * 80)
        grand = sum(len(splits[s]) for s in SPLIT_NAMES)
        print(
            f"{'IMAGES':<28} | {len(splits['train']):>7} | "
            f"{len(splits['val']):>7} | {len(splits['test']):>7} | {grand:>7}"
        )
        if grand:
            print(
                f"{'%':<28} | {100 * len(splits['train']) / grand:>6.1f}% | "
                f"{100 * len(splits['val']) / grand:>6.1f}% | "
                f"{100 * len(splits['test']) / grand:>6.1f}% |"
            )
        print("=" * 80)

        # Patient counts
        if self.col_patient_id:
            pcol = self.col_patient_id
            has_col = all(pcol in splits[s].columns for s in SPLIT_NAMES)
            if has_col:
                print("\nPatients per split:")
                for name in SPLIT_NAMES:
                    n = splits[name][pcol].nunique()
                    print(f"  {name}: {n} unique patients")
        print()


# ===================================================================
# Folder-based organizer (class-per-folder)
# ===================================================================
class FolderOrganizer(BaseOrganizer):
    """Organise a directory-based dataset.

    Supports **flat** layouts (``source/class/img``) and **pre-split**
    layouts (``source/train/class/img``).

    When ``--patient-id-regex`` is provided, patient-level splitting is
    enforced to prevent data leakage.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.patient_id_regex: Optional[str] = args.patient_id_regex

    # ---------------------------------------------------------------- public
    def run(self) -> None:
        """Full folder pipeline: detect layout → collect → split → write."""
        layout = self._detect_layout()
        logger.info("Source layout: %s", layout)

        if layout == "pre-split":
            classes, splits = self._collect_pre_split()
            logger.info("Pre-split data: %d classes", len(classes))

            # Warn about potential leakage in pre-split data
            if self.patient_id_regex:
                self._check_presplit_leakage(splits)
        else:
            classes, paths, labels = self._collect_flat()
            logger.info(
                "Collected %d images across %d classes.", len(paths), len(classes)
            )

            if self.patient_id_regex:
                logger.info(
                    "Patient-level folder split (regex: %s).",
                    self.patient_id_regex,
                )
                splits = self._patient_level_split_folder(paths, labels)
            else:
                splits = self._image_level_split_folder(paths, labels)

        for s in SPLIT_NAMES:
            logger.info("Split %s: %d items", s, len(splits[s]))

        self._transfer_files(splits)
        self._print_multiclass_summary(splits, classes)

    # -------------------------------------------------------- layout detect
    def _detect_layout(self) -> str:
        """Return ``'pre-split'`` or ``'flat'``.

        Pre-split means the top-level contains ``train/``, ``val/``,
        ``test/`` directories each holding class sub-folders.
        """
        top_dirs = {
            d.name
            for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }
        if set(SPLIT_NAMES).issubset(top_dirs):
            train_dir = self.source_dir / "train"
            if train_dir.exists() and any(
                d.is_dir()
                for d in train_dir.iterdir()
                if not d.name.startswith(".")
            ):
                return "pre-split"
        return "flat"

    # ---------------------------------------------------------- collectors
    def _collect_flat(self) -> Tuple[List[str], List[Path], List[str]]:
        """Collect images from flat ``source/class/images`` layout.

        Returns ``(classes, paths, labels)`` where class names are
        normalised (normal aliases → ``'no finding'``).
        """
        paths: List[Path] = []
        labels: List[str] = []

        for cls_dir in sorted(self.source_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            normalised = self.normalize_class_label(cls_dir.name)
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    paths.append(f)
                    labels.append(normalised)

        if not paths:
            raise ValueError(f"No images found in {self.source_dir}")

        classes = sorted(set(labels))
        return classes, paths, labels

    def _collect_pre_split(
        self,
    ) -> Tuple[List[str], Dict[str, List[Tuple[Path, str]]]]:
        """Collect images from ``source/split/class/images`` layout.

        Returns ``(classes, splits_dict)`` with normalised class names.
        """
        splits: Dict[str, List[Tuple[Path, str]]] = {s: [] for s in SPLIT_NAMES}
        all_classes: Set[str] = set()

        for split_name in SPLIT_NAMES:
            split_dir = self.source_dir / split_name
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                    continue
                normalised = self.normalize_class_label(cls_dir.name)
                all_classes.add(normalised)
                for f in sorted(cls_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        splits[split_name].append((f, normalised))

        total = sum(len(v) for v in splits.values())
        if total == 0:
            raise ValueError(
                f"No images found in pre-split source {self.source_dir}"
            )

        classes = sorted(all_classes)
        return classes, splits

    # ----------------------------------------------------- patient helpers
    def _extract_patient_id(self, path: Path) -> Optional[str]:
        """Extract patient ID from a filename using the configured regex.

        The regex should contain one capture group for the patient ID.
        """
        if not self.patient_id_regex:
            return None
        match = re.search(self.patient_id_regex, path.stem)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        return None

    # ----------------------------------------------------------- splitting
    def _patient_level_split_folder(
        self, paths: List[Path], labels: List[str],
    ) -> Dict[str, List[Tuple[Path, str]]]:
        """Patient-level stratified split for folder data.

        Extracts patient IDs via regex, groups images, then splits at
        the patient level using iterative stratification.
        """
        # Extract patient IDs
        patient_ids: List[Optional[str]] = [
            self._extract_patient_id(p) for p in paths
        ]
        valid_mask = [pid is not None for pid in patient_ids]

        n_no_id = sum(not v for v in valid_mask)
        if n_no_id > 0:
            logger.warning(
                "%d / %d images: no patient-ID extracted (regex: %s).",
                n_no_id, len(paths), self.patient_id_regex,
            )

        if all(not v for v in valid_mask):
            logger.warning(
                "No patient IDs extracted — falling back to image-level split."
            )
            return self._image_level_split_folder(paths, labels)

        # Group images by patient
        patient_images: Dict[str, List[int]] = defaultdict(list)
        no_id_indices: List[int] = []

        for i, (pid, valid) in enumerate(zip(patient_ids, valid_mask)):
            if valid:
                patient_images[pid].append(i)  # type: ignore[arg-type]
            else:
                no_id_indices.append(i)

        # Build patient-level label matrix
        all_classes = sorted(set(labels))
        class_to_idx = {c: ci for ci, c in enumerate(all_classes)}
        patient_list = list(patient_images.keys())
        patient_labels = np.zeros(
            (len(patient_list), len(all_classes)), dtype=int,
        )
        for pi, pid in enumerate(patient_list):
            for idx in patient_images[pid]:
                ci = class_to_idx[labels[idx]]
                patient_labels[pi, ci] = 1

        # Stratify patients
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        fold_indices = self.iterative_stratification(
            patient_labels, ratios, self.seed,
        )

        splits: Dict[str, List[Tuple[Path, str]]] = {s: [] for s in SPLIT_NAMES}
        for i, split_name in enumerate(SPLIT_NAMES):
            for pi in fold_indices[i]:
                pid = patient_list[pi]
                for idx in patient_images[pid]:
                    splits[split_name].append((paths[idx], labels[idx]))

        # Distribute images without patient-ID proportionally
        if no_id_indices:
            rng = np.random.RandomState(self.seed)
            indices_arr = np.array(no_id_indices)
            rng.shuffle(indices_arr)
            ratios_arr = np.array(ratios)
            targets = (ratios_arr * len(indices_arr)).astype(int)
            diff = len(indices_arr) - targets.sum()
            for j in range(abs(diff)):
                targets[j % 3] += 1 if diff > 0 else -1
            ptr = 0
            for fi, split_name in enumerate(SPLIT_NAMES):
                for idx in indices_arr[ptr : ptr + targets[fi]]:
                    splits[split_name].append((paths[idx], labels[idx]))
                ptr += targets[fi]

        # Verify no patient leakage
        self._verify_folder_leakage(splits)

        logger.info(
            "Patient-level folder split: %d patients, %d images.",
            len(patient_list), len(paths),
        )
        return splits

    def _image_level_split_folder(
        self, paths: List[Path], labels: List[str],
    ) -> Dict[str, List[Tuple[Path, str]]]:
        """Image-level stratified split using sklearn."""
        train_p, temp_p, train_l, temp_l = train_test_split(
            paths,
            labels,
            test_size=self.val_ratio + self.test_ratio,
            stratify=labels,
            random_state=self.seed,
        )
        val_frac = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_p, test_p, val_l, test_l = train_test_split(
            temp_p,
            temp_l,
            test_size=1 - val_frac,
            stratify=temp_l,
            random_state=self.seed,
        )
        return {
            "train": list(zip(train_p, train_l)),
            "val": list(zip(val_p, val_l)),
            "test": list(zip(test_p, test_l)),
        }

    # --------------------------------------------------- leakage checks
    def _verify_folder_leakage(
        self, splits: Dict[str, List[Tuple[Path, str]]],
    ) -> None:
        """Verify zero patient overlap across splits."""
        split_pids: Dict[str, Set[str]] = {s: set() for s in SPLIT_NAMES}
        for split_name, items in splits.items():
            for p, _ in items:
                pid = self._extract_patient_id(p)
                if pid:
                    split_pids[split_name].add(pid)

        for i, s1 in enumerate(SPLIT_NAMES):
            for s2 in SPLIT_NAMES[i + 1 :]:
                overlap = split_pids[s1] & split_pids[s2]
                if overlap:
                    raise RuntimeError(
                        f"DATA LEAKAGE: {len(overlap)} patients in both "
                        f"'{s1}' and '{s2}': {sorted(overlap)[:5]}…"
                    )
        logger.info("Folder leakage check PASSED.")

    def _check_presplit_leakage(
        self, splits: Dict[str, List[Tuple[Path, str]]],
    ) -> None:
        """Warn (don't error) about leakage in pre-split data."""
        split_pids: Dict[str, Set[str]] = {s: set() for s in SPLIT_NAMES}
        for split_name, items in splits.items():
            for p, _ in items:
                pid = self._extract_patient_id(p)
                if pid:
                    split_pids[split_name].add(pid)

        for i, s1 in enumerate(SPLIT_NAMES):
            for s2 in SPLIT_NAMES[i + 1 :]:
                overlap = split_pids[s1] & split_pids[s2]
                if overlap:
                    logger.warning(
                        "POTENTIAL LEAKAGE in pre-split data: %d patients "
                        "in both '%s' and '%s'. First 5: %s",
                        len(overlap), s1, s2, sorted(overlap)[:5],
                    )

    # ------------------------------------------------------- file transfer
    def _transfer_files(
        self, splits: Dict[str, List[Tuple[Path, str]]],
    ) -> None:
        """Copy / symlink files into ``dest/{split}/{class}/``."""
        succeeded = skipped = 0
        for split_name, items in splits.items():
            for src, cls in tqdm(items, desc=f"Transferring {split_name}"):
                dest_folder = self.dest_dir / split_name / cls
                dest_folder.mkdir(parents=True, exist_ok=True)
                if self._transfer_file(src, dest_folder / src.name):
                    succeeded += 1
                else:
                    skipped += 1
        if skipped:
            logger.warning("%d images skipped.", skipped)
        logger.info("Transfer: %d OK, %d skipped.", succeeded, skipped)

    # ------------------------------------------------------------ summary
    def _print_multiclass_summary(
        self,
        splits: Dict[str, List[Tuple[Path, str]]],
        classes: List[str],
    ) -> None:
        """Print a class × split summary table (folder mode)."""
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for split_name, items in splits.items():
            for _, cls in items:
                counts[cls][split_name] += 1

        print()
        print("=" * 80)
        print(
            f"{'CLASS':<28} | {'TRAIN':>7} | {'VAL':>7} "
            f"| {'TEST':>7} | {'TOTAL':>7}"
        )
        print("-" * 80)

        for cls in classes:
            c = counts[cls]
            total = c["train"] + c["val"] + c["test"]
            print(
                f"{cls:<28} | {c['train']:>7} | {c['val']:>7} | "
                f"{c['test']:>7} | {total:>7}"
            )

        print("-" * 80)
        grand = sum(len(splits[s]) for s in SPLIT_NAMES)
        sizes = {s: len(splits[s]) for s in SPLIT_NAMES}
        print(
            f"{'IMAGES':<28} | {sizes['train']:>7} | {sizes['val']:>7} | "
            f"{sizes['test']:>7} | {grand:>7}"
        )
        if grand:
            print(
                f"{'%':<28} | {100 * sizes['train'] / grand:>6.1f}% | "
                f"{100 * sizes['val'] / grand:>6.1f}% | "
                f"{100 * sizes['test'] / grand:>6.1f}% |"
            )
        print("=" * 80)

        # Patient counts if regex provided
        if self.patient_id_regex:
            print("\nPatients per split:")
            for split_name in SPLIT_NAMES:
                pids: Set[str] = set()
                for p, _ in splits[split_name]:
                    pid = self._extract_patient_id(p)
                    if pid:
                        pids.add(pid)
                print(f"  {split_name}: {len(pids)} unique patients")
        print()


# ===================================================================
# CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments."""
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
        help="Source format: 'csv' (default) or 'folder'.",
    )

    # CSV-specific -----------------------------------------------------------
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
        help="Min radiologists for positive label (default: 1 = OR).",
    )
    csv_grp.add_argument(
        "--col-image-id", type=str, default="image_id",
        help="CSV column name for image ID (default: image_id).",
    )
    csv_grp.add_argument(
        "--col-class-name", type=str, default="class_name",
        help="CSV column name for class / disease label (default: class_name).",
    )
    csv_grp.add_argument(
        "--col-patient-id", type=str, default=None,
        help="CSV column for patient ID (enables patient-level splitting).",
    )
    csv_grp.add_argument(
        "--col-rad-id", type=str, default="rad_id",
        help="CSV column for annotator / radiologist ID (default: rad_id).",
    )

    # Folder-specific --------------------------------------------------------
    folder_grp = p.add_argument_group("Folder source options")
    folder_grp.add_argument(
        "--patient-id-regex", type=str, default=None,
        help=(
            "Regex applied to filename stems to extract patient ID. "
            "Should contain one capture group, e.g. '^(patient\\d+)'."
        ),
    )

    # Split ratios -----------------------------------------------------------
    split_grp = p.add_argument_group("Split ratios")
    split_grp.add_argument(
        "--train-ratio", type=float, default=0.7, metavar="R",
    )
    split_grp.add_argument(
        "--val-ratio", type=float, default=0.15, metavar="R",
    )
    split_grp.add_argument(
        "--test-ratio", type=float, default=0.15, metavar="R",
    )
    split_grp.add_argument(
        "--seed", type=int, default=42, metavar="S",
        help="Random seed (default: 42).",
    )

    # File transfer ----------------------------------------------------------
    p.add_argument(
        "--copy", action="store_true", default=False,
        help="Copy files instead of symlinks (slower, more disk).",
    )
    p.add_argument(
        "--mode", choices=["append", "replace"], default="append",
        help="append: skip existing. replace: wipe dest first.",
    )
    return p.parse_args()


# ===================================================================
# Entry point
# ===================================================================
def main() -> None:
    """Dispatch to the appropriate organizer based on --source-format."""
    args = parse_args()

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(
            f"Aborting: source directory does not exist: {args.source}"
        )

    if args.source_format == "csv":
        organizer: BaseOrganizer = CSVOrganizer(args)
    else:
        organizer = FolderOrganizer(args)

    organizer.run()
    logger.info("Done.")


if __name__ == "__main__":
    main()

