"""Image processing pipeline for Chest X-ray Classification.

Scans raw data, runs smart cropping and quality-control checks, preprocesses 
clean images into ``data/processed/``, and quarantines corrupted or problematic 
files into ``data/quarantine/``.

Typical usage
-------------
From the CLI::

    python processing.py --raw-dir data/raw --processed-dir data/processed \
        --quarantine-dir data/quarantine --target-size 384
"""
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from xclinvision.config import PipelineConfig

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom"}

# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------

class ProcessCategory(str, Enum):
    SUCCESS = "success"
    FILTERED = "filtered" 
    ERROR = "error"   


@dataclass
class XrayProcessResult:
    image: Optional[np.ndarray]
    category: ProcessCategory
    reason: str


@dataclass
class PipelineReport:
    total_images: int = 0
    processed: int = 0
    quarantined_corrupted: int = 0
    quarantined_flagged: int = 0
    duplicates_same_class: int = 0
    duplicates_cross_class: int = 0
    duplicates_cross_split: int = 0


# ---------------------------------------------------------------------------
# DICOM / standard image reader
# ---------------------------------------------------------------------------

def read_image_grayscale(source: Union[str, Path, bytes]) -> Optional[np.ndarray]:
    """Read an image as 8-bit grayscale, with transparent DICOM support."""
    try:
        if isinstance(source, bytes):
            return _read_bytes_grayscale(source)
        path = Path(source)
        if path.suffix.lower() in {".dcm", ".dicom"}:
            return _read_dicom_grayscale(path)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img  
    except Exception as exc:
        logger.error("read_image_grayscale failed for %s: %s", source if not isinstance(source, bytes) else "<bytes>", exc)
        return None

def _decode_dicom_dataset(ds) -> np.ndarray:
    """Convert a loaded pydicom Dataset to an 8-bit grayscale numpy array."""
    arr = ds.pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__getitem__") else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__getitem__") else float(ww)
        arr = np.clip(arr, wc - ww / 2, wc + ww / 2)

    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) * 255.0 if mx - mn > 0 else np.zeros_like(arr)
    img = arr.astype(np.uint8)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = 255 - img

    return img

def _read_dicom_grayscale(path: Path) -> Optional[np.ndarray]:
    """Decode a DICOM file to 8-bit grayscale via pydicom."""
    import pydicom 
    return _decode_dicom_dataset(pydicom.dcmread(path))

def _read_bytes_grayscale(data: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes to grayscale — tries DICOM first, then cv2."""
    if len(data) > 132 and data[128:132] == b"DICM":
        import pydicom
        import io as _io
        return _decode_dicom_dataset(pydicom.dcmread(_io.BytesIO(data)))

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img

DEFAULT_MIN_AREA_RATIO = 0.15
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 3.0

# Metadata-only version tag — written into processing_metadata.json
# for reproducibility but NOT used in directory naming.
PROCESSING_VERSION = "v2"

# ---------------------------------------------------------------------------
# Shared processed-data directory resolution
# ---------------------------------------------------------------------------
# Directory naming convention: processed_{target_size}
# e.g. data/processed_1024/ for images processed at 1024px.
# No version or hash suffixes — one directory per resolution, period.
# If class names change, use --force-reprocess to regenerate.
# ---------------------------------------------------------------------------

def compute_pixel_hash(image_path: Path) -> str:
    img = read_image_grayscale(image_path)
    if img is None: return ""
    return hashlib.sha256(img.tobytes()).hexdigest()

def get_processed_dir_for_size(
    base_processed_dir: str,
    image_size: int,
    class_names: list | None = None,
) -> Path:
    """Return the processed data directory for a given resolution.

    Convention: ``{base_parent}/processed_{image_size}``
    e.g. ``data/processed_1024`` when *base_processed_dir* is ``data/processed``.

    The *class_names* parameter is accepted for backward-compatibility but
    is no longer encoded into the directory name.
    """
    base = Path(base_processed_dir)
    return base.parent / f"{base.name}_{image_size}"

def compute_image_hash(image_path: Path) -> str:
    """Compute SHA-256 hash of image file bytes in chunks to cap memory footprint."""
    hasher = hashlib.sha256()
    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ---------------------------------------------------------------------------
# 1. Core Processing & Filtering Function
# ---------------------------------------------------------------------------
def is_side_by_side_double(img: np.ndarray, min_aspect: float = 1.5) -> bool:
    """Detect side-by-side duplicate X-rays (two views stitched horizontally)."""
    h, w = img.shape[:2]
    aspect = w / h
    if aspect < min_aspect:
        return False

    strip_w = max(int(w * 0.05), 3)
    cx = w // 2
    centre_strip = img[:, cx - strip_w : cx + strip_w]
    strip_mean = centre_strip.mean()
    overall_mean = img.mean()

    if overall_mean > 10 and strip_mean < overall_mean * 0.45:
        return True

    col_means = img.mean(axis=0).astype(np.float64)
    kernel_size = max(w // 40, 3) | 1
    smoothed = np.convolve(col_means, np.ones(kernel_size) / kernel_size, mode='same')

    left_bound, right_bound = int(w * 0.35), int(w * 0.65)
    centre_region = smoothed[left_bound:right_bound]

    if len(centre_region) == 0:
        return False

    dip_val = centre_region.min()
    side_val = max(smoothed[:left_bound].mean(), smoothed[right_bound:].mean(), 1.0)

    if dip_val < side_val * 0.50:
        return True

    return False

def clean_dark_overlays(
        image: np.ndarray, 
        dark_thresh: int = 30, 
        min_area_ratio: float = 0.005, 
        border_margin_ratio: float = 0.15
        ) -> np.ndarray:
    """Detects and removes dark rectangular clinical markers/annotations near borders."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    h, w = image.shape[:2]
    total_area = h * w
    
    _, thresh = cv2.threshold(image, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min_area_ratio * total_area) or area > (0.1 * total_area):
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        margin_h, margin_w = int(border_margin_ratio * h), int(border_margin_ratio * w)
        near_border = (y < margin_h or y + ch > h - margin_h or x < margin_w or x + cw > w - margin_w)
        
        if not near_border:
            continue
        
        if area / (cw * ch) > 0.6:
            cv2.drawContours(image, [cnt], -1, 0, -1)

    return image


def process_and_filter_xray(
        input_data: Union[str, Path, np.ndarray], 
        target_size: int = 384,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
        ) -> XrayProcessResult:
    """Smart cropping, artifact cleaning, CLAHE, and resizing to square."""
    if isinstance(input_data, (str, Path)):
        img = read_image_grayscale(input_data)
    else:
        img = input_data if len(input_data.shape) == 2 else cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
            
    if img is None: 
        return XrayProcessResult(None, ProcessCategory.ERROR, "Load Failed (Corrupted or empty file)")
    if img.std() < 2.0: 
        return XrayProcessResult(None, ProcessCategory.ERROR, f"Image is blank or near-uniform (std={img.std():.2f})")

    img = clean_dark_overlays(img)

    if is_side_by_side_double(img):
        return XrayProcessResult(None, ProcessCategory.FILTERED, "Side-by-side double image detected")

    # Smart Crop (Broadened thresholds for inverse-intensity DICOM safety)
    mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    mask_white = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)[1]
    combined_mask = cv2.bitwise_and(mask, mask_white)

    coords = cv2.findNonZero(combined_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        area_ratio = (w * h) / (img.shape[0] * img.shape[1])
        
        if area_ratio < min_area_ratio:
            return XrayProcessResult(None, ProcessCategory.FILTERED, f"X-ray area too small ({area_ratio:.2f})")

        crop_ar = w / h
        if crop_ar < MIN_ASPECT_RATIO or crop_ar > MAX_ASPECT_RATIO:
            return XrayProcessResult(None, ProcessCategory.FILTERED, f"Abnormal Aspect Ratio ({crop_ar:.2f})")
        
        img = img[y:y+h, x:x+w]
    else:
        return XrayProcessResult(None, ProcessCategory.FILTERED, "No content detected (Blank image)")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    old_h, old_w = img.shape[:2]
    scale = target_size / max(old_h, old_w)
    new_w, new_h = int(old_w * scale), int(old_h * scale)
    
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    delta_w, delta_h = target_size - new_w, target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return XrayProcessResult(img, ProcessCategory.SUCCESS, "Success")


# ---------------------------------------------------------------------------
# 2. Visualization Tool
# ---------------------------------------------------------------------------
def visualize_crop_step(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print("Could not load image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, mask_white = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    combined_mask = cv2.bitwise_and(mask_black, mask_white)

    coords = cv2.findNonZero(combined_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cropped_img = img[y:y+h, x:x+w]
    else:
        print("No X-ray region detected.")
        return

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
    plt.title(f"1. Detection Box\nOriginal Size: {img.shape[:2]}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(combined_mask, cmap='gray')
    plt.title("2. Segmentation Mask\n(White = Useful Area)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Final Crop\nNew Size: {cropped_img.shape[:2]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_advanced_processing(image_path):
    orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if orig is None: return
    
    cleaned = clean_dark_overlays(orig)
    result = process_and_filter_xray(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(cleaned, cmap='gray'); axes[1].set_title("After Dark Overlay Removal")
    if result.image is not None:
        axes[2].imshow(result.image, cmap='gray'); axes[2].set_title(f"Final (Status: {result.reason})")
    else:
        axes[2].text(0.5, 0.5, f"QUARANTINED\n{result.reason}", ha='center', color='red')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# 3. Pipeline Orchestrator
# ---------------------------------------------------------------------------
def load_dataset_metadata(
    raw_dir: Union[Path, str],
    mode: Optional[str] = None,
    *,
    config: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    if config is not None:
        mode = config.classification_mode
    elif mode is None:
        from xclinvision.config import PipelineConfig as _PC
        mode = _PC.from_yaml().classification_mode
        
    raw_dir = Path(raw_dir)
    records: list[dict] = []

    if mode == "multilabel":
        labels_path = raw_dir / "labels.csv"
        if not labels_path.exists():
            logger.error("labels.csv not found in %s", raw_dir)
            return pd.DataFrame(columns=["split", "class", "filepath", "filename"])

        label_df = pd.read_csv(labels_path)
        valid_filenames = set(label_df["filename"].astype(str))

        for split in ["train", "val", "test"]:
            split_file = raw_dir / "splits" / f"{split}.txt"
            if not split_file.exists():
                logger.warning("Split file not found: %s", split_file)
                continue

            filenames = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
            for fname in tqdm(filenames, desc=f"Scanning {split}", leave=False):
                if fname not in valid_filenames:
                    continue
                img_path = raw_dir / "images" / fname
                if not img_path.exists():
                    continue
                records.append({
                    "split": split, "class": "multi",
                    "filepath": str(img_path), "filename": fname,
                })
    else:
        for split in ["train", "val", "test"]:
            split_path = raw_dir / split
            if not split_path.exists():
                continue

            class_dirs = [d for d in sorted(split_path.iterdir()) if d.is_dir()]
            for class_dir in tqdm(class_dirs, desc=f"Scanning {split}", leave=False):
                class_name = class_dir.name.lower()
                img_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                for img_file in tqdm(img_files, desc=f"  {class_name}", leave=False):
                    records.append({
                        "split": split, "class": class_name,
                        "filepath": str(img_file), "filename": img_file.name,
                    })

    df = pd.DataFrame(records, columns=["split", "class", "filepath", "filename"])

    if not df.empty:
        logger.info("Found %d images across %d splits (mode=%s)", len(df), df["split"].nunique(), mode)
    else:
        logger.error("No images found in %s!", raw_dir)

    return df

def _process_image_worker(args):
    """Exception-safe top-level worker for multiprocessing.

    Performs ALL disk I/O inside the worker so only lightweight metadata
    crosses the IPC boundary (no numpy arrays serialised via pickle).
    """
    cv2.setNumThreads(0)  # prevent OpenCV threading deadlocks under fork()

    row_dict, target_size, min_area_ratio, output_format, processed_dir, quarantine_dir, is_multilabel = args

    try:
        result = process_and_filter_xray(row_dict["filepath"], target_size, min_area_ratio)
    except Exception as exc:
        logger.debug("Worker exception on %s: %s", row_dict["filepath"], exc)
        result = XrayProcessResult(None, ProcessCategory.ERROR, f"Worker exception: {exc}")

    saved_path: Optional[str] = None

    if result.category == ProcessCategory.SUCCESS and result.image is not None:
        out_name = f"{Path(row_dict['filename']).stem}.{output_format}"
        if is_multilabel:
            dest = Path(processed_dir) / "images" / out_name
        else:
            dest = Path(processed_dir) / row_dict["split"] / row_dict["class"] / out_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), result.image)
        saved_path = str(dest)
    else:
        q_folder = "corrupted" if result.category == ProcessCategory.ERROR else "flagged"
        if is_multilabel:
            dest = Path(quarantine_dir) / q_folder / row_dict["filename"]
        else:
            dest = Path(quarantine_dir) / q_folder / row_dict["split"] / row_dict["class"] / row_dict["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row_dict["filepath"], dest)
        saved_path = str(dest)

    return {
        "status": result.category.name,
        "reason": result.reason,
        "row": row_dict,
        "saved_path": saved_path,
    }

def _merge_multilabels_into_manifest(manifest_df: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    manifest_df = manifest_df.copy()
    # 1. Merge safely using the exact original filename (prevents .jpg vs .dcm mixups)
    manifest_df["_original_filename"] = manifest_df["filepath_original"].apply(lambda x: Path(x).name)
    merged = manifest_df.merge(labels_df, left_on="_original_filename", right_on="filename",  how="left")

    # 2. Validate that no labels were lost during the merge
    label_cols = [c for c in labels_df.columns if c != "filename"]
    for col in label_cols:
        if merged[col].isna().any():
            missing = merged[merged[col].isna()]["_original_filename"].tolist()[:5]
            raise ValueError(f"NaN in label column '{col}' after merge. Unmatched files: {missing}")
        merged[col] = merged[col].astype(int)

    # 3. Clean up columns
    drop_cols = [
        c for c in ("class", "source_dataset", "filepath_original", "_original_filename", "filename") 
        if c in merged.columns
    ]
    
    merged.drop(columns=drop_cols, inplace=True)
    return merged


def run_processing_pipeline(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    quarantine_dir: Union[str, Path],
    duplicate_dir: Optional[Union[str, Path]] = None,
    target_size: int = 384,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    output_format: str = "png",
    detect_duplicates: bool = True,
    mode: Optional[str] = None,
    *,
    config: Optional[PipelineConfig] = None,
) -> PipelineReport:
    
    raw_dir, quarantine_dir = Path(raw_dir), Path(quarantine_dir)
    report = PipelineReport()
    
    if config is not None:
        mode = config.classification_mode
    elif mode is None:
        from xclinvision.config import PipelineConfig as _PC
        mode = _PC.from_yaml().classification_mode
    is_multilabel = mode == "multilabel"

    logger.info("Scanning raw dataset at %s (mode=%s) …", raw_dir, mode)
    df = load_dataset_metadata(raw_dir, mode=mode)
    report.total_images = len(df) 

    unique_classes = df['class'].unique().tolist() if not df.empty else None
    processed_dir = get_processed_dir_for_size(str(processed_dir), target_size, unique_classes)
    duplicate_dir = Path(duplicate_dir) if duplicate_dir else processed_dir.parent / "duplicate"

    source_meta_path = raw_dir / "source_metadata.csv"
    if source_meta_path.exists():
        try:
            src_df = pd.read_csv(source_meta_path)
            df = df.merge(src_df[["filename", "source_dataset"]], on="filename", how="left")
            logger.info("Merged source_metadata.csv: %d rows matched", df["source_dataset"].notna().sum())
        except Exception as exc:
            logger.warning("Could not merge source_metadata.csv: %s", exc)
    if "source_dataset" not in df.columns:
        df["source_dataset"] = "unknown"

    if report.total_images == 0:
        logger.warning("No images found. Aborting.")
        return report

    # -------------------------------------------------------------------
    # PHASE 1: Scan, hash, and resolve duplicates BEFORE processing
    # -------------------------------------------------------------------
    keeper_filepaths: set = set()          
    processed_records = []
    quarantine_records = []
    duplicate_records = []

    if detect_duplicates:
        logger.info("Hashing all images (single pass)…")
        hashes = []
        for fp in tqdm(df['filepath'], desc="Hashing", leave=False):
            hashes.append(compute_image_hash(Path(fp)))
        df['hash'] = hashes

        for img_hash, group_df in df.groupby('hash'):
            entries = group_df.to_dict('records')
            unique_classes = group_df['class'].unique()

            if len(unique_classes) > 1 and not is_multilabel:
                report.duplicates_cross_class += len(entries)
                for entry in entries:
                    dest = duplicate_dir / "conflicts" / entry['split'] / entry['class'] / entry['filename']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry['filepath'], dest)
                    duplicate_records.append({
                        "filepath_original": entry['filepath'], "filepath_duplicate": str(dest),
                        "split": entry['split'], "class": entry['class'],
                        "reason": f"Cross-class conflict (classes: {', '.join(sorted(unique_classes))})"
                    })
                continue 

            # Same-class duplicates
            if len(entries) > 1:
                report.duplicates_same_class += len(entries) - 1
                keeper = entries[0]
                keeper_filepaths.add(keeper['filepath'])

                for dup_entry in entries[1:]:
                    dest = duplicate_dir / "same_class" / dup_entry['split'] / dup_entry['class'] / dup_entry['filename']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dup_entry['filepath'], dest)
                    duplicate_records.append({
                        "filepath_original": dup_entry['filepath'], "filepath_duplicate": str(dest),
                        "split": dup_entry['split'], "class": dup_entry['class'],
                        "reason": f"Same-class duplicate (kept: {keeper['filename']})"
                    })
            else:
                keeper_filepaths.add(entries[0]['filepath'])
                
        # Cross-split leakage guard
        _split_priority = {"train": 0, "val": 1, "test": 2}
        keeper_df = df[df['filepath'].isin(keeper_filepaths)].copy()
        for img_hash, grp in keeper_df.groupby('hash'):
            if grp['split'].nunique() > 1:
                grp_sorted = grp.sort_values('split', key=lambda s: s.map(_split_priority))
                leak_fps = set(grp_sorted.iloc[1:]['filepath'])
                report.duplicates_cross_split += len(leak_fps)
                keeper_filepaths -= leak_fps
                for _, row in grp_sorted.iloc[1:].iterrows():
                    dest = duplicate_dir / "cross_split" / row['split'] / row['class'] / row['filename']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(row['filepath'], dest)
                    duplicate_records.append({
                        "filepath_original": row['filepath'], "filepath_duplicate": str(dest),
                        "split": row['split'], "class": row['class'],
                        "reason": f"Cross-split leak (kept in {grp_sorted.iloc[0]['split']})",
                    })
    else:
        keeper_filepaths = set(df['filepath'].tolist())

    # -------------------------------------------------------------------
    # PHASE 2: Process the unique / clean set
    # -------------------------------------------------------------------
    df_keepers = df[df['filepath'].isin(keeper_filepaths)].copy()
    logger.info("Processing %d unique images…", len(df_keepers))

    worker_args = [
        (row.to_dict(), target_size, min_area_ratio, output_format,
         str(processed_dir), str(quarantine_dir), is_multilabel)
        for _, row in df_keepers.iterrows()
    ]

    # I/O heavy process: crank up the workers
    n_workers = min(12, cpu_count())
    logger.info("Launching pool with %d workers…", n_workers)

    with Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_image_worker, worker_args),
                        total=len(worker_args), desc="Processing X-rays"):
            row = res["row"]
            filepath = row["filepath"]
            status = res["status"]
            saved_path = res["saved_path"]

            if status == ProcessCategory.SUCCESS.name:
                processed_records.append({
                    "filepath_original": filepath,
                    "filepath_processed": saved_path,
                    "split": row["split"],
                    "class": row["class"],
                    "source_dataset": row.get("source_dataset", "unknown"),
                })
                report.processed += 1

            else:
                if status == ProcessCategory.ERROR.name:
                    report.quarantined_corrupted += 1
                else:
                    report.quarantined_flagged += 1

                quarantine_records.append({
                    "filepath_original": filepath,
                    "filepath_quarantine": saved_path,
                    "split": row["split"],
                    "class": row["class"],
                    "reason": res["reason"],
                })

    # Save manifests
    if processed_records:
        processed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(processed_records).to_csv(processed_dir / "manifest.csv", index=False)
        
    if quarantine_records:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(quarantine_records).to_csv(quarantine_dir / "quarantine_manifest.csv", index=False)
        
    if duplicate_records:
        duplicate_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(duplicate_records).to_csv(duplicate_dir / "duplicate_manifest.csv", index=False)

    # Multilabel adjustments
    if is_multilabel:
        labels_src = raw_dir / "labels.csv"
        manifest_path = processed_dir / "manifest.csv"
        if labels_src.exists() and manifest_path.exists():
            manifest_df = pd.read_csv(manifest_path)
            manifest_df = _merge_multilabels_into_manifest(manifest_df, labels_src)
            manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)
            manifest_df.to_csv(manifest_path, index=False)
            logger.info("Merged multilabel targets into %s", manifest_path)
        splits_src = raw_dir / "splits"
        if splits_src.exists():
            dest_splits = processed_dir / "splits"
            if dest_splits.exists():
                shutil.rmtree(dest_splits)
            shutil.copytree(splits_src, dest_splits)

    # Save metadata JSON for reproducibility
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_info = {
        "processing_version": PROCESSING_VERSION,
        "target_size": target_size,
        "min_area_ratio": min_area_ratio,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "report": {
            "total_images": report.total_images,
            "processed": report.processed,
            "quarantined_corrupted": report.quarantined_corrupted,
            "quarantined_flagged": report.quarantined_flagged,
            "duplicates_same_class": report.duplicates_same_class,
            "duplicates_cross_class": report.duplicates_cross_class,
            "duplicates_cross_split": report.duplicates_cross_split
        }
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    logger.info(
        "PROCESSING PIPELINE COMPLETE | Processed: %d | Quarantined: %d (Corrupted: %d, Flagged: %d) | Duplicates removed: %d",
        report.processed,
        report.quarantined_corrupted + report.quarantined_flagged,
        report.quarantined_corrupted,
        report.quarantined_flagged,
        report.duplicates_same_class + report.duplicates_cross_class + report.duplicates_cross_split
    )

    return report

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chest X-Ray processing pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data root")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Output dir for clean images")
    parser.add_argument("--quarantine-dir", type=str, default="data/quarantine", help="Output dir for bad images")
    parser.add_argument("--target-size", type=int, default=1024, help="Target H=W in pixels")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA_RATIO, help="Minimum acceptable X-ray area ratio")
    parser.add_argument("--output-format", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--duplicate-dir", type=str, default="data/duplicate", help="Output dir for cross-class duplicate images")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip duplicate detection across splits")
    parser.add_argument("--mode", type=str, default=None, choices=["multiclass", "multilabel"], help="Override classification_mode from system.yaml")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from xclinvision.config import PipelineConfig
    cfg = PipelineConfig.from_yaml()
    mode = args.mode if args.mode else cfg.classification_mode
    logger.info("Classification mode: %s", mode)

    run_processing_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        quarantine_dir=args.quarantine_dir,
        duplicate_dir=args.duplicate_dir,
        target_size=args.target_size,
        min_area_ratio=args.min_area,
        output_format=args.output_format,
        detect_duplicates=not args.skip_duplicate_check,
        mode=mode,
    )

if __name__ == "__main__":
    main()


