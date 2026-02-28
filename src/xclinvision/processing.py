"""Image processing pipeline for Chest X-ray Classification.

Scans raw data, runs smart cropping and quality-control checks, preprocesses 
clean images into ``data/processed/``, and quarantines corrupted or problematic 
files into ``data/quarantine/``.

Typical usage
-------------
From the CLI::

    python processing.py --raw-dir data/raw --processed-dir data/processed \
        --quarantine-dir data/quarantine --target-size 224
"""
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

def compute_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of image file for duplicate detection."""
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MIN_AREA_RATIO = 0.15
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 3.0

# ---------------------------------------------------------------------------
# 1. Core Processing & Filtering Function
# ---------------------------------------------------------------------------

def is_side_by_side_double(img: np.ndarray, min_aspect: float = 1.5) -> bool:
    """Detect side-by-side duplicate X-rays (two views stitched horizontally).

    Strategy:
    1. Only consider images whose aspect ratio is wide enough to plausibly
       contain two views (width >= 1.5 * height).
    2. Look at the vertical intensity strip in the middle 10% of the image.
       Side-by-side doubles have a dark vertical gap (or seam) between the
       two views, so the mean intensity of the centre strip is significantly
       lower than the overall mean.
    3. Also check for a sharp intensity *dip* in the horizontal projection
       (column-wise mean) near the centre — a telltale sign of a separator.
    """
    h, w = img.shape[:2]
    aspect = w / h
    if aspect < min_aspect:
        return False

    # --- Check 1: dark centre strip ---
    strip_w = max(int(w * 0.05), 3)  # 10% of width, centred
    cx = w // 2
    centre_strip = img[:, cx - strip_w : cx + strip_w]
    strip_mean = centre_strip.mean()
    overall_mean = img.mean()

    # The centre gap is typically much darker than the lung fields
    if overall_mean > 10 and strip_mean < overall_mean * 0.45:
        return True

    # --- Check 2: horizontal projection dip ---
    col_means = img.mean(axis=0).astype(np.float64)
    # Smooth to avoid pixel noise triggering false positives
    kernel_size = max(w // 40, 3) | 1  # ensure odd
    smoothed = np.convolve(col_means, np.ones(kernel_size) / kernel_size, mode='same')

    # Only inspect the middle 30% of the image
    left_bound = int(w * 0.35)
    right_bound = int(w * 0.65)
    centre_region = smoothed[left_bound:right_bound]

    if len(centre_region) == 0:
        return False

    dip_val = centre_region.min()
    side_val = max(smoothed[:left_bound].mean(), smoothed[right_bound:].mean(), 1.0)

    # A strong dip (< 50% of the flanking brightness) signals a separator
    if dip_val < side_val * 0.50:
        return True

    return False


def clean_dark_overlays(image, dark_thresh=30, min_area_ratio=0.005, border_margin_ratio=0.15):
    """
    Detects and removes dark rectangular clinical markers/annotations near borders.
    Optimized for grayscale X-ray processing.
    """
    h, w = image.shape[:2]
    total_area = h * w
    
    # 1. Threshold for dark regions (Markers are usually near-black)
    _, thresh = cv2.threshold(image, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Cleanup noise (small dots)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find potential marker contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    found_any = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter: Must be large enough to be a marker, but not half the image
        if area < (min_area_ratio * total_area) or area > (0.1 * total_area):
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Filter: Must be near the border (where tags usually live)
        margin_h, margin_w = int(border_margin_ratio * h), int(border_margin_ratio * w)
        near_border = (y < margin_h or y + ch > h - margin_h or 
                       x < margin_w or x + cw > w - margin_w)
        
        if not near_border:
            continue
        
        # Filter: Must be roughly rectangular (0.6 fill ratio)
        if area / (cw * ch) > 0.6:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            found_any = True
    
    # 4. Only inpaint if we actually found something to fix
    if found_any:
        # INPAINT_TELEA is generally faster for small text/marker removal
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return image

def process_and_filter_xray(input_data, target_size=224, min_area_ratio=DEFAULT_MIN_AREA_RATIO) -> Tuple[Optional[np.ndarray], str]:
    """
    Advanced processing for Chest X-rays:
    - Removes large white or black margins (Auto-Crop)
    - Filters images where the X-ray is too small (Quality Check)
    - Contrast Enhancement (CLAHE)
    - Square Resizing (Letterboxing)
    """
    # 1. Load Input
    if isinstance(input_data, (str, Path)):
        img = cv2.imread(str(input_data), cv2.IMREAD_GRAYSCALE)
    else:
        img = input_data
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    # --- INTEGRITY CHECKS ---
    if img is None: return None, "Error: Load Failed (Corrupted or empty file)"
    if img.std() < 2.0: return None, f"Error: Image is blank or near-uniform (std={img.std():.2f})"

    img = clean_dark_overlays(img)

    # --- SIDE-BY-SIDE DOUBLE CHECK ---
    if is_side_by_side_double(img):
        return None, "Filtered: Side-by-side double image detected"

    # 2. Smart Cropping (Handles BOTH white and black margins)
    mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
    mask_white = cv2.threshold(img, 253, 255, cv2.THRESH_BINARY_INV)[1]
    combined_mask = cv2.bitwise_and(mask, mask_white)

    coords = cv2.findNonZero(combined_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # QUALITY CHECK: Ratio of useful content vs total image
        total_area = img.shape[0] * img.shape[1]
        useful_area = w * h
        area_ratio = useful_area / total_area
        
        if area_ratio < min_area_ratio:
            return None, f"Filtered: X-ray area too small ({area_ratio:.2f})"

        crop_ar = w / h
        if crop_ar < MIN_ASPECT_RATIO or crop_ar > MAX_ASPECT_RATIO:
            return None, f"Filtered: Abnormal Aspect Ratio ({crop_ar:.2f})"
        
        # Apply the crop
        img = img[y:y+h, x:x+w]
    else:
        return None, "Filtered: No content detected (Blank image)"

    # 3. CLAHE (Essential for Pneumonia/TB features)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. Aspect-Ratio Aware Resize (Letterboxing)
    old_h, old_w = img.shape[:2]
    scale = target_size / max(old_h, old_w)
    new_w, new_h = int(old_w * scale), int(old_h * scale)
    
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 5. Padding to Square
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return img, "Success"

# ---------------------------------------------------------------------------
# 2. Visualization Tool
# ---------------------------------------------------------------------------

def visualize_crop_step(image_path):
    """Visualizes the auto-cropping logic for EDA and debugging."""
    img = cv2.imread(str(image_path))
    if img is None:
        print("Could not load image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, mask_black = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    _, mask_white = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV)
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
    processed, status = process_and_filter_xray(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(cleaned, cmap='gray'); axes[1].set_title("After Dark Overlay Removal")
    if processed is not None:
        axes[2].imshow(processed, cmap='gray'); axes[2].set_title(f"Final (Status: {status})")
    else:
        axes[2].text(0.5, 0.5, f"QUARANTINED\n{status}", ha='center', color='red')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# 3. Pipeline Orchestrator
# ---------------------------------------------------------------------------

def load_dataset_metadata(raw_dir: Path | str) -> pd.DataFrame:
    """Load metadata by scanning raw_dir/{split}/{class}/ for images."""
    raw_dir = Path(raw_dir)
    records =[]
    
    for split in ["train", "val", "test"]:
        split_path = raw_dir / split
        if not split_path.exists():
            logger.warning("Split directory not found: %s", split_path)
            continue
            
        class_dirs = [d for d in sorted(split_path.iterdir()) if d.is_dir()]
        
        for class_dir in tqdm(class_dirs, desc=f"Scanning {split}", leave=False):
            # Standardize class names
            class_name = class_dir.name.lower()
            
            img_files = [f for f in class_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            
            for img_file in tqdm(img_files, desc=f"  {class_name}", leave=False):
                records.append({
                    "split": split,
                    "class": class_name,
                    "filepath": str(img_file),
                    "filename": img_file.name,
                })
    
    # Pre-define columns to prevent errors on empty sets
    df = pd.DataFrame(records, columns=["split", "class", "filepath", "filename"])
    
    if not df.empty:
        logger.info("Found %d images across %d splits", len(df), df["split"].nunique())
    else:
        logger.error("No images found in %s!", raw_dir)
        
    return df

@dataclass
class PipelineReport:
    total_images: int = 0
    processed: int = 0
    quarantined_corrupted: int = 0
    quarantined_flagged: int = 0
    duplicates_same_class: int = 0
    duplicates_cross_class: int = 0

def run_processing_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path,
    quarantine_dir: str | Path,
    duplicate_dir: str | Path = None,
    target_size: int = 224,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    output_format: str = "png",
    detect_duplicates: bool = True,
) -> PipelineReport:
    
    raw_dir, processed_dir, quarantine_dir = Path(raw_dir), Path(processed_dir), Path(quarantine_dir)
    duplicate_dir = Path(duplicate_dir) if duplicate_dir else processed_dir.parent / "duplicate"
    report = PipelineReport()

    logger.info("Scanning raw dataset at %s …", raw_dir)
    df = load_dataset_metadata(raw_dir)
    report.total_images = len(df)

    if report.total_images == 0:
        logger.warning("No images found. Aborting.")
        return report

    # -------------------------------------------------------------------
    # PHASE 1: Scan, hash, and resolve duplicates BEFORE processing
    # -------------------------------------------------------------------
    cross_class_hashes: set = set()       # hashes to quarantine entirely
    keeper_filepaths: set = set()          # filepaths we will actually process
    hash_to_entries: dict = {}             # hash -> list of row-dicts

    processed_records = []
    quarantine_records = []
    duplicate_records = []

    if detect_duplicates:
        logger.info("Hashing all images (single pass)…")
        hashes = []
        for fp in tqdm(df['filepath'], desc="Hashing", leave=False):
            hashes.append(compute_image_hash(Path(fp)))
        df['hash'] = hashes

        # Group by hash and categorize
        for img_hash, group_df in df.groupby('hash'):
            entries = group_df.to_dict('records')
            hash_to_entries[img_hash] = entries
            unique_classes = group_df['class'].unique()

            # --- Cross-class conflict: same pixels, different labels ---
            if len(unique_classes) > 1:
                cross_class_hashes.add(img_hash)
                report.duplicates_cross_class += len(entries)
                logger.warning(
                    "Cross-class duplicate (%s): appears in classes %s (%d files)",
                    entries[0]['filename'], set(unique_classes), len(entries)
                )
                for entry in entries:
                    dest = duplicate_dir / "conflicts" / entry['split'] / entry['class'] / entry['filename']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry['filepath'], dest)
                    duplicate_records.append({
                        "filepath_original": entry['filepath'],
                        "filepath_duplicate": str(dest),
                        "split": entry['split'],
                        "class": entry['class'],
                        "reason": f"Cross-class conflict (classes: {', '.join(sorted(unique_classes))})"
                    })
                continue  # none of these go to processing

            # --- Same-class duplicates: keep first, move rest ---
            if len(entries) > 1:
                report.duplicates_same_class += len(entries) - 1
                keeper = entries[0]
                keeper_filepaths.add(keeper['filepath'])

                for dup_entry in entries[1:]:
                    dest = duplicate_dir / "same_class" / dup_entry['split'] / dup_entry['class'] / dup_entry['filename']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dup_entry['filepath'], dest)
                    duplicate_records.append({
                        "filepath_original": dup_entry['filepath'],
                        "filepath_duplicate": str(dest),
                        "split": dup_entry['split'],
                        "class": dup_entry['class'],
                        "reason": f"Same-class duplicate (kept: {keeper['filename']})"
                    })
            else:
                keeper_filepaths.add(entries[0]['filepath'])
    else:
        # No dedup — every file is a keeper
        keeper_filepaths = set(df['filepath'].tolist())

    # -------------------------------------------------------------------
    # PHASE 2: Process the unique / clean set
    # -------------------------------------------------------------------
    df_keepers = df[df['filepath'].isin(keeper_filepaths)].copy()
    logger.info("Processing %d unique images…", len(df_keepers))
    for _, row in tqdm(df_keepers.iterrows(), total=len(df_keepers), desc="Processing X-rays"):
        filepath = row["filepath"]
        processed_img, status = process_and_filter_xray(filepath, target_size, min_area_ratio)

        # Handle SUCCESS
        if processed_img is not None:
            out_name = f"{Path(row['filename']).stem}.{output_format}"
            dest = processed_dir / row["split"] / row["class"] / out_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(dest), processed_img)
            
            processed_records.append({
                "filepath_original": filepath,
                "filepath_processed": str(dest),
                "split": row["split"],
                "class": row["class"]
            })
            report.processed += 1
            
        # Handle QUARANTINE (Failure/Filtered)
        else:
            is_error = status.startswith("Error")
            q_folder = "corrupted" if is_error else "flagged"
            
            if is_error: report.quarantined_corrupted += 1
            else: report.quarantined_flagged += 1

            dest = quarantine_dir / q_folder / row["split"] / row["class"] / row["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dest) # Copy the original bad file
            
            quarantine_records.append({
                "filepath_original": filepath,
                "filepath_quarantine": str(dest),
                "split": row["split"],
                "class": row["class"],
                "reason": status
            })

    # Save Manifests
    if processed_records:
        processed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(processed_records).to_csv(processed_dir / "manifest.csv", index=False)
        
    if quarantine_records:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(quarantine_records).to_csv(quarantine_dir / "quarantine_manifest.csv", index=False)
        
    if duplicate_records:
        duplicate_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(duplicate_records).to_csv(duplicate_dir / "duplicate_manifest.csv", index=False)

    # Print Summary
    print("\n" + "=" * 60)
    print("PROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    unique_count = len(keeper_filepaths)
    print(f"  Total raw images:              {report.total_images}")
    print(f"  Cross-class conflicts moved:   {report.duplicates_cross_class}")
    print(f"  Same-class duplicates moved:   {report.duplicates_same_class}")
    print(f"  Unique images to process:      {unique_count}")
    print("-" * 60)
    print(f"  Processed (clean):             {report.processed}")
    print(f"  Quarantined (corrupted):       {report.quarantined_corrupted}")
    print(f"  Quarantined (flagged):         {report.quarantined_flagged}")
    print("=" * 60)

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
    parser.add_argument("--target-size", type=int, default=224, help="Target H=W in pixels")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA_RATIO, help="Minimum acceptable X-ray area ratio")
    parser.add_argument("--output-format", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--duplicate-dir", type=str, default="data/duplicate", help="Output dir for cross-class duplicate images")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip duplicate detection across splits")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    run_processing_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        quarantine_dir=args.quarantine_dir,
        duplicate_dir=args.duplicate_dir,
        target_size=args.target_size,
        min_area_ratio=args.min_area,
        output_format=args.output_format,
        detect_duplicates=not args.skip_duplicate_check
    )

if __name__ == "__main__":
    main()

