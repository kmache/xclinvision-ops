import kagglehub
import shutil
import os
from pathlib import Path
import yaml


EXPECTED_SPLITS = ["train", "val", "test"]

# Read class names from system.yaml (single source of truth)
_SYSTEM_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "system.yaml"
try:
    with open(_SYSTEM_CONFIG, "r") as _fh:
        _cfg = yaml.safe_load(_fh) or {}
    EXPECTED_CLASSES = [c.lower() for c in _cfg.get("model", {}).get("class_names", [])]
except Exception:
    EXPECTED_CLASSES = []
if len(EXPECTED_CLASSES) < 2:
    EXPECTED_CLASSES = ["normal", "pneumonia", "cardiomegaly"]


def download_dataset(
    dataset_slug: str = "muhammadrehan00/chest-xray-dataset",
    output_dir: str = "data/raw",
    force: bool = False,
):
    """
    Downloads the Chest X-Ray Dataset (Normal / Pneumonia / Cardiomegaly)
    via KaggleHub and organises it into data/raw/{train,val,test}/{class}/.

    If force=True, the Kaggle cache for this dataset is wiped first so a
    completely fresh copy is pulled.
    """
    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Quick check: is the data already in place? ---
    already_exists = all(
        (dest_path / split / cls).exists()
        and any((dest_path / split / cls).iterdir())
        for split in EXPECTED_SPLITS
        for cls in EXPECTED_CLASSES
    )

    if already_exists and not force:
        print(f"✅ Data already exists at {dest_path}. Skipping.", flush=True)
        return

    print("⬇️  Initializing download (via KaggleHub)...", flush=True)

    try:
        # --- 2. Download (instant when already cached) ---
        cache_path = Path(kagglehub.dataset_download(dataset_slug))

        # --- 3. Force re-download if requested ---
        if force:
            print(f"🧹 Force enabled: cleaning cache at {cache_path}...", flush=True)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print("   -> Cache deleted.", flush=True)

            print("⬇️  Re-downloading fresh copy from Kaggle...", flush=True)
            cache_path = Path(kagglehub.dataset_download(dataset_slug))
            print(f"✅ Downloaded fresh to: {cache_path}", flush=True)
        else:
            print(f"✅ Using cached version at: {cache_path}", flush=True)

        # --- 4. Discover the dataset root inside the cache ---
        dataset_root = _find_dataset_root(cache_path)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Could not locate train/val/test splits inside {cache_path}. "
                "The dataset layout may have changed."
            )
        print(f"📂 Dataset root found at: {dataset_root}", flush=True)

        # --- 5. Copy images into data/raw/{split}/{class}/ ---
        total_copied = 0
        for split in EXPECTED_SPLITS:
            src_split = dataset_root / split
            if not src_split.exists():
                # Try capitalised variant (e.g. Train, Test)
                src_split = dataset_root / split.capitalize()
            if not src_split.exists():
                print(f"⚠️  Split '{split}' not found – skipping.", flush=True)
                continue

            for cls in EXPECTED_CLASSES:
                src_cls = _resolve_class_dir(src_split, cls)
                if src_cls is None:
                    print(
                        f"⚠️  Class '{cls}' not found in {src_split} – skipping.",
                        flush=True,
                    )
                    continue

                dst_cls = dest_path / split / cls
                dst_cls.mkdir(parents=True, exist_ok=True)

                copied = _copy_images(src_cls, dst_cls)
                total_copied += copied
                print(
                    f"   -> {split}/{cls}: {copied} images copied.",
                    flush=True,
                )

        print(f"🎉 Done! {total_copied} images total → {dest_path}", flush=True)

    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        print("   (Make sure you have authorised your Kaggle token.)", flush=True)
        raise


# ── helpers ──────────────────────────────────────────────────────────────────
def _find_dataset_root(base: Path) -> Path | None:
    """Walk down until we find a directory that contains train/test/val."""
    if _has_splits(base):
        return base
    for root, dirs, _ in os.walk(base):
        if _has_splits(Path(root)):
            return Path(root)
    return None


def _has_splits(p: Path) -> bool:
    children = {d.name.lower() for d in p.iterdir() if d.is_dir()}
    return children >= {"train", "test"}  # val may be optional


def _resolve_class_dir(split_dir: Path, cls_name: str) -> Path | None:
    """Case-insensitive lookup for a class folder inside a split."""
    for d in split_dir.iterdir():
        if d.is_dir() and d.name.lower() == cls_name.lower():
            return d
    return None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _copy_images(src: Path, dst: Path) -> int:
    """Copy image files from src → dst, skipping duplicates."""
    count = 0
    for f in src.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            dest_file = dst / f.name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                count += 1
    return count


if __name__ == "__main__":
    download_dataset(output_dir="data/raw", force=False)
