import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from xclinvision.processing import process_and_filter_xray

img_path = Path("/home/karim-mache/Documents/nouncode-ai/portfolio_projects/xclinvision-ops/data/raw/train/tuberculosis/tuberculosis-9349.jpg")
img, status = process_and_filter_xray(img_path)
print(f"Status: {status}")
if img is not None:
    print(f"Processed shape: {img.shape}")
    cv2.imwrite("test_processed_output.png", img)
    print("Saved processed image as test_processed_output.png")
else:
    print("Image was filtered/quarantined.")
