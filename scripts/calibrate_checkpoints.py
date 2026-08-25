#!/usr/bin/env python
"""Fit per-class calibration on the validation split and write it into the
exported checkpoints, then re-derive decision thresholds from the calibrated
probabilities.

Why this exists
---------------
Every shipped checkpoint was exported with ``temperature: null``, so
``InferencePipeline._apply_temperature`` was a no-op and served probabilities
were raw sigmoid outputs. Measured per-class ECE on the validation split ran
0.15-0.44 while ROC-AUC ran 0.845-0.959: the models rank well and report
badly.

A single global temperature does not fix it — fitting one on this task gives
T ~ 1.0 (measured: 1.0146, 1.1059, 0.9815, 1.0029 for the four checkpoints),
because one scalar averages four different per-class errors. Nor does per-class
*temperature* alone: the models trained with pos_weight 5.1-15.2, which biases
every head toward positive, and temperature can only pull logits toward or away
from zero, never shift them. This script therefore fits a per-class affine map,
``p = sigmoid(z / T + b)``, of which temperature scaling is the b = 0 case.

Calibration is fitted on the validation split only. The test split is scored
afterwards purely as held-out confirmation and never influences a parameter or
the calibrated/uncalibrated verdict.

Usage
-----
    python scripts/calibrate_checkpoints.py --dry-run     # report, write nothing
    python scripts/calibrate_checkpoints.py               # write the payloads
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xclinvision.agent.guardrails import CRITICAL_CONDITIONS  # noqa: E402
from xclinvision.config import get_class_names  # noqa: E402
from xclinvision.dataset import get_val_transforms  # noqa: E402
from xclinvision.evaluator import CalibrationAnalyzer, TemperatureScaler  # noqa: E402
from xclinvision.modeling import build_model, get_model_normalization  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calibrate")

#: A fit only earns the "calibrated" label if held-out error is genuinely low.
#: Merely reducing ECE is not enough — 0.26 is an improvement on 0.29 and still
#: means a reported 0.80 is nowhere near an 80% chance of the finding, which is
#: exactly the overstatement the flag exists to prevent.
DEFAULT_ECE_MAX = 0.05

#: Sensitivity target for life-threatening findings; everything else is tuned
#: for F1. Mirrors the intent of the clinical floor in agent/guardrails.py.
CRITICAL_MIN_RECALL = 0.90


def _load_split(manifest: Path, split: str, class_names: List[str]):
    import pandas as pd

    df = pd.read_csv(manifest)
    d = df[df["split"] == split].reset_index(drop=True)
    return d["filepath_processed"].tolist(), d[class_names].values.astype(np.float32)


def _collect_logits(
    model, paths: List[str], mean, std, image_size: int, device: str, batch: int = 48
) -> np.ndarray:
    """Raw pre-activation logits for every image, in manifest order."""
    tf = get_val_transforms(image_size, mean, std)
    out = np.zeros(
        (len(paths), model(torch.zeros(1, 3, image_size, image_size, device=device)).shape[1]),
        dtype=np.float32,
    )
    buf: List[np.ndarray] = []
    idx: List[int] = []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        with torch.no_grad():
            out[idx] = model(x).float().cpu().numpy()
        buf.clear()
        idx.clear()

    for i, p in enumerate(paths):
        buf.append(tf(image=np.array(Image.open(p).convert("RGB")))["image"].numpy())
        idx.append(i)
        if len(buf) == batch:
            flush()
    flush()
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _derive_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Per-class thresholds from calibrated probabilities.

    Critical findings are tuned to the lowest threshold reaching
    CRITICAL_MIN_RECALL; every other class is tuned for maximum F1. Returns the
    thresholds and, for the report, which rule produced each one.
    """
    from sklearn.metrics import precision_recall_curve

    thresholds: Dict[str, float] = {}
    rule: Dict[str, str] = {}

    for i, name in enumerate(class_names):
        yt, p = y_true[:, i], probs[:, i]
        if name in CRITICAL_CONDITIONS:
            t = float(p.min())
            for cand in np.unique(p)[::-1]:
                if ((p >= cand) & (yt == 1)).sum() / max(1, (yt == 1).sum()) >= CRITICAL_MIN_RECALL:
                    t = float(cand)
                    break
            thresholds[name] = round(t, 4)
            rule[name] = f"recall>={CRITICAL_MIN_RECALL:.2f} (critical)"
        else:
            prec, rec, thr = precision_recall_curve(yt, p)
            f1 = np.divide(
                2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0
            )
            best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
            thresholds[name] = round(float(thr[best]) if len(thr) else 0.5, 4)
            rule[name] = "max-F1"
    return thresholds, rule


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models-dir", default="models/best_models")
    ap.add_argument("--manifest", default="data/processed_384/manifest.csv")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument(
        "--ece-max",
        type=float,
        default=DEFAULT_ECE_MAX,
        help="Max mean validation ECE that still counts as calibrated.",
    )
    ap.add_argument(
        "--no-bias",
        action="store_true",
        help="Fit temperature only (b=0). Textbook temperature scaling.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = ap.parse_args()

    class_names = get_class_names()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyzer = CalibrationAnalyzer(num_bins=15)
    manifest = Path(args.manifest)

    val_paths, y_val = _load_split(manifest, "val", class_names)
    test_paths, y_test = _load_split(manifest, "test", class_names)
    logger.info("val n=%d  test n=%d  classes=%s", len(val_paths), len(test_paths), class_names)

    pos = y_val.sum(axis=0).astype(int)
    logger.info("val positive support: %s", dict(zip(class_names, pos.tolist())))
    if pos.min() < 50:
        logger.error(
            "Smallest positive support is %d, too few for a stable per-class fit. "
            "Refusing to fit; report the size and stop.",
            int(pos.min()),
        )
        return 2

    summary = {}
    for meta_path in sorted(Path(args.models_dir).glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        arch = meta["model_name"]
        pth = meta_path.with_name(meta_path.name.replace("_meta.json", ".pth"))
        logger.info("\n%s\n%s\n%s", "=" * 78, arch, "=" * 78)

        model = (
            build_model(
                arch, num_classes=len(class_names), pretrained=False, img_size=args.image_size
            )
            .to(device)
            .eval()
        )
        payload = torch.load(pth, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict", payload)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            logger.error("%s: %d missing keys, skipping", arch, len(incompatible.missing_keys))
            continue

        norm = get_model_normalization(model, arch)
        zv = _collect_logits(
            model, val_paths, list(norm["mean"]), list(norm["std"]), args.image_size, device
        )
        zt = _collect_logits(
            model, test_paths, list(norm["mean"]), list(norm["std"]), args.image_size, device
        )

        scaler = TemperatureScaler()
        temps, biases = scaler.fit_per_class(zv, y_val, with_bias=not args.no_bias)

        t_arr = np.asarray(temps, dtype=np.float32)
        b_arr = np.asarray(biases, dtype=np.float32)
        pv_raw, pv_cal = _sigmoid(zv), _sigmoid(zv / t_arr + b_arr)
        pt_raw, pt_cal = _sigmoid(zt), _sigmoid(zt / t_arr + b_arr)

        ece = {"val_before": {}, "val_after": {}, "test_before": {}, "test_after": {}}
        logger.info(
            "%-22s %6s %8s | %10s %9s | %11s %10s",
            "class",
            "T",
            "bias",
            "ECE val",
            "-> after",
            "ECE test",
            "-> after",
        )
        for i, name in enumerate(class_names):
            ece["val_before"][name] = round(analyzer._bin_ece(pv_raw[:, i], y_val[:, i]), 4)
            ece["val_after"][name] = round(analyzer._bin_ece(pv_cal[:, i], y_val[:, i]), 4)
            ece["test_before"][name] = round(analyzer._bin_ece(pt_raw[:, i], y_test[:, i]), 4)
            ece["test_after"][name] = round(analyzer._bin_ece(pt_cal[:, i], y_test[:, i]), 4)
            logger.info(
                "  %-20s %6.3f %8.3f | %10.4f %9.4f | %11.4f %10.4f",
                name,
                temps[i],
                biases[i],
                ece["val_before"][name],
                ece["val_after"][name],
                ece["test_before"][name],
                ece["test_after"][name],
            )

        mean_val_after = float(np.mean(list(ece["val_after"].values())))
        mean_val_before = float(np.mean(list(ece["val_before"].values())))
        mean_test_after = float(np.mean(list(ece["test_after"].values())))
        mean_test_before = float(np.mean(list(ece["test_before"].values())))

        # Verdict is decided on validation only. Test is reported beside it as
        # independent confirmation and never selects anything.
        improved = mean_val_after < mean_val_before
        status = "calibrated" if (improved and mean_val_after <= args.ece_max) else "uncalibrated"
        logger.info(
            "  mean val  %.4f -> %.4f     mean test %.4f -> %.4f",
            mean_val_before,
            mean_val_after,
            mean_test_before,
            mean_test_after,
        )
        logger.info("  verdict: %s (improved=%s, bar=%.3f)", status.upper(), improved, args.ece_max)

        thresholds, rule = _derive_thresholds(y_val, pv_cal, class_names)
        logger.info(
            "  thresholds from calibrated probs: %s",
            {k: f"{v} [{rule[k]}]" for k, v in thresholds.items()},
        )

        summary[arch] = {
            "temperature": [round(t, 6) for t in temps],
            "calibration_bias": [round(b, 6) for b in biases],
            "calibration_status": status,
            "calibration_ece": ece,
            "thresholds": thresholds,
            "threshold_rule": rule,
        }

        if args.dry_run:
            logger.info("  --dry-run: payload not written")
            continue

        payload["temperature"] = summary[arch]["temperature"]
        payload["calibration_bias"] = summary[arch]["calibration_bias"]
        payload["calibration_status"] = status
        payload["calibration_ece"] = ece
        payload["thresholds"] = thresholds
        torch.save(payload, pth)

        meta_out = {k: v for k, v in payload.items() if k != "model_state_dict"}
        meta_path.write_text(json.dumps(meta_out, indent=2))
        logger.info("  wrote %s and %s", pth.name, meta_path.name)

    out = Path("outputs/calibration_summary.json")
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        logger.info("\nwrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
