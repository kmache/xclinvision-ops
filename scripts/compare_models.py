#!/usr/bin/env python3
"""Compare all trained models by computing metrics from prediction files."""
import json, os, glob, numpy as np
from sklearn.metrics import roc_auc_score, f1_score

CLASS_NAMES = ["Cardiomegaly", "Aortic enlargement", "Pleural thickening", "Pulmonary fibrosis"]
SHORT_NAMES = ["Cardi", "Aortic", "Pleur", "PulmF"]

def compute_metrics(probs, targets):
    probs = np.array(probs)
    targets = np.array(targets)
    n_classes = probs.shape[1]
    aucs = []
    for i in range(n_classes):
        if len(np.unique(targets[:, i])) > 1:
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
        else:
            aucs.append(float('nan'))
    auc_macro = np.nanmean(aucs)
    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(targets, preds_05, average='macro', zero_division=0)
    best_thresholds = []
    for i in range(n_classes):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            preds_t = (probs[:, i] >= t).astype(int)
            f1_t = f1_score(targets[:, i], preds_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = t
        best_thresholds.append(best_t)
    preds_opt = np.zeros_like(probs)
    for i in range(n_classes):
        preds_opt[:, i] = (probs[:, i] >= best_thresholds[i]).astype(int)
    f1_opt = f1_score(targets, preds_opt, average='macro', zero_division=0)
    per_class_f1 = [f1_score(targets[:, i], preds_opt[:, i], zero_division=0) for i in range(n_classes)]
    return {'auc_macro': auc_macro, 'aucs': aucs, 'f1_05': f1_05, 'f1_opt': f1_opt,
            'thresholds': best_thresholds, 'per_class_f1': per_class_f1}

def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    # Get training times
    train_times = {}
    if os.path.exists('outputs/train_all_results.txt'):
        with open('outputs/train_all_results.txt') as f:
            for line in f:
                if 'SUCCESS' in line:
                    parts = line.strip()
                    name = parts.split('] ')[1].split(' —')[0].replace('train_', '')
                    secs = int(parts.split('(')[1].split('s)')[0])
                    train_times[name] = secs

    results = []
    for model_dir in sorted(glob.glob('models/*_2026*')):
        model = os.path.basename(model_dir).rsplit('_2026', 1)[0]

        # Find best checkpoint epoch (ckpt is in model_dir directly)
        ckpts = glob.glob(f'{model_dir}/*-epoch=*-val_f1_macro=*.ckpt')
        best_epoch = None
        ckpt_f1 = None
        for c in ckpts:
            bn = os.path.basename(c)
            ep = int(bn.split('epoch=')[1].split('-')[0])
            f1_val = float(bn.split('val_f1_macro=')[1].split('.ckpt')[0])
            if best_epoch is None or f1_val > ckpt_f1:
                best_epoch = ep
                ckpt_f1 = f1_val

        pred_files = sorted(glob.glob(f'{model_dir}/preds/val_predictions_*.json'))
        if not pred_files:
            continue

        # Scan all prediction files, compute metrics, find best AUC epoch
        best_auc = -1
        best_result = None
        for pf in pred_files:
            with open(pf) as f:
                d = json.load(f)
            m = compute_metrics(d['probabilities'], d['targets'])
            if m['auc_macro'] > best_auc:
                best_auc = m['auc_macro']
                best_result = {**m, 'pred_epoch': d['epoch']}

        results.append({
            'model': model,
            'best_ckpt_epoch': best_epoch,
            'ckpt_f1': ckpt_f1,
            'train_time': train_times.get(model, 0),
            **best_result
        })

    results.sort(key=lambda x: x['auc_macro'], reverse=True)

    W = 105
    print(f"\n{'=' * W}")
    print(f"{'COMPREHENSIVE MODEL COMPARISON — 14 Models @ 1024px Processing':^{W}}")
    print(f"{'=' * W}")

    # Main table
    hdr = f"{'Model':<20} {'BestEp':>6} {'AUC':>8} {'F1@0.5':>8} {'F1_opt':>8}"
    for s in SHORT_NAMES:
        hdr += f" {s:>7}"
    hdr += f" {'Time':>7}"
    print(f"\n{hdr}")
    print('-' * len(hdr))
    for r in results:
        mins = r['train_time'] // 60
        secs = r['train_time'] % 60
        row = f"{r['model']:<20} {r['pred_epoch']:>6} {r['auc_macro']:>8.4f} {r['f1_05']:>8.4f} {r['f1_opt']:>8.4f}"
        for f1 in r['per_class_f1']:
            row += f" {f1:>7.3f}"
        row += f" {mins:>3}m{secs:02d}s"
        print(row)

    # Per-class AUC
    print(f"\nPer-Class AUC:")
    hdr2 = f"{'Model':<20} {'AUC_macro':>9}"
    for s in SHORT_NAMES:
        hdr2 += f" {s:>8}"
    print(hdr2)
    print('-' * len(hdr2))
    for r in results:
        row = f"{r['model']:<20} {r['auc_macro']:>9.4f}"
        for a in r['aucs']:
            row += f" {a:>8.4f}"
        print(row)

    # Optimal thresholds
    print(f"\nOptimal Thresholds:")
    hdr3 = f"{'Model':<20}"
    for s in SHORT_NAMES:
        hdr3 += f" {s:>8}"
    print(hdr3)
    print('-' * len(hdr3))
    for r in results:
        row = f"{r['model']:<20}"
        for t in r['thresholds']:
            row += f" {t:>8.2f}"
        print(row)

    # Summary
    print(f"\n{'=' * W}")
    print(f"{'TOP 5 MODELS BY AUC':^{W}}")
    print(f"{'=' * W}")
    for i, r in enumerate(results[:5], 1):
        print(f"  #{i}: {r['model']:<20} AUC={r['auc_macro']:.4f}  F1_opt={r['f1_opt']:.4f}  (best epoch {r['pred_epoch']})")

    # Check if best_models/ exists
    best_exports = glob.glob('models/best_models/*.pth')
    if best_exports:
        print(f"\nExported best models: {len(best_exports)}")
        for p in sorted(best_exports):
            print(f"  {os.path.basename(p)}")
    else:
        print(f"\nNote: No .pth exports found in models/best_models/")

if __name__ == '__main__':
    main()
