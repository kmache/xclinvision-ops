#!/usr/bin/env python3
"""Diagnostic: Simple training loop on FULL dataset (no PL) to isolate AUC issue.

Tests:
- Full train set (10K+), eval on val set (2K+)
- FP32 (no mixed precision)
- No weighted sampler
- No gradient accumulation
- Simple constant LR with head-only then full fine-tune
"""
import sys, os
sys.path.insert(0, 'src')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from xclinvision.modeling import build_model, _init_classifier_bias, freeze_backbone, unfreeze_layers
from xclinvision.dataset import ChestXrayDataModule
from xclinvision.config import PipelineConfig, get_class_names
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True

# Config
config = PipelineConfig.from_yaml()
class_names = get_class_names()
disease_names = [n for n in class_names if n.lower() != 'no finding']
num_classes = len(disease_names)
disease_config = PipelineConfig(
    class_names=class_names,
    classification_mode=config.classification_mode,
    clinical_rules={k: v for k, v in config.clinical_rules.items() if k != 'no finding'},
)

print(f"Classes: {disease_names} (n={num_classes})")

# Load data - use 224px to avoid OOM, no weighted sampler
dm = ChestXrayDataModule(
    manifest_path='data/processed_1024_v2_4779b5/manifest.csv',
    config=disease_config,
    batch_size=32,
    num_workers=4,
    image_size=224,
    cache_size=0,
    use_weighted_sampler=False,
)
dm.setup(stage='fit')

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
print(f"Train: {len(dm.train_dataset)} samples, {len(train_loader)} batches")
print(f"Val: {len(dm.val_dataset)} samples, {len(val_loader)} batches")

# Build model
model = build_model('densenet', num_classes=num_classes, pretrained=True, dropout=0.3, img_size=224)
_init_classifier_bias(model, bias_value=-2.0)
model.cuda()

# Class weights
cw = dm.get_class_weights()
print(f"pos_weight: {cw.tolist()}")
criterion = nn.BCEWithLogitsLoss(pos_weight=cw.cuda())


def evaluate(model, loader):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    aucs = []
    for i in range(num_classes):
        if len(np.unique(all_targets[:, i])) > 1:
            aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        else:
            aucs.append(0.5)
    return np.mean(aucs), aucs, all_probs, all_targets


# ============ Phase 1: Head only, 5 epochs ============
print("\n=== Phase 1: Head-only training, 5 epochs, lr=5e-4, FP32 ===")
freeze_backbone(model, unfreeze_head=True)
head_params = [p for p in model.parameters() if p.requires_grad]
trainable = sum(p.numel() for p in head_params)
print(f"Trainable params: {trainable:,}")

optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=1e-4)

for epoch in range(5):
    model.train()
    total_loss = 0
    n_batches = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    val_auc, per_class, val_probs, val_targets = evaluate(model, val_loader)

    pos_means = [val_probs[val_targets[:, i] == 1, i].mean() for i in range(num_classes)]
    neg_means = [val_probs[val_targets[:, i] == 0, i].mean() for i in range(num_classes)]
    gap = np.mean(pos_means) - np.mean(neg_means)

    print(f"  Epoch {epoch}: loss={avg_loss:.4f} val_AUC={val_auc:.4f} "
          f"per_class=[{','.join(f'{a:.3f}' for a in per_class)}] "
          f"prob_gap={gap:.4f}")

# ============ Phase 2: Full fine-tune, 10 epochs ============
print("\n=== Phase 2: Full fine-tune, 10 epochs, head=5e-4, backbone=5e-5 ===")
unfreeze_layers(model, num_layers=0)
all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {all_trainable:,}")

backbone_params, head_params_ft = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(k in name for k in ("head", "fc", "classifier", "last_linear")):
        head_params_ft.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-5},
    {"params": head_params_ft, "lr": 5e-4},
], weight_decay=1e-4)

best_val_auc = 0
for epoch in range(10):
    model.train()
    total_loss = 0
    n_batches = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    val_auc, per_class, val_probs, val_targets = evaluate(model, val_loader)
    best_val_auc = max(best_val_auc, val_auc)

    pos_means = [val_probs[val_targets[:, i] == 1, i].mean() for i in range(num_classes)]
    neg_means = [val_probs[val_targets[:, i] == 0, i].mean() for i in range(num_classes)]
    gap = np.mean(pos_means) - np.mean(neg_means)

    print(f"  Epoch {epoch+5}: loss={avg_loss:.4f} val_AUC={val_auc:.4f} "
          f"per_class=[{','.join(f'{a:.3f}' for a in per_class)}] "
          f"prob_gap={gap:.4f}")

print(f"\n=== RESULT: Best val AUC = {best_val_auc:.4f} ===")
if best_val_auc > 0.75:
    print("CONCLUSION: Simple loop works! Issue is in PL pipeline (scheduler/callbacks/etc).")
elif best_val_auc > 0.60:
    print("CONCLUSION: Partial learning. May need more epochs or tuning.")
else:
    print("CONCLUSION: Even simple loop fails. Issue is in data/labels/transforms.")
