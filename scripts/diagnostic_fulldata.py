#!/usr/bin/env python3
"""Diagnostic: train on FULL dataset, compare simple loop vs PL conditions."""
import sys, os
sys.path.insert(0, 'src')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from xclinvision.modeling import build_model, _init_classifier_bias, freeze_backbone, unfreeze_layers
from xclinvision.dataset import ChestXrayDataModule
from xclinvision.config import PipelineConfig, get_class_names
from sklearn.metrics import roc_auc_score

config = PipelineConfig.from_yaml()

# Same setup as actual training
dm = ChestXrayDataModule(
    manifest_path='data/processed_1024_v2_4779b5/manifest.csv',
    config=config,
    batch_size=24,
    num_workers=4,
    image_size=384,
    use_weighted_sampler=True,
)
dm.setup(stage='fit')

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

model = build_model('densenet', num_classes=4, pretrained=True, dropout=0.4, img_size=384)
_init_classifier_bias(model, bias_value=-2.0)
model.cuda()
model.train()

cw = dm.get_class_weights()
print(f"Class weights: {cw.tolist()}")
criterion = nn.BCEWithLogitsLoss(pos_weight=cw.cuda())

# Phase 1: Head only (epochs 0-2)
freeze_backbone(model, unfreeze_head=True)
head_params = [p for p in model.parameters() if p.requires_grad]
print(f"Head trainable params: {sum(p.numel() for p in head_params)}")
optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=1e-4)

def evaluate(model, val_loader):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.cuda()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    aucs = []
    for i in range(4):
        if len(np.unique(all_targets[:, i])) > 1:
            aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        else:
            aucs.append(0.5)
    model.train()
    return np.mean(aucs), aucs

ACCUMULATE = 3

print("\n=== Phase 1: Head only, 3 epochs ===")
for epoch in range(3):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float()) / ACCUMULATE
        loss.backward()
        total_loss += loss.item() * ACCUMULATE
        n_batches += 1
        if (batch_idx + 1) % ACCUMULATE == 0:
            optimizer.step()
            optimizer.zero_grad()
    # Handle remaining gradients
    if n_batches % ACCUMULATE != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    val_auc, per_class = evaluate(model, val_loader)
    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch}: loss={avg_loss:.4f} val_AUC={val_auc:.4f} per_class={[f'{a:.3f}' for a in per_class]}")

# Phase 2: Full fine-tuning
print("\n=== Phase 2: Full fine-tuning, 7 more epochs ===")
unfreeze_layers(model, num_layers=0)  # Unfreeze all
backbone_params, head_params_new = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(k in name for k in ("head", "fc", "classifier", "last_linear")):
        head_params_new.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-5},
    {"params": head_params_new, "lr": 5e-4},
], weight_decay=1e-4)

for epoch in range(3, 10):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float()) / ACCUMULATE
        loss.backward()
        total_loss += loss.item() * ACCUMULATE
        n_batches += 1
        if (batch_idx + 1) % ACCUMULATE == 0:
            optimizer.step()
            optimizer.zero_grad()
    if n_batches % ACCUMULATE != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    val_auc, per_class = evaluate(model, val_loader)
    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch}: loss={avg_loss:.4f} val_AUC={val_auc:.4f} per_class={[f'{a:.3f}' for a in per_class]}")

print("\nDone!")
