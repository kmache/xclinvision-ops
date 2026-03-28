#!/usr/bin/env python3
"""Quick diagnostic: try to overfit a small dataset subset."""
import sys, os
sys.path.insert(0, 'src')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from xclinvision.modeling import build_model, _init_classifier_bias, freeze_backbone
from xclinvision.dataset import ChestXrayDataModule
from xclinvision.config import PipelineConfig, get_class_names
from sklearn.metrics import roc_auc_score

# Config
config = PipelineConfig.from_yaml()
class_names = get_class_names()
print(f"Class names: {class_names}")
print(f"Multilabel: {config.multilabel}")

# Load data
dm = ChestXrayDataModule(
    manifest_path='data/processed_1024_v2_4779b5/manifest.csv',
    config=config,
    batch_size=16,
    num_workers=0,
    image_size=384,
)
dm.setup(stage='fit')

# Take a small subset (first 200 train samples)
subset_indices = list(range(200))
subset = torch.utils.data.Subset(dm.train_dataset, subset_indices)
train_loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)

# Build model
model = build_model('densenet', num_classes=4, pretrained=True, dropout=0.4, img_size=384)
_init_classifier_bias(model, bias_value=-2.0)
model.cuda()

# Get class weights
cw = dm.get_class_weights()
print(f"Class weights (pos_weight): {cw.tolist()}")
criterion = nn.BCEWithLogitsLoss(pos_weight=cw.cuda())

# Optimizer: only head params first (like progressive unfreezing)
freeze_backbone(model, unfreeze_head=True)
head_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=1e-4)

# Train 10 epochs on 200 samples — should OVERFIT easily
print("\n=== Overfit test: 200 samples, 10 epochs, head-only ===")
for epoch in range(10):
    model.train()
    total_loss = 0
    all_probs, all_targets = [], []
    
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Compute AUC
    aucs = []
    for i in range(4):
        if len(np.unique(all_targets[:, i])) > 1:
            aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        else:
            aucs.append(0.5)
    avg_loss = total_loss / len(train_loader)
    mean_auc = np.mean(aucs)
    
    # Probability stats
    pos_means = [all_probs[all_targets[:, i]==1, i].mean() if (all_targets[:, i]==1).sum() > 0 else 0 
                 for i in range(4)]
    neg_means = [all_probs[all_targets[:, i]==0, i].mean() if (all_targets[:, i]==0).sum() > 0 else 0
                 for i in range(4)]
    
    print(f"Epoch {epoch:>2}: loss={avg_loss:.4f} AUC={mean_auc:.4f} "
          f"per_class=[{aucs[0]:.3f},{aucs[1]:.3f},{aucs[2]:.3f},{aucs[3]:.3f}] "
          f"pos_mean={np.mean(pos_means):.4f} neg_mean={np.mean(neg_means):.4f}")

# Now try with FULL model unfrozen
print("\n=== Now unfreezing backbone ===")
for p in model.parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0
    all_probs, all_targets = [], []
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    aucs = []
    for i in range(4):
        if len(np.unique(all_targets[:, i])) > 1:
            aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        else:
            aucs.append(0.5)
    avg_loss = total_loss / len(train_loader)
    mean_auc = np.mean(aucs)
    print(f"Epoch {epoch+10:>2}: loss={avg_loss:.4f} AUC={mean_auc:.4f} "
          f"per_class=[{aucs[0]:.3f},{aucs[1]:.3f},{aucs[2]:.3f},{aucs[3]:.3f}]")
