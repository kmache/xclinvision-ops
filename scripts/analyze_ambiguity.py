#!/usr/bin/env python3
"""Analyze ambiguity in VinBigData annotations for our 4 selected diseases.

Categorizes images into:
  - unanimous_normal: ALL 3 rads say "No finding" (truly normal)
  - consensus_disease: >= 2 rads agree on one of our 4 diseases
  - ambiguous: 1 rad sees our disease but no consensus, AND some rads say normal
  - other_disease_only: rads annotated diseases outside our 4, none say normal
"""
import pandas as pd
import numpy as np

OUR_DISEASES = {'cardiomegaly', 'aortic enlargement', 'pleural thickening', 'pulmonary fibrosis'}
NORMAL_ALIASES = {'no finding', 'normal', 'no_finding', 'healthy'}
MIN_RADS = 2

df = pd.read_csv('data/vinbigdata-chest-xray-abnormalities-detection/train.csv')
print(f'Raw annotations: {len(df)} rows, {df["image_id"].nunique()} images')
print(f'Rad IDs: {sorted(df["rad_id"].unique())}')
print()

df['cn_lower'] = df['class_name'].str.strip().str.lower()

# For each image, categorize per-rad annotations
results = []
for img_id, grp in df.groupby('image_id'):
    # What each rad annotated
    rads_normal = set()        # Rads that annotated "No finding"
    rads_our_disease = {}      # rad -> set of our diseases they found
    rads_other_disease = set() # Rads that annotated diseases NOT in our selected 4
    
    for _, row in grp.iterrows():
        cn = row['cn_lower']
        rid = row['rad_id']
        if cn in NORMAL_ALIASES:
            rads_normal.add(rid)
        elif cn in OUR_DISEASES:
            rads_our_disease.setdefault(rid, set()).add(cn)
        else:
            rads_other_disease.add(rid)
    
    # Count votes per disease (for our 4)
    disease_votes = {}
    for rid, diseases in rads_our_disease.items():
        for d in diseases:
            disease_votes[d] = disease_votes.get(d, 0) + 1
    
    # Check consensus
    has_consensus = any(v >= MIN_RADS for v in disease_votes.values())
    has_any_our_disease = len(rads_our_disease) > 0
    has_normal = len(rads_normal) > 0
    has_other = len(rads_other_disease) > 0
    
    n_rads_total = len(set(r for _, r in grp[['rad_id']].drop_duplicates().itertuples()))
    
    if has_consensus:
        cat = 'consensus_disease'
    elif has_any_our_disease and has_normal:
        # Some rads say disease (our 4), others say normal → AMBIGUOUS
        cat = 'ambiguous_mixed'
    elif has_any_our_disease and not has_normal:
        # 1 rad sees our disease but others see OTHER diseases (not normal)
        cat = 'ambiguous_subthreshold'
    elif not has_any_our_disease and not has_normal:
        # Only other diseases
        cat = 'other_disease_only'
    else:
        # All normal (might also have other diseases)
        if has_other and has_normal:
            cat = 'normal_with_other_disease'
        else:
            cat = 'unanimous_normal'
    
    results.append({
        'image_id': img_id,
        'category': cat,
        'n_rads_normal': len(rads_normal),
        'n_rads_our_disease': len(rads_our_disease),
        'n_rads_other': len(rads_other_disease),
        'disease_votes': str(disease_votes) if disease_votes else '',
    })

df_cat = pd.DataFrame(results)
print('=== Image categorization (15,000 VinBigData images) ===')
print(df_cat['category'].value_counts().to_string())
print()

# Now check what the CURRENT labels.csv did with these
labels = pd.read_csv('data/raw/labels.csv')
disease_cols = ['Cardiomegaly','Aortic enlargement','Pleural thickening','Pulmonary fibrosis']
labels['is_disease'] = labels[disease_cols].sum(axis=1) > 0

# Map image_id from filename
labels['image_id'] = labels['filename'].str.replace(r'\.\w+$', '', regex=True)

# Cross-reference
labels_merged = labels.merge(df_cat[['image_id', 'category']], on='image_id', how='left')

print('=== Current labels.csv: how many images per category are labeled disease? ===')
for cat in df_cat['category'].unique():
    sub = labels_merged[labels_merged['category'] == cat]
    if len(sub) == 0:
        print(f'  {cat}: 0 images in labels.csv (not included)')
    else:
        n_disease = sub['is_disease'].sum()
        print(f'  {cat}: {len(sub)} images, {n_disease} labeled disease ({100*n_disease/len(sub):.1f}%)')

print()

# KEY METRIC: How many ambiguous images are mislabeled?
for cat in ['ambiguous_mixed', 'ambiguous_subthreshold', 'normal_with_other_disease']:
    sub = labels_merged[labels_merged['category'] == cat]
    if len(sub) > 0:
        n_as_normal = (~sub['is_disease']).sum()
        print(f'{cat}: {len(sub)} total, {n_as_normal} currently labeled as normal (all-zero)')

print()

# Also check how many images are NOT in labels.csv
in_labels = set(labels_merged['image_id'])
all_imgs = set(df_cat['image_id'])
not_in_labels = all_imgs - in_labels
print(f'Images NOT in labels.csv: {len(not_in_labels)}')
if not_in_labels:
    cats_missing = df_cat[df_cat['image_id'].isin(not_in_labels)]['category'].value_counts()
    print(cats_missing.to_string())

# Show some ambiguous examples
print('\n=== Example ambiguous images (mixed: some rads say normal, 1 says our disease) ===')
ambig = df_cat[df_cat['category'] == 'ambiguous_mixed'].head(10)
for _, row in ambig.iterrows():
    img_id = row['image_id']
    sub = df[df['image_id']==img_id][['rad_id','cn_lower']].drop_duplicates()
    lab = labels_merged[labels_merged['image_id']==img_id]
    lab_str = 'NOT IN LABELS' if len(lab)==0 else str(lab[disease_cols].values[0])
    print(f'  {img_id[:30]}: current_label={lab_str}')
    for _, r in sub.iterrows():
        print(f'    rad={r["rad_id"]}: {r["cn_lower"]}')
