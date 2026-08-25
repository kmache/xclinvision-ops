# Model evaluation and calibration

Measured results for the four checkpoints in `models/best_models/`. Every number here
was produced by executed code, not estimated. Reproduce with:

```bash
python scripts/calibrate_checkpoints.py --dry-run     # report, write nothing
python scripts/calibrate_checkpoints.py               # refit and write the payloads
```

- **Dataset** — VinBigData Chest X-ray, 14,304 images, 4 findings, multilabel.
- **Splits** — 10,020 train / 2,133 val / 2,151 test (70/15/15), iterative stratification.
  Prevalence agrees across splits to within 0.15 pp on every class.
- **Calibration is fitted on validation only.** Test is scored afterwards as held-out
  confirmation and never influences a parameter or the calibrated/uncalibrated verdict.

## Class support (validation)

| Class | positives | prevalence | AP baseline |
|---|---:|---:|---:|
| Cardiomegaly | 273 / 2,133 | 12.80% | 0.128 |
| Aortic enlargement | 352 / 2,133 | 16.50% | 0.165 |
| Pleural thickening | 132 / 2,133 | 6.19% | 0.062 |
| Pulmonary fibrosis | 153 / 2,133 | 7.17% | 0.072 |

Read average precision against prevalence, never against 0.5.

## Discrimination — unchanged by calibration

| Model | Cardiomegaly | Aortic enl. | Pleural thick. | Pulm. fibrosis |
|---|---:|---:|---:|---:|
| convnext_small | 0.925 | 0.937 | 0.881 | 0.858 |
| densenet | 0.929 | 0.928 | 0.855 | 0.845 |
| efficientnet_b0 | 0.924 | 0.927 | 0.864 | 0.870 |
| vit_base | **0.957** | **0.959** | **0.904** | 0.898 |

Per-class ROC-AUC. Calibration is a strictly increasing per-class map, so it cannot
change ranking — these values are identical before and after, which is the point: the
models always ranked well. What was broken was the number attached to the ranking.

## The calibration problem

Every checkpoint shipped with `temperature: null`, so `_apply_temperature` was a no-op
and served probabilities were raw sigmoid outputs. Measured per-class ECE ran 0.15–0.44
while AUC ran 0.845–0.959, and mean predicted probability exceeded the true positive rate
by 2.0–7.5× on every class.

Two things did **not** fix it:

- **A single global temperature.** Fitted values were 1.0146 / 1.1059 / 0.9815 / 1.0029 —
  a no-op. One scalar averages four different per-class errors.
- **Per-class temperature alone.** Improves mean ECE, but only to 0.17–0.28, and three
  individual classes get *worse*. Temperature divides logits; it cannot shift them, and
  the error here is a shift.

The shift is a training artefact. `MultilabelFocalLoss` received `pos_weight` of
6.88 / 5.10 / 15.21 / 13.07 (n_neg/n_pos on the train split), which biases every head
toward positive. The fitted bias term recovers exactly that: **negative on all 16
model×class cells**, largest where `pos_weight` was largest.

## What is fitted

Per class, `p = sigmoid(z / T + b)` — temperature scaling is the `b = 0` special case.
Fitted by L-BFGS on per-class binary NLL over the validation split, stored in the
checkpoint payload as `temperature` (list), `calibration_bias` (list),
`calibration_status`, and `calibration_ece`.

| Model | T (per class) | bias (per class) |
|---|---|---|
| convnext_small | 0.721, 0.583, 0.875, 0.804 | −1.913, −1.704, −2.084, −2.411 |
| densenet | 0.575, 0.552, 0.654, 0.663 | −1.976, −1.697, −3.410, −2.909 |
| efficientnet_b0 | 0.599, 0.542, 0.650, 0.542 | −2.763, −2.144, −2.578, −2.753 |
| vit_base | 0.531, 0.463, 0.567, 0.492 | −1.720, −1.667, −2.121, −4.177 |

Class order: Cardiomegaly, Aortic enlargement, Pleural thickening, Pulmonary fibrosis.

## ECE before and after — held out on test

| Model / class | val before | val after | **test before** | **test after** |
|---|---:|---:|---:|---:|
| convnext_small · Cardiomegaly | 0.1904 | 0.0249 | 0.1906 | **0.0137** |
| convnext_small · Aortic enl. | 0.1712 | 0.0177 | 0.1706 | **0.0216** |
| convnext_small · Pleural thick. | 0.1821 | 0.0095 | 0.1784 | **0.0111** |
| convnext_small · Pulm. fibrosis | 0.2569 | 0.0093 | 0.2568 | **0.0125** |
| densenet · Cardiomegaly | 0.2124 | 0.0217 | 0.2091 | **0.0208** |
| densenet · Aortic enl. | 0.1891 | 0.0171 | 0.1887 | **0.0203** |
| densenet · Pleural thick. | 0.4009 | 0.0112 | 0.4029 | **0.0076** |
| densenet · Pulm. fibrosis | 0.3607 | 0.0093 | 0.3646 | **0.0158** |
| efficientnet_b0 · Cardiomegaly | 0.2952 | 0.0194 | 0.2874 | **0.0171** |
| efficientnet_b0 · Aortic enl. | 0.2399 | 0.0171 | 0.2352 | **0.0217** |
| efficientnet_b0 · Pleural thick. | 0.2974 | 0.0127 | 0.2957 | **0.0131** |
| efficientnet_b0 · Pulm. fibrosis | 0.3278 | 0.0082 | 0.3226 | **0.0073** |
| vit_base · Cardiomegaly | 0.1502 | 0.0160 | 0.1499 | **0.0144** |
| vit_base · Aortic enl. | 0.1663 | 0.0147 | 0.1690 | **0.0119** |
| vit_base · Pleural thick. | 0.2218 | 0.0117 | 0.2220 | **0.0148** |
| vit_base · Pulm. fibrosis | 0.4378 | 0.0112 | 0.4326 | **0.0194** |

Mean per model:

| Model | val before → after | test before → after | verdict |
|---|---|---|---|
| convnext_small | 0.2002 → 0.0153 | 0.1991 → **0.0147** | calibrated |
| densenet | 0.2908 → 0.0148 | 0.2913 → **0.0161** | calibrated |
| efficientnet_b0 | 0.2901 → 0.0144 | 0.2852 → **0.0148** | calibrated |
| vit_base | 0.2440 → 0.0134 | 0.2434 → **0.0151** | calibrated |

All 16 cells improve. Test tracks validation to ~0.005, so the fit generalises rather
than memorising the split it was fitted on.

### When a checkpoint stays "uncalibrated"

`calibration_status` flips to `calibrated` only when mean **validation** ECE both improves
*and* falls at or below **0.05** (`--ece-max`). Improvement alone is not enough: 0.26 is
better than 0.29 and still means a reported 0.80 is nowhere near an 80% chance of the
finding. A checkpoint that fails the bar keeps its fitted parameters — they are still
applied — but continues to report `uncalibrated`, so `ClinicalReporter` keeps rendering
"Raw model probability: … (not calibrated)" instead of clinical certainty language.

All four checkpoints currently pass.

## Thresholds re-derived from calibrated probabilities

Critical findings are tuned to the lowest threshold reaching recall ≥ 0.90; every other
class is tuned for maximum F1.

**`CRITICAL_CONDITIONS` is `{Pneumothorax, Consolidation}`, and neither is a served
class**, so all four classes currently take the max-F1 rule and the recall branch is
unreachable. The clinical floor in `agent/guardrails.py` is likewise a no-op for this
class list — it still applies, it simply has nothing to cap. Both activate unchanged if a
critical class is added to `model.class_names`.

| Model | Cardiomegaly | Aortic enl. | Pleural thick. | Pulm. fibrosis |
|---|---:|---:|---:|---:|
| convnext_small | 0.240 | 0.366 | 0.279 | 0.242 |
| densenet | 0.257 | 0.282 | 0.184 | 0.163 |
| efficientnet_b0 | 0.297 | 0.336 | 0.216 | 0.241 |
| vit_base | 0.321 | 0.420 | 0.306 | 0.348 |

Thresholds dropped from the previous 0.61–0.79 range because the probabilities they act
on are no longer inflated. A lower number on a calibrated scale is not a laxer test.

## Operating-point effect (validation, macro F1)

| Model | before | after | Δ |
|---|---:|---:|---:|
| convnext_small | 0.537 | 0.563 | +0.027 |
| densenet | 0.506 | 0.530 | +0.024 |
| efficientnet_b0 | 0.523 | 0.539 | +0.016 |
| vit_base | **0.595** | **0.626** | +0.031 |

Calibration was not undertaken to raise F1 — ranking is unchanged, so the gain comes only
from thresholds now sitting in the right place. The real result is that a served
probability now means what it says.

## What this does not fix

Calibration corrects the *number*, not the *separability*. From the diagnostic:
Cardiomegaly and Aortic enlargement remain individually usable (AP 0.63–0.81); Pleural
thickening and Pulmonary fibrosis remain weak (AP 0.30–0.52, precision 0.14–0.18 at 90%
recall). Those two classes are now honestly calibrated *and* still not clinically usable
on their own. No threshold or calibration choice changes that — it needs more positive
support or a stronger model.

Other open items carried from the diagnostic: backbones are ImageNet-pretrained, not
medical-domain; VinBigData ships no patient identifier, so splits are image-level and
patient-level leakage cannot be positively excluded (no duplicate or near-duplicate
imagery was found across splits).
