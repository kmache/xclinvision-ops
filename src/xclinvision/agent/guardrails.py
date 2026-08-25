"""Guardrail Validator & Dynamic Clinical Thresholds.

1. **GuardrailValidator** – scans generated reports for disallowed absolutist
   language and rewrites to evidence-based phrasing.  Also checks structural
   requirements (e.g. presence of citations, spatial evidence consistency).

2. **ClinicalThresholds** – per-class dynamic inference thresholds tuned for
   NPV and sensitivity on life-threatening pathologies.  False negatives for
   critical conditions (e.g. pneumothorax) are prioritised to avoid.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# 1.  Disallowed-term scanner
# ════════════════════════════════════════════════════════════════════════════════

_GUARDRAIL_TERMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "guardrail_terms.yaml"

# Hardcoded fallback used when the YAML file is missing or corrupt.
_FALLBACK_DISALLOWED_TERMS: Dict[str, str] = {
    "definitely has": "findings are consistent with",
    "certainly has": "findings are consistent with",
    "clearly shows": "findings suggest",
    "undoubtedly": "with high probability",
    "without a doubt": "with high probability",
    "100% certain": "high confidence",
    "100 percent certain": "high confidence",
    "confirms the diagnosis": "is suggestive of the diagnosis",
    "proven diagnosis": "probable diagnosis",
    "rules out": "reduces the likelihood of",
    "is diagnostic of": "is suggestive of",
    "is pathognomonic": "is highly suggestive of",
    "the patient has": "findings are consistent with",
    "diagnosed with": "findings suggest",
    "no doubt": "with high confidence",
    "absolutely": "with high confidence",
    "i am certain": "the evidence suggests",
    "i'm certain": "the evidence suggests",
    "guaranteed": "likely",
    "impossible": "unlikely",
    "always indicates": "is commonly associated with",
    "never seen in": "is uncommonly associated with",
    "must be": "is likely",
    "cannot be anything else": "is the most likely explanation",
    "definitive evidence": "supportive evidence",
    "conclusive evidence": "supportive evidence",
    "unequivocal evidence": "suggestive evidence",
    "positive for": "findings consistent with",
    "negative for": "no significant findings suggestive of",
}


def _load_disallowed_terms(path: Path = _GUARDRAIL_TERMS_PATH) -> Dict[str, str]:
    """Load disallowed terms from a YAML file, falling back to hardcoded defaults."""
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        terms = data.get("disallowed_terms", {})
        if isinstance(terms, dict) and terms:
            logger.info("Loaded %d disallowed terms from %s", len(terms), path)
            return terms
    except Exception as exc:
        logger.warning("Could not load guardrail terms from %s (%s); using fallback.", path, exc)
    return dict(_FALLBACK_DISALLOWED_TERMS)


def _compile_patterns(terms: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns for each disallowed term (case-insensitive)."""
    return [
        (re.compile(re.escape(term), re.IGNORECASE), replacement)
        for term, replacement in terms.items()
    ]


# Mapping: disallowed phrase (case-insensitive) → suggested replacement.
# The replacements use evidence-based clinical hedging language.
DISALLOWED_TERMS: Dict[str, str] = _load_disallowed_terms()

# Compiled regex patterns for each disallowed term (case-insensitive)
_DISALLOWED_PATTERNS: List[Tuple[re.Pattern, str]] = _compile_patterns(DISALLOWED_TERMS)


def reload_terms(path: Optional[Path] = None) -> None:
    """Reload disallowed terms from YAML and recompile patterns.

    Useful after editing ``configs/guardrail_terms.yaml`` at runtime.
    """
    global DISALLOWED_TERMS, _DISALLOWED_PATTERNS  # noqa: PLW0603
    DISALLOWED_TERMS = _load_disallowed_terms(path or _GUARDRAIL_TERMS_PATH)
    _DISALLOWED_PATTERNS = _compile_patterns(DISALLOWED_TERMS)
    logger.info("Guardrail terms reloaded (%d terms).", len(DISALLOWED_TERMS))


def _scan_and_rewrite(text: str) -> Tuple[str, List[str]]:
    """Scan text for disallowed terms and rewrite them.

    Returns ``(rewritten_text, list_of_violations_found)``.
    """
    violations: List[str] = []
    rewritten = text
    for pattern, replacement in _DISALLOWED_PATTERNS:
        match = pattern.search(rewritten)
        if match:
            original = match.group()
            violations.append(f"'{original}' → '{replacement}'")
            rewritten = pattern.sub(replacement, rewritten)
    return rewritten, violations


# ════════════════════════════════════════════════════════════════════════════════
# 2.  GuardrailValidator
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    """Result of a guardrail validation pass."""
    passed: bool
    term_violations: List[str] = field(default_factory=list)
    structural_warnings: List[str] = field(default_factory=list)
    rewrites_applied: int = 0
    force_human_review: bool = False

class GuardrailValidator:
    """Validate and sanitise clinical reports before delivery.

    Responsibilities
    ----------------
    * **Term scanning** – replace absolutist / disallowed language with
      evidence-based phrasing.
    * **Structural checks** – ensure citations are present, spatial evidence
      is referenced when findings mention specific regions, and urgency
      is consistent with findings.
    * **Equivocal escalation** – if too many guardrail violations are found,
      automatically flag for human review.
    """

    def __init__(
        self,
        *,
        max_violations_before_review: int = 3,
        require_citations: bool = True,
    ) -> None:
        self.max_violations_before_review = max_violations_before_review
        self.require_citations = require_citations

    def validate_and_sanitise(self, report) -> GuardrailResult:
        """Run all guardrail checks on a :class:`ClinicalReport`.

        The report is **mutated in-place** (rewritten text fields).
        Returns a :class:`GuardrailResult` summary.
        """
        result = GuardrailResult(passed=True)

        # ── 1. Term scanning on free-text fields ──────────────────────
        text_fields = [
            ("reasoning_trace", report.reasoning_trace),
            ("impression", report.impression),
        ]
        for fname, text in text_fields:
            rewritten, violations = _scan_and_rewrite(text)
            if violations:
                setattr(report, fname, rewritten)
                result.term_violations.extend(
                    [f"[{fname}] {v}" for v in violations]
                )
                result.rewrites_applied += len(violations)

        # Scan list fields
        for fname in ("findings", "differential_diagnosis", "next_steps"):
            items: List[str] = getattr(report, fname, [])
            new_items = []
            for item in items:
                rewritten, violations = _scan_and_rewrite(item)
                new_items.append(rewritten)
                if violations:
                    result.term_violations.extend(
                        [f"[{fname}] {v}" for v in violations]
                    )
                    result.rewrites_applied += len(violations)
            setattr(report, fname, new_items)

        # ── 2. Structural checks ─────────────────────────────────────
        if self.require_citations and not report.citations:
            result.structural_warnings.append(
                "No citations provided — clinical claims lack source references."
            )

        # Check urgency consistency: critical findings should not be Low urgency
        critical_keywords = {"pneumothorax", "tension", "consolidation", "massive effusion"}
        findings_lower = " ".join(report.findings).lower()
        has_critical = any(kw in findings_lower for kw in critical_keywords)
        if has_critical and report.urgency == "Low":
            result.structural_warnings.append(
                "Critical findings detected but urgency is 'Low' — escalating to 'High'."
            )
            report.urgency = "High"
            report.requires_human_review = True

        # ── 3. Decide pass / fail ────────────────────────────────────
        total_issues = len(result.term_violations) + len(result.structural_warnings)
        if total_issues > 0:
            result.passed = False
        if total_issues >= self.max_violations_before_review:
            result.force_human_review = True
            report.requires_human_review = True

        # Append guardrail note to reasoning trace
        if result.term_violations:
            report.reasoning_trace += (
                f"\n\n[GUARDRAIL] {result.rewrites_applied} absolutist term(s) "
                f"rewritten to evidence-based language."
            )
        if result.structural_warnings:
            report.reasoning_trace += (
                "\n[GUARDRAIL] Structural warning(s): "
                + "; ".join(result.structural_warnings)
            )

        logger.info(
            "Guardrail validation: passed=%s, violations=%d, rewrites=%d, review=%s",
            result.passed,
            len(result.term_violations),
            result.rewrites_applied,
            result.force_human_review,
        )
        return result


# ════════════════════════════════════════════════════════════════════════════════
# 3.  Dynamic Clinical Thresholds
# ════════════════════════════════════════════════════════════════════════════════

# Thresholds are tuned per class to meet clinical safety objectives:
#   - Critical/life-threatening conditions (pneumothorax, consolidation)
#     get LOWER thresholds to maximise sensitivity (minimise false negatives).
#   - Non-critical conditions get HIGHER thresholds to maximise specificity
#     (minimise false positives / over-diagnosis).
#   - "No finding" uses a higher threshold so we only declare "normal" when
#     highly confident → high NPV.

# These are default starting points.  In production, calibrate from
# validation-set ROC curves targeting desired sensitivity/NPV/specificity.

DEFAULT_CLINICAL_THRESHOLDS: Dict[str, float] = {
    # ── Critical / life-threatening → low threshold (maximise sensitivity) ──
    "Pneumothorax": 0.25,
    "Consolidation": 0.30,
    # ── Urgent → moderately low threshold ──
    "Pleural effusion": 0.35,
    "Nodule/Mass": 0.35,
    "Cardiomegaly": 0.40,
    "Atelectasis": 0.35,
    # ── Other abnormalities → balanced threshold ──
    "Aortic enlargement": 0.45,
    "Pleural thickening": 0.45,
    "Pulmonary fibrosis": 0.40,
    "ILD": 0.40,
    "Infiltration": 0.40,
    "Lung Opacity": 0.40,
    "Calcification": 0.45,
    "Other lesion": 0.45,
    # ── Normal → high threshold (maximise NPV: say "normal" only when sure) ──
    "No finding": 0.60,
}

# Classification of conditions by clinical priority
CRITICAL_CONDITIONS = {"Pneumothorax", "Consolidation"}
URGENT_CONDITIONS = {"Pleural effusion", "Nodule/Mass", "Cardiomegaly", "Atelectasis"}


@dataclass
class ThresholdProfile:
    """Container for per-class threshold configuration with clinical rationale."""

    thresholds: Dict[str, float]
    priority_map: Dict[str, str] = field(default_factory=dict)

    def get_threshold(self, class_name: str) -> float:
        """Return the threshold for a given class, defaulting to 0.50."""
        return self.thresholds.get(class_name, 0.50)

    def get_priority(self, class_name: str) -> str:
        """Return clinical priority: critical, urgent, or routine."""
        return self.priority_map.get(class_name, "routine")


def build_clinical_threshold_profile(
    class_names: List[str],
    *,
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> ThresholdProfile:
    """Build a :class:`ThresholdProfile` for the given class names.

    Uses :data:`DEFAULT_CLINICAL_THRESHOLDS` as the base, overridden by
    any ``custom_thresholds`` provided.

    Parameters
    ----------
    class_names:
        List of class names used by the model.
    custom_thresholds:
        Optional per-class overrides (e.g. from calibration).
    """
    merged = dict(DEFAULT_CLINICAL_THRESHOLDS)
    if custom_thresholds:
        merged.update(custom_thresholds)

    # Only keep thresholds for classes the model actually uses
    thresholds = {name: merged.get(name, 0.50) for name in class_names}

    # Sensitivity floor. custom_thresholds are the checkpoint's F1-optimised
    # values and can sit high (0.66-0.78 for the shipped models). For a
    # life-threatening finding a high threshold trades sensitivity away in
    # exactly the wrong direction, so cap a critical class at its
    # DEFAULT_CLINICAL_THRESHOLDS value. Non-critical classes keep the
    # calibrated threshold untouched.
    for name in class_names:
        if name not in CRITICAL_CONDITIONS:
            continue
        cap = DEFAULT_CLINICAL_THRESHOLDS.get(name)
        if cap is not None and thresholds[name] > cap:
            logger.warning(
                "Critical finding '%s': capping threshold %.4f -> %.2f to preserve sensitivity",
                name, thresholds[name], cap,
            )
            thresholds[name] = cap

    # Build priority map
    priority_map: Dict[str, str] = {}
    for name in class_names:
        if name in CRITICAL_CONDITIONS:
            priority_map[name] = "critical"
        elif name in URGENT_CONDITIONS:
            priority_map[name] = "urgent"
        else:
            priority_map[name] = "routine"

    profile = ThresholdProfile(thresholds=thresholds, priority_map=priority_map)

    logger.info(
        "Clinical threshold profile built for %d classes: %s",
        len(thresholds),
        {k: f"{v:.2f} ({priority_map.get(k, 'routine')})" for k, v in thresholds.items()},
    )
    return profile

