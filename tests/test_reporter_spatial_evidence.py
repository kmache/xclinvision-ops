"""Regression test for the clinical report spatial-evidence panel.

Issue #3: the template hard-coded raw region keys ('right_upper', etc.)
but ClinicalReport.spatial_evidence is keyed by human-readable labels
('right upper lobe', etc.) produced by interpret_xai_regions, so the
panel always rendered fixed '0.00' placeholders. This test renders
the template directly and asserts populated regions reach the output.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader


def _render_minimal(spatial_evidence):
    template_dir = pathlib.Path(__file__).resolve().parent.parent / (
        "src/xclinvision/agent/templates"
    )
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    tpl = env.get_template("clinical_report.html")
    # Mirror the attributes the template touches; SimpleNamespace lets the
    # template's `report.spatial_evidence` / `report.findings` etc. resolve
    # without instantiating the full Pydantic ClinicalReport.
    report = SimpleNamespace(
        findings=[],
        spatial_evidence=spatial_evidence,
        reasoning_trace="",
        differential_diagnosis=[],
        impression="",
        urgency="Low",
        next_steps=[],
        citations=[],
        requires_human_review=False,
        temporal_changes=None,
    )
    return tpl.render(
        report=report,
        calibrated=[],
        original_b64="",
        gradcam_b64="",
        scorecam_b64="",
        attention_b64="",
        radar_b64="",
        patient_meta={},
        indication="",
        conversation_log=[],
        comments="",
        report_id="test-id",
        generated_at="2026-05-08 00:00 UTC",
    )


def test_spatial_evidence_human_readable_keys_render():
    spatial = {
        "right upper lobe": "high activation (0.72)",
        "cardiac silhouette": "moderate activation (0.20)",
    }
    html = _render_minimal(spatial)

    # The actual region labels and their descriptions must appear in HTML
    assert "right upper lobe" in html
    assert "high activation (0.72)" in html
    assert "cardiac silhouette" in html
    assert "moderate activation (0.20)" in html


def test_spatial_evidence_panel_hidden_when_empty():
    html = _render_minimal({})
    # Outer {% if report.spatial_evidence %} should suppress the heading
    assert "Spatial Evidence (XAI)" not in html


def test_spatial_evidence_no_stale_zero_placeholders():
    """Raw region keys with literal '0.00' must not appear when data is present."""
    spatial = {"right upper lobe": "high activation (0.72)"}
    html = _render_minimal(spatial)
    # The old template emitted hard-coded labels regardless of data; ensure
    # none of them leak into the output.
    for stale in ("right_upper:", "right_middle:", "left_upper:", "cardiac:", "apical:"):
        assert stale not in html
