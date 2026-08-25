"""Unit tests for the persistent storage layer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make app/backend importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from storage import Storage  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = Storage(
        db_path=tmp_path / "x.db",
        image_dir=tmp_path / "img",
        analysis_max=3,
        feedback_max=3,
        image_max_count=3,
        image_max_bytes=200,
    )
    yield s
    s.close()


def test_analysis_roundtrip(store):
    store.put_analysis("a1", {"patient_id": "P1", "timestamp": "2024-01-01", "k": 1})
    got = store.get_analysis("a1")
    assert got["k"] == 1 and got["patient_id"] == "P1"
    assert store.exists_analysis("a1") and not store.exists_analysis("nope")
    assert store.count_analyses() == 1


def test_analysis_count_eviction(store):
    for i in range(5):
        store.put_analysis(f"a{i}", {"i": i, "patient_id": "P", "timestamp": str(i)})
    # max=3, so first two should be evicted
    assert store.count_analyses() == 3
    assert not store.exists_analysis("a0")
    assert not store.exists_analysis("a1")
    assert store.exists_analysis("a4")


def test_by_patient(store):
    store.put_analysis("a1", {"patient_id": "P1", "timestamp": "2024-01-01"})
    store.put_analysis("a2", {"patient_id": "P1", "timestamp": "2024-02-01"})
    store.put_analysis("a3", {"patient_id": "P2", "timestamp": "2024-01-01"})
    p1 = store.by_patient("P1")
    assert len(p1) == 2
    # Sorted DESC by timestamp
    assert p1[0]["timestamp"] == "2024-02-01"


def test_feedback_roundtrip_and_eviction(store):
    for i in range(5):
        store.append_feedback({"feedback_type": "correct" if i % 2 else "incorrect", "i": i})
    fb = store.all_feedback()
    assert len(fb) == 3
    # Oldest two should be evicted
    assert all(f["i"] >= 2 for f in fb)
    assert store.count_feedback() == 3


def test_all_analyses_snapshot_is_independent(store):
    store.put_analysis("a1", {"patient_id": "P", "timestamp": "t"})
    snap = store.all_analyses()
    snap["a1"]["mutated"] = True
    # Snapshot mutation should not affect the store
    assert "mutated" not in store.get_analysis("a1")


def test_image_roundtrip_and_eviction(store):
    store.put_image("a1", b"x" * 80)
    store.put_image("a2", b"x" * 80)
    assert store.get_image("a1") == b"x" * 80
    # Adding another 80 bytes hits the byte cap (200), should evict oldest.
    store.put_image("a3", b"x" * 80)
    # 80*3 = 240 > 200, so oldest evicted
    assert store.get_image("a1") is None
    assert store.get_image("a3") == b"x" * 80


def test_image_path_traversal_rejected(store):
    with pytest.raises(ValueError):
        store.put_image("../../etc/passwd", b"data")
    with pytest.raises(ValueError):
        store.put_image("a/b", b"data")


def test_persistence_across_reopen(tmp_path):
    db = tmp_path / "x.db"
    img = tmp_path / "img"
    s1 = Storage(db_path=db, image_dir=img)
    s1.put_analysis("a1", {"patient_id": "P", "timestamp": "t", "k": 7})
    s1.append_feedback({"feedback_type": "correct", "analysis_id": "a1"})
    s1.put_image("a1", b"hello")
    s1.close()

    s2 = Storage(db_path=db, image_dir=img)
    assert s2.get_analysis("a1")["k"] == 7
    assert len(s2.all_feedback()) == 1
    assert s2.get_image("a1") == b"hello"
    s2.close()


def test_numpy_serialization(store):
    import numpy as np
    store.put_analysis(
        "a1",
        {
            "patient_id": "P",
            "timestamp": "t",
            "scores": np.array([0.1, 0.2, 0.3]),
            "label": np.int64(2),
        },
    )
    got = store.get_analysis("a1")
    assert got["scores"] == [0.1, 0.2, 0.3]
    assert got["label"] == 2


def test_by_patient_respects_limit(store):
    store.put_analysis("a1", {"patient_id": "P1", "timestamp": "2024-01-01"})
    store.put_analysis("a2", {"patient_id": "P1", "timestamp": "2024-02-01"})
    store.put_analysis("a3", {"patient_id": "P1", "timestamp": "2024-03-01"})

    assert len(store.by_patient("P1")) == 3
    limited = store.by_patient("P1", limit=2)
    assert len(limited) == 2
    # LIMIT is applied after ORDER BY timestamp DESC, so we keep the newest.
    assert [r["timestamp"] for r in limited] == ["2024-03-01", "2024-02-01"]


def test_summaries_projects_scalars_only(store):
    """Issue 10: the chat context must not decode stored base64 heatmaps."""
    store.put_analysis("a1", {
        "analysis_id": "a1",
        "patient_id": "P1",
        "timestamp": "2024-01-01",
        "prediction": "Cardiomegaly",
        "confidence": 0.9,
        "model_version": "vit_base",
        "heatmap_gradcam": "x" * 4096,
        "heatmap_overlay": "y" * 4096,
        "top_k_predictions": [{"class_name": "Cardiomegaly", "probability": 0.9}],
    })

    summary = store.summaries()["a1"]
    assert summary == {
        "analysis_id": "a1",
        "patient_id": "P1",
        "timestamp": "2024-01-01",
        "prediction": "Cardiomegaly",
        "confidence": 0.9,
        "model_version": "vit_base",
    }
    # The heavy fields are still retrievable through the full accessor.
    assert store.get_analysis("a1")["heatmap_gradcam"] == "x" * 4096


def test_by_patient_uses_the_patient_index(store):
    """Issue 10: the point of by_patient is the index, not just the filter."""
    store.put_analysis("a1", {"patient_id": "P1", "timestamp": "2024-01-01"})

    plan = store._conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT data_json FROM analyses WHERE patient_id = ? ORDER BY timestamp DESC",
        ("P1",),
    ).fetchall()

    detail = " ".join(str(row[3]) for row in plan)
    assert "idx_analyses_patient" in detail, detail
    assert "SCAN analyses" not in detail, detail


def test_summaries_filters_by_patient_and_limits(store):
    for i in range(4):
        store.put_analysis(f"a{i}", {
            "analysis_id": f"a{i}",
            "patient_id": "P1" if i % 2 else "P2",
            "timestamp": f"2024-01-0{i + 1}",
            "prediction": "Cardiomegaly",
            "confidence": 0.5,
            "model_version": "vit_base",
        })

    assert set(store.summaries(patient_id="P1")) == {"a1", "a3"}
    assert len(store.summaries(limit=2)) == 2


def test_summaries_query_plan_uses_the_index_when_filtering(store):
    store.put_analysis("a1", {"patient_id": "P1", "timestamp": "2024-01-01"})

    plan = store._conn.execute(
        f"EXPLAIN QUERY PLAN SELECT {store._SUMMARY_SQL} FROM analyses WHERE patient_id = ?",
        ("P1",),
    ).fetchall()

    assert "idx_analyses_patient" in " ".join(str(r[3]) for r in plan)


# ---------------------------------------------------------------------------
# Per-kind blob retention (heatmaps vs uploaded images)
# ---------------------------------------------------------------------------

from storage import BLOB_KIND_HEATMAP, BLOB_KIND_UPLOAD  # noqa: E402


@pytest.fixture
def split_store(tmp_path):
    """Small upload budget, generous heatmap budget — the shipped shape."""
    s = Storage(
        db_path=tmp_path / "x.db",
        image_dir=tmp_path / "img",
        image_max_count=3,
        image_max_bytes=10 ** 9,
        heatmap_max_count=1000,
        heatmap_max_bytes=10 ** 9,
    )
    yield s
    s.close()


def test_heatmaps_survive_past_the_old_shared_image_cap(split_store):
    """The regression: heatmaps shared the 200-file upload cap at 4 per analysis.

    Writing more heatmap blobs than the old cap must now retain all of them.
    """
    n = 250  # comfortably past the old DEFAULT_IMAGE_MAX_COUNT of 200
    for i in range(n):
        split_store.put_image(f"XCL-{i:05d}__heatmap_gradcam", b"h" * 32,
                              kind=BLOB_KIND_HEATMAP)

    assert split_store.count_images(BLOB_KIND_HEATMAP) == n
    for i in range(n):
        assert split_store.get_image(
            f"XCL-{i:05d}__heatmap_gradcam", kind=BLOB_KIND_HEATMAP
        ) == b"h" * 32, f"heatmap {i} was evicted under the new budget"


def test_upload_and_heatmap_eviction_are_independent(split_store):
    """Filling one budget must not evict the other kind."""
    for i in range(3):
        split_store.put_image(f"HM-{i}", b"h" * 32, kind=BLOB_KIND_HEATMAP)

    # Overrun the upload cap (3) many times over.
    for i in range(30):
        split_store.put_image(f"IMG-{i:03d}", b"u" * 32, kind=BLOB_KIND_UPLOAD)

    assert split_store.count_images(BLOB_KIND_UPLOAD) == 3, "upload cap not enforced"
    for i in range(3):
        assert split_store.get_image(f"HM-{i}", kind=BLOB_KIND_HEATMAP) is not None, (
            "uploads evicted a heatmap — the budgets are not independent"
        )

    # And the reverse direction.
    kept_uploads = {
        f"IMG-{i:03d}"
        for i in range(30)
        if split_store.get_image(f"IMG-{i:03d}", kind=BLOB_KIND_UPLOAD) is not None
    }
    for i in range(500):
        split_store.put_image(f"HM2-{i:04d}", b"h" * 32, kind=BLOB_KIND_HEATMAP)
    for key in kept_uploads:
        assert split_store.get_image(key, kind=BLOB_KIND_UPLOAD) is not None, (
            "heatmaps evicted an upload — the budgets are not independent"
        )


def test_count_cap_retains_exactly_max_count(split_store):
    """Exactly image_max_count files are kept, not max_count - 1."""
    cap = split_store.image_max_count
    for i in range(cap * 3):
        split_store.put_image(f"IMG-{i:03d}", b"u" * 16, kind=BLOB_KIND_UPLOAD)
    assert split_store.count_images(BLOB_KIND_UPLOAD) == cap


def test_replacing_an_existing_blob_does_not_evict_another(split_store):
    """A same-key rewrite adds no file, so it must not push anything out.

    _evict_blobs_if_needed counted the file it was about to replace as a new
    arrival, so at the cap a rewrite dropped the store to max_count - 1.
    """
    cap = split_store.image_max_count
    for i in range(cap):
        split_store.put_image(f"IMG-{i}", b"u" * 16, kind=BLOB_KIND_UPLOAD)
    assert split_store.count_images(BLOB_KIND_UPLOAD) == cap

    split_store.put_image(f"IMG-{cap - 1}", b"v" * 16, kind=BLOB_KIND_UPLOAD)

    assert split_store.count_images(BLOB_KIND_UPLOAD) == cap, (
        "rewriting an existing key evicted an unrelated blob"
    )
    assert split_store.get_image("IMG-0", kind=BLOB_KIND_UPLOAD) is not None


def test_replacing_a_blob_reclaims_its_bytes(tmp_path):
    """The byte cap must not charge the incoming write on top of what it replaces."""
    s = Storage(
        db_path=tmp_path / "b.db",
        image_dir=tmp_path / "img",
        image_max_count=10 ** 6,
        image_max_bytes=200,
    )
    try:
        s.put_image("A", b"x" * 80, kind=BLOB_KIND_UPLOAD)
        s.put_image("B", b"x" * 80, kind=BLOB_KIND_UPLOAD)
        s.put_image("B", b"y" * 80, kind=BLOB_KIND_UPLOAD)  # same size, in place
        assert s.get_image("A", kind=BLOB_KIND_UPLOAD) is not None, (
            "a same-size rewrite double-counted its bytes and evicted A"
        )
        assert s.get_image("B", kind=BLOB_KIND_UPLOAD) == b"y" * 80
    finally:
        s.close()


def test_has_image_distinguishes_retained_from_evicted(split_store):
    """has_image reports current retention without reading the payload back."""
    for i in range(split_store.image_max_count * 3):
        split_store.put_image(f"IMG-{i:03d}", b"u" * 16, kind=BLOB_KIND_UPLOAD)

    assert split_store.has_image("IMG-000", kind=BLOB_KIND_UPLOAD) is False
    newest = f"IMG-{split_store.image_max_count * 3 - 1:03d}"
    assert split_store.has_image(newest, kind=BLOB_KIND_UPLOAD) is True
    assert split_store.has_image("NEVER-WRITTEN", kind=BLOB_KIND_UPLOAD) is False


def test_legacy_flat_blobs_are_still_readable(tmp_path):
    """Blobs written before the per-kind split live flat and must still load."""
    image_dir = tmp_path / "img"
    image_dir.mkdir(parents=True)
    (image_dir / "OLD-1.bin").write_bytes(b"legacy")

    s = Storage(db_path=tmp_path / "l.db", image_dir=image_dir)
    try:
        assert s.get_image("OLD-1", kind=BLOB_KIND_UPLOAD) == b"legacy"
        assert s.get_image("OLD-1", kind=BLOB_KIND_HEATMAP) == b"legacy"
    finally:
        s.close()
