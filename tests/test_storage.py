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
