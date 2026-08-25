"""Persistent storage layer for analyses, feedback, and images.

Replaces the in-memory ``_analysis_store`` / ``_feedback_store`` /
``_image_store`` dicts with:
  * SQLite-backed analyses + feedback tables (queryable, persistent
    across restarts, FIFO-evicted by row count).
  * Filesystem-backed blob store (one file per blob) with file-count and
    aggregate-byte caps applied *per blob kind*. Uploaded source images and
    derived heatmaps sit in separate subdirectories with separate budgets, so
    a burst of uploads cannot evict the heatmaps older analyses reference.

All public methods are thread-safe. Snapshots (``all_analyses``,
``all_feedback``) materialize lists/dicts so callers iterating outside
the lock cannot trip mid-mutation.

Production migration path: swap ``Storage`` for a Postgres-backed
implementation honoring the same interface; nothing in the app/backend
or xclinvision/agent layers reaches into the dicts directly anymore.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_MAX = 5000
DEFAULT_FEEDBACK_MAX = 10000
DEFAULT_IMAGE_MAX_COUNT = 200
DEFAULT_IMAGE_MAX_BYTES = 256 * 1024 * 1024  # 256 MB

#: Blob kinds. Each has its own retention budget and its own subdirectory, so
#: filling one cannot evict the other. Before the split, heatmaps moved
#: out-of-band shared the 200-file upload cap at 4 blobs per analysis, which cut
#: heatmap retention from every analyses row to roughly 40 of them.
BLOB_KIND_UPLOAD = "uploads"
BLOB_KIND_HEATMAP = "heatmaps"
_BLOB_KINDS = (BLOB_KIND_UPLOAD, BLOB_KIND_HEATMAP)

#: Heatmaps are sized against the analyses row cap, not the upload cap: at four
#: heatmap fields per analysis, DEFAULT_ANALYSIS_MAX rows need 4x that many
#: blobs for retention to match the rows that reference them.
DEFAULT_HEATMAP_MAX_COUNT = 4 * DEFAULT_ANALYSIS_MAX  # 20 000 blobs
DEFAULT_HEATMAP_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB


class Storage:
    """SQLite + filesystem storage for analyses, feedback, and images."""

    def __init__(
        self,
        db_path: Path,
        image_dir: Path,
        *,
        analysis_max: int = DEFAULT_ANALYSIS_MAX,
        feedback_max: int = DEFAULT_FEEDBACK_MAX,
        image_max_count: int = DEFAULT_IMAGE_MAX_COUNT,
        image_max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
        heatmap_max_count: int = DEFAULT_HEATMAP_MAX_COUNT,
        heatmap_max_bytes: int = DEFAULT_HEATMAP_MAX_BYTES,
    ) -> None:
        self.db_path = Path(db_path)
        self.image_dir = Path(image_dir)
        self.analysis_max = analysis_max
        self.feedback_max = feedback_max
        self.image_max_count = image_max_count
        self.image_max_bytes = image_max_bytes
        self.heatmap_max_count = heatmap_max_count
        self.heatmap_max_bytes = heatmap_max_bytes

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN/COMMIT below
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------ schema
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    timestamp TEXT,
                    inserted_at INTEGER NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_analyses_patient
                    ON analyses(patient_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_analyses_inserted
                    ON analyses(inserted_at);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT,
                    feedback_type TEXT,
                    timestamp TEXT,
                    inserted_at INTEGER NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_analysis
                    ON feedback(analysis_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_inserted
                    ON feedback(inserted_at);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --------------------------------------------------------------- analyses
    def put_analysis(self, analysis_id: str, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, default=_json_default)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO analyses "
                "(id, patient_id, timestamp, inserted_at, data_json) "
                "VALUES (?, ?, ?, strftime('%s','now')*1000, ?)",
                (
                    analysis_id,
                    str(data.get("patient_id") or ""),
                    str(data.get("timestamp") or ""),
                    payload,
                ),
            )
            self._evict_analyses_if_needed()

    def _evict_analyses_if_needed(self) -> None:
        cur = self._conn.execute("SELECT COUNT(*) FROM analyses")
        (count,) = cur.fetchone()
        if count > self.analysis_max:
            n_drop = count - self.analysis_max
            self._conn.execute(
                "DELETE FROM analyses WHERE id IN ("
                "  SELECT id FROM analyses ORDER BY inserted_at ASC LIMIT ?"
                ")",
                (n_drop,),
            )

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT data_json FROM analyses WHERE id = ?", (analysis_id,)
            )
            row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def exists_analysis(self, analysis_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)
            )
            return cur.fetchone() is not None

    def count_analyses(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM analyses")
            (n,) = cur.fetchone()
        return int(n)

    def all_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Return a dict-snapshot of all analyses keyed by id (insertion order)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, data_json FROM analyses ORDER BY inserted_at ASC"
            )
            rows = cur.fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def by_patient(
        self, patient_id: str, *, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyses for one patient, newest first, via idx_analyses_patient."""
        sql = (
            "SELECT data_json FROM analyses WHERE patient_id = ? "
            "ORDER BY timestamp DESC"
        )
        params: List[Any] = [patient_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    #: Scalar fields the agent tools actually read off the analysis store.
    #: Projecting to these keeps every row's base64 heatmaps out of Python.
    _SUMMARY_KEYS = (
        "analysis_id", "patient_id", "timestamp",
        "prediction", "confidence", "model_version",
    )
    #: id / patient_id / timestamp are real columns; the rest live in the JSON.
    _SUMMARY_SQL = (
        "id AS analysis_id, patient_id, timestamp, "
        "json_extract(data_json, '$.prediction')    AS prediction, "
        "json_extract(data_json, '$.confidence')    AS confidence, "
        "json_extract(data_json, '$.model_version') AS model_version"
    )

    def summaries(
        self,
        patient_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Scalar-only snapshot of analyses, keyed by id.

        ``all_analyses()`` json.loads every row, each carrying base64 heatmap
        PNGs. The reasoning tools (``_tool_get_monitoring_status``,
        ``_tool_compare_with_history``) need only :attr:`_SUMMARY_KEYS`, so
        project to those in SQL and never decode the payload.

        Filtering by ``patient_id`` uses ``idx_analyses_patient``.
        """
        sql = f"SELECT {self._SUMMARY_SQL} FROM analyses"
        params: List[Any] = []
        if patient_id is not None:
            sql += " WHERE patient_id = ?"
            params.append(patient_id)
        sql += " ORDER BY inserted_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {row[0]: dict(zip(self._SUMMARY_KEYS, row)) for row in rows}

    # --------------------------------------------------------------- feedback
    def append_feedback(self, entry: Dict[str, Any]) -> None:
        payload = json.dumps(entry, default=_json_default)
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback "
                "(analysis_id, feedback_type, timestamp, inserted_at, data_json) "
                "VALUES (?, ?, ?, strftime('%s','now')*1000, ?)",
                (
                    str(entry.get("analysis_id") or ""),
                    str(entry.get("feedback_type") or ""),
                    str(entry.get("timestamp") or ""),
                    payload,
                ),
            )
            self._evict_feedback_if_needed()

    def _evict_feedback_if_needed(self) -> None:
        cur = self._conn.execute("SELECT COUNT(*) FROM feedback")
        (count,) = cur.fetchone()
        if count > self.feedback_max:
            n_drop = count - self.feedback_max
            self._conn.execute(
                "DELETE FROM feedback WHERE id IN ("
                "  SELECT id FROM feedback ORDER BY inserted_at ASC LIMIT ?"
                ")",
                (n_drop,),
            )

    def count_feedback(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM feedback")
            (n,) = cur.fetchone()
        return int(n)

    def all_feedback(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT data_json FROM feedback ORDER BY inserted_at ASC"
            )
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]

    # ------------------------------------------------------------------ blobs
    @staticmethod
    def _safe_key(analysis_id: str) -> str:
        # Defend against path-traversal in analysis_id by sanitizing strictly.
        safe = "".join(c for c in analysis_id if c.isalnum() or c in ("-", "_"))
        if safe != analysis_id or not safe:
            raise ValueError(f"Invalid analysis_id for filesystem: {analysis_id!r}")
        return safe

    def _blob_dir(self, kind: str) -> Path:
        if kind not in _BLOB_KINDS:
            raise ValueError(f"Unknown blob kind: {kind!r}")
        d = self.image_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _blob_caps(self, kind: str) -> tuple:
        if kind == BLOB_KIND_HEATMAP:
            return self.heatmap_max_count, self.heatmap_max_bytes
        return self.image_max_count, self.image_max_bytes

    def _image_path(self, analysis_id: str, kind: str = BLOB_KIND_UPLOAD) -> Path:
        return self._blob_dir(kind) / f"{self._safe_key(analysis_id)}.bin"

    def _legacy_image_path(self, analysis_id: str) -> Path:
        """Flat pre-split location. Read-only: never written to again."""
        return self.image_dir / f"{self._safe_key(analysis_id)}.bin"

    def put_image(self, analysis_id: str, data: bytes, kind: str = BLOB_KIND_UPLOAD) -> None:
        with self._lock:
            path = self._image_path(analysis_id, kind)
            self._evict_blobs_if_needed(path, incoming=len(data), kind=kind)
            path.write_bytes(data)

    def _evict_blobs_if_needed(self, target: Path, incoming: int, kind: str) -> None:
        """FIFO-evict within a single blob kind until the incoming write fits.

        Scoped to ``kind``: uploads and heatmaps hold independent budgets, so a
        burst of uploads can no longer evict the heatmaps that older analyses
        depend on for their spatial evidence.

        ``target`` is excluded from both the file count and the byte total.
        Replacing an existing key adds no file and frees that file's bytes, so
        counting it evicted an unrelated blob for nothing — at the count cap a
        same-key rewrite dropped the store to max_count - 1, and under the byte
        cap it charged the incoming bytes on top of the bytes it was about to
        replace.
        """
        max_count, max_bytes = self._blob_caps(kind)
        directory = self._blob_dir(kind)
        files = sorted(directory.glob("*.bin"), key=lambda p: p.stat().st_mtime)
        # Everything that will still be on disk alongside the incoming write.
        candidates = [p for p in files if p != target]
        total = sum(p.stat().st_size for p in candidates)
        # After the write there will be len(candidates) + 1 files.
        while candidates and (len(candidates) + 1 > max_count or total + incoming > max_bytes):
            oldest = candidates.pop(0)
            try:
                total -= oldest.stat().st_size
                oldest.unlink()
            except FileNotFoundError:
                pass

    def get_image(self, analysis_id: str, kind: str = BLOB_KIND_UPLOAD) -> Optional[bytes]:
        with self._lock:
            try:
                p = self._image_path(analysis_id, kind)
            except ValueError:
                return None
            if p.exists():
                return p.read_bytes()
            # Blobs written before the per-kind split still live flat.
            legacy = self._legacy_image_path(analysis_id)
            if legacy.exists():
                return legacy.read_bytes()
            return None

    def has_image(self, analysis_id: str, kind: str = BLOB_KIND_UPLOAD) -> bool:
        """Whether a blob is retained right now.

        Lets a caller tell "evicted" from "never written" without paying to
        read the payload back.
        """
        with self._lock:
            try:
                p = self._image_path(analysis_id, kind)
            except ValueError:
                return False
            return p.exists() or self._legacy_image_path(analysis_id).exists()

    def count_images(self, kind: str = BLOB_KIND_UPLOAD) -> int:
        with self._lock:
            return len(list(self._blob_dir(kind).glob("*.bin")))

    def image_dir_size(self, kind: Optional[str] = None) -> int:
        """Bytes on disk for one kind, or for every kind plus legacy blobs."""
        with self._lock:
            if kind is not None:
                return sum(p.stat().st_size for p in self._blob_dir(kind).glob("*.bin"))
            total = sum(p.stat().st_size for p in self.image_dir.glob("*.bin"))
            for k in _BLOB_KINDS:
                total += sum(p.stat().st_size for p in self._blob_dir(k).glob("*.bin"))
            return total


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays / sets used in analyses."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        # Bytes don't belong in analysis JSON; fail loudly.
        raise TypeError("bytes payloads must go through put_image, not put_analysis")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_storage_from_env() -> Storage:
    """Construct a Storage using environment defaults.

    Env:
      XCLINVISION_DB_PATH      - SQLite file path (default: data/xclinvision.db)
      XCLINVISION_IMAGE_DIR    - Image directory  (default: data/images/)
      ANALYSIS_STORE_MAX       - row cap for analyses (default 5000)
      FEEDBACK_STORE_MAX       - row cap for feedback (default 10000)
      IMAGE_STORE_MAX          - uploaded-image file-count cap (default 200)
      IMAGE_STORE_MAX_BYTES    - uploaded-image byte cap (default 256 MB)
      HEATMAP_STORE_MAX        - heatmap blob-count cap (default 20 000)
      HEATMAP_STORE_MAX_BYTES  - heatmap byte cap (default 4 GB)

    Heatmaps are budgeted separately from uploads: they are derived artefacts,
    four per analysis, and are referenced by every analyses row, so they need a
    budget proportional to ANALYSIS_STORE_MAX rather than to the upload cap.
    """
    db_path = Path(os.environ.get("XCLINVISION_DB_PATH", "data/xclinvision.db"))
    image_dir = Path(os.environ.get("XCLINVISION_IMAGE_DIR", "data/images"))
    return Storage(
        db_path=db_path,
        image_dir=image_dir,
        analysis_max=int(os.environ.get("ANALYSIS_STORE_MAX", DEFAULT_ANALYSIS_MAX)),
        feedback_max=int(os.environ.get("FEEDBACK_STORE_MAX", DEFAULT_FEEDBACK_MAX)),
        image_max_count=int(os.environ.get("IMAGE_STORE_MAX", DEFAULT_IMAGE_MAX_COUNT)),
        image_max_bytes=int(os.environ.get("IMAGE_STORE_MAX_BYTES", DEFAULT_IMAGE_MAX_BYTES)),
        heatmap_max_count=int(os.environ.get("HEATMAP_STORE_MAX", DEFAULT_HEATMAP_MAX_COUNT)),
        heatmap_max_bytes=int(os.environ.get("HEATMAP_STORE_MAX_BYTES", DEFAULT_HEATMAP_MAX_BYTES)),
    )


__all__ = [
    "Storage",
    "build_storage_from_env",
    "BLOB_KIND_UPLOAD",
    "BLOB_KIND_HEATMAP",
]
