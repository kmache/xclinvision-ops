"""DICOM ingestion: photometric handling, de-identification, and hard failure.

Every DICOM here is synthesised in memory. No DICOM fixture is committed, so the
repository never carries imaging data — real or synthetic — that someone would
later have to audit.

Context: application/dicom was already an accepted upload type, and PIL cannot
open a DICOM, so _safe_open_rgb raised and the caller's `except Exception`
branch persisted the ORIGINAL bytes with every identifying tag intact. These
tests pin that shut.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app" / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from xclinvision.processing import (  # noqa: E402
    DicomError,
    is_dicom,
    read_dicom_safe,
    read_image_grayscale,
)

#: Identifiers stamped into every synthetic file. If any of these survives into
#: a stored record, an audit entry or a persisted blob, a test must fail.
PHI = {
    "PatientName": "DOE^JANE",
    "PatientID": "MRN-12345",
    "PatientBirthDate": "19700101",
    "PatientSex": "F",
    "AccessionNumber": "ACC-99887766",
    "InstitutionName": "St Elsewhere General",
    "ReferringPhysicianName": "HOUSE^GREGORY",
    "StudyDate": "20260101",
}

#: Values long enough to substring-search without false positives. PatientSex is
#: excluded deliberately: "F" occurs inside "False" in any serialised record, so
#: searching for it proves nothing. It is asserted by tag-key absence instead.
PHI_SEARCHABLE = {k: v for k, v in PHI.items() if len(v) > 3}


def make_dicom(
    *,
    photometric: str = "MONOCHROME2",
    modality: str = "CR",
    rows: int = 16,
    cols: int = 16,
    bits_allocated: int = 16,
    bits_stored: int = 12,
    pixels: np.ndarray | None = None,
    with_phi: bool = True,
    **extra,
) -> bytes:
    """Build a minimal but valid single-frame DICOM in memory."""
    ds = Dataset()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CR image storage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = fm

    if with_phi:
        for tag, value in PHI.items():
            setattr(ds, tag, value)

    ds.Modality = modality
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PixelRepresentation = 0
    for key, value in extra.items():
        setattr(ds, key, value)

    if pixels is None:
        pixels = np.linspace(0, (1 << bits_stored) - 1, rows * cols)
    dtype = "<u2" if bits_allocated == 16 else "u1"
    ds.PixelData = pixels.astype(dtype).tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Photometric interpretation
# ---------------------------------------------------------------------------


def test_monochrome1_is_the_inverse_of_monochrome2():
    """MONOCHROME1 means low value = white, so it must be inverted.

    Asserting the two are *inverses* rather than merely different: an
    implementation that inverted both, or neither, would pass a difference check
    while rendering every MONOCHROME1 study upside-down in intensity.
    """
    pixels = np.linspace(0, 4095, 256)
    mono2, _ = read_dicom_safe(make_dicom(photometric="MONOCHROME2", pixels=pixels))
    mono1, _ = read_dicom_safe(make_dicom(photometric="MONOCHROME1", pixels=pixels))

    assert mono2.dtype == np.uint8 and mono1.dtype == np.uint8
    assert np.array_equal(mono1.astype(int), 255 - mono2.astype(int))
    # And not accidentally symmetric (which would make the assertion vacuous).
    assert not np.array_equal(mono1, mono2)


# ---------------------------------------------------------------------------
# 2. Bit depth and rescale
# ---------------------------------------------------------------------------


def test_16bit_12bit_stored_rescales_to_full_8bit_range():
    pixels = np.linspace(0, 4095, 256)
    img, meta = read_dicom_safe(make_dicom(bits_allocated=16, bits_stored=12, pixels=pixels))

    assert img.dtype == np.uint8
    assert img.min() == 0 and img.max() == 255, "12-bit range not mapped onto 0-255"
    assert meta["BitsAllocated"] == 16 and meta["BitsStored"] == 12


def test_rescale_slope_and_intercept_are_applied():
    """A monotonic rescale must not change the normalised output.

    Slope/intercept shift the value range; min-max normalisation then maps it
    back, so the image is identical. What matters is that applying them does not
    corrupt the result — and that the parameters are kept for reproducibility.
    """
    pixels = np.linspace(0, 4095, 256)
    plain, _ = read_dicom_safe(make_dicom(pixels=pixels))
    scaled, meta = read_dicom_safe(
        make_dicom(pixels=pixels, RescaleSlope=2.0, RescaleIntercept=-1024.0)
    )
    assert np.array_equal(plain, scaled)
    assert meta["RescaleSlope"] == 2.0
    assert meta["RescaleIntercept"] == -1024.0


def test_window_centre_and_width_clip_the_range():
    pixels = np.linspace(0, 4095, 256)
    windowed, meta = read_dicom_safe(
        make_dicom(pixels=pixels, WindowCenter=2048.0, WindowWidth=1024.0)
    )
    unwindowed, _ = read_dicom_safe(make_dicom(pixels=pixels))
    assert not np.array_equal(windowed, unwindowed), "window was ignored"
    assert meta["WindowCenter"] == 2048.0 and meta["WindowWidth"] == 1024.0


# ---------------------------------------------------------------------------
# 3. De-identification
# ---------------------------------------------------------------------------


def test_returned_metadata_contains_no_identifying_tag():
    _, meta = read_dicom_safe(make_dicom())
    for tag in PHI:
        assert tag not in meta, f"{tag} survived into the returned metadata"
    for value in PHI_SEARCHABLE.values():
        assert value not in str(meta), f"{value!r} leaked into metadata values"
    # Only decoding parameters carried forward.
    assert set(meta) <= {
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "PhotometricInterpretation",
        "PixelRepresentation",
        "WindowCenter",
        "WindowWidth",
        "RescaleSlope",
        "RescaleIntercept",
        "PixelSpacing",
        "Modality",
    }


def test_private_and_unlisted_tags_do_not_survive():
    """The allowlist must drop tags nobody enumerated, including private ones."""
    raw = make_dicom(DeviceSerialNumber="SN-SECRET-42", StudyDescription="VIP PATIENT")
    _, meta = read_dicom_safe(raw)
    assert "DeviceSerialNumber" not in meta
    assert "StudyDescription" not in meta
    assert "SN-SECRET-42" not in str(meta)
    assert "VIP PATIENT" not in str(meta)


def test_stored_record_and_audit_entry_contain_no_identifiers(client, dummy_image_bytes):
    """End-to-end: upload a tagged DICOM, inspect what was persisted.

    This is the test that fails against the pre-fix code, where the analyze
    endpoint stored the original DICOM bytes verbatim.
    """
    from unittest.mock import MagicMock, patch

    import main  # type: ignore[import-not-found]
    from test_api_integration import _fake_pipeline, _stub_agent  # noqa

    raw = make_dicom(rows=64, cols=64)
    assert is_dicom(raw)
    # The identifiers really are in the upload we are about to send.
    assert b"DOE^JANE" in raw and b"MRN-12345" in raw

    with (
        patch("main.get_pipeline", return_value=_fake_pipeline()),
        patch("main._get_agent", return_value=_stub_agent()),
    ):
        r = client.post(
            "/api/v2/analyze",
            files={"file": ("study.dcm", raw, "application/dicom")},
            data={"patient_id": "ANON-1", "model_name": "efficientnet_b0"},
        )
    assert r.status_code == 200, r.text
    analysis_id = r.json()["analysis_id"]

    stored_blob = main.storage.get_image(analysis_id)
    assert stored_blob is not None, "nothing persisted"
    for needle in (b"DOE^JANE", b"MRN-12345", b"St Elsewhere", b"HOUSE^GREGORY", b"DICM"):
        assert needle not in stored_blob, f"{needle!r} persisted to the blob store"
    assert stored_blob[:8] == b"\x89PNG\r\n\x1a\n", "expected a tag-free PNG re-encode"

    record = main._analysis_store[analysis_id]
    blob = repr(record)
    for value in PHI_SEARCHABLE.values():
        assert value not in blob, f"{value!r} reached the stored analysis record"
    # Short-valued tags are checked by key, not by substring.
    for tag in PHI:
        assert tag not in record, f"{tag} reached the stored analysis record"


# ---------------------------------------------------------------------------
# 4. Modality gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["MR", "CT", "US", "OT", ""])
def test_non_cxr_modality_is_refused(modality):
    with pytest.raises(DicomError) as exc:
        read_dicom_safe(make_dicom(modality=modality))
    assert "modality" in str(exc.value).lower()


@pytest.mark.parametrize("modality", ["CR", "DX"])
def test_cxr_modalities_are_accepted(modality):
    img, meta = read_dicom_safe(make_dicom(modality=modality))
    assert img.size > 0 and meta["Modality"] == modality


def test_analyze_rejects_mr_with_a_clear_message(client):
    from unittest.mock import patch

    from test_api_integration import _fake_pipeline  # noqa

    with patch("main.get_pipeline", return_value=_fake_pipeline()):
        r = client.post(
            "/api/v2/analyze",
            files={"file": ("brain.dcm", make_dicom(modality="MR"), "application/dicom")},
        )
    assert r.status_code == 400
    assert "modality" in r.json()["detail"].lower()
    assert "MR" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Pixel cap
# ---------------------------------------------------------------------------


def test_oversized_pixel_array_is_refused_before_decoding():
    """Rows x Columns is checked from the header, before pixel data is touched.

    The file below declares 9000x9000 but carries a tiny pixel buffer. If the
    guard ran after decoding, the decode would fail first and raise a different
    error; getting the cap message proves the header check ran first.
    """
    raw = make_dicom(rows=9000, cols=9000, pixels=np.zeros(16))
    with pytest.raises(DicomError) as exc:
        read_dicom_safe(raw, max_pixels=1_000_000)
    assert "pixel cap" in str(exc.value)
    assert "9000x9000" in str(exc.value)


def test_pixel_cap_is_enforced_through_the_endpoint(client):
    from unittest.mock import patch

    from test_api_integration import _fake_pipeline  # noqa

    raw = make_dicom(rows=9000, cols=9000, pixels=np.zeros(16))
    with patch("main.get_pipeline", return_value=_fake_pipeline()):
        r = client.post(
            "/api/v2/analyze",
            files={"file": ("huge.dcm", raw, "application/dicom")},
        )
    assert r.status_code in (400, 413)


# ---------------------------------------------------------------------------
# 6. Parse failure must raise, never degrade
# ---------------------------------------------------------------------------


def test_malformed_dicom_raises_instead_of_falling_through():
    """A truncated DICOM must not be resized down the plain-raster path.

    Returning None here is what previously routed a broken DICOM into the
    PNG/JPEG branch, which skips rescale, windowing and photometric handling —
    silently diverging preprocessing from training.
    """
    truncated = b"\x00" * 128 + b"DICM" + b"\x00" * 64
    with pytest.raises(DicomError):
        read_dicom_safe(truncated)
    with pytest.raises(DicomError):
        read_image_grayscale(truncated)


def test_dicom_with_undecodable_pixel_data_raises():
    raw = bytearray(make_dicom(rows=32, cols=32))
    del raw[-200:]  # amputate the pixel buffer, keep a valid header
    with pytest.raises(DicomError):
        read_dicom_safe(bytes(raw))


def test_non_dicom_bytes_are_not_treated_as_dicom(dummy_image_bytes):
    """The raster path keeps its Optional contract; only DICOM raises."""
    assert not is_dicom(dummy_image_bytes)
    assert read_image_grayscale(dummy_image_bytes) is not None
    assert read_image_grayscale(b"not an image at all") is None
