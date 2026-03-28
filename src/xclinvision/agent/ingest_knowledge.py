#!/usr/bin/env python3
"""
Clinical Knowledge Base Ingestion Pipeline
==========================================
Phase 1 of the ClinicalReasoningAgent: builds a high-quality, searchable
vector store that enables reliable retrieval for clinical reasoning.

Creates:
    data/vectorstore/chroma/         ← ChromaDB persistent store (semantic)
    data/vectorstore/bm25_index.pkl  ← Serialised BM25 index (keyword)

Data sources (auto-discovered from --data-dirs):
    - XML        : Indiana University CXR reports (FINDINGS + IMPRESSION)
    - PDF        : Clinical guidelines, textbooks, research papers
    - CSV / Excel: Tabular data (annotations, metadata, datasets)
    - JSON / JSONL: Structured clinical records, API exports
    - HTML       : Web-scraped clinical resources, Radiopaedia pages
    - Images     : DICOM metadata extraction, OCR on scanned documents
    - Built-in   : Radiopaedia summaries, clinical guidelines, synthetic signals

Usage:
    python scripts/ingest_knowledge.py \
        --reports-dir NLMCXR_reports/ecgen-radiology \
        --data-dirs data/raw/knowledge \
        --output-dir data/vectorstore \
        --embedding-model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import mimetypes
import os
import pickle
import re
import sys
import pandas as pd
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import tiktoken

# Optional heavy dependencies — gracefully degrade when absent
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]
    BS4_AVAILABLE = False

try:
    import fitz as pymupdf  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    pymupdf = None  # type: ignore[assignment]
    PYMUPDF_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    pydicom = None  # type: ignore[assignment]
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_knowledge")

# ════════════════════════════════════════════════════════════════════════════════
# Domain constants – the 14 VinBigData chest-X-ray abnormalities
# ════════════════════════════════════════════════════════════════════════════════

VINBIG_CLASSES: List[str] = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

# ════════════════════════════════════════════════════════════════════════════════
# Data models
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class ClinicalDocument:
    """Semi-structured clinical document."""

    doc_id: str
    source: str  # iu_cxr | radiopaedia | guideline | synthetic
    findings: str
    impression: str
    indication: str = ""
    comparison: str = ""
    conditions: List[str] = field(default_factory=list)
    anatomy: List[str] = field(default_factory=list)
    urgency: str = "routine"  # critical | urgent | routine
    mesh_terms: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A chunk ready for vector-store ingestion."""

    chunk_id: str
    text: str
    chunk_type: str  # narrative | structured
    metadata: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════════
# 1.  Data-source parsers
# ════════════════════════════════════════════════════════════════════════════════

# ── 1a. Indiana University CXR XML reports ────────────────────────────────────

MESH_TO_CONDITION: Dict[str, str] = {
    "cardiomegaly": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "aortic enlargement": "Aortic enlargement",
    "aortic ectasia": "Aortic enlargement",
    "tortuous aorta": "Aortic enlargement",
    "atelectasis": "Atelectasis",
    "calcification": "Calcification",
    "calcified": "Calcification",
    "consolidation": "Consolidation",
    "airspace disease": "Consolidation",
    "interstitial lung disease": "ILD",
    "ild": "ILD",
    "interstitial": "ILD",
    "infiltrate": "Infiltration",
    "infiltration": "Infiltration",
    "lung opacity": "Lung Opacity",
    "opacity": "Lung Opacity",
    "opacification": "Lung Opacity",
    "nodule": "Nodule/Mass",
    "mass": "Nodule/Mass",
    "nodule/mass": "Nodule/Mass",
    "pleural effusion": "Pleural effusion",
    "effusion": "Pleural effusion",
    "pleural thickening": "Pleural thickening",
    "pneumothorax": "Pneumothorax",
    "pulmonary fibrosis": "Pulmonary fibrosis",
    "fibrosis": "Pulmonary fibrosis",
}

ANATOMY_PATTERNS: Dict[str, str] = {
    "right upper lobe": "right_upper",
    "right middle lobe": "right_middle",
    "right lower lobe": "right_lower",
    "left upper lobe": "left_upper",
    "left lower lobe": "left_lower",
    "lingula": "left_middle",
    "right lung": "right_lung",
    "left lung": "left_lung",
    "bilateral": "bilateral",
    "right hemithorax": "right_lung",
    "left hemithorax": "left_lung",
    "right hilum": "right_hilum",
    "left hilum": "left_hilum",
    "hilar": "bilateral_hilum",
    "mediastinum": "mediastinum",
    "mediastinal": "mediastinum",
    "cardiac": "cardiac",
    "heart": "cardiac",
    "aorta": "aorta",
    "aortic": "aorta",
    "pleural": "pleural",
    "costophrenic": "costophrenic",
    "diaphragm": "diaphragm",
    "apex": "apex",
    "apical": "apex",
    "base": "base",
    "basal": "base",
}

CRITICAL_CONDITIONS = {"Pneumothorax", "Consolidation"}
URGENT_CONDITIONS = {"Pleural effusion", "Nodule/Mass", "Cardiomegaly", "Atelectasis"}
CRITICAL_KEYWORDS = [
    "tension",
    "massive",
    "large effusion",
    "emergency",
    "acute respiratory",
    "severe",
    "life-threatening",
]


def _map_mesh_to_conditions(mesh_terms: List[str]) -> List[str]:
    conditions: set[str] = set()
    for term in mesh_terms:
        term_lower = term.lower()
        for key, cond in MESH_TO_CONDITION.items():
            if key in term_lower:
                conditions.add(cond)
    return sorted(conditions)


def _extract_conditions_from_text(text: str) -> List[str]:
    """Fallback: scan free text for condition mentions, ignoring negated phrases."""
    conditions: set[str] = set()
    text_lower = text.lower()
    
    # Simple negation window: look at the 4-5 words before a term
    # E.g., "no evidence of acute pneumonia"
    negation_pattern = re.compile(r'\b(no|not|without|negative|clear|resolved)\b')
    
    for key, cond in MESH_TO_CONDITION.items():
        for match in re.finditer(re.escape(key), text_lower):
            start_idx = max(0, match.start() - 30) # Look back ~30 chars
            context_window = text_lower[start_idx:match.start()]
            
            if not negation_pattern.search(context_window):
                conditions.add(cond)
                
    return sorted(conditions)


def _extract_anatomy(text: str) -> List[str]:
    text_lower = text.lower()
    return sorted({region for pattern, region in ANATOMY_PATTERNS.items() if pattern in text_lower})


def _assess_urgency(conditions: List[str], findings: str, impression: str) -> str:
    text = (findings + " " + impression).lower()
    if any(kw in text for kw in CRITICAL_KEYWORDS):
        return "critical"
    if any(c in CRITICAL_CONDITIONS for c in conditions):
        return "critical"
    if any(c in URGENT_CONDITIONS for c in conditions):
        return "urgent"
    return "routine"


def parse_iu_cxr_reports(reports_dir: Path) -> List[ClinicalDocument]:
    """Parse Indiana University CXR XML reports into ClinicalDocuments."""
    xml_files = sorted(reports_dir.glob("*.xml"))
    logger.info("Found %d IU CXR XML files in %s", len(xml_files), reports_dir)

    docs: List[ClinicalDocument] = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            pmcid_el = root.find(".//pmcId")
            doc_id = (
                f"iu_cxr_{pmcid_el.get('id', xml_path.stem)}"
                if pmcid_el is not None
                else f"iu_cxr_{xml_path.stem}"
            )

            abstract = root.find(".//Abstract")
            if abstract is None:
                continue

            sections: Dict[str, str] = {}
            for text_el in abstract.findall("AbstractText"):
                label = (text_el.get("Label") or "").upper()
                content = (text_el.text or "").strip()
                if content:
                    sections[label] = content

            findings = sections.get("FINDINGS", "")
            impression = sections.get("IMPRESSION", "")
            if not findings and not impression:
                continue

            indication = sections.get("INDICATION", "")
            comparison = sections.get("COMPARISON", "")

            # MeSH terms
            mesh_terms: List[str] = []
            mesh_el = root.find(".//MeSH")
            if mesh_el is not None:
                for tag in ("major", "automatic"):
                    for el in mesh_el.findall(tag):
                        if el.text:
                            mesh_terms.append(el.text.strip())

            conditions = _map_mesh_to_conditions(mesh_terms)
            # Augment from free text if MeSH is sparse
            if not conditions:
                conditions = _extract_conditions_from_text(findings + " " + impression)

            full_text = findings + " " + impression
            anatomy = _extract_anatomy(full_text)
            urgency = _assess_urgency(conditions, findings, impression)

            docs.append(
                ClinicalDocument(
                    doc_id=doc_id,
                    source="iu_cxr",
                    findings=findings,
                    impression=impression,
                    indication=indication,
                    comparison=comparison,
                    conditions=conditions,
                    anatomy=anatomy,
                    urgency=urgency,
                    mesh_terms=mesh_terms,
                )
            )
        except ET.ParseError:
            logger.warning("XML parse error: %s", xml_path)
        except Exception as exc:
            logger.warning("Error processing %s: %s", xml_path, exc)

    logger.info("Parsed %d valid IU CXR reports", len(docs))
    return docs


# ── 1a-ii. Multi-format document loaders ─────────────────────────────────────
# Each loader converts a specific file type into List[ClinicalDocument].
# They share a common contract: func(Path) -> List[ClinicalDocument]

# ............. PDF loader .................................................

def load_pdf(file_path: Path) -> List[ClinicalDocument]:
    """Extract text from a PDF file using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not installed — skipping %s", file_path)
        return []
    docs: List[ClinicalDocument] = []
    try:
        pdf = pymupdf.open(str(file_path))
        pages_text: List[str] = []
        for page in pdf:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text.strip())
        pdf.close()
        if not pages_text:
            logger.warning("PDF has no extractable text: %s", file_path)
            return []
        full_text = "\n\n".join(pages_text)
        conditions = _extract_conditions_from_text(full_text)
        anatomy = _extract_anatomy(full_text)
        urgency = _assess_urgency(conditions, full_text, "")
        docs.append(ClinicalDocument(
            doc_id=f"pdf_{file_path.stem}",
            source="pdf",
            findings=full_text,
            impression="",
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
    except Exception as exc:
        logger.warning("Error reading PDF %s: %s", file_path, exc)
    return docs


# ............. CSV / Excel loader .........................................

def _dataframe_to_docs(
    df: pd.DataFrame, file_path: Path, fmt: str,
) -> List[ClinicalDocument]:
    """Convert a pandas DataFrame into ClinicalDocuments."""
    docs: List[ClinicalDocument] = []
    cols_lower = {c.lower().strip(): c for c in df.columns}

    text_col = None
    for candidate in ("findings", "report", "text", "description",
                       "narrative", "content", "clinical_text", "report_text"):
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break

    impression_col = None
    for candidate in ("impression", "conclusion", "summary", "diagnosis"):
        if candidate in cols_lower:
            impression_col = cols_lower[candidate]
            break

    if not text_col:
        logger.warning("CSV/Excel %s missing a valid text column. Skipping.", file_path.name)
        return []

    for idx, row in df.iterrows():
        findings = str(row.get(text_col, "")).strip()
        impression = str(row.get(impression_col, "")).strip() if impression_col else ""

        if not findings or len(findings) < 20:
            continue

        conditions = _extract_conditions_from_text(findings + " " + impression)
        anatomy = _extract_anatomy(findings + " " + impression)
        urgency = _assess_urgency(conditions, findings, impression)
        docs.append(ClinicalDocument(
            doc_id=f"{fmt}_{file_path.stem}_row{idx}",
            source=fmt,
            findings=findings,
            impression=impression,
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
    return docs


def load_csv(file_path: Path) -> List[ClinicalDocument]:
    """Load a CSV file into ClinicalDocuments."""
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not installed — skipping %s", file_path)
        return []
    try:
        df = pd.read_csv(file_path, low_memory=False)
        docs = _dataframe_to_docs(df, file_path, "csv")
        logger.info("CSV %s: %d rows -> %d docs", file_path.name, len(df), len(docs))
        return docs
    except Exception as exc:
        logger.warning("Error reading CSV %s: %s", file_path, exc)
        return []


def load_excel(file_path: Path) -> List[ClinicalDocument]:
    """Load an Excel (.xlsx/.xls) file into ClinicalDocuments."""
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not installed — skipping %s", file_path)
        return []
    try:
        xls = pd.ExcelFile(file_path)
        all_docs: List[ClinicalDocument] = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            all_docs.extend(_dataframe_to_docs(df, file_path, f"excel_{sheet}"))
        logger.info("Excel %s: %d sheets -> %d docs", file_path.name, len(xls.sheet_names), len(all_docs))
        return all_docs
    except Exception as exc:
        logger.warning("Error reading Excel %s: %s", file_path, exc)
        return []


# ............. JSON / JSONL loader ........................................

def _json_obj_to_doc(
    obj: Dict[str, Any], file_path: Path, idx: int,
) -> Optional[ClinicalDocument]:
    """Convert a single JSON object to a ClinicalDocument."""
    text_keys = ("findings", "report", "text", "description", "content",
                 "narrative", "clinical_text", "body", "report_text")
    impression_keys = ("impression", "conclusion", "summary", "diagnosis")

    findings = ""
    for key in text_keys:
        if key in obj and obj[key]:
            findings = str(obj[key]).strip()
            break

    impression = ""
    for key in impression_keys:
        if key in obj and obj[key]:
            impression = str(obj[key]).strip()
            break

    if not findings:
        parts = []
        for k, v in obj.items():
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}: {v.strip()}")
            elif isinstance(v, (int, float)):
                parts.append(f"{k}: {v}")
        findings = "\n".join(parts)

    if not findings or len(findings) < 20:
        return None

    conditions = _extract_conditions_from_text(findings + " " + impression)
    anatomy = _extract_anatomy(findings + " " + impression)
    urgency = _assess_urgency(conditions, findings, impression)
    return ClinicalDocument(
        doc_id=f"json_{file_path.stem}_{idx}",
        source="json",
        findings=findings,
        impression=impression,
        conditions=conditions,
        anatomy=anatomy,
        urgency=urgency,
    )


def load_json(file_path: Path) -> List[ClinicalDocument]:
    """Load a JSON file (object or array of objects)."""
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            logger.warning("JSON %s: unexpected root type %s", file_path, type(data))
            return []
        for idx, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            doc = _json_obj_to_doc(obj, file_path, idx)
            if doc:
                docs.append(doc)
        logger.info("JSON %s: %d objects -> %d docs", file_path.name, len(data), len(docs))
    except Exception as exc:
        logger.warning("Error reading JSON %s: %s", file_path, exc)
    return docs


def load_jsonl(file_path: Path) -> List[ClinicalDocument]:
    """Load a JSONL (JSON-Lines) file — one JSON object per line."""
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    doc = _json_obj_to_doc(obj, file_path, idx)
                    if doc:
                        docs.append(doc)
        logger.info("JSONL %s: -> %d docs", file_path.name, len(docs))
    except Exception as exc:
        logger.warning("Error reading JSONL %s: %s", file_path, exc)
    return docs


# ............. HTML loader ................................................

def load_html(file_path: Path) -> List[ClinicalDocument]:
    """Extract text from an HTML file using BeautifulSoup."""
    if not BS4_AVAILABLE:
        logger.warning("beautifulsoup4 not installed — skipping %s", file_path)
        return []
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        text = soup.get_text(separator="\n", strip=True)
        if not text.strip() or len(text.strip()) < 20:
            return []

        # Try structured heading-based extraction
        sections: Dict[str, str] = {}
        for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
            heading_text = heading.get_text(strip=True).upper()
            content_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3", "h4"]:
                    break
                t = sibling.get_text(strip=True)
                if t:
                    content_parts.append(t)
            if content_parts:
                sections[heading_text] = " ".join(content_parts)

        findings = sections.get("FINDINGS", "") or text
        impression = sections.get("IMPRESSION", "") or sections.get("CONCLUSION", "")
        conditions = _extract_conditions_from_text(text)
        anatomy = _extract_anatomy(text)
        urgency = _assess_urgency(conditions, findings, impression)
        docs.append(ClinicalDocument(
            doc_id=f"html_{file_path.stem}",
            source="html",
            findings=findings,
            impression=impression if impression != findings else "",
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
        logger.info("HTML %s: title='%s', %d chars", file_path.name, title[:60], len(text))
    except Exception as exc:
        logger.warning("Error reading HTML %s: %s", file_path, exc)
    return docs


# ............. Plain text / Markdown loader ...............................

def load_text(file_path: Path) -> List[ClinicalDocument]:
    """Load a plain text or Markdown file."""
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if not text or len(text) < 20:
            return []
        conditions = _extract_conditions_from_text(text)
        anatomy = _extract_anatomy(text)
        urgency = _assess_urgency(conditions, text, "")
        docs.append(ClinicalDocument(
            doc_id=f"text_{file_path.stem}",
            source="text",
            findings=text,
            impression="",
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
    except Exception as exc:
        logger.warning("Error reading text %s: %s", file_path, exc)
    return docs


# ............. DICOM metadata loader ......................................

_DICOM_TEXT_TAGS = [
    "StudyDescription", "SeriesDescription", "ImageComments",
    "PatientComments", "AdditionalPatientHistory",
    "RequestedProcedureDescription", "ClinicalTrialProtocolName",
    "InstitutionalDepartmentName",
]


def load_dicom(file_path: Path) -> List[ClinicalDocument]:
    """Extract metadata text from a DICOM file (not pixel data)."""
    if not PYDICOM_AVAILABLE:
        logger.warning("pydicom not installed — skipping %s", file_path)
        return []
    docs: List[ClinicalDocument] = []
    try:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
        parts: List[str] = []
        for tag_name in _DICOM_TEXT_TAGS:
            val = getattr(ds, tag_name, None)
            if val and str(val).strip():
                parts.append(f"{tag_name}: {str(val).strip()}")
        for attr, label in [("PatientAge", "PatientAge"),
                            ("PatientSex", "PatientSex"),
                            ("Modality", "Modality"),
                            ("BodyPartExamined", "BodyPartExamined")]:
            val = getattr(ds, attr, "")
            if val:
                parts.append(f"{label}: {val}")
        text = "\n".join(parts)
        if not text.strip() or len(text.strip()) < 20:
            return []
        conditions = _extract_conditions_from_text(text)
        anatomy = _extract_anatomy(text)
        docs.append(ClinicalDocument(
            doc_id=f"dicom_{file_path.stem}",
            source="dicom",
            findings=text,
            impression="",
            conditions=conditions,
            anatomy=anatomy,
            urgency="routine",
        ))
    except Exception as exc:
        logger.warning("Error reading DICOM %s: %s", file_path, exc)
    return docs


# ............. Image OCR loader ...........................................

def load_image(file_path: Path) -> List[ClinicalDocument]:
    """Extract text from a scanned document image using Tesseract OCR."""
    if not OCR_AVAILABLE:
        logger.warning("PIL/pytesseract not installed — skipping %s", file_path)
        return []
    docs: List[ClinicalDocument] = []
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img).strip()
        if not text or len(text) < 20:
            logger.debug("Image %s: insufficient OCR text (%d chars)", file_path, len(text))
            return []
        conditions = _extract_conditions_from_text(text)
        anatomy = _extract_anatomy(text)
        urgency = _assess_urgency(conditions, text, "")
        docs.append(ClinicalDocument(
            doc_id=f"image_ocr_{file_path.stem}",
            source="image_ocr",
            findings=text,
            impression="",
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
        logger.info("Image OCR %s: %d chars extracted", file_path.name, len(text))
    except Exception as exc:
        logger.warning("Error processing image %s: %s", file_path, exc)
    return docs


# ............. XML (generic, non-IU CXR) loader ..........................

def load_generic_xml(file_path: Path) -> List[ClinicalDocument]:
    """Extract text content from a generic XML file."""
    docs: List[ClinicalDocument] = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        texts: List[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())
        full_text = "\n".join(texts)
        if not full_text.strip() or len(full_text.strip()) < 20:
            return []
        conditions = _extract_conditions_from_text(full_text)
        anatomy = _extract_anatomy(full_text)
        urgency = _assess_urgency(conditions, full_text, "")
        docs.append(ClinicalDocument(
            doc_id=f"xml_{file_path.stem}",
            source="xml",
            findings=full_text,
            impression="",
            conditions=conditions,
            anatomy=anatomy,
            urgency=urgency,
        ))
    except Exception as exc:
        logger.warning("Error reading XML %s: %s", file_path, exc)
    return docs


# ............. Loader registry & auto-discovery ...........................

FILE_LOADERS: Dict[str, Any] = {
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".tsv": load_csv,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".json": load_json,
    ".jsonl": load_jsonl,
    ".ndjson": load_jsonl,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
    ".md": load_text,
    ".rst": load_text,
    ".xml": load_generic_xml,
    ".dcm": load_dicom,
    ".dicom": load_dicom,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".tiff": load_image,
    ".tif": load_image,
    ".bmp": load_image,
}


def scan_directory(directory: Path, recursive: bool = True) -> List[ClinicalDocument]:
    """Auto-discover and load all supported files from a directory."""
    if not directory.is_dir():
        logger.warning("Data directory does not exist: %s", directory)
        return []

    docs: List[ClinicalDocument] = []
    pattern = "**/*" if recursive else "*"
    files = sorted(f for f in directory.glob(pattern) if f.is_file())

    logger.info("Scanning %s: found %d files", directory, len(files))
    format_counts: Dict[str, int] = {}
    skipped = 0

    for file_path in files:
        ext = file_path.suffix.lower()
        loader = FILE_LOADERS.get(ext)
        if loader is None:
            skipped += 1
            continue
        try:
            loaded = loader(file_path)
            docs.extend(loaded)
            format_counts[ext] = format_counts.get(ext, 0) + len(loaded)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)

    summary = ", ".join(f"{ext}={count}" for ext, count in sorted(format_counts.items()))
    logger.info(
        "Directory scan complete: %d docs loaded (%s), %d files skipped",
        len(docs), summary or "none", skipped,
    )
    return docs


# ── 1b. Radiopaedia / RadLex structured summaries ────────────────────────────

RADIOPAEDIA_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Aortic enlargement": {
        "findings": (
            "Widening of the mediastinal silhouette with prominence of the aortic knob. "
            "The thoracic aorta appears tortuous or dilated, sometimes with mural calcification. "
            "The descending aorta may be displaced laterally. On lateral view the retrosternal "
            "clear space may be reduced."
        ),
        "impression": (
            "Aortic enlargement, likely representing aortic ectasia or aneurysm. "
            "Differential includes atherosclerotic disease, connective tissue disorder "
            "(Marfan, Ehlers-Danlos), chronic hypertension, or post-stenotic dilatation."
        ),
        "anatomy": ["aorta", "mediastinum"],
        "urgency": "urgent",
        "differentials": [
            "Atherosclerotic aortic aneurysm",
            "Marfan syndrome",
            "Aortic dissection",
            "Post-stenotic dilatation",
        ],
        "management": (
            "CT angiography recommended for precise measurement and morphological assessment. "
            "Cardiology or vascular surgery referral if diameter exceeds 5.5 cm or shows rapid "
            "growth (>5 mm/year). Strict blood pressure control essential."
        ),
    },
    "Atelectasis": {
        "findings": (
            "Volume loss in the affected lobe with displacement of fissures toward the collapsed "
            "segment. Ipsilateral mediastinal shift may be present. Compensatory hyperinflation "
            "of adjacent lobes. May manifest as linear, plate-like, or rounded opacity."
        ),
        "impression": (
            "Atelectasis, likely subsegmental or lobar. Consider mucus plugging, post-operative "
            "changes, endobronchial lesion, or external compression as potential causes."
        ),
        "anatomy": ["right_upper", "right_middle", "right_lower", "left_upper", "left_lower"],
        "urgency": "routine",
        "differentials": [
            "Mucus plugging",
            "Post-surgical atelectasis",
            "Endobronchial tumor",
            "Foreign body aspiration",
        ],
        "management": (
            "Incentive spirometry and chest physiotherapy for post-operative or mucus-related "
            "atelectasis. If persistent or recurrent, consider bronchoscopy to exclude an "
            "endobronchial lesion."
        ),
    },
    "Calcification": {
        "findings": (
            "Focal areas of increased density within the lung parenchyma, pleura, or mediastinal "
            "structures consistent with calcification. Morphology may be punctate, dense, "
            "ring-like, eggshell, or popcorn-pattern. Distribution varies by aetiology."
        ),
        "impression": (
            "Calcifications identified; pattern suggests granulomatous disease, prior infection, "
            "or chronic process. Distribution and morphology should be correlated with clinical "
            "history and prior imaging."
        ),
        "anatomy": ["right_lung", "left_lung", "mediastinum", "pleural"],
        "urgency": "routine",
        "differentials": [
            "Granulomatous disease (TB, histoplasmosis)",
            "Calcified pleural plaques (asbestos exposure)",
            "Calcified lymph nodes",
            "Dystrophic calcification",
        ],
        "management": (
            "Compare with prior imaging when available. CT if calcification pattern is atypical "
            "or associated with a soft tissue mass. Occupational history for asbestos exposure."
        ),
    },
    "Cardiomegaly": {
        "findings": (
            "The cardiac silhouette is enlarged with a cardiothoracic ratio exceeding 0.5 on "
            "PA projection. The shape may be globular (suggesting pericardial effusion) or show "
            "specific chamber enlargement. Pulmonary vascular redistribution may be present."
        ),
        "impression": (
            "Cardiomegaly. Consider congestive heart failure, dilated cardiomyopathy, valvular "
            "heart disease, or pericardial effusion. Correlate with clinical signs of volume "
            "overload."
        ),
        "anatomy": ["cardiac"],
        "urgency": "urgent",
        "differentials": [
            "Congestive heart failure",
            "Dilated cardiomyopathy",
            "Valvular heart disease",
            "Pericardial effusion",
            "Hypertensive heart disease",
        ],
        "management": (
            "Echocardiography recommended for cardiac function and structure assessment. "
            "BNP/NT-proBNP if heart failure suspected. Cardiology referral for new-onset "
            "cardiomegaly."
        ),
    },
    "Consolidation": {
        "findings": (
            "Homogeneous opacification of lung parenchyma with air bronchograms. Distribution "
            "may be lobar, segmental, or multifocal. Silhouette sign may be present depending "
            "on location (e.g., obscured right heart border with right middle lobe consolidation)."
        ),
        "impression": (
            "Consolidation suggesting pneumonia as the most likely aetiology. Differential "
            "includes pulmonary haemorrhage, organising pneumonia (COP), or obstructive "
            "pneumonitis distal to an endobronchial lesion."
        ),
        "anatomy": ["right_upper", "right_middle", "right_lower", "left_upper", "left_lower"],
        "urgency": "urgent",
        "differentials": [
            "Community-acquired pneumonia",
            "Aspiration pneumonia",
            "Pulmonary haemorrhage",
            "Organising pneumonia (COP)",
            "Obstructive pneumonitis (lung cancer)",
        ],
        "management": (
            "Sputum culture and blood cultures. Empiric antibiotics per institutional guidelines. "
            "Follow-up imaging in 6-8 weeks to confirm resolution. CT if non-resolving to "
            "exclude underlying malignancy."
        ),
    },
    "ILD": {
        "findings": (
            "Diffuse reticular or reticulonodular pattern throughout the lungs. May see "
            "honeycombing, ground-glass opacities, or traction bronchiectasis. Typically "
            "bilateral with basal or peripheral predominance depending on subtype."
        ),
        "impression": (
            "Interstitial lung disease pattern. Broad differential includes idiopathic pulmonary "
            "fibrosis (UIP pattern), hypersensitivity pneumonitis, connective tissue disease-"
            "related ILD, sarcoidosis, and drug-related toxicity."
        ),
        "anatomy": ["right_lung", "left_lung", "bilateral", "base"],
        "urgency": "urgent",
        "differentials": [
            "Idiopathic pulmonary fibrosis (UIP)",
            "Hypersensitivity pneumonitis",
            "Nonspecific interstitial pneumonia (NSIP)",
            "Sarcoidosis",
            "Drug-induced lung disease",
        ],
        "management": (
            "High-resolution CT (HRCT) for pattern characterisation. Pulmonary function tests "
            "(FVC, DLCO). Multidisciplinary discussion (MDD) for definitive diagnosis. "
            "Rheumatology workup if connective tissue disease suspected."
        ),
    },
    "Infiltration": {
        "findings": (
            "Increased opacity within the lung parenchyma, which may be patchy, diffuse, or "
            "peribronchovascular in distribution. Margins are often ill-defined. Air bronchograms "
            "may or may not be present."
        ),
        "impression": (
            "Pulmonary infiltrates identified. Differential includes infection, inflammatory "
            "process, pulmonary oedema, or haemorrhage depending on distribution and clinical context."
        ),
        "anatomy": ["right_lung", "left_lung"],
        "urgency": "routine",
        "differentials": [
            "Pneumonia",
            "Pulmonary oedema",
            "Pulmonary haemorrhage",
            "Eosinophilic pneumonia",
            "Drug reaction",
        ],
        "management": (
            "Clinical correlation with fever, WBC, and symptoms. Empiric treatment as indicated. "
            "Follow-up imaging to confirm resolution."
        ),
    },
    "Lung Opacity": {
        "findings": (
            "Area of increased opacity within the lung, which may be focal, multifocal, or "
            "diffuse. Borders may be well-defined or ill-defined. Internal characteristics "
            "(air bronchograms, cavitation) should be assessed."
        ),
        "impression": (
            "Lung opacity identified. Wide differential depending on acuity, distribution, and "
            "clinical context. Infection, atelectasis, and neoplasm are primary considerations."
        ),
        "anatomy": ["right_lung", "left_lung"],
        "urgency": "routine",
        "differentials": [
            "Pneumonia",
            "Atelectasis",
            "Lung mass",
            "Pleural effusion (layering)",
            "Pulmonary oedema",
        ],
        "management": (
            "Correlate with prior imaging. CT for further characterisation if persistent or "
            "suspicious. Follow-up chest X-ray in 4-6 weeks for indeterminate opacities."
        ),
    },
    "Nodule/Mass": {
        "findings": (
            "Well-circumscribed rounded opacity within the lung parenchyma. Classified as "
            "nodule if <3 cm, mass if >=3 cm. Assess for calcification, cavitation, margins "
            "(smooth vs spiculated), and satellite lesions."
        ),
        "impression": (
            "Pulmonary nodule/mass identified. Differential includes primary lung neoplasm, "
            "metastatic disease, granuloma, or hamartoma. Size, morphology, and growth rate "
            "are critical for risk stratification."
        ),
        "anatomy": ["right_lung", "left_lung"],
        "urgency": "urgent",
        "differentials": [
            "Primary lung cancer (adenocarcinoma, squamous cell)",
            "Metastasis",
            "Granuloma (TB, fungal)",
            "Hamartoma",
            "Carcinoid tumor",
        ],
        "management": (
            "CT for detailed characterisation. Apply Fleischner Society or Lung-RADS guidelines. "
            "PET-CT for metabolic assessment of indeterminate lesions. Biopsy for tissue "
            "diagnosis if intermediate or high risk."
        ),
    },
    "Other lesion": {
        "findings": (
            "Abnormality identified that does not fit standard pulmonary categories. May include "
            "mediastinal mass, chest wall lesion, foreign body, rib fracture, subcutaneous "
            "emphysema, or other incidental findings."
        ),
        "impression": (
            "Other thoracic abnormality detected. Further characterisation with cross-sectional "
            "imaging and clinical correlation recommended."
        ),
        "anatomy": ["mediastinum", "cardiac", "pleural"],
        "urgency": "routine",
        "differentials": [
            "Mediastinal mass",
            "Chest wall mass",
            "Rib fracture",
            "Surgical hardware",
            "Foreign body",
        ],
        "management": (
            "CT for further evaluation. Correlate with clinical history. Subspecialty referral "
            "as indicated by specific finding."
        ),
    },
    "Pleural effusion": {
        "findings": (
            "Blunting of the costophrenic angle with meniscus sign. In larger effusions, "
            "opacification of the lower hemithorax with fluid tracking up the lateral chest "
            "wall. Massive effusion may cause contralateral mediastinal shift."
        ),
        "impression": (
            "Pleural effusion. Consider heart failure (bilateral), parapneumonic effusion, "
            "malignant effusion, or other systemic process (hepatic, renal, hypoalbuminaemia)."
        ),
        "anatomy": ["costophrenic", "pleural", "base"],
        "urgency": "urgent",
        "differentials": [
            "Congestive heart failure (transudative)",
            "Parapneumonic effusion / empyema",
            "Malignant effusion",
            "Pulmonary embolism",
            "Hepatic hydrothorax",
        ],
        "management": (
            "Lateral decubitus film or ultrasound to confirm and quantify. Diagnostic "
            "thoracentesis if new, unilateral, or atypical. Light's criteria to classify as "
            "transudative vs exudative."
        ),
    },
    "Pleural thickening": {
        "findings": (
            "Thickening of the pleural surface, which may be focal or diffuse, unilateral or "
            "bilateral. May be smooth or irregular. Associated calcification (asbestos-related "
            "pleural plaques) may be present."
        ),
        "impression": (
            "Pleural thickening identified. Differential includes prior infection or inflammation, "
            "asbestos exposure, post-traumatic changes, or malignant pleural disease."
        ),
        "anatomy": ["pleural", "costophrenic"],
        "urgency": "routine",
        "differentials": [
            "Post-inflammatory (resolved effusion or empyema)",
            "Asbestos-related pleural disease",
            "Mesothelioma",
            "Post-traumatic",
            "Post-radiation",
        ],
        "management": (
            "CT for detailed characterisation if new or progressive. Occupational history for "
            "asbestos exposure. PET-CT if malignancy suspected."
        ),
    },
    "Pneumothorax": {
        "findings": (
            "Visible visceral pleural line with absence of lung markings peripheral to this "
            "line. May be subtle at the apex on an upright film or present as increased "
            "lucency at the base on supine imaging. Assess for mediastinal shift (tension)."
        ),
        "impression": (
            "Pneumothorax identified. Assess size and clinical stability. Tension pneumothorax "
            "is a medical emergency requiring immediate decompression."
        ),
        "anatomy": ["apex", "pleural", "right_lung", "left_lung"],
        "urgency": "critical",
        "differentials": [
            "Primary spontaneous pneumothorax",
            "Secondary spontaneous (COPD, cystic fibrosis)",
            "Traumatic pneumothorax",
            "Iatrogenic (post-procedure)",
            "Tension pneumothorax",
        ],
        "management": (
            "Small (<2 cm rim): observation with serial imaging if clinically stable. "
            "Large or symptomatic: chest tube insertion or needle aspiration. Tension "
            "pneumothorax: immediate needle decompression (2nd intercostal space, midclavicular "
            "line) followed by chest tube."
        ),
    },
    "Pulmonary fibrosis": {
        "findings": (
            "Reticular pattern predominantly at the lung bases with possible honeycombing. "
            "Reduced lung volumes. Traction bronchiectasis may be present. Distribution "
            "typically peripheral and basal-predominant (UIP pattern)."
        ),
        "impression": (
            "Findings consistent with pulmonary fibrosis, most likely usual interstitial "
            "pneumonia (UIP) pattern if basal-predominant with honeycombing. Correlate with "
            "PFTs and multidisciplinary discussion for definitive diagnosis."
        ),
        "anatomy": ["base", "bilateral", "right_lower", "left_lower"],
        "urgency": "urgent",
        "differentials": [
            "Idiopathic pulmonary fibrosis",
            "Connective tissue disease-related UIP",
            "Chronic hypersensitivity pneumonitis",
            "Asbestosis",
            "Drug-related fibrosis",
        ],
        "management": (
            "HRCT to confirm UIP pattern. Pulmonary function tests (FVC, DLCO). "
            "Multidisciplinary discussion for definitive classification. Consider antifibrotic "
            "therapy (nintedanib or pirfenidone) if IPF confirmed."
        ),
    },
}


def build_radiopaedia_summaries() -> List[ClinicalDocument]:
    """Generate Radiopaedia/RadLex-style ClinicalDocuments for each VinBigData condition."""
    docs: List[ClinicalDocument] = []
    for condition, info in RADIOPAEDIA_KNOWLEDGE.items():
        full_text = (
            f"Radiopaedia Reference – {condition}\n\n"
            f"Radiographic Findings:\n{info['findings']}\n\n"
            f"Impression:\n{info['impression']}\n\n"
            f"Differential Diagnosis:\n"
            + "\n".join(f"  - {d}" for d in info["differentials"])
            + f"\n\nManagement:\n{info['management']}"
        )
        docs.append(
            ClinicalDocument(
                doc_id=f"radiopaedia_{condition.lower().replace('/', '_').replace(' ', '_')}",
                source="radiopaedia",
                findings=info["findings"],
                impression=info["impression"],
                conditions=[condition],
                anatomy=info["anatomy"],
                urgency=info["urgency"],
            )
        )
    logger.info("Built %d Radiopaedia summaries", len(docs))
    return docs


# ── 1c. Clinical management guidelines ───────────────────────────────────────

CLINICAL_GUIDELINES: Dict[str, str] = {
    "Aortic enlargement": (
        "Clinical Guideline – Aortic Enlargement\n"
        "1. Confirm with CT angiography for precise diameter measurement.\n"
        "2. If diameter <4.5 cm: surveillance imaging every 12 months.\n"
        "3. If diameter 4.5-5.5 cm: surveillance every 6 months; vascular surgery consultation.\n"
        "4. If diameter >5.5 cm or growth >5 mm/year: surgical or endovascular repair indicated.\n"
        "5. Optimise blood pressure control (target <130/80 mmHg); beta-blocker preferred.\n"
        "6. Screen for connective tissue disorders if age <50 or family history."
    ),
    "Atelectasis": (
        "Clinical Guideline – Atelectasis\n"
        "1. Post-operative: incentive spirometry q1h while awake, early mobilisation.\n"
        "2. Mucus plugging: chest physiotherapy, suctioning if intubated.\n"
        "3. If persistent >48h or recurrent: consider bronchoscopy.\n"
        "4. Exclude endobronchial lesion in smokers or high-risk patients.\n"
        "5. Follow-up CXR to confirm resolution."
    ),
    "Calcification": (
        "Clinical Guideline – Pulmonary/Pleural Calcification\n"
        "1. Characterise pattern: popcorn (hamartoma), central (granuloma), eggshell (silicosis/sarcoid).\n"
        "2. If granulomatous: assess TB risk, consider Quantiferon or PPD.\n"
        "3. If pleural plaques with occupational history: asbestos surveillance programme.\n"
        "4. If associated with a soft-tissue component: CT and possible biopsy.\n"
        "5. Compare with prior imaging to assess stability."
    ),
    "Cardiomegaly": (
        "Clinical Guideline – Cardiomegaly\n"
        "1. Echocardiography: assess ejection fraction, chamber sizes, valvular function.\n"
        "2. Labs: BNP/NT-proBNP, basic metabolic panel, thyroid function.\n"
        "3. If new-onset HF: cardiology referral within 2 weeks.\n"
        "4. Initiate guideline-directed medical therapy (ACEi/ARB + beta-blocker + diuretic).\n"
        "5. If pericardial effusion suspected: urgent echo; pericardiocentesis if tamponade.\n"
        "6. Lifestyle modification: sodium restriction, fluid management, daily weights."
    ),
    "Consolidation": (
        "Clinical Guideline – Consolidation / Pneumonia\n"
        "1. Obtain sputum and blood cultures before antibiotics if possible.\n"
        "2. Empiric antibiotics per CURB-65 or PSI severity stratification.\n"
        "3. Outpatient (CURB-65 0-1): amoxicillin or doxycycline.\n"
        "4. Inpatient (CURB-65 2+): IV co-amoxiclav + macrolide or respiratory fluoroquinolone.\n"
        "5. ICU admission criteria: septic shock, respiratory failure requiring ventilation.\n"
        "6. Follow-up CXR at 6-8 weeks; CT if non-resolving to exclude underlying malignancy."
    ),
    "ILD": (
        "Clinical Guideline – Interstitial Lung Disease\n"
        "1. HRCT is the cornerstone investigation – characterise pattern (UIP, NSIP, etc.).\n"
        "2. PFTs: FVC and DLCO at baseline and serial monitoring.\n"
        "3. Multidisciplinary discussion (MDD) with pulmonologist, radiologist, pathologist.\n"
        "4. Rheumatology workup: ANA, RF, anti-CCP, myositis panel if CTD-ILD suspected.\n"
        "5. If definite UIP/IPF: antifibrotic therapy (nintedanib or pirfenidone).\n"
        "6. Pulmonary rehabilitation referral. Assess for lung transplant eligibility if advanced."
    ),
    "Infiltration": (
        "Clinical Guideline – Pulmonary Infiltrates\n"
        "1. Correlate with clinical picture: fever → infectious; orthopnoea → cardiogenic.\n"
        "2. If infectious: cultures, empiric antibiotics based on community vs nosocomial.\n"
        "3. If cardiogenic oedema: diuretics, preload/afterload reduction.\n"
        "4. If eosinophilic: consider peripheral eosinophil count, BAL if indicated.\n"
        "5. Follow-up imaging to document resolution or progression."
    ),
    "Lung Opacity": (
        "Clinical Guideline – Lung Opacity (Indeterminate)\n"
        "1. Review prior imaging for comparison – assess stability.\n"
        "2. If new and symptomatic: treat underlying cause (infection most common).\n"
        "3. If persistent >6 weeks: CT characterisation recommended.\n"
        "4. If associated with suspicious features (spiculation, growth): urgent CT + PET.\n"
        "5. Follow-up CXR at 4-6 weeks for indeterminate opacities."
    ),
    "Nodule/Mass": (
        "Clinical Guideline – Pulmonary Nodule / Mass\n"
        "1. Measure size precisely; apply Fleischner Society guidelines for incidental nodules.\n"
        "2. Solid nodule <6 mm: no routine follow-up (low risk); optional 12-month CT (high risk).\n"
        "3. Solid nodule 6-8 mm: CT at 6-12 months; consider PET if high risk.\n"
        "4. Solid nodule >8 mm: CT at 3 months, PET-CT, or tissue sampling.\n"
        "5. Mass (>=3 cm): high suspicion for malignancy; urgent CT + PET + biopsy.\n"
        "6. Ground-glass nodules: longer follow-up intervals per Fleischner GGN guidelines.\n"
        "7. MDT discussion for all intermediate/high-risk lesions."
    ),
    "Other lesion": (
        "Clinical Guideline – Other Thoracic Lesion\n"
        "1. CT for cross-sectional characterisation of the abnormality.\n"
        "2. Correlate with clinical history and presentation.\n"
        "3. Subspecialty referral based on specific finding (thoracic surgery, oncology, etc.).\n"
        "4. If incidental and benign-appearing: document and monitor."
    ),
    "Pleural effusion": (
        "Clinical Guideline – Pleural Effusion\n"
        "1. Confirm with lateral decubitus CXR or bedside ultrasound.\n"
        "2. If bilateral and clinical HF: treat heart failure; thoracentesis if asymmetric.\n"
        "3. If unilateral or atypical: diagnostic thoracentesis.\n"
        "4. Apply Light's criteria (protein, LDH) to classify transudative vs exudative.\n"
        "5. Exudative: send cytology, culture, glucose, pH, cell count + differential.\n"
        "6. If malignant: oncology referral; consider indwelling pleural catheter or pleurodesis.\n"
        "7. If empyema: chest tube drainage + IV antibiotics; thoracic surgery if loculated."
    ),
    "Pleural thickening": (
        "Clinical Guideline – Pleural Thickening\n"
        "1. CT for characterisation: smooth vs nodular, circumferential vs focal.\n"
        "2. Occupational history: asbestos, construction, shipbuilding.\n"
        "3. If calcified plaques without symptoms: reassurance; periodic surveillance.\n"
        "4. If nodular or circumferential: PET-CT to evaluate for mesothelioma.\n"
        "5. Biopsy (thoracoscopic preferred) if mesothelioma suspected."
    ),
    "Pneumothorax": (
        "Clinical Guideline – Pneumothorax\n"
        "1. Tension pneumothorax: IMMEDIATE needle decompression (2nd ICS MCL), then chest tube.\n"
        "2. Large (>2 cm on CXR) or symptomatic: chest tube (intercostal drain).\n"
        "3. Small (<2 cm) and stable: observation, high-flow O2, repeat CXR at 4-6h.\n"
        "4. If recurrent: thoracic surgery referral for pleurodesis or bullectomy.\n"
        "5. Iatrogenic (post-procedure): manage based on size and symptoms.\n"
        "6. Advise against air travel and diving until fully resolved and cleared."
    ),
    "Pulmonary fibrosis": (
        "Clinical Guideline – Pulmonary Fibrosis\n"
        "1. HRCT to confirm pattern: UIP (honeycombing, traction bronchiectasis, basal-predominant).\n"
        "2. PFTs baseline: FVC, DLCO – repeat every 3-6 months.\n"
        "3. Multidisciplinary discussion for definitive IPF diagnosis.\n"
        "4. If IPF confirmed: antifibrotic therapy (nintedanib or pirfenidone).\n"
        "5. Pulmonary rehabilitation to improve exercise tolerance and quality of life.\n"
        "6. Assess supplemental O2 needs. Refer for lung transplant evaluation if FVC <80% predicted.\n"
        "7. Vaccinations: annual influenza, pneumococcal, COVID-19."
    ),
}


def build_clinical_guidelines() -> List[ClinicalDocument]:
    """Generate clinical guideline ClinicalDocuments for each VinBigData condition."""
    docs: List[ClinicalDocument] = []
    for condition, guideline_text in CLINICAL_GUIDELINES.items():
        info = RADIOPAEDIA_KNOWLEDGE.get(condition, {})
        docs.append(
            ClinicalDocument(
                doc_id=f"guideline_{condition.lower().replace('/', '_').replace(' ', '_')}",
                source="guideline",
                findings=guideline_text,
                impression=f"Management guideline for {condition}.",
                conditions=[condition],
                anatomy=info.get("anatomy", []),
                urgency=info.get("urgency", "routine"),
            )
        )
    logger.info("Built %d clinical guideline documents", len(docs))
    return docs


# ── 1d. Synthetic structured summaries (dense retrieval signals) ──────────────

SYNTHETIC_SUMMARIES: Dict[str, List[str]] = {
    "Aortic enlargement": [
        "aortic enlargement | dilated aorta | mediastinal widening | aortic knob prominence | tortuous aorta",
        "aortic aneurysm | aortic ectasia | atherosclerotic aorta | aortic calcification | CT angiography",
        "thoracic aortic dilatation | ascending aorta enlarged | descending aorta displaced | vascular surgery referral",
    ],
    "Atelectasis": [
        "atelectasis | volume loss | fissure displacement | lobar collapse | subsegmental atelectasis",
        "plate-like atelectasis | linear atelectasis | rounded atelectasis | mucus plugging | post-operative",
        "mediastinal shift | compensatory hyperinflation | bronchoscopy | incentive spirometry",
    ],
    "Calcification": [
        "calcification | calcified nodule | granulomatous calcification | eggshell calcification | popcorn calcification",
        "calcified pleural plaque | calcified lymph node | dystrophic calcification | TB calcification | histoplasmosis",
        "punctate calcification | ring calcification | dense calcification | chronic granulomatous disease",
    ],
    "Cardiomegaly": [
        "cardiomegaly | enlarged cardiac silhouette | CTR > 0.5 | heart failure | valvular disease",
        "pericardial effusion | dilated cardiomyopathy | left ventricular enlargement | pulmonary congestion",
        "globular heart | cardiac decompensation | echocardiography | BNP elevated | fluid overload",
    ],
    "Consolidation": [
        "consolidation | air bronchograms | lobar pneumonia | airspace opacification | silhouette sign",
        "community-acquired pneumonia | aspiration pneumonia | right lower lobe consolidation | left lower lobe consolidation",
        "segmental consolidation | multifocal consolidation | non-resolving consolidation | organising pneumonia",
    ],
    "ILD": [
        "interstitial lung disease | ILD | reticular pattern | reticulonodular | honeycombing",
        "ground-glass opacity | traction bronchiectasis | UIP pattern | NSIP | basal predominant fibrosis",
        "idiopathic pulmonary fibrosis | hypersensitivity pneumonitis | sarcoidosis | HRCT characterisation",
    ],
    "Infiltration": [
        "infiltration | pulmonary infiltrate | ill-defined opacity | patchy opacity | peribronchovascular",
        "infectious infiltrate | inflammatory infiltrate | pulmonary oedema infiltrate | eosinophilic infiltrate",
        "diffuse infiltrate | bilateral infiltrate | ground-glass infiltrate | alveolar infiltrate",
    ],
    "Lung Opacity": [
        "lung opacity | focal opacity | multifocal opacity | diffuse opacity | indeterminate opacity",
        "airspace opacity | ground-glass opacity | mixed opacity | persistent opacity | new opacity",
        "hazy opacity | patchy opacity | peripheral opacity | central opacity | follow-up imaging",
    ],
    "Nodule/Mass": [
        "pulmonary nodule | lung mass | solitary pulmonary nodule | SPN | spiculated nodule",
        "lung cancer screening | Fleischner guidelines | Lung-RADS | PET-CT avid | metabolically active",
        "ground-glass nodule | part-solid nodule | solid nodule | nodule growth | biopsy indicated",
    ],
    "Other lesion": [
        "mediastinal mass | chest wall lesion | rib fracture | foreign body | subcutaneous emphysema",
        "surgical hardware | pacemaker | central line | thoracic abnormality | incidental finding",
        "chest wall mass | sternal fracture | vertebral lesion | anterior mediastinal mass | thymoma",
    ],
    "Pleural effusion": [
        "pleural effusion | costophrenic blunting | meniscus sign | fluid layering | lateral decubitus",
        "transudative effusion | exudative effusion | parapneumonic effusion | malignant effusion | empyema",
        "Light's criteria | thoracentesis | bilateral effusion | massive effusion | pleural fluid analysis",
    ],
    "Pleural thickening": [
        "pleural thickening | thickened pleura | pleural plaque | calcified plaque | asbestos exposure",
        "diffuse pleural thickening | focal pleural thickening | mesothelioma | post-inflammatory thickening",
        "circumferential pleural thickening | nodular pleural thickening | pleural peel | trapped lung",
    ],
    "Pneumothorax": [
        "pneumothorax | visceral pleural line | absent lung markings | tension pneumothorax | deep sulcus sign",
        "spontaneous pneumothorax | traumatic pneumothorax | iatrogenic pneumothorax | chest tube | needle decompression",
        "large pneumothorax | small pneumothorax | recurrent pneumothorax | pleurodesis | subcutaneous emphysema",
    ],
    "Pulmonary fibrosis": [
        "pulmonary fibrosis | UIP pattern | honeycombing | traction bronchiectasis | basal fibrosis",
        "idiopathic pulmonary fibrosis | IPF | nintedanib | pirfenidone | antifibrotic therapy",
        "reduced lung volumes | peripheral fibrosis | progressive fibrosis | FVC decline | DLCO decreased",
    ],
}


def build_synthetic_summaries() -> List[ClinicalDocument]:
    """Build synthetic structured summaries — dense keyword signals for retrieval alignment."""
    docs: List[ClinicalDocument] = []
    for condition, summaries in SYNTHETIC_SUMMARIES.items():
        info = RADIOPAEDIA_KNOWLEDGE.get(condition, {})
        for idx, summary in enumerate(summaries):
            docs.append(
                ClinicalDocument(
                    doc_id=f"synthetic_{condition.lower().replace('/', '_').replace(' ', '_')}_{idx}",
                    source="synthetic",
                    findings=summary,
                    impression=f"Structured keywords for {condition}.",
                    conditions=[condition],
                    anatomy=info.get("anatomy", []),
                    urgency=info.get("urgency", "routine"),
                )
            )
    logger.info("Built %d synthetic summary documents", len(docs))
    return docs


# ════════════════════════════════════════════════════════════════════════════════
# 2.  Chunking strategies
# ════════════════════════════════════════════════════════════════════════════════

_enc = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    return len(_enc.encode(text))


def _make_chunk_id(text: str, doc_id: str, idx: int) -> str:
    """Deterministic chunk ID from full content hash."""
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    return f"{doc_id}_chunk_{idx}_{h}"


def _build_metadata(doc: ClinicalDocument, chunk_type: str) -> Dict[str, Any]:
    return {
        "source": doc.source,
        "doc_id": doc.doc_id,
        "condition": "|".join(doc.conditions) if doc.conditions else "unspecified",
        "anatomy": "|".join(doc.anatomy) if doc.anatomy else "unspecified",
        "urgency": doc.urgency,
        "chunk_type": chunk_type,
    }


def create_narrative_chunks(
    docs: List[ClinicalDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """Create narrative chunks from ClinicalDocuments using token-based splitting.

    Uses an 800-token window with 100-token overlap for context preservation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", ", ", " "],
    )

    chunks: List[Chunk] = []
    for doc in docs:
        # Build a coherent narrative from the document
        parts: List[str] = []
        if doc.indication:
            parts.append(f"Indication: {doc.indication}")
        if doc.comparison:
            parts.append(f"Comparison: {doc.comparison}")
        if doc.findings:
            parts.append(f"Findings: {doc.findings}")
        if doc.impression:
            parts.append(f"Impression: {doc.impression}")

        full_text = "\n".join(parts)
        if not full_text.strip():
            continue

        metadata = _build_metadata(doc, "narrative")

        split_texts = splitter.split_text(full_text)
        for idx, text in enumerate(split_texts):
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(text, doc.doc_id, idx),
                    text=text,
                    chunk_type="narrative",
                    metadata=metadata,
                )
            )

    logger.info("Created %d narrative chunks", len(chunks))
    return chunks


def create_structured_chunks(docs: List[ClinicalDocument]) -> List[Chunk]:
    """Create short, dense structured chunks from synthetic summaries.

    These are already concise — no splitting needed.  They improve retrieval
    precision for exact medical terminology.
    """
    chunks: List[Chunk] = []
    for doc in docs:
        if doc.source != "synthetic":
            continue
        text = doc.findings.strip()
        if not text:
            continue
        metadata = _build_metadata(doc, "structured")
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(text, doc.doc_id, 0),
                text=text,
                chunk_type="structured",
                metadata=metadata,
            )
        )

    logger.info("Created %d structured chunks", len(chunks))
    return chunks


# ════════════════════════════════════════════════════════════════════════════════
# 3.  Ingestion — ChromaDB + BM25
# ════════════════════════════════════════════════════════════════════════════════


def build_chromadb(
    chunks: List[Chunk],
    output_dir: Path,
    embedding_model: str = "BAAI/bge-m3",
    collection_name: str = "clinical_knowledge",
    batch_size: int = 256,
) -> chromadb.Collection:
    """Embed all chunks and persist to ChromaDB using idempotent upserts."""
    chroma_path = output_dir / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        trust_remote_code=True,
    )

    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Ingest in batches using upsert
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        logger.info("ChromaDB: upserted %d / %d chunks", end, total)

    logger.info(
        "ChromaDB collection '%s' ready with %d total chunks at %s",
        collection_name,
        collection.count(),
        chroma_path,
    )
    return collection


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokeniser for BM25 indexing (lowercased, alphanumeric tokens)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: List[Chunk], output_dir: Path) -> BM25Okapi:
    """Build a BM25 keyword index from all chunks and serialise to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = [c.chunk_id for c in chunks]
    corpus = [_tokenize_for_bm25(c.text) for c in chunks]
    chunk_texts = [c.text for c in chunks]
    chunk_metadatas = [c.metadata for c in chunks]

    bm25 = BM25Okapi(corpus)

    index_path = output_dir / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": chunk_ids,
                "chunk_texts": chunk_texts,
                "chunk_metadatas": chunk_metadatas,
                "corpus": corpus,
            },
            f,
        )

    logger.info("BM25 index built (%d documents) → %s", len(chunk_ids), index_path)
    return bm25


# ════════════════════════════════════════════════════════════════════════════════
# 4.  Hybrid Retrieval — Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════════════════════════════


class HybridRetriever:
    """Combines ChromaDB semantic search with BM25 keyword search using
    Reciprocal Rank Fusion (RRF).

    Score for each document:
        RRF_score = sum_over_systems(  1 / (k + rank_i)  )
    where k=60 (standard RRF constant) and rank_i is the 1-based rank
    from each retrieval system.

    Alternatively, a weighted-average mode blends normalised scores:
        score = alpha * semantic_score + (1 - alpha) * bm25_score
    """

    def __init__(
        self,
        vectorstore_dir: str | Path,
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "clinical_knowledge",
    ):
        vectorstore_dir = Path(vectorstore_dir)

        # ── ChromaDB ──
        chroma_path = vectorstore_dir / "chroma"
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            trust_remote_code=True,
        )
        client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

        # ── BM25 ──
        bm25_path = vectorstore_dir / "bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.bm25: BM25Okapi = data["bm25"]
        self.bm25_chunk_ids: List[str] = data["chunk_ids"]
        self.bm25_chunk_texts: List[str] = data["chunk_texts"]
        self.bm25_chunk_metadatas: List[Dict] = data["chunk_metadatas"]

        logger.info(
            "HybridRetriever loaded: %d ChromaDB docs, %d BM25 docs",
            self.collection.count(),
            len(self.bm25_chunk_ids),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 10,
        method: str = "rrf",
        alpha: float = 0.7,
        rrf_k: int = 60,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid retrieval.

        Args:
            text:   Query string.
            top_k:  Number of results to return.
            method: "rrf" (Reciprocal Rank Fusion) or "weighted" (score blending).
            alpha:  Weight for semantic score when method="weighted" (0-1).
            rrf_k:  RRF constant (default 60, per Cormack et al.).
            where:  Optional ChromaDB metadata filter dict
                    (e.g. {"condition": {"$contains": "Cardiomegaly"}}).

        Returns:
            List of dicts: [{text, chunk_id, metadata, score}, ...]
        """
        n_candidates = top_k * 3  # over-retrieve then fuse

        semantic_results = self._semantic_search(text, n_candidates, where=where)
        bm25_results = self._bm25_search(text, n_candidates)

        if method == "rrf":
            fused = self._reciprocal_rank_fusion(
                semantic_results, bm25_results, k=rrf_k
            )
        elif method == "weighted":
            fused = self._weighted_fusion(semantic_results, bm25_results, alpha=alpha)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        return fused[:top_k]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _semantic_search(
        self, text: str, n: int, where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "query_texts": [text],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        out: List[Dict[str, Any]] = []
        if results and results["ids"]:
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance → similarity = 1 - distance
                out.append(
                    {
                        "chunk_id": cid,
                        "text": doc,
                        "metadata": meta,
                        "score": 1.0 - dist,
                    }
                )
        return out

    def _bm25_search(self, text: str, n: int) -> List[Dict[str, Any]]:
        tokens = _tokenize_for_bm25(text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :n
        ]

        out: List[Dict[str, Any]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            out.append(
                {
                    "chunk_id": self.bm25_chunk_ids[idx],
                    "text": self.bm25_chunk_texts[idx],
                    "metadata": self.bm25_chunk_metadatas[idx],
                    "score": float(scores[idx]),
                }
            )
        return out

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF).

        For each document seen in either result list:
            rrf_score = sum_i( 1 / (k + rank_i) )
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, item in enumerate(semantic_results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            doc_map[cid] = item

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in doc_map:
                doc_map[cid] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**doc_map[cid], "score": score, "fusion": "rrf"}
            for cid, score in ranked
        ]

    @staticmethod
    def _weighted_fusion(
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Weighted average fusion with min-max normalisation.

        Final score = alpha * norm_semantic + (1 - alpha) * norm_bm25
        """
        def _normalise(results: List[Dict]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r["score"] for r in results]
            lo, hi = min(scores), max(scores)
            rng = hi - lo if hi != lo else 1.0
            return {r["chunk_id"]: (r["score"] - lo) / rng for r in results}

        sem_norm = _normalise(semantic_results)
        bm25_norm = _normalise(bm25_results)

        doc_map: Dict[str, Dict] = {}
        all_ids = set(sem_norm) | set(bm25_norm)

        combined: Dict[str, float] = {}
        for cid in all_ids:
            combined[cid] = (
                alpha * sem_norm.get(cid, 0.0)
                + (1.0 - alpha) * bm25_norm.get(cid, 0.0)
            )

        # Build doc_map from both result sets
        for item in semantic_results:
            doc_map[item["chunk_id"]] = item
        for item in bm25_results:
            if item["chunk_id"] not in doc_map:
                doc_map[item["chunk_id"]] = item

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [
            {**doc_map[cid], "score": score, "fusion": "weighted"}
            for cid, score in ranked
        ]


# ════════════════════════════════════════════════════════════════════════════════
# 5.  Validation — quick smoke tests
# ════════════════════════════════════════════════════════════════════════════════


def run_validation(retriever: HybridRetriever) -> None:
    """Run a few sample queries and print results for manual inspection."""
    test_queries = [
        ("cardiomegaly enlarged heart failure", {"condition": {"$contains": "Cardiomegaly"}}),
        ("pneumothorax tension emergency", {"urgency": "critical"}),
        ("consolidation right lower lobe pneumonia", None),
        ("pulmonary fibrosis honeycombing basal", None),
        ("pleural effusion bilateral", None),
    ]
    print("\n" + "=" * 80)
    print("  VALIDATION — Hybrid Retrieval Smoke Tests")
    print("=" * 80)

    for query_text, where_filter in test_queries:
        print(f"\n{'─' * 70}")
        print(f"  Query: \"{query_text}\"")
        if where_filter:
            print(f"  Filter: {where_filter}")
        print(f"{'─' * 70}")

        results = retriever.query(query_text, top_k=5, method="rrf", where=where_filter)
        for i, r in enumerate(results, 1):
            text_preview = r["text"][:120].replace("\n", " ")
            print(
                f"  [{i}] score={r['score']:.4f}  "
                f"src={r['metadata'].get('source', '?'):12s}  "
                f"cond={r['metadata'].get('condition', '?'):20s}  "
                f"urgency={r['metadata'].get('urgency', '?'):8s}"
            )
            print(f"      {text_preview}…")

    print("\n" + "=" * 80)
    print("  Validation complete.")
    print("=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# 6.  CLI entry-point
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clinical knowledge base (ChromaDB + BM25).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "NLMCXR_reports" / "ecgen-radiology",
        help="Path to Indiana University CXR XML reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "vectorstore",
        help="Directory for ChromaDB and BM25 index output.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-m3",
        help="Sentence-transformers model name for embeddings.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="clinical_knowledge",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Narrative chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Narrative chunk overlap in tokens.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run smoke-test queries after building the index.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip IU CXR report parsing (use only built-in knowledge).",
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Additional directories to scan for documents (PDF, CSV, JSON, "
            "HTML, DICOM, images, etc.).  Accepts multiple paths."
        ),
    )
    args = parser.parse_args()

    logger.info("═" * 60)
    logger.info("Clinical Knowledge Base — Ingestion Pipeline")
    logger.info("═" * 60)
    logger.info("Reports dir   : %s", args.reports_dir)
    logger.info("Output dir    : %s", args.output_dir)
    logger.info("Data dirs     : %s", args.data_dirs or "(none)")
    logger.info("Embedding     : %s", args.embedding_model)
    logger.info("Chunk size    : %d tokens (overlap %d)", args.chunk_size, args.chunk_overlap)

    # ── Collect documents ──
    all_docs: List[ClinicalDocument] = []

    if not args.skip_reports:
        all_docs.extend(parse_iu_cxr_reports(args.reports_dir))

    # Scan additional data directories (multi-format auto-discovery)
    for data_dir in args.data_dirs:
        all_docs.extend(scan_directory(data_dir))

    all_docs.extend(build_radiopaedia_summaries())
    all_docs.extend(build_clinical_guidelines())
    all_docs.extend(build_synthetic_summaries())

    logger.info("Total documents collected: %d", len(all_docs))

    # ── Chunk ──
    narrative_chunks = create_narrative_chunks(
        [d for d in all_docs if d.source != "synthetic"],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    structured_chunks = create_structured_chunks(all_docs)
    all_chunks = narrative_chunks + structured_chunks
    logger.info("Total chunks: %d (narrative=%d, structured=%d)",
                len(all_chunks), len(narrative_chunks), len(structured_chunks))

    # ── Ingest ──
    build_chromadb(
        all_chunks,
        args.output_dir,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
    )
    build_bm25_index(all_chunks, args.output_dir)

    # ── Validate ──
    if args.validate:
        retriever = HybridRetriever(
            vectorstore_dir=args.output_dir,
            embedding_model=args.embedding_model,
            collection_name=args.collection_name,
        )
        run_validation(retriever)

    logger.info("Done. Knowledge base written to %s", args.output_dir)


if __name__ == "__main__":
    main()