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
    - PDF        : Clinical guidelines, textbooks (using unstructured for tables)
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
        --embedding-model BAAI/bge-m3 \
        --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import tiktoken

# ── Required Domain Imports ──
import pandas as pd
from unstructured.partition.pdf import partition_pdf
import pydicom
from PIL import Image
import pytesseract

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]
    BS4_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_knowledge")

# ════════════════════════════════════════════════════════════════════════════════
# Domain constants & NLP heuristics
# ════════════════════════════════════════════════════════════════════════════════

VINBIG_CLASSES: List[str] = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]

MESH_TO_CONDITION: Dict[str, str] = {
    "cardiomegaly": "Cardiomegaly", "cardiac enlargement": "Cardiomegaly", "enlarged heart": "Cardiomegaly",
    "aortic enlargement": "Aortic enlargement", "aortic ectasia": "Aortic enlargement", "tortuous aorta": "Aortic enlargement",
    "atelectasis": "Atelectasis", "calcification": "Calcification", "calcified": "Calcification",
    "consolidation": "Consolidation", "airspace disease": "Consolidation",
    "interstitial lung disease": "ILD", "interstitial abnormality": "ILD",
    "infiltrate": "Infiltration", "infiltration": "Infiltration",
    "lung opacity": "Lung Opacity", "opacity": "Lung Opacity", "opacification": "Lung Opacity",
    "nodule": "Nodule/Mass", "mass": "Nodule/Mass", 
    "pleural effusion": "Pleural effusion", "effusion": "Pleural effusion",
    "pleural thickening": "Pleural thickening", "pneumothorax": "Pneumothorax",
    "pulmonary fibrosis": "Pulmonary fibrosis", "fibrosis": "Pulmonary fibrosis",
}

ANATOMY_PATTERNS: Dict[str, str] = {
    "right upper lobe": "right_upper", "right middle lobe": "right_middle", "right lower lobe": "right_lower",
    "left upper lobe": "left_upper", "left lower lobe": "left_lower", "lingula": "left_middle",
    "right lung": "right_lung", "left lung": "left_lung", "bilateral": "bilateral",
    "right hemithorax": "right_lung", "left hemithorax": "left_lung",
    "right hilum": "right_hilum", "left hilum": "left_hilum", "hilar": "bilateral_hilum",
    "mediastinum": "mediastinum", "mediastinal": "mediastinum",
    "cardiac": "cardiac", "heart": "cardiac", "aorta": "aorta", "aortic": "aorta",
    "pleural": "pleural", "costophrenic": "costophrenic", "diaphragm": "diaphragm",
    "apex": "apex", "apical": "apex", "base": "base", "basal": "base",
}

CRITICAL_CONDITIONS = {"Pneumothorax", "Consolidation"}
URGENT_CONDITIONS = {"Pleural effusion", "Nodule/Mass", "Cardiomegaly", "Atelectasis"}
CRITICAL_KEYWORDS = ["tension", "massive", "emergency", "acute respiratory", "severe", "life-threatening", "stat"]

# ════════════════════════════════════════════════════════════════════════════════
# Data models
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ClinicalDocument:
    """Semi-structured clinical document."""
    doc_id: str
    source: str  # iu_cxr | radiopaedia | guideline | synthetic | pdf | etc.
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
# Extraction Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _map_mesh_to_conditions(mesh_terms: List[str]) -> List[str]:
    conditions: set[str] = set()
    for term in mesh_terms:
        for key, cond in MESH_TO_CONDITION.items():
            if key in term.lower():
                conditions.add(cond)
    return sorted(conditions)

def _extract_conditions_from_text(text: str) -> List[str]:
    """Fallback: scan free text for conditions, with robust negation handling."""
    conditions: set[str] = set()
    text_lower = text.lower()
    
    # Advanced clinical negation patterns
    negation_pattern = re.compile(
        r'\b(no|not|without|negative for|clear of|resolved|unremarkable|'
        r'normal|free of|no evidence of|rule out|to exclude)\b'
    )
    
    for key, cond in MESH_TO_CONDITION.items():
        for match in re.finditer(r'\b' + re.escape(key) + r'\b', text_lower):
            start_idx = max(0, match.start() - 40)
            context_window = text_lower[start_idx:match.start()]
            
            # Check if a negation term exists in the immediate leading text
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

# ════════════════════════════════════════════════════════════════════════════════
# 1. Document Parsers
# ════════════════════════════════════════════════════════════════════════════════

def parse_iu_cxr_reports(reports_dir: Path) -> List[ClinicalDocument]:
    """Parse Indiana University CXR XML reports."""
    xml_files = sorted(reports_dir.glob("*.xml"))
    logger.info("Found %d IU CXR XML files in %s", len(xml_files), reports_dir)

    docs: List[ClinicalDocument] = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            pmcid_el = root.find(".//pmcId")
            doc_id = f"iu_cxr_{pmcid_el.get('id', xml_path.stem)}" if pmcid_el is not None else f"iu_cxr_{xml_path.stem}"

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

            mesh_terms: List[str] = []
            mesh_el = root.find(".//MeSH")
            if mesh_el is not None:
                for tag in ("major", "automatic"):
                    for el in mesh_el.findall(tag):
                        if el.text:
                            mesh_terms.append(el.text.strip())

            conditions = _map_mesh_to_conditions(mesh_terms)
            if not conditions:
                conditions = _extract_conditions_from_text(findings + " " + impression)

            full_text = findings + " " + impression
            docs.append(
                ClinicalDocument(
                    doc_id=doc_id, source="iu_cxr", findings=findings, impression=impression,
                    indication=indication, comparison=comparison, conditions=conditions,
                    anatomy=_extract_anatomy(full_text), urgency=_assess_urgency(conditions, findings, impression),
                    mesh_terms=mesh_terms,
                )
            )
        except ET.ParseError:
            logger.warning("XML parse error: %s", xml_path)
        except Exception as exc:
            logger.warning("Error processing %s: %s", xml_path, exc)

    logger.info("Parsed %d valid IU CXR reports", len(docs))
    return docs


def load_pdf(file_path: Path) -> List[ClinicalDocument]:
    """Extract text and tables from a PDF using unstructured.io."""
    docs: List[ClinicalDocument] = []
    try:
        logger.info("Parsing PDF with unstructured (this may take a moment): %s", file_path.name)
        
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            chunking_strategy="by_title"  # Keeps paragraphs grouped under their headers
        )
        
        text_parts: List[str] = []
        for el in elements:
            category = getattr(el, "category", "")
            
            if category == "Table" and hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                text_parts.append(f"\n[TABLE]\n{el.metadata.text_as_html}\n[/TABLE]\n")
            elif category in ["Title", "Header"]:
                text_parts.append(f"\n### {el.text}\n")
            else:
                text_parts.append(str(el.text))
                
        full_text = "\n".join(text_parts).strip()
        
        if not full_text or len(full_text) < 20:
            logger.warning("PDF has no extractable text: %s", file_path)
            return []
            
        conditions = _extract_conditions_from_text(full_text)
        docs.append(ClinicalDocument(
            doc_id=f"pdf_{file_path.stem}", source="pdf", findings=full_text, impression="",
            conditions=conditions, anatomy=_extract_anatomy(full_text),
            urgency=_assess_urgency(conditions, full_text, ""),
        ))
    except Exception as exc:
        logger.warning("Error reading PDF %s: %s", file_path, exc)
        
    return docs

def _dataframe_to_docs(df: pd.DataFrame, file_path: Path, source_tag: str) -> List[ClinicalDocument]:
    """Convert a pandas DataFrame into ClinicalDocuments safely."""
    docs: List[ClinicalDocument] = []
    cols_lower = {c.lower().strip(): c for c in df.columns}

    text_col = next((cols_lower[c] for c in ("findings", "report", "text", "description", "narrative", "content", "clinical_text") if c in cols_lower), None)
    impression_col = next((cols_lower[c] for c in ("impression", "conclusion", "summary", "diagnosis") if c in cols_lower), None)

    if not text_col:
        logger.warning("CSV/Excel %s missing a valid clinical text column. Skipping.", file_path.name)
        return []

    for idx, row in df.iterrows():
        findings = str(row.get(text_col, "")).strip()
        impression = str(row.get(impression_col, "")).strip() if impression_col else ""

        if not findings or len(findings) < 20:
            continue

        conditions = _extract_conditions_from_text(findings + " " + impression)
        docs.append(ClinicalDocument(
            doc_id=f"{source_tag}_{file_path.stem}_row{idx}", source=source_tag,
            findings=findings, impression=impression, conditions=conditions,
            anatomy=_extract_anatomy(findings + " " + impression),
            urgency=_assess_urgency(conditions, findings, impression),
        ))
    return docs

def load_csv(file_path: Path) -> List[ClinicalDocument]:
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return _dataframe_to_docs(df, file_path, "csv")
    except Exception as exc:
        logger.warning("Error reading CSV %s: %s", file_path, exc)
        return []

def load_excel(file_path: Path) -> List[ClinicalDocument]:
    try:
        xls = pd.ExcelFile(file_path)
        all_docs = []
        for sheet in xls.sheet_names:
            all_docs.extend(_dataframe_to_docs(xls.parse(sheet), file_path, "excel"))
        return all_docs
    except Exception as exc:
        logger.warning("Error reading Excel %s: %s", file_path, exc)
        return []

def load_json(file_path: Path) -> List[ClinicalDocument]:
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for idx, obj in enumerate(data):
            if not isinstance(obj, dict): continue
            text = str(obj.get("findings", obj.get("text", obj.get("content", "")))).strip()
            if len(text) > 20:
                conds = _extract_conditions_from_text(text)
                docs.append(ClinicalDocument(
                    doc_id=f"json_{file_path.stem}_{idx}", source="json",
                    findings=text, impression="", conditions=conds,
                    anatomy=_extract_anatomy(text), urgency=_assess_urgency(conds, text, "")
                ))
    except Exception as exc:
        logger.warning("Error reading JSON %s: %s", file_path, exc)
    return docs

def load_jsonl(file_path: Path) -> List[ClinicalDocument]:
    docs: List[ClinicalDocument] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        text = str(obj.get("findings", obj.get("text", ""))).strip()
                        if len(text) > 20:
                            conds = _extract_conditions_from_text(text)
                            docs.append(ClinicalDocument(
                                doc_id=f"jsonl_{file_path.stem}_{idx}", source="jsonl",
                                findings=text, impression="", conditions=conds,
                                anatomy=_extract_anatomy(text), urgency=_assess_urgency(conds, text, "")
                            ))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Error reading JSONL %s: %s", file_path, exc)
    return docs

def load_html(file_path: Path) -> List[ClinicalDocument]:
    if not BS4_AVAILABLE: return []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 20:
            conds = _extract_conditions_from_text(text)
            return [ClinicalDocument(
                doc_id=f"html_{file_path.stem}", source="html",
                findings=text, impression="", conditions=conds,
                anatomy=_extract_anatomy(text), urgency=_assess_urgency(conds, text, "")
            )]
    except Exception as exc:
        logger.warning("Error reading HTML %s: %s", file_path, exc)
    return []

def load_text(file_path: Path) -> List[ClinicalDocument]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if len(text) > 20:
            conds = _extract_conditions_from_text(text)
            return [ClinicalDocument(
                doc_id=f"text_{file_path.stem}", source="text",
                findings=text, impression="", conditions=conds,
                anatomy=_extract_anatomy(text), urgency=_assess_urgency(conds, text, "")
            )]
    except Exception:
        pass
    return []

_DICOM_TEXT_TAGS = ["StudyDescription", "SeriesDescription", "ImageComments", "PatientComments", "RequestedProcedureDescription"]

def load_dicom(file_path: Path) -> List[ClinicalDocument]:
    try:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
        parts = [f"{t}: {getattr(ds, t, '')}" for t in _DICOM_TEXT_TAGS if getattr(ds, t, '')]
        text = "\n".join(parts).strip()
        if len(text) > 20:
            conds = _extract_conditions_from_text(text)
            return [ClinicalDocument(
                doc_id=f"dicom_{file_path.stem}", source="dicom",
                findings=text, impression="", conditions=conds,
                anatomy=_extract_anatomy(text), urgency="routine"
            )]
    except Exception as exc:
        logger.warning("Error reading DICOM %s: %s", file_path, exc)
    return []

def load_image(file_path: Path) -> List[ClinicalDocument]:
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img).strip()
        if len(text) > 20:
            conds = _extract_conditions_from_text(text)
            return [ClinicalDocument(
                doc_id=f"image_ocr_{file_path.stem}", source="image_ocr",
                findings=text, impression="", conditions=conds,
                anatomy=_extract_anatomy(text), urgency=_assess_urgency(conds, text, "")
            )]
    except Exception as exc:
        logger.warning("Error processing image %s: %s", file_path, exc)
    return []

FILE_LOADERS: Dict[str, Any] = {
    ".pdf": load_pdf, ".csv": load_csv, ".tsv": load_csv, ".xlsx": load_excel, ".xls": load_excel,
    ".json": load_json, ".jsonl": load_jsonl, ".ndjson": load_jsonl, ".html": load_html, ".htm": load_html,
    ".txt": load_text, ".md": load_text, ".dcm": load_dicom, ".dicom": load_dicom,
    ".png": load_image, ".jpg": load_image, ".jpeg": load_image
}

def scan_directory(directory: Path, recursive: bool = True) -> List[ClinicalDocument]:
    if not directory.is_dir(): return []
    docs, format_counts, skipped = [], {}, 0
    pattern = "**/*" if recursive else "*"
    files = sorted(f for f in directory.glob(pattern) if f.is_file())

    for file_path in files:
        ext = file_path.suffix.lower()
        loader = FILE_LOADERS.get(ext)
        if not loader:
            skipped += 1
            continue
        try:
            loaded = loader(file_path)
            docs.extend(loaded)
            format_counts[ext] = format_counts.get(ext, 0) + len(loaded)
        except Exception:
            logger.warning("Failed to load %s", file_path)

    summary = ", ".join(f"{ext}={c}" for ext, c in sorted(format_counts.items()))
    logger.info("Scan complete: %d docs loaded (%s), %d files skipped", len(docs), summary, skipped)
    return docs

# ════════════════════════════════════════════════════════════════════════════════
# 2. Domain Knowledge Builders (Radiopaedia, Guidelines, Synthetic)
# ════════════════════════════════════════════════════════════════════════════════

RADIOPAEDIA_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Aortic enlargement": {
        "findings": "Widening of the mediastinal silhouette with prominence of the aortic knob.",
        "impression": "Aortic enlargement, likely representing aortic ectasia or aneurysm.",
        "anatomy": ["aorta", "mediastinum"], "urgency": "urgent",
        "differentials": ["Atherosclerotic aortic aneurysm", "Marfan syndrome", "Aortic dissection"],
        "management": "CT angiography recommended."
    },
    "Atelectasis": {
        "findings": "Volume loss in the affected lobe with displacement of fissures.",
        "impression": "Atelectasis, likely subsegmental or lobar.",
        "anatomy": ["right_upper", "right_middle", "right_lower", "left_upper", "left_lower"], "urgency": "routine",
        "differentials": ["Mucus plugging", "Post-surgical atelectasis", "Endobronchial tumor"],
        "management": "Incentive spirometry."
    },
    "Calcification": {
        "findings": "Focal areas of increased density within the lung parenchyma or pleura.",
        "impression": "Calcifications identified.",
        "anatomy": ["right_lung", "left_lung", "mediastinum", "pleural"], "urgency": "routine",
        "differentials": ["Granulomatous disease", "Calcified pleural plaques"],
        "management": "Compare with prior imaging."
    },
    "Cardiomegaly": {
        "findings": "The cardiac silhouette is enlarged with a cardiothoracic ratio exceeding 0.5.",
        "impression": "Cardiomegaly. Consider congestive heart failure.",
        "anatomy": ["cardiac"], "urgency": "urgent",
        "differentials": ["Congestive heart failure", "Dilated cardiomyopathy", "Pericardial effusion"],
        "management": "Echocardiography, BNP."
    },
    "Consolidation": {
        "findings": "Homogeneous opacification of lung parenchyma with air bronchograms.",
        "impression": "Consolidation suggesting pneumonia.",
        "anatomy": ["right_upper", "right_middle", "right_lower", "left_upper", "left_lower"], "urgency": "urgent",
        "differentials": ["Pneumonia", "Pulmonary haemorrhage", "Organising pneumonia"],
        "management": "Empiric antibiotics."
    },
    "ILD": {
        "findings": "Diffuse reticular or reticulonodular pattern throughout the lungs.",
        "impression": "Interstitial lung disease pattern.",
        "anatomy": ["right_lung", "left_lung", "bilateral", "base"], "urgency": "urgent",
        "differentials": ["Idiopathic pulmonary fibrosis", "Hypersensitivity pneumonitis", "Sarcoidosis"],
        "management": "HRCT."
    },
    "Infiltration": {
        "findings": "Increased opacity within the lung parenchyma, patchy or diffuse.",
        "impression": "Pulmonary infiltrates identified.",
        "anatomy": ["right_lung", "left_lung"], "urgency": "routine",
        "differentials": ["Pneumonia", "Pulmonary oedema", "Pulmonary haemorrhage"],
        "management": "Clinical correlation with fever, WBC."
    },
    "Lung Opacity": {
        "findings": "Area of increased opacity within the lung.",
        "impression": "Lung opacity identified.",
        "anatomy": ["right_lung", "left_lung"], "urgency": "routine",
        "differentials": ["Pneumonia", "Atelectasis", "Lung mass"],
        "management": "Follow-up chest X-ray in 4-6 weeks."
    },
    "Nodule/Mass": {
        "findings": "Well-circumscribed rounded opacity within the lung parenchyma.",
        "impression": "Pulmonary nodule/mass identified.",
        "anatomy": ["right_lung", "left_lung"], "urgency": "urgent",
        "differentials": ["Primary lung cancer", "Metastasis", "Granuloma"],
        "management": "CT characterisation, apply Fleischner guidelines."
    },
    "Other lesion": {
        "findings": "Abnormality identified that does not fit standard pulmonary categories.",
        "impression": "Other thoracic abnormality detected.",
        "anatomy": ["mediastinum", "cardiac", "pleural"], "urgency": "routine",
        "differentials": ["Mediastinal mass", "Chest wall mass", "Rib fracture"],
        "management": "CT for further evaluation."
    },
    "Pleural effusion": {
        "findings": "Blunting of the costophrenic angle with meniscus sign.",
        "impression": "Pleural effusion.",
        "anatomy": ["costophrenic", "pleural", "base"], "urgency": "urgent",
        "differentials": ["Heart failure", "Parapneumonic effusion", "Malignant effusion"],
        "management": "Diagnostic thoracentesis or ultrasound."
    },
    "Pleural thickening": {
        "findings": "Thickening of the pleural surface, focal or diffuse.",
        "impression": "Pleural thickening identified.",
        "anatomy": ["pleural", "costophrenic"], "urgency": "routine",
        "differentials": ["Post-inflammatory", "Asbestos-related pleural disease", "Mesothelioma"],
        "management": "CT for detailed characterisation."
    },
    "Pneumothorax": {
        "findings": "Visible visceral pleural line with absence of lung markings peripheral to it.",
        "impression": "Pneumothorax identified.",
        "anatomy": ["apex", "pleural", "right_lung", "left_lung"], "urgency": "critical",
        "differentials": ["Spontaneous pneumothorax", "Traumatic pneumothorax", "Tension pneumothorax"],
        "management": "Chest tube insertion or needle aspiration."
    },
    "Pulmonary fibrosis": {
        "findings": "Reticular pattern predominantly at the lung bases with possible honeycombing.",
        "impression": "Findings consistent with pulmonary fibrosis.",
        "anatomy": ["base", "bilateral", "right_lower", "left_lower"], "urgency": "urgent",
        "differentials": ["Idiopathic pulmonary fibrosis", "Connective tissue disease-related UIP"],
        "management": "HRCT to confirm UIP pattern."
    },
}

def build_radiopaedia_summaries() -> List[ClinicalDocument]:
    docs = []
    for condition, info in RADIOPAEDIA_KNOWLEDGE.items():
        text = f"Radiopaedia – {condition}\nFindings: {info['findings']}\nImpression: {info['impression']}\nDiff: {', '.join(info['differentials'])}\nManagement: {info['management']}"
        docs.append(ClinicalDocument(
            doc_id=f"radiopaedia_{condition.replace('/', '_').replace(' ', '_').lower()}", source="radiopaedia",
            findings=text, impression="", conditions=[condition], anatomy=info["anatomy"], urgency=info["urgency"],
        ))
    return docs

def build_clinical_guidelines() -> List[ClinicalDocument]:
    docs = []
    for condition, info in RADIOPAEDIA_KNOWLEDGE.items():
        text = f"Clinical Guideline for {condition}:\nManagement includes {info['management']}. Differentials include {', '.join(info['differentials'])}."
        docs.append(ClinicalDocument(
            doc_id=f"guideline_{condition.replace('/', '_').replace(' ', '_').lower()}", source="guideline",
            findings=text, impression="", conditions=[condition], anatomy=info["anatomy"], urgency=info["urgency"],
        ))
    return docs

SYNTHETIC_SUMMARIES: Dict[str, List[str]] = {
    "Pneumothorax": ["pneumothorax | visceral pleural line | absent lung markings | tension pneumothorax", "spontaneous pneumothorax | chest tube | needle decompression"],
    "Consolidation": ["consolidation | air bronchograms | lobar pneumonia | airspace opacification", "community-acquired pneumonia | aspiration pneumonia"],
    "Cardiomegaly": ["cardiomegaly | enlarged cardiac silhouette | CTR > 0.5 | heart failure", "pericardial effusion | left ventricular enlargement"],
    "Pulmonary fibrosis": ["pulmonary fibrosis | UIP pattern | honeycombing | traction bronchiectasis", "idiopathic pulmonary fibrosis | IPF | antifibrotic therapy"],
}

def build_synthetic_summaries() -> List[ClinicalDocument]:
    docs = []
    for condition, summaries in SYNTHETIC_SUMMARIES.items():
        info = RADIOPAEDIA_KNOWLEDGE.get(condition, {})
        for idx, text in enumerate(summaries):
            docs.append(ClinicalDocument(
                doc_id=f"synthetic_{condition.replace('/', '_').replace(' ', '_').lower()}_{idx}", source="synthetic",
                findings=text, impression="", conditions=[condition], anatomy=info.get("anatomy", []), urgency=info.get("urgency", "routine"),
            ))
    return docs

# ════════════════════════════════════════════════════════════════════════════════
# 3.  Chunking strategies
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

def create_narrative_chunks(docs: List[ClinicalDocument], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=_token_length, separators=["\n\n", "\n", ". ", ", ", " "],
    )
    chunks = []
    for doc in docs:
        parts = [f"{k}: {v}" for k, v in [("Indication", doc.indication), ("Comparison", doc.comparison), ("Findings", doc.findings), ("Impression", doc.impression)] if v]
        full_text = "\n".join(parts)
        if not full_text.strip(): continue
        meta = _build_metadata(doc, "narrative")
        for idx, text in enumerate(splitter.split_text(full_text)):
            chunks.append(Chunk(chunk_id=_make_chunk_id(text, doc.doc_id, idx), text=text, chunk_type="narrative", metadata=meta))
    logger.info("Created %d narrative chunks", len(chunks))
    return chunks

def create_structured_chunks(docs: List[ClinicalDocument]) -> List[Chunk]:
    """
    Intentionally restricts structured chunking to 'synthetic' dense keywords.
    These bypass the text splitter entirely to maintain strict phrase boundaries 
    (e.g., 'pneumonia | consolidation | right lower lobe') for high-precision retrieval.
    """
    chunks = []
    for doc in docs:
        if doc.source != "synthetic" or not doc.findings.strip(): continue
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.findings, doc.doc_id, 0), 
            text=doc.findings, chunk_type="structured", 
            metadata=_build_metadata(doc, "structured")
        ))
    logger.info("Created %d structured chunks", len(chunks))
    return chunks

# ════════════════════════════════════════════════════════════════════════════════
# 4.  Ingestion — ChromaDB + BM25
# ════════════════════════════════════════════════════════════════════════════════

def build_chromadb(chunks: List[Chunk], output_dir: Path, embedding_model: str = "BAAI/bge-m3", collection_name: str = "clinical_knowledge", batch_size: int = 256) -> chromadb.Collection:
    chroma_path = output_dir / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model, trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [c.chunk_id for c in chunks], [c.text for c in chunks], [c.metadata for c in chunks]
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.upsert(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
    
    logger.info("ChromaDB '%s' ready with %d chunks", collection_name, collection.count())
    return collection

def _tokenize_for_bm25(text: str) -> List[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(t) > 1]

def build_bm25_index(chunks: List[Chunk], output_dir: Path) -> BM25Okapi:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids, corpus, chunk_texts, chunk_metadatas = [], [], [], []
    for c in chunks:
        chunk_ids.append(c.chunk_id)
        corpus.append(_tokenize_for_bm25(c.text))
        chunk_texts.append(c.text)
        chunk_metadatas.append(c.metadata)
        
    bm25 = BM25Okapi(corpus)
    index_path = output_dir / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "chunk_texts": chunk_texts, "chunk_metadatas": chunk_metadatas, "corpus": corpus}, f)
    
    logger.info("BM25 index built (%d docs)", len(chunk_ids))
    return bm25

# ════════════════════════════════════════════════════════════════════════════════
# 5.  Hybrid Retrieval (RRF)
# ════════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    def __init__(self, vectorstore_dir: str | Path, embedding_model: str = "BAAI/bge-m3", collection_name: str = "clinical_knowledge"):
        vectorstore_dir = Path(vectorstore_dir)
        self.collection = chromadb.PersistentClient(path=str(vectorstore_dir / "chroma")).get_collection(
            name=collection_name, embedding_function=SentenceTransformerEmbeddingFunction(model_name=embedding_model, trust_remote_code=True)
        )
        with open(vectorstore_dir / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25, self.bm25_ids, self.bm25_texts, self.bm25_metas = data["bm25"], data["chunk_ids"], data["chunk_texts"], data["chunk_metadatas"]

    def query(self, text: str, top_k: int = 10, rrf_k: int = 60, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        n = top_k * 3
        
        # 1. Semantic (ChromaDB)
        sem_res = []
        db_query = self.collection.query(
            query_texts=[text], n_results=n, where=where, include=["documents", "metadatas", "distances"]
        )
        if db_query and db_query["ids"] and db_query["ids"][0]:
            for cid, doc, meta, dist in zip(db_query["ids"][0], db_query["documents"][0], db_query["metadatas"][0], db_query["distances"][0]):
                sem_res.append({"chunk_id": cid, "text": doc, "metadata": meta, "score": 1.0 - dist})
        
        # 2. Keyword (BM25)
        bm25_res = []
        tokens = _tokenize_for_bm25(text)
        if tokens:
            bm25_scores = self.bm25.get_scores(tokens)
            top_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:n]
            for i in top_idx:
                if bm25_scores[i] > 0:
                    bm25_res.append({"chunk_id": self.bm25_ids[i], "text": self.bm25_texts[i], "metadata": self.bm25_metas[i], "score": float(bm25_scores[i])})
        
        # 3. Reciprocal Rank Fusion (RRF)
        scores, doc_map = {}, {}
        for rank, r in enumerate(sem_res, 1):
            scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
            doc_map[r["chunk_id"]] = r
            
        for rank, r in enumerate(bm25_res, 1):
            scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
            if r["chunk_id"] not in doc_map: 
                doc_map[r["chunk_id"]] = r
            
        # Sort and trim
        ranked_fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{**doc_map[cid], "score": score} for cid, score in ranked_fused][:top_k]

def run_validation(retriever: HybridRetriever) -> None:
    test_queries = [
        ("cardiomegaly enlarged heart failure", {"condition": {"$contains": "Cardiomegaly"}}),
        ("tension pneumothorax", {"urgency": "critical"}),
        ("consolidation right lower lobe pneumonia", None),
    ]
    print("\n" + "=" * 80 + "\n  VALIDATION — Hybrid Retrieval Smoke Tests\n" + "=" * 80)
    for q, w in test_queries:
        print(f"\nQuery: '{q}' | Filter: {w}")
        for i, r in enumerate(retriever.query(q, top_k=3, where=w), 1):
            print(f"  [{i}] src={r['metadata'].get('source')} | cond={r['metadata'].get('condition')} | score={r['score']:.4f}\n      {r['text'][:120]}...")
    print("\n" + "=" * 80)

# ════════════════════════════════════════════════════════════════════════════════
# 6.  CLI entry-point
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Build clinical KB.")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "NLMCXR_reports/ecgen-radiology")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/vectorstore")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--collection-name", type=str, default="clinical_knowledge")
    parser.add_argument("--data-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    all_docs = parse_iu_cxr_reports(args.reports_dir)
    for d in args.data_dirs: all_docs.extend(scan_directory(d))
    all_docs.extend(build_radiopaedia_summaries() + build_clinical_guidelines() + build_synthetic_summaries())

    chunks = create_narrative_chunks([d for d in all_docs if d.source != "synthetic"]) + create_structured_chunks(all_docs)
    build_chromadb(chunks, args.output_dir, args.embedding_model, args.collection_name)
    build_bm25_index(chunks, args.output_dir)

    if args.validate: run_validation(HybridRetriever(args.output_dir, args.embedding_model, args.collection_name))

if __name__ == "__main__":
    main()