# -*- coding: utf-8 -*-
from __future__ import annotations
import re, os
from typing import Optional, Dict, List
import pandas as pd
from langchain_core.documents import Document

# ====== Stałe pól (jedna prawda) ======
F_DATASET = "dataset"
F_MEASURE = "measure"
F_VALUE   = "value"
F_REGION  = "region"
F_PERIOD  = "period"  # surowe z CSV
F_OKRES   = "okres"   # ujednolicony (YYYY lub YYYY-Qx)
F_TYPE    = "type"    # z CSV 'typ', w meta trzymamy 'type'
F_SRC     = "source_file"
F_ROW     = "row_index"

# ====== Normalizacje / parsery ======
def parse_value(x) -> Optional[float]:
    if x is None: return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "brak", "null"}:
        return None
    s = s.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def norm_period(p: Optional[str]) -> Optional[str]:
    if not p or str(p).strip()=="":
        return None
    s = str(p).replace("\u00A0"," ").strip().upper()
    s = re.sub(r"[R]\.?$", "", s)
    s = re.sub(r"\s+", "", s).replace("/", "-").replace("_", "-")
    if re.match(r"^20\d{2}-Q[1-4]$", s): return s
    m = re.match(r"^(20\d{2})Q([1-4])$", s)
    if m: return f"{m.group(1)}-Q{m.group(2)}"
    m = re.match(r"^Q([1-4])[- ]?(20\d{2})$", s)
    if m: return f"{m.group(2)}-Q{m.group(1)}"
    return s

def clean_measure_and_type(measure: str, typ: Optional[str]) -> tuple[str, Optional[str]]:
    m = str(measure or "").strip()
    t = None if (typ is None or str(typ).strip()=="") else str(typ).strip()
    m = m.replace("przciętna", "przeciętna").replace(" w zl", " w zł")
    paren = re.search(r"\(([^)]+)\)", m)
    if paren and (t is None or t==""):
        t = paren.group(1).strip()
    m = re.sub(r"\s*\([^)]+\)\s*", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m, t

# ====== Reprezentacja page_content (kanoniczna) ======
def make_page_text(rec: Dict[str, object]) -> str:
    """Kanoniczny string – zgodny z tabdata: kluczowe pary oddzielone ' | '."""
    return " | ".join([
        f"{F_DATASET}: {rec.get(F_DATASET)}",
        f"{F_MEASURE}: {rec.get('measure_clean')}",
        f"{F_VALUE}: {rec.get('value_float')}",
        f"{F_REGION}: {rec.get(F_REGION) or '-'}",
        f"{F_PERIOD}: {rec.get('period_norm') or rec.get('period_raw') or '-'}",
        f"typ: {rec.get('typ_clean') or '-'}",
    ])

# ====== Budowa dokumentu z jednego wiersza CSV ======
def row_to_document(row: dict, source_file: str, row_index: int) -> Document:
    dataset = str(row.get("dataset") or "").strip()
    measure_raw = row.get("measure")
    typ_raw     = row.get("typ")
    measure_clean, typ_clean = clean_measure_and_type(measure_raw, typ_raw)

    value_float = parse_value(row.get("value"))
    region_raw  = None if pd.isna(row.get("region")) else str(row.get("region")).strip()
    period_raw  = None if pd.isna(row.get("period")) else str(row.get("period")).strip()
    period_norm_ = norm_period(period_raw)

    rec = {
        "dataset": dataset,
        "measure_clean": measure_clean,
        "typ_clean": typ_clean,
        "value_float": value_float,
        "region": region_raw,
        "period_raw": period_raw,
        "period_norm": period_norm_,
    }
    page_text = make_page_text(rec)
    meta = {
        F_DATASET: dataset,
        F_MEASURE: measure_clean,
        F_TYPE:    typ_clean,
        F_VALUE:   value_float,
        F_REGION:  region_raw,
        F_OKRES:   period_norm_ or period_raw,
        F_SRC:     os.path.basename(source_file),
        F_ROW:     int(row_index),
    }
    return Document(page_content=page_text, metadata=meta)

# ====== Autouzdrawianie meta z page_content (runtime) ======
_KV_PAT = re.compile(r"\b(dataset|measure|value|region|period|typ)\s*:\s*(.*?)(?:\s*\|\s*|\s*$)", re.I)

def ensure_core_meta(d: Document) -> None:
    """Jeśli meta uboższa, wyciągnij z page_content."""
    m = d.metadata or {}
    need_ds = not m.get(F_DATASET)
    need_me = not m.get(F_MEASURE)
    need_va = (m.get(F_VALUE) is None)
    need_rg = (m.get(F_REGION) is None)
    need_ok = (m.get(F_OKRES)  is None) and (m.get(F_PERIOD) is None)
    if not (need_ds or need_me or need_va or need_rg or need_ok):
        return
    text = d.page_content or ""
    kvs = {}
    for k, v in _KV_PAT.findall(text):
        kvs[k.lower()] = v.strip()
    if need_ds and "dataset" in kvs: m[F_DATASET] = kvs["dataset"]
    if need_me and "measure" in kvs: m[F_MEASURE] = kvs["measure"]
    if need_va and "value"   in kvs: m[F_VALUE]   = parse_value(kvs["value"])
    if need_rg and "region"  in kvs: m[F_REGION]  = kvs["region"]
    if need_ok:
        per = kvs.get("period")
        if per: m[F_OKRES] = norm_period(per) or per
    d.metadata = m

# ====== Walidacje szybkie ======
def has_core_fields(d: Document) -> bool:
    m = d.metadata or {}
    return bool(m.get(F_DATASET)) and bool(m.get(F_MEASURE))

def valid_value(d: Document) -> bool:
    v = (d.metadata or {}).get(F_VALUE, None)
    try:
        import math
        return (v is not None) and not (isinstance(v, float) and (math.isnan(v)))
    except Exception:
        return v is not None
