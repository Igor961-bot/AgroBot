#format.py
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from data_schema import F_DATASET, F_MEASURE, F_VALUE, F_REGION, F_OKRES, F_TYPE
from .common import norm_text
# ---------------------- Formatowanie i odpowiedź ----------------------
def _pl_number(x: Optional[float]) -> str:
    if x is None: return "-"
    try:
        xv = float(x)
    except Exception:
        return str(x)
    if abs(xv - int(xv)) < 1e-6:
        s = f"{int(round(xv)):,}".replace(",", " ")
    else:
        s = f"{xv:,.2f}".replace(",", " ").replace(".", ",")
    return s

def _unit_from_measure(measure: str) -> str:
    m = norm_text(measure or "")
    if " zł" in (measure or "") or "zł" in m:
        return " zł"
    if "osób" in m or "liczba" in m or "osoby" in m:
        return ""
    return ""

def build_sources_rows(docs: List[Document]) -> List[Dict[str, Any]]:
    rows = []
    for d in docs:
        m = d.metadata or {}
        rows.append({
            "value":  m.get(F_VALUE),
            "dataset": m.get(F_DATASET),
            "measure": m.get(F_MEASURE),
            "type":    m.get(F_TYPE),
            "okres":   m.get(F_OKRES) or "-",
            "region":  m.get(F_REGION) or "-",
        })
    return rows
_all_ = [_pl_number, _unit_from_measure, build_sources_rows]
