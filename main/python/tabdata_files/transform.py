# -*- coding: utf-8 -*-
# transform.py
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from data_schema import F_DATASET, F_MEASURE, F_VALUE, F_REGION, F_OKRES, F_TYPE, valid_value
from .common import norm_text, _is_national, _soft_match

# ---- Sort/klastrowanie oraz preferencje ----
def period_key(p: Optional[str]) -> Tuple[int,int,int]:
    if not p or str(p).strip()=="": return (0,0,0)
    s = str(p).lower().strip()
    m = re.match(r"^(20\d{2})[-_/ ]?q([1-4])$", s)
    if m: return (int(m.group(1)), int(m.group(2)), 0)
    m = re.match(r"^q([1-4])[-_/ ]?(20\d{2})$", s)
    if m: return (int(m.group(2)), int(m.group(1)), 0)
    m = re.match(r"^(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})$", s)
    if m:
        y, mo, d = map(int, m.groups())
        daynum = (mo-1)*31 + d
        q = (mo-1)//3 + 1
        return (y, q, daynum)
    m = re.match(r"^(20\d{2})$", s)
    if m: return (int(m.group(1)), 0, 0)
    return (0,0,0)

def _key4(d: Document) -> Tuple[str,str,str,str]:
    m = d.metadata or {}
    return (
        norm_text(m.get(F_DATASET) or ""),
        norm_text(m.get(F_MEASURE) or ""),
        norm_text(m.get(F_TYPE) or ""),
        norm_text(m.get(F_REGION) or ""),
    )

def pick_latest_per_cluster(
    docs_scored: List[Tuple[Document, float]],
    k_clusters: int = 6,
    prefer_typ: Optional[str] = None,
    prefer_national_first: bool = False
) -> List[Document]:
    bykey: Dict[Tuple[str,str,str,str], List[Tuple[Document,float,int]]] = {}
    for d, s in docs_scored:
        sup = int(d.metadata.get("rrf_support", 1))
        bykey.setdefault(_key4(d), []).append((d, s, sup))

    best_per: List[Tuple[Document,float,int]] = []
    for key, items in bykey.items():
        items.sort(key=lambda x: (x[2], x[1], period_key(x[0].metadata.get(F_OKRES))), reverse=True)
        items_valid = [it for it in items if valid_value(it[0])]
        chosen = (sorted(items_valid, key=lambda x: period_key(x[0].metadata.get(F_OKRES)), reverse=True)[0]
                  if items_valid else items[0])
        best_per.append(chosen)

    def _prio(meta: Dict[str, Any]) -> Tuple[int, int]:
        region_prio = 1 if (prefer_national_first and _is_national(meta.get(F_REGION))) else 0
        typ_prio = 1 if (prefer_typ and _soft_match(prefer_typ, meta.get(F_TYPE))) else 0
        return (region_prio, typ_prio)

    best_per.sort(
        key=lambda x: (
            _prio(x[0].metadata),
            x[2],                                  # rrf_support
            x[1],                                  # score = CE
            period_key(x[0].metadata.get(F_OKRES)) # świeżość
        ),
        reverse=True
    )
    return [d for d,_,_ in best_per[:k_clusters]]
_all_ = [period_key, _key4, pick_latest_per_cluster]