# -*- coding: utf-8 -*-
# common.py

import os, re, unicodedata, logging, time
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from data_schema import (
    F_DATASET, F_MEASURE, F_REGION, F_TYPE
)

# ===================== KONFIG / ENV =====================
CSV_DIR: str = os.getenv("CSV_DIR", ".data/all_data.csv")

# Włączniki dla modułu danych
LLM_MQ_ENABLED = True          
MQ_BASE_VARIANTS = 4
MQ_LLM_VARIANTS  = 8

USE_HYDE = True
HYDE_VARIANTS_BASE = 3
HYDE_VARIANTS_LLM  = 3
HYDE_STRIP_NUMBERS = True

# Wagi dla fuzji RRF
RRF_WEIGHTS = {"dense_q": 1.0, "dense_mq": 0.95, "hyde": 0.7, "bm25": 1.0}

MIN_CE_FOR_CONTEXT = 0.06       # próg CE
USE_LLM = True                  # czy używać LLM (dla MQ/HyDE)

# Tryb szybki (wyłącz MQ/HyDE, próg CE)
FAST_DATA_MODE = os.getenv("FAST_DATA_MODE", "0").lower() in ("1", "true", "yes")
if FAST_DATA_MODE:
    LLM_MQ_ENABLED = False
    USE_HYDE = False
    MIN_CE_FOR_CONTEXT = float("-inf")

# Opcjonalne limity z ENV (ścięcie wejścia CE / rozmiaru RRF)
try:
    TAB_CE_MAX = int(os.getenv("TAB_CE_MAX", "").strip() or "0") or None
except Exception:
    TAB_CE_MAX = None
try:
    TAB_RRF_K = int(os.getenv("TAB_RRF_K", "").strip() or "0") or 160
except Exception:
    TAB_RRF_K = 160

# ===================== DEBUG / LOGGING (ON/OFF z ENV) =====================
TABDATA_DEBUG = os.getenv("TABDATA_DEBUG", "0").lower() in ("1", "true", "yes")
TABDATA_LOGFILE = os.getenv("TABDATA_LOGFILE", "")  # np. ./logs/tabdata.log

_log = logging.getLogger("tabdata")
if TABDATA_DEBUG and not _log.handlers:
    _log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); sh.setLevel(logging.DEBUG); _log.addHandler(sh)
    if TABDATA_LOGFILE:
        os.makedirs(os.path.dirname(TABDATA_LOGFILE), exist_ok=True)
        fh = logging.FileHandler(TABDATA_LOGFILE, encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); _log.addHandler(fh)
else:
    _log.addHandler(logging.NullHandler())

def dbg(msg: str, **kv):
    if not TABDATA_DEBUG:
        return
    if kv:
        try:
            msg = msg + " | " + " ".join(f"{k}={v}" for k, v in kv.items())
        except Exception:
            pass
    _log.debug(msg)

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dbg(f"[TIMER] {name}", ms=int((time.perf_counter() - t0) * 1000))

dbg("CFG",
    CSV_DIR=CSV_DIR,
    LLM_MQ_ENABLED=LLM_MQ_ENABLED,
    MQ_BASE_VARIANTS=MQ_BASE_VARIANTS,
    MQ_LLM_VARIANTS=MQ_LLM_VARIANTS,
    USE_HYDE=USE_HYDE,
    HYDE_BASE=HYDE_VARIANTS_BASE,
    HYDE_LLM=HYDE_VARIANTS_LLM,
    MIN_CE_FOR_CONTEXT=MIN_CE_FOR_CONTEXT,
    RRF_WEIGHTS=RRF_WEIGHTS,
    FAST_DATA_MODE=FAST_DATA_MODE,
    TAB_CE_MAX=TAB_CE_MAX,
    TAB_RRF_K=TAB_RRF_K,
)

# ===================== UTYLITY TEKSTOWE =====================
def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def norm_text(s: str) -> str:
    s = strip_acc(str(s)).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

NATIONAL_TOKENS = {
    "ogolem", "ogółem", "-", "", "polska", "kraj", "poland",
    "cały kraj", "cala polska", "cała polska", "caly kraj"
}

def _is_national(region: Optional[str]) -> bool:
    if region is None:
        return True
    s = norm_text(region)
    s2 = re.sub(r"[^\w ]", "", s)
    return (s2 in NATIONAL_TOKENS) or s2.startswith("ogolem")
# ===================== REGIONY I DOPASOWYWANIE =====================
REGION_CANON = {
    "dolnoslaskie","kujawsko-pomorskie","lubelskie","lubuskie","lodzkie","malopolskie",
    "mazowieckie","opolskie","podkarpackie","podlaskie","pomorskie","slaskie",
    "swietokrzyskie","warminsko-mazurskie","wielkopolskie","zachodniopomorskie",
    "austria","belgia","bulgaria","chorwacja","cypr","czechy","dania","estonia","finlandia",
    "francja","grecja","hiszpania","holandia","irlandia","islandia",
    "lichtenstein","liechtenstein","litwa","luksemburg","lotwa","łotwa","malta",
    "niemcy","norwegia","portugalia","rumunia","slowacja","słowacja","slowenia","słowenia",
    "szwajcaria","szwecja","wegry","węgry","wielka brytania","wlochy","włochy",
}
REGION_INDEX: List[Dict[str, Any]] = []
_REGION_STOPWORDS = {
    "woj", "woj.", "wojewodztwo", "region", "kraj", "panstwo", "panstwie", "ogolem", "razem",
    "dane", "danych", "danego", "daną", "danym", "dana"
}
_PL_VOWELS = set("aeiouy")
_PL_SUFFIXES = [
    "skiego","skimi","skich","skim","skie","ckiego","ckimi","ckich","ckim","ckie",
    "owego","owemu","owych","owym","owa","owe","owy",
    "iego","iemu","owej","owe","owy","owa","owym","owych",
    "ami","ach","owi","ego","emu","om","ym","im","em","ie","ia","iu","a","e","i","y","o","u","ą","ę"
]

def _is_region_token(s: Optional[str]) -> bool:
    if not s: return False
    return norm_text(s) in REGION_CANON

def _is_special_internal_region(s: Optional[str]) -> bool:
    if not s: return False
    t = norm_text(s)
    return t in {"mswia","ms","msw","msz", "mon"} or t.startswith("ms ")

REGION_TAXONOMY = {"national": set(), "voivodeship": set(), "country": set()}

def rebuild_region_taxonomy() -> None:
    regs = {norm_text(r) for r in FIELD_VOCAB[F_REGION] if r is not None}
    national = {r for r in regs if _is_national(r)}
    voiv     = {r for r in regs if _is_region_token(r)}
    specials = {r for r in regs if _is_special_internal_region(r)}
    country  = {r for r in regs if r not in national and r not in voiv and r not in specials and r not in {"", "-"}}
    REGION_TAXONOMY["national"]    = national
    REGION_TAXONOMY["voivodeship"] = voiv
    REGION_TAXONOMY["country"]     = country

# ===================== SŁOWNIKI / VOCAB =====================
FIELD_VOCAB: Dict[str, set[str]] = {F_DATASET: set(), F_MEASURE: set(), F_REGION: set(), "typ": set()}

def _add_to_vocab(val, field):
    if val is None: return
    s = str(val).strip()
    if not s: return
    # tu 'typ' w data_schema to F_TYPE; dla zgodności trzymajmy mapę:
    fld = "typ" if field == F_TYPE else field
    FIELD_VOCAB[fld].add(norm_text(s))



def _tokenize_words(s: str) -> list[str]:
    s = norm_text(s)
    s = re.sub(r"[-_/]", " ", s)
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t and t not in _REGION_STOPWORDS]

def _strip_pl_suffix(tok: str) -> str:
    t = tok
    for suf in sorted(_PL_SUFFIXES, key=len, reverse=True):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t

def _approx_syllables(word: str) -> list[str]:
    w = re.sub(r"[^a-z]", "", norm_text(word))
    if len(w) <= 3:
        return [w]
    syl, cur = [], ""
    for ch in w:
        if ch in _PL_VOWELS and cur and cur[-1] not in _PL_VOWELS:
            syl.append(cur); cur = ch
        else:
            cur += ch
    if cur: syl.append(cur)
    ngr = set()
    for n in (3, 4):
        for i in range(0, max(0, len(w) - n + 1)):
            ngr.add(w[i:i+n])
    return list(set(syl) | ngr)

def _roots_from_text(s: str) -> set[str]:
    return {_strip_pl_suffix(t) for t in _tokenize_words(s) if len(_strip_pl_suffix(t)) >= 3}

def _syllset_from_text(s: str) -> set[str]:
    out = set()
    for t in _tokenize_words(s):
        for seg in _approx_syllables(t):
            if len(seg) >= 2:
                out.add(seg)
    return out

def _chargrams_from_text(s: str) -> set[str]:
    w = re.sub(r"[^a-z]", "", norm_text(s))
    out = set()
    for n in (3, 4):
        for i in range(0, max(0, len(w) - n + 1)):
            out.add(w[i:i+n])
    return out

def _canon_parts(canon: str) -> List[str]:
    parts = re.split(r"[\s-]+", norm_text(canon))
    stems = [_strip_pl_suffix(p) for p in parts if p]
    return [s for s in stems if s]

def _match_canon_with_inflection(q_norm: str, canon: str) -> bool:
    stems = _canon_parts(canon)
    if not stems:
        return False
    part_patterns = [re.escape(st) + r"\w*" for st in stems]
    pat = r"\b" + r"(?:[\s_-]+)".join(part_patterns) + r"\b"
    return re.search(pat, q_norm) is not None

def match_region_text(q: str, min_score: float = 0.58) -> Optional[str]:
    # Bezpieczne wykrywanie regionu/kraju z tekstu (zabezpiecza przed 'dane'→'Dania').
    if not REGION_INDEX:
        return None
    q_norm  = norm_text(q)
    q_roots = _roots_from_text(q_norm)
    q_syll  = _syllset_from_text(q_norm)
    q_ngr   = _chargrams_from_text(q_norm)

    # 1) fleksja
    for e in REGION_INDEX:
        if _match_canon_with_inflection(q_norm, e["canon"]):
            return e["canon"]
    # 2) norma z separatorami
    for e in REGION_INDEX:
        pat = r"\b" + re.escape(e["norm"]).replace(r"\ ", r"[\s_-]+") + r"\b"
        if re.search(pat, q_norm):
            return e["canon"]
    # 3) fuzzy tylko gdy są przesłanki
    hint = re.search(r"\b(woj|wojew|region|powiat|gmina|w\s+(polsce|kraju|ue|efta|[a-ząćęłńóśźż-]{5,}))\b", q_norm)
    if not hint:
        return None
    def jacc(a: set[str], b: set[str]) -> float:
        if not a or not b: return 0.0
        i = len(a & b); u = len(a | b)
        return i / u if u else 0.0
    best_canon, best_score = None, 0.0
    for e in REGION_INDEX:
        s = 0.5*jacc(q_syll, e["syll"]) + 0.35*jacc(q_roots, e["roots"]) + 0.15*jacc(q_ngr, e["chargrams"])
        if s > best_score:
            best_score, best_canon = s, e["canon"]
    return best_canon if best_score >= min_score else None

def build_region_match_index() -> None:
    REGION_INDEX.clear()
    seen = set()
    for reg in FIELD_VOCAB[F_REGION]:
        if not reg: continue
        canon = str(reg)
        if canon in seen: continue
        seen.add(canon)
        REGION_INDEX.append({
            "canon": canon,
            "norm": norm_text(canon),
            "roots": _roots_from_text(canon),
            "syll":  _syllset_from_text(canon),
            "chargrams": _chargrams_from_text(canon),
        })

def _dataset_is_foreign(name: Optional[str]) -> bool:
    return bool(re.search(r"\b(ue|efta|wielk(?:a|iej)\s*bryt|dwustronn|transferowan)\b", str(name or ""), re.I))

def _query_mentions_foreign(q: str) -> bool:
    return bool(re.search(r"\b(ue|efta|wielk(?:a|iej)\s*bryt|dwustronn|transferowan)\b", norm_text(q), re.I))

def _is_country_name(name: Optional[str]) -> bool:
    return norm_text(name or "") in REGION_TAXONOMY["country"]

# Fuzzy (rapidfuzz → difflib fallback)
try:
    from rapidfuzz import fuzz
    def _ratio(a,b): return fuzz.token_set_ratio(a,b) / 100.0
except Exception:
    import difflib
    def _ratio(a,b): return difflib.SequenceMatcher(None, a, b).ratio()
def _soft_match(a: Optional[str], b: Optional[str], thr: float = 0.72) -> bool:
    if not a: return True
    if not b: return False
    return _ratio(norm_text(a), norm_text(b)) >= thr


