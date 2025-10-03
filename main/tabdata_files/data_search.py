# -*- coding: utf-8 -*-
# data_search.py

# stdlib
import os, re
from typing import List, Dict, Any, Tuple, Optional

# third-party
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

# projekt
from data_schema import (
    F_DATASET, F_MEASURE, F_REGION, F_OKRES, F_TYPE, F_SRC, F_ROW,
    row_to_document, norm_period, ensure_core_meta, has_core_fields, valid_value,
)
from resources import vectorstore_T as vectorstore
from resources import cross_encoder_T as reranker_ce
from resources import emb_T as _emb_model
from .common import (
    dbg, timer, norm_text, strip_acc,
    FIELD_VOCAB, REGION_TAXONOMY,
    rebuild_region_taxonomy, build_region_match_index, match_region_text,
    CSV_DIR, LLM_MQ_ENABLED, MQ_BASE_VARIANTS, MQ_LLM_VARIANTS, USE_HYDE,
    HYDE_VARIANTS_BASE, HYDE_VARIANTS_LLM, HYDE_STRIP_NUMBERS, RRF_WEIGHTS,
    MIN_CE_FOR_CONTEXT, TAB_CE_MAX, TAB_RRF_K,_add_to_vocab,
    _is_national, _soft_match, _ratio, USE_LLM, _query_mentions_foreign, _dataset_is_foreign,
    _is_special_internal_region 
)
from .transform import period_key, pick_latest_per_cluster

# ===================== CSV → Documents (dla BM25/vocab lokalnie) =====================
def _read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp1250", "iso-8859-2", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _looks_standard(df: pd.DataFrame) -> bool:
    need = {"dataset","measure","value","region","period","typ"}
    return need.issubset(set(df.columns))

# Zbierz źródła CSV (katalog lub plik)
CSV_SOURCES: Dict[str, str] = {}
if os.path.isdir(CSV_DIR):
    for fn in os.listdir(CSV_DIR):
        if fn.lower().endswith(".csv"):
            CSV_SOURCES[os.path.splitext(fn)[0]] = os.path.join(CSV_DIR, fn)
elif os.path.isfile(CSV_DIR) and CSV_DIR.lower().endswith(".csv"):
    CSV_SOURCES[os.path.splitext(os.path.basename(CSV_DIR))[0]] = CSV_DIR
else:
    print(f"[UWAGA] {CSV_DIR} nie jest ani katalogiem z CSV, ani plikiem CSV.")

# Budowa dokumentów z CSV (do BM25 + słowniki)
all_docs: List[Document] = []
for name, path in CSV_SOURCES.items():
    try:
        with timer(f"CSV_build:{os.path.basename(path)}"):
            df = _read_csv_smart(path)
            if not _looks_standard(df):
                print(f"[UWAGA] Pomijam '{path}' — niestandardowy schemat kolumn.")
                continue
            docs_local: List[Document] = []
            for i, row in df.iterrows():
                d = row_to_document(row.to_dict(), source_file=path, row_index=int(i))
                docs_local.append(d)
                # słowniki do best_match/region taxonomy
                _add_to_vocab(d.metadata.get(F_DATASET), F_DATASET)
                _add_to_vocab(d.metadata.get(F_MEASURE), F_MEASURE)
                _add_to_vocab(d.metadata.get(F_REGION),  F_REGION)
                _add_to_vocab(d.metadata.get(F_TYPE),    F_TYPE)
            all_docs.extend(docs_local)
    except Exception as e:
        print(f"[BŁĄD] {path}: {e}")

print(f"Zbudowano dokumentów: {len(all_docs)}")
dbg("DOCS_BUILT", count=len(all_docs))
dbg("VOCAB_SIZES",
    dataset=len(FIELD_VOCAB[F_DATASET]),
    measure=len(FIELD_VOCAB[F_MEASURE]),
    region=len(FIELD_VOCAB[F_REGION]),
    typ=len(FIELD_VOCAB["typ"])
)

# Po zbudowaniu słowników uzupełniamy taksonomię i indeks
rebuild_region_taxonomy()

build_region_match_index()

# ===================== CHROMA / RETRIEVERS =====================
from resources import vectorstore_T as vectorstore

def _collection_count(vs) -> int:
    coll = getattr(vs, "_collection", None)
    try:
        return coll.count() if coll is not None else 0
    except Exception:
        return 0

# Gdy kolekcja pusta – dociśnij dokumenty (ad-hoc) i zapisz
if _collection_count(vectorstore) == 0 and all_docs:
    with timer("CHROMA_add_documents"):
        vectorstore.add_documents(all_docs)
        getattr(vectorstore, "persist", lambda: None)()
dbg("CHROMA_STAT", vectors=_collection_count(vectorstore))

# BM25 na pełnym korpusie
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 80
dbg("BM25_INIT", docs=len(all_docs))

# Dense search helpers (embeddingowe z Chroma)
def dense_search(query: str, k: int = 80) -> List[Document]:
    with timer("dense_search"):
        retr = vectorstore.as_retriever(search_kwargs={"k": k})
        out = retr.invoke(query)
        dbg("dense_search", k=k, got=len(out))
        return out

def dense_search_on(texts: List[str], k: int = 40) -> List[Document]:
    with timer("dense_search_on"):
        retr = vectorstore.as_retriever(search_kwargs={"k": k})
        out: List[Document] = []
        for t in texts:
            r = retr.invoke(t)
            out.extend(r)
        # dedupe
        seen, uniq = set(), []
        for d in out:
            key = (d.metadata.get(F_SRC,"?"), d.metadata.get(F_ROW,-1))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        dbg("dense_search_on", inputs=len(texts), per_k=k, raw=len(out), uniq=len(uniq))
        return uniq


# ===================== CZĘŚĆ 2/2: MQ+HyDE (LM Studio), RRF, CE, Retrieve, Answer =====================
from langchain_openai import ChatOpenAI

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "bielik-11b-v2.6-instruct")

_llm_chat = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    temperature=0.2,
    max_tokens=256,
)

# ---- Parsowanie zapytań ----
_QUARTER_WORDS = {
    "1":"1","i":"1","pierwszy":"1","pierwszym":"1",
    "2":"2","ii":"2","drugi":"2","drugim":"2",
    "3":"3","iii":"3","trzeci":"3","trzecim":"3",
    "4":"4","iv":"4","czwarty":"4","czwartym":"4",
}
_KW_PAT = r"kw(?:\.|art(?:ał|ale|al|)|)"

def _extract_pl_quarter_period(qn: str) -> Optional[str]:
    t = norm_text(qn)
    m = re.search(rf"\b({'|'.join(_QUARTER_WORDS.keys())})\s*{_KW_PAT}\b.*?\b(20\d{{2}})\b", t)
    if m:
        q = _QUARTER_WORDS.get(m.group(1)); y = m.group(2)
        return f"{y}-Q{q}" if q else None
    m = re.search(rf"\b(20\d{{2}})\b.*?\b({'|'.join(_QUARTER_WORDS.keys())})\s*{_KW_PAT}\b", t)
    if m:
        y = m.group(1); q = _QUARTER_WORDS.get(m.group(2))
        return f"{y}-Q{q}" if q else None
    return None

def best_match(token: str, candidates: set[str], min_ratio: float = 0.62) -> Optional[str]:
    if not token or not candidates: return None
    tn = norm_text(token)
    best, score = None, 0.0
    for c in candidates:
        r = _ratio(tn, c)
        if r > score:
            best, score = c, r
    return best if score >= min_ratio else None

def _is_country_name(name: Optional[str]) -> bool:
    return norm_text(name or "") in REGION_TAXONOMY["country"]


def _region_matches(want: Optional[str], have: Optional[str]) -> bool:
    if not want: return True
    if _is_national(want): return _is_national(have)
    return _soft_match(want, have)

def parse_query_fields(q: str) -> Dict[str, Optional[str]]:
    qn = norm_text(q)

    # okres
    m = re.search(r"(20\d{2}\s*[-_/ ]?\s*q[1-4])", qn)
    period = norm_period(m.group(1)) if m else None
    if period is None:
        period = _extract_pl_quarter_period(qn)
    if period is None:
        years = set()
        years.update(re.findall(r"\b(20\d{2})\b", qn))
        years.update(re.findall(r"\b(20\d{2})\s*r\.?\b", qn))
        years.update(re.findall(r"\b(20\d{2})r\.?\b", qn))
        if years:
            period = str(sorted({int(y) for y in years})[-1])

    dataset = best_match(qn, FIELD_VOCAB[F_DATASET])
    measure = best_match(qn, FIELD_VOCAB[F_MEASURE])
    typ     = best_match(qn, FIELD_VOCAB["typ"])
    region  = match_region_text(q)

    # „dane …” vs „Dania”
    if region and norm_text(region) == "dania" and re.search(r"\bdane\b", qn):
        region = None

    if typ and measure and norm_text(typ) == norm_text(measure):
        typ = None

    return { "dataset": dataset, "measure": measure, "region": region, "typ": typ, "period": period }


# ---- MQ / HyDE ----
def _safe_unique(xs: List[str], limit: int = 20) -> List[str]:
    seen, out = set(), []
    for x in xs:
        s = norm_text(x)
        if s and s not in seen:
            seen.add(s); out.append(x.strip())
        if len(out) >= limit:
            break
    return out

def _take_closest_vocab(target: Optional[str], pool: set[str], k: int = 20) -> List[str]:
    if not target or not pool: return []
    scored = [(cand, _ratio(target, cand)) for cand in pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:k]]

def _base_mq_variants(q: str, n: int = 4) -> List[str]:
    qn = norm_text(q)
    base = {q.strip(), strip_acc(q), qn.replace("ile", "jaka jest liczba"),
            qn.replace("ile wynosi", "podaj wartość")}
    if "ubezpieczonych" in qn:
        base.update({"liczba ubezpieczonych ogółem", "liczba ubezpieczonych w krus ogółem"})
    if "płatnik" in qn or "platnik" in qn:
        base.update({"liczba płatników składek ogółem"})
    if "emerytura" in qn or "emerytur" in qn:
        base.update({"przeciętne świadczenie emerytalne ogółem"})
    if "wojew" not in qn and "region" not in qn:
        base.add(q.strip() + " ogółem")
    return list(base)[:n]

from langchain_openai import ChatOpenAI
def _llm_generate_query_expansions(
    question: str,
    measure_hint: Optional[str],
    typ_hint: Optional[str],
    dataset_hint: Optional[str],
    k_variants: int = MQ_LLM_VARIANTS
) -> List[str]:
    if not (USE_LLM and LLM_MQ_ENABLED):
        return []
    measure_cands = _take_closest_vocab(measure_hint, FIELD_VOCAB[F_MEASURE], k=20)
    typ_cands     = _take_closest_vocab(typ_hint,     FIELD_VOCAB["typ"],     k=20)
    dataset_cands = _take_closest_vocab(dataset_hint, FIELD_VOCAB[F_DATASET], k=20)

    sys_rules = (
        "Jesteś generatorem wariantów zapytań do wyszukiwarki danych KRUS.\n"
        "Zasady: zwięzłe warianty PL (jedna linia = jedno zapytanie). "
        "Używaj TYLKO podanych etykiet measure/typ/dataset (jeśli podane). "
        "Zachowuj znaczenie pytania. Nie zgaduj, gdy brak kandydatów."
    )
    m_line = " | ".join(measure_cands) if measure_cands else "(brak)"
    t_line = " | ".join(typ_cands)     if typ_cands     else "(brak)"
    d_line = " | ".join(dataset_cands) if dataset_cands else "(brak)"
    user_block = (
        f"Pytanie:\n{question}\n\n"
        f"Kandydaci 'measure': {m_line}\n"
        f"Kandydaci 'typ': {t_line}\n"
        f"Kandydaci 'dataset': {d_line}\n"
        f"Wygeneruj do {k_variants} wariantów. Zwróć TYLKO linie zapytań."
    )
    prompt = f"<s>[INST] <<SYS>>{sys_rules}<</SYS>>\n{user_block}\n[/INST]"
    try:
        with timer("MQ_llm"):
            out = _llm_chat.invoke(prompt)

        lines = [ln.strip(" -•\t") for ln in (out.content or "").splitlines()]
        lines = [re.sub(r'^\s*\d+[\).\s-]*', '', ln) for ln in lines]  # usuń numeracje
        lines = [ln.strip(' "„”') for ln in lines]                      # usuń cudzysłowy
        lines = [ln for ln in lines if ln]                              # puste out
        return _safe_unique(lines, limit=k_variants)
    except Exception as e:
        dbg("MQ_llm_error", err=str(e))
        return []

def make_mq_prompts_llm(q: str) -> List[str]:
    with timer("MQ_total"):
        parsed = parse_query_fields(q)
        base = _safe_unique(_base_mq_variants(q, n=MQ_BASE_VARIANTS))
        dbg("MQ_base", n=len(base), examples=base[:3])
        gen  = _llm_generate_query_expansions(
            question=q,
            measure_hint=parsed.get("measure"),
            typ_hint=parsed.get("typ"),
            dataset_hint=parsed.get("dataset"),
            k_variants=MQ_LLM_VARIANTS
        )
        out = _safe_unique(base + gen, limit=MQ_BASE_VARIANTS + MQ_LLM_VARIANTS)
        dbg("MQ_final", n=len(out), examples=out[:5])
        return out

def _pick_or_default(val: Optional[str], pool: set[str], k: int = 3) -> List[str]:
    if not pool and not val:
        return ["-"]
    base = [val] if val else []
    rest = _take_closest_vocab(val, pool, k=k) if pool else []
    out = [x for x in base + rest if x]
    return _safe_unique(out, limit=max(1, k+1)) or ["-"]

NUM_PAT = re.compile(r"(?<![A-Za-z])[-+]?\d[\d\s,./%]*")

def make_hyde_texts(question: str, n_base: int = HYDE_VARIANTS_BASE, n_llm: int = HYDE_VARIANTS_LLM) -> List[str]:
    with timer("HyDE_total"):
        parsed = parse_query_fields(question)
        m  = parsed.get("measure")
        t  = parsed.get("typ")
        ds = parsed.get("dataset")
        rg = parsed.get("region") or ("ogółem" if ("wojew" not in norm_text(question) and "region" not in norm_text(question)) else "-")
        pr = parsed.get("period") or "-"

        cand_meas = _pick_or_default(m,  FIELD_VOCAB[F_MEASURE], k=3)
        cand_typ  = _pick_or_default(t,  FIELD_VOCAB["typ"],     k=3)
        cand_ds   = _pick_or_default(ds, FIELD_VOCAB[F_DATASET], k=3)

        base: List[str] = []
        for i in range(max(1, n_base)):
            base.append(" | ".join([
                f"dataset: {cand_ds[i % len(cand_ds)]}",
                f"measure: {cand_meas[i % len(cand_meas)]}",
                "value: <X>",
                f"region: {rg}",
                f"period: {pr}",
                f"typ: {cand_typ[i % len(cand_typ)]}",
            ]))

        gen: List[str] = []
        if USE_LLM and n_llm > 0:
            sys = (
                "Jesteś pomocnikiem generującym hipotetyczne opisy danych KRUS. "
                "Napisz 1–2 zdania po polsku, bez liczb/%, bez konkretnych wartości. "
                "Jeśli pojawia się liczba, zastąp ją tokenem <X>."
            )
            user = (
                f"Pytanie: {question}\n"
                f"Wskazówki: dataset={ds or '-'}, measure={m or '-'}, typ={t or '-'}, region={rg}, period={pr}.\n"
                f"Wypisz {n_llm} wariantów, każdy w osobnej linii."
            )
            prompt = f"<s>[INST] <<SYS>>{sys}<</SYS>>\n{user}\n[/INST]"
            try:
                with timer("HyDE_llm"):
                    raw = _llm_chat.invoke(prompt).content or ""
                lines = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
                gen = lines[:n_llm]
            except Exception as e:
                dbg("HyDE_llm_error", err=str(e))
                gen = []

        texts = base + gen
        if HYDE_STRIP_NUMBERS:
            texts = [re.sub(NUM_PAT, "<X>", t) for t in texts]
        out = _safe_unique(texts, limit=n_base + n_llm)
        dbg("HyDE_final", n=len(out), sample=out[:3])
        return out


# ---- RRF ----
def rrf_merge_with_support_weighted(
    runs: List[List[Document]],
    labels: List[str],
    weights: Dict[str, float],
    k: int = 120,
    k_rrf: int = 60
) -> List[Document]:
    scores, support_w, best = {}, {}, {}
    pos_maps: List[Dict[Tuple[str,int], int]] = []
    for run in runs:
        m = {}
        for i, d in enumerate(run):
            key = (d.metadata.get(F_SRC,"?"), d.metadata.get(F_ROW,-1))
            m[key] = i
        pos_maps.append(m)

    keys = set().union(*[set(m.keys()) for m in pos_maps])
    for key in keys:
        s = 0.0; supw = 0.0
        for m, lab in zip(pos_maps, labels):
            if key in m:
                w = float(weights.get(lab, 1.0))
                s   += w * (1.0 / (k_rrf + m[key] + 1))
                supw += w
        scores[key]    = s
        support_w[key] = supw

    for run in runs:
        for d in run:
            key = (d.metadata.get(F_SRC,"?"), d.metadata.get(F_ROW,-1))
            if key not in best:
                best[key] = d

    merged = sorted(best.values(),
                    key=lambda d: scores[(d.metadata.get(F_SRC,"?"), d.metadata.get(F_ROW,-1))],
                    reverse=True)
    for d in merged:
        key = (d.metadata.get(F_SRC,"?"), d.metadata.get(F_ROW,-1))
        d.metadata["rrf_score"]   = scores[key]
        d.metadata["rrf_support"] = support_w[key]

    dbg("RRF_merge", inputs=[len(run) for run in runs], out=len(merged))
    return merged[:k]


# ---- Reranker CE + fallback cosine ----
from resources import cross_encoder_T as reranker_ce
from resources import emb_T as _emb_model

def _to_float_score(sc) -> float:
    try:
        if hasattr(sc, "shape"):
            return float(np.squeeze(sc))
    except Exception:
        pass
    if isinstance(sc, (list, tuple)):
        return 0.0 if not sc else _to_float_score(sc[0])
    try:
        return float(sc)
    except Exception:
        return 0.0

def _ce_text(d: Document, n: int = 800) -> str:
    s = d.page_content or ""
    return s if len(s) <= n else s[:n]

def _fallback_dense_similarity(query: str, docs: List[Document]) -> List[Tuple[Document,float]]:
    qv = np.asarray(_emb_model.embed_query(query), dtype=np.float32)
    dvs = np.asarray([_emb_model.embed_documents([d.page_content])[0] for d in docs], dtype=np.float32)
    qn = qv / max(1e-12, np.linalg.norm(qv))
    dn = dvs / np.clip(np.linalg.norm(dvs, axis=1, keepdims=True), 1e-12, None)
    sims = (dn @ qn).tolist()
    ranked = sorted([(d, s) for d, s in zip(docs, sims)], key=lambda x: x[1], reverse=True)
    for d, s in ranked:
        try: d.metadata["ce_score"] = float(s)
        except Exception: pass
    return ranked

def rerank_with_scores(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    if not docs:
        return []
    with timer("CE_total"):
        if reranker_ce is None:
            dbg("CE_disabled")
            for d in docs:
                d.metadata["ce_score"] = 0.0
            return [(d, 0.0) for d in docs]

        pairs = [(query, _ce_text(d)) for d in docs]
        try:
            with timer("CE_predict"):
                scores = reranker_ce.predict(pairs)
        except Exception as e:
            dbg("CE_error", err=str(e))
            ranked = _fallback_dense_similarity(query, docs)
            dbg("CE_fallback_cosine", out=len(ranked))
            return ranked

        out = []
        for d, sc in zip(docs, scores):
            scf = _to_float_score(sc)
            d.metadata["ce_score"] = scf
            out.append((d, scf))

        vals = [s for _, s in out]
        dbg("CE_stats", n=len(vals), mi=min(vals) if vals else None,
            ma=max(vals) if vals else None, mean=(float(np.mean(vals)) if vals else None))
        ranked = sorted(out, key=lambda x: x[1], reverse=True)

        if len(vals) >= 3 and (max(vals) - min(vals) < 1e-6):
            dbg("CE_flat_detected")
            ranked = _fallback_dense_similarity(query, docs)
            dbg("CE_fallback_cosine", out=len(ranked))
        return ranked
# ---------------------- GŁÓWNY RETRIEVE ----------------------
def retrieve(query: str, k_final: int = 24) -> List[Document]:
    dbg("Q", q=query)
    with timer("retrieve_total"):
        with timer("parse_query_fields"):
            parsed = parse_query_fields(query)
        prefs = derive_preferences(query, parsed)
        dbg("PARSED", **{k: v for k, v in parsed.items() if v})
        dbg("PREFS", **prefs)
        
        # Dense & MQ
        d1 = []
        if LLM_MQ_ENABLED:
            mq = make_mq_prompts_llm(query)
            d1 = dense_search_on(mq, k=60)

        # HyDE
        d2 = []
        if USE_HYDE:
            hy = make_hyde_texts(query, n_base=HYDE_VARIANTS_BASE, n_llm=HYDE_VARIANTS_LLM)
            d2 = dense_search_on(hy, k=60)

        # Bezpośrednie + BM25
        d0 = dense_search(query, k=100)
        bm25.k = 100
        with timer("bm25_invoke"):
            d3 = bm25.invoke(query)
        dbg("BM25", got=len(d3))

        # RRF merge
        fused = rrf_merge_with_support_weighted(
            [d0, d1, d2, d3],
            labels=["dense_q", "dense_mq", "hyde", "bm25"],
            weights=RRF_WEIGHTS,
            k=TAB_RRF_K,
            k_rrf=60
        )

        # CE rerank
        fused_for_ce = fused[:TAB_CE_MAX] if TAB_CE_MAX else fused
        if TAB_CE_MAX:
            dbg("CE_input", n=len(fused_for_ce))
        docs_scored = rerank_with_scores(query, fused_for_ce)
        dbg("AFTER_CE", n=len(docs_scored),
            above_thr=sum(1 for _, s in docs_scored if (s or 0.0) >= MIN_CE_FOR_CONTEXT),
            thr=MIN_CE_FOR_CONTEXT)

        # Autouzdrawianie meta zanim cokolwiek potniemy
        for d, _ in docs_scored:
            ensure_core_meta(d)

        # Wstępny kandydat set (po progu CE)
        cand = docs_scored
        
        if not re.search(r"\b(mswia|mon|msz|ms)\b", norm_text(query)):
            cand = [(d, s) for (d, s) in cand
                    if not _is_special_internal_region((d.metadata or {}).get(F_REGION))]

        # Odrzuć UE/EFTA/transfery gdy pytanie nie sugeruje zagranicy
        if not _query_mentions_foreign(query):
            cand = [(d, s) for (d, s) in cand if not _dataset_is_foreign((d.metadata or {}).get(F_DATASET))]

        # Wymuszenie 'ogółem'
        if prefs.get("force_national"):
            nat_only = [(d, s) for (d, s) in cand if _is_national((d.metadata or {}).get(F_REGION))]
            if nat_only:
                cand = nat_only

        # Próg CE
        cand_pairs: List[Tuple[Document, float]] = []
        for d, s in cand:
            ce = float(s or 0.0)
            d.metadata["ce_score"] = ce
            if ce >= MIN_CE_FOR_CONTEXT:
                cand_pairs.append((d, ce))
        dbg("CE_threshold", kept=len(cand_pairs), thr=MIN_CE_FOR_CONTEXT, cand=len(cand))

        # Kluczowe filtry jakości
        before = len(cand_pairs)
        cand_pairs = [(d, ce) for (d, ce) in cand_pairs if has_core_fields(d)]
        dbg("FILTER_core_fields", before=before, after=len(cand_pairs))

        before = len(cand_pairs)
        cand_pairs = [(d, ce) for (d, ce) in cand_pairs if valid_value(d)]
        dbg("FILTER_valid_value", before=before, after=len(cand_pairs))

        # Fallback: weź z RRF coś sensownego (z value) jeśli pusto
        if not cand_pairs:
            N = max(24, k_final * 2)
            dbg("CE_fallback_no_thr_with_value", take=N)
            tmp = [(d, float((d.metadata or {}).get("ce_score") or 0.0)) for d in fused[:N]]
            for d, _ in tmp:
                ensure_core_meta(d)
            tmp = [(d, ce) for (d, ce) in tmp if has_core_fields(d) and valid_value(d)]
            if tmp:
                cand_pairs = tmp[:max(1, k_final)]
            else:
                cand_pairs = [(d, float((d.metadata or {}).get("ce_score") or 0.0)) for d in fused[:max(1, k_final)]]

        # Filtry po okresie/regionie
        req_period = parsed.get("period")
        if req_period:
            req_pn = norm_period(req_period)
            if req_pn and re.fullmatch(r"20\d{2}", req_pn):
                def _year_of(p: Optional[str]) -> Optional[str]:
                    ps = norm_period(p)
                    if not ps: return None
                    m = re.match(r"^(20\d{2})", ps)
                    return m.group(1) if m else None
                cand_pairs = [(d, ce) for (d, ce) in cand_pairs if _year_of((d.metadata or {}).get(F_OKRES)) == req_pn]
                if not cand_pairs:
                    return []
            else:
                cand_pairs = [(d, ce) for (d, ce) in cand_pairs if norm_period((d.metadata or {}).get(F_OKRES)) == req_pn]
                if not cand_pairs:
                    return []

        if prefs.get("is_country_query") and parsed.get("region"):
            want_country = parsed["region"]
            cand_pairs = [(d, ce) for (d, ce) in cand_pairs if _soft_match(want_country, (d.metadata or {}).get(F_REGION))]
            if not cand_pairs:
                return []

        req_region = parsed.get("region")
        if req_region:
            cand_pairs = [(d, ce) for (d, ce) in cand_pairs if _region_matches(req_region, (d.metadata or {}).get(F_REGION))]
            if not cand_pairs:
                return []

        if prefs.get("force_national") and not prefs.get("is_country_query"):
            nat_pairs = [(d, ce) for (d, ce) in cand_pairs if _is_national((d.metadata or {}).get(F_REGION))]
            if nat_pairs:
                cand_pairs = nat_pairs

        # Klasteryzacja + wybór najświeższych
        docs_for_llm = pick_latest_per_cluster(
            cand_pairs,
            k_clusters=max(1, k_final),
            prefer_typ=prefs.get("prefer_typ"),
            prefer_national_first=prefs.get("force_national", False)
        )
        docs_for_llm.sort(key=lambda d: period_key(d.metadata.get(F_OKRES)), reverse=True)

        dbg("RETRIEVE_DONE", returned=min(k_final, len(docs_for_llm)))
        return docs_for_llm[:k_final]

def derive_preferences(query: str, parsed: Dict[str, Optional[str]]) -> Dict[str, Any]:
    qn = norm_text(query)
    prefer_typ, force_typ = None, False
    if "emerytaln" in qn or "emerytur" in qn or "emerytura" in qn:
        prefer_typ = "emerytury"
        force_typ = not ("rent" in qn or "renty" in qn or "rentę" in qn or "rentowe" in qn)

    is_country_query = _is_country_name(parsed.get("region"))
    force_national = ("wojew" not in qn and "region" not in qn and not is_country_query)

    return {
        "prefer_typ": prefer_typ,
        "force_typ": force_typ,
        "force_national": force_national,
        "is_country_query": is_country_query
    }



def _choose_best_doc(docs: List[Document], query: str, parsed: Dict[str, Optional[str]], prefs: Dict[str, Any]) -> Optional[Document]:
    if not docs:
        return None
    pool = list(docs)
    if prefs.get("force_national") and not prefs.get("is_country_query"):
        nat = [d for d in pool if _is_national((d.metadata or {}).get(F_REGION))]
        if nat:
            pool = nat

    def _key(d: Document):
        ce = float((d.metadata or {}).get("ce_score") or 0.0)
        meas = 1.0 if _soft_match(parsed.get("measure"), (d.metadata or {}).get(F_MEASURE)) else 0.0
        typm = 1.0 if _soft_match(parsed.get("typ"),     (d.metadata or {}).get(F_TYPE))    else 0.0
        return (ce, meas, typm, period_key((d.metadata or {}).get(F_OKRES)))

    return max(pool, key=_key) if pool else None



