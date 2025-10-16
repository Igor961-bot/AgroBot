# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# ===== ENV (zachowujemy nazwy używane w reszcie projektu) =====
EMBEDDER_MODEL_T: Optional[str] = os.getenv("EMBEDDER_MODEL_T")
EMBEDDER_MODEL_U: Optional[str] = os.getenv("EMBEDDER_MODEL_U")
RERANKER_MODEL_T: Optional[str] = os.getenv("RERANKER_MODEL_T")
RERANKER_MODEL_U: Optional[str] = os.getenv("RERANKER_MODEL_U")

PERSIST_DIR:  str = os.getenv("PERSIST_DIR",  "./chroma_statystyki")
PERSIST_PATH: str = os.getenv("PERSIST_PATH", "./chroma_ustawa")

# OpenShift endpoints
BIELIK_BASE_URL: str = os.getenv("BIELIK_BASE_URL", "").rstrip("/")
BIELIK_MODEL_ID: str = os.getenv("BIELIK_MODEL_ID", "speakleash/Bielik-11B-v2.6-Instruct")
BIELIK_API_KEY: str = os.getenv("BIELIK_API_KEY", "dummy")

E5_BASE_URL: str = os.getenv("E5_BASE_URL", "").rstrip("/")
E5_API_KEY: str = os.getenv("E5_API_KEY", "dummy")

BGE_BASE_URL: str = os.getenv("BGE_BASE_URL", "").rstrip("/")
BGE_API_KEY: str = os.getenv("BGE_API_KEY", "dummy")
RADLAB_BASE_URL: str = os.getenv("RADLAB_BASE_URL", "").rstrip("/")
RADLAB_API_KEY: str = os.getenv("RADLAB_API_KEY", "dummy")

# Batching / timeouty (zostają)
_EMBED_BATCH_ITEMS      = int(os.getenv("EMBED_BATCH", "64"))
_EMBED_MAX_TOTAL_CHARS  = int(os.getenv("EMBED_MAX_TOTAL_CHARS", "50000"))
_EMBED_CONNECT_S        = float(os.getenv("EMBED_CONNECT_TIMEOUT_S", "5"))
_EMBED_READ_S           = float(os.getenv("EMBED_READ_TIMEOUT_S",  "300"))
_EMBED_WRITE_S          = float(os.getenv("EMBED_WRITE_TIMEOUT_S", "300"))

# ====== Helpers ======
def _headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def _post(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, headers=headers,
                      timeout=(_EMBED_CONNECT_S, max(_EMBED_READ_S, _EMBED_WRITE_S)))
    r.raise_for_status()
    return r.json()

def _normalize_e5_model_id(name: str) -> str:
    # pozwala zostawić starą nazwę 'text-embedding-intfloat-multilingual-e5-large-instruct'
    if name and "intfloat-multilingual-e5-large-instruct" in name:
        return "intfloat/multilingual-e5-large-instruct"
    return name

def _chunk_by_payload(texts: List[str]) -> List[List[str]]:
    out, cur, cur_len = [], [], 0
    for t in texts:
        tl = len(t)
        if cur and (len(cur)+1 > _EMBED_BATCH_ITEMS or cur_len + tl > _EMBED_MAX_TOTAL_CHARS):
            out.append(cur); cur, cur_len = [t], tl
        else:
            cur.append(t); cur_len += tl
    if cur: out.append(cur)
    return out

# ====== Bielik – prosty klient chat/completions (używaj jeśli potrzebujesz) ======
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
    url = f"{BIELIK_BASE_URL}/chat/completions"
    payload = {"model": BIELIK_MODEL_ID, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    data = _post(url, _headers(BIELIK_API_KEY), payload)
    return data["choices"][0]["message"]["content"]

# ====== Embeddings przez OpenShift E5 ======
from langchain_core.embeddings import Embeddings

class OpenShiftE5Embeddings(Embeddings):
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model = _normalize_e5_model_id(model_name)
        self.base = base_url.rstrip("/")
        self.key = api_key

    def _prefix(self, texts: List[str], is_query: bool) -> List[str]:
        # E5 wymaga prefiksów
        p = "query: " if is_query else "passage: "
        return [p + t for t in texts]

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        url = f"{self.base}/embeddings"
        payload = {"model": self.model, "input": batch}
        data = _post(url, _headers(self.key), payload)
        return [d["embedding"] for d in data["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = self._prefix(texts, is_query=False)
        out: List[List[float]] = []
        for sub in _chunk_by_payload(texts):
            out.extend(self._embed_batch(sub))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch(self._prefix([text], is_query=True))[0]

def build_embeddings(model_name: Optional[str]) -> Embeddings:
    if not model_name:
        raise ValueError("Brak EMBEDDER_MODEL_* w ENV.")
    if not E5_BASE_URL:
        raise RuntimeError("Brak E5_BASE_URL w ENV.")
    return OpenShiftE5Embeddings(model_name, E5_BASE_URL, E5_API_KEY)

emb_T = build_embeddings(EMBEDDER_MODEL_T)
emb_U = build_embeddings(EMBEDDER_MODEL_U)

# ====== Cross-encodery: /v1/rerank (nowe) lub /v1/score (legacy) ======
class RemoteCE:
    """
    Uniwersalny klient:
    - tryb 'rerank': POST /v1/rerank  {query, documents[], top_k, return_scores}
    - tryb 'score' : POST /v1/score   {model, input: [{text1, text2}, ...]} (lub równoważne)
    predict(pairs) zwraca numpy.array w tej samej kolejności co wejście.
    """
    def __init__(self, base_url: str, api_key: str = "", model_id: str = ""):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.model = model_id
        # autodetekcja trybu po ścieżce
        if self.base.endswith("/v1/rerank"):
            self.mode = "rerank"
            self.url = self.base
        elif self.base.endswith("/v1/score"):
            self.mode = "score"
            self.url = self.base
        else:
            # brak ścieżki -> spróbuj score (legacy)
            self.mode = "score"
            self.url = self.base + "/v1/score"
        self._headers = {"Content-Type": "application/json"}
        if self.key:
            self._headers["Authorization"] = f"Bearer {self.key}"

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=(_EMBED_CONNECT_S, max(_EMBED_READ_S, _EMBED_WRITE_S)),
        )
        r.raise_for_status()
        return r.json()

    def _predict_score(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[float]:
        out: List[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            payload = {
                "model": self.model,
                "input": [{"text1": a, "text2": b} for (a, b) in chunk],
            }
            data = self._post(self.url, payload)
            if "data" in data and isinstance(data["data"], list) and "score" in data["data"][0]:
                out.extend(float(x["score"]) for x in data["data"])
            elif "scores" in data:
                out.extend(float(s) for s in data["scores"])
            else:
                raise RuntimeError(f"Nieoczekiwana odpowiedź /v1/score: {data}")
        return out

    def _predict_rerank(self, pairs: List[Tuple[str, str]], batch_size_docs: int = 128) -> List[float]:
        """
        /v1/rerank działa na {query, documents[]}, więc grupujemy pary po query.
        Zwracamy listę score'ów w kolejności wejściowych par.
        """
        # 1) zbuduj kolejkę pozycji do odtworzenia kolejności
        #    idx_map: (query, doc_text) -> [list of indices w pairs]
        from collections import defaultdict
        idx_map: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, (q, d) in enumerate(pairs):
            idx_map[(q, d)].append(idx)

        # 2) grupuj dokumenty po query
        by_query: Dict[str, List[str]] = defaultdict(list)
        for (q, d) in pairs:
            by_query[q].append(d)

        # 3) strzelaj per query (chunkując dokumenty gdy trzeba)
        scores_out: List[Optional[float]] = [None] * len(pairs)
        for q, docs in by_query.items():
            # ewentualny dedupe w obrębie query, żeby nie oceniać tego samego d wiele razy
            seen = {}
            dedup_docs = []
            for d in docs:
                if d not in seen:
                    seen[d] = len(seen)
                    dedup_docs.append(d)

            # porcjuj, jeśli dokumentów jest bardzo dużo
            for i in range(0, len(dedup_docs), batch_size_docs):
                chunk_docs = dedup_docs[i : i + batch_size_docs]
                payload = {
                    "query": q,
                    "documents": chunk_docs,
                    "top_k": len(chunk_docs),
                    "return_scores": True,
                }
                data = self._post(self.url, payload)

                # spodziewamy się: {"results": [{"index": i, "document": {"text": ...}, "relevance_score": float}, ...]}
                results = data.get("results") or data.get("data") or []
                # zbuduj mapa: doc_text -> score
                text2score: Dict[str, float] = {}
                for r in results:
                    doc = r.get("document") or {}
                    text = (doc.get("text") or "").strip()
                    score = r.get("relevance_score")
                    if text:
                        text2score[text] = float(score)

                # rozlej po wszystkich wystąpieniach (q, d) w oryginalnych parach
                for d in chunk_docs:
                    if d in text2score:
                        for orig_idx in idx_map.get((q, d), []):
                            scores_out[orig_idx] = text2score[d]

        # sanity: żadne None
        for k, v in enumerate(scores_out):
            if v is None:
                # nie znaleziono doc w odpowiedzi (np. provider nie zwrócił tekstu) – awaryjnie 0.0
                scores_out[k] = 0.0
        return [float(x) for x in scores_out]

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        if self.mode == "score":
            vals = self._predict_score(pairs, batch_size=batch_size)
        else:
            vals = self._predict_rerank(pairs, batch_size_docs=max(32, batch_size))
        return np.asarray(vals, dtype=np.float32)

# Użycie: podstawiamy te same zmienne ENV co miałeś
cross_encoder_T = RemoteCE(BGE_BASE_URL or "https://bge-reranker-v2-krus-chatbox.apps.core.symmetry.pl/v1/rerank",
                           BGE_API_KEY, RERANKER_MODEL_T or "BAAI/bge-reranker-v2-m3")
cross_encoder_U = RemoteCE(RADLAB_BASE_URL or "https://polish-cross-encoder-krus-chatbox.apps.core.symmetry.pl/v1/rerank",
                           RADLAB_API_KEY, RERANKER_MODEL_U or "radlab/polish-cross-encoder")


# ====== Chroma (bez zmian) ======
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

vectorstore_T = Chroma(collection_name="statystyki", embedding_function=emb_T, persist_directory=PERSIST_DIR)
vectorstore_U = Chroma(collection_name="ustawa",     embedding_function=emb_U, persist_directory=PERSIST_PATH)

def get_embedder(model_name: Optional[str]):
    return build_embeddings(model_name)

__all__ = [
    "emb_T", "emb_U",
    "vectorstore_T", "vectorstore_U",
    "cross_encoder_T", "cross_encoder_U",
    "PERSIST_DIR", "PERSIST_PATH",
    "build_embeddings", "get_embedder",
    "llm_chat",
]
