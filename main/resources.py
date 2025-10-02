from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()

# --- ENV ---
USE_LMSTUDIO: bool = os.getenv("USE_LMSTUDIO", "1").lower() in ("1","true","yes")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

EMBEDDER_MODEL_T: Optional[str] = os.getenv("EMBEDDER_MODEL_T")
EMBEDDER_MODEL_U: Optional[str] = os.getenv("EMBEDDER_MODEL_U")

RERANKER_MODEL_T: Optional[str] = os.getenv("RERANKER_MODEL_T")  # np. BAAI/bge-reranker-v2-m3
RERANKER_MODEL_U: Optional[str] = os.getenv("RERANKER_MODEL_U")  # np. radlab/polish-cross-encoder

PERSIST_DIR: str  = os.getenv("PERSIST_DIR",  "./chroma_statystyki")
PERSIST_PATH: str = os.getenv("PERSIST_PATH", "./chroma_ustawa")

# Możesz wyłączyć ONNX CE na Windows (eliminuje PermissionError na temp .onnx_data)
USE_ONNX_CE = 0

# --- LM Studio clienty (tylko embeddings potrzeba tutaj) ---
# --- LM Studio client (długie timeouty, bez keep-alive, bez przycinania treści) ---
import httpx
from openai import OpenAI
import os

_EMBED_CONNECT_S = float(os.getenv("EMBED_CONNECT_TIMEOUT_S", "5"))
_EMBED_READ_S    = float(os.getenv("EMBED_READ_TIMEOUT_S",  "300"))  # spokojnie nawet 300s
_EMBED_WRITE_S   = float(os.getenv("EMBED_WRITE_TIMEOUT_S", "300"))

_httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=_EMBED_CONNECT_S, read=_EMBED_READ_S, write=_EMBED_WRITE_S, pool=None),
    limits=httpx.Limits(max_keepalive_connections=0, max_connections=5), 
    headers={
        "Connection": "close",         
        "Accept-Encoding": "gzip",      
    },
)

_client_embed = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    http_client=_httpx_client,
)

# --- Embeddings (LM Studio + fallback) ---
from langchain_core.embeddings import Embeddings

import time
from typing import List
from langchain_core.embeddings import Embeddings

# Sterowanie wyłącznie batchingiem (brak jakiegokolwiek cięcia treści):
_EMBED_BATCH_ITEMS      = int(os.getenv("EMBED_BATCH", "8"))             # max elementów w jednym żądaniu
_EMBED_MAX_TOTAL_CHARS  = int(os.getenv("EMBED_MAX_TOTAL_CHARS", "50000"))  # max łączna liczba znaków w żądaniu
_EMBED_RETRIES          = int(os.getenv("EMBED_RETRIES", "4"))

def _chunk_by_payload(texts: List[str]) -> List[List[str]]:
    """Dzielenie na sub-batche wg: limit elementów i łącznego rozmiaru znaków (bez obcinania)."""
    out, cur, cur_len = [], [], 0
    for t in texts:
        t_len = len(t)
        # jeśli dodanie t przekroczy limit elementów lub łącznej długości → zamknij bieżący batch
        if cur and (len(cur) + 1 > _EMBED_BATCH_ITEMS or cur_len + t_len > _EMBED_MAX_TOTAL_CHARS):
            out.append(cur)
            cur, cur_len = [t], t_len
        else:
            cur.append(t); cur_len += t_len
    if cur:
        out.append(cur)
    return out

class LMStudioEmbeddings(Embeddings):
    """Brak przycinania. E5 dostaje prefix query/passage, a wysyłka dzielona na bezpieczne porcje."""
    def __init__(self, model: str, batch_size: int = None):
        self.model = model
        self.batch_size = batch_size or _EMBED_BATCH_ITEMS  # kompatybilność, realnie dzieli _chunk_by_payload

    def _is_e5(self) -> bool:
        return "e5" in (self.model or "").lower()

    def _prefix(self, texts: List[str], is_query: bool) -> List[str]:
        if not self._is_e5():
            return texts
        p = "query: " if is_query else "passage: "
        return [p + t for t in texts]

    def _post_embed_once(self, batch: List[str]) -> List[List[float]]:
        # pojedyncza próba, bez bisekcji
        resp = _client_embed.embeddings.create(model=self.model, input=batch)
        return [d.embedding for d in resp.data]

    def _post_embed_with_retry(self, batch: List[str]) -> List[List[float]]:
        last_err = None
        for attempt in range(_EMBED_RETRIES):
            try:
                return self._post_embed_once(batch)
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s, 4s
        # po wyczerpaniu retry spróbuj zmniejszyć batch (bisekcja)
        if len(batch) > 1:
            mid = len(batch) // 2
            left = self._post_embed_with_retry(batch[:mid])
            right = self._post_embed_with_retry(batch[mid:])
            return left + right
        # jedynka też nie przeszła → rzuć błąd
        raise last_err

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = self._prefix(texts, is_query=False)
        out: List[List[float]] = []
        for sub in _chunk_by_payload(texts):
            # podgląd: ile wysyłamy, ile znaków łącznie (bez obcinania)
            print(f"[EMB] send: {len(sub)} items, {sum(len(t) for t in sub)} chars")
            out.extend(self._post_embed_with_retry(sub))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._post_embed_with_retry(self._prefix([text], is_query=True))[0]


def build_embeddings(model_name: Optional[str]) -> Embeddings:
    if not model_name:
        raise ValueError("Brak nazwy modelu embeddings w ENV.")
    if USE_LMSTUDIO:
        try:
            _client_embed.embeddings.create(model=model_name, input=["ping"])
            return LMStudioEmbeddings(model_name)
        except Exception as e:
            print(f"[EMB] LM Studio '{model_name}' niedostępne → fallback lokalny: {e}")
    # fallback lokalny: Sentence-Transformers → HF BGE
    try:
        from sentence_transformers import SentenceTransformer
        class ST_Emb(Embeddings):
            def __init__(self, name: str):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(name, device=device)
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
                return v.astype(np.float32).tolist()
            def embed_query(self, text: str) -> List[float]:
                v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
                return v[0].astype(np.float32).tolist()
        return ST_Emb(model_name)
    except Exception:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

emb_T = build_embeddings(EMBEDDER_MODEL_T)
emb_U = build_embeddings(EMBEDDER_MODEL_U)

# --- Cross-Encoder (ONNX → ST) ---
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

class OptimizedONNXCrossEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.fallback_model = None

        if not USE_ONNX_CE:
            self.fallback_model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            print(f"[CE] ONNX wyłączony (USE_ONNX_CE=0) → CrossEncoder({model_name})")
            return

        try:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = ORTModelForSequenceClassification.from_pretrained(
                model_name, provider="CPUExecutionProvider", export=True
            )
            print(f"[CE] ONNX OK: {model_name}")
        except Exception as e:
            print(f"[CE] Fallback ST dla {model_name}: {e}")
            self.fallback_model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        if self.model is None:
            return np.array(self.fallback_model.predict(pairs, batch_size=batch_size))
        import torch as _torch
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i+batch_size]
            t1 = [p[0] for p in b]; t2 = [p[1] for p in b]
            enc = self.tok(t1, t2, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with _torch.no_grad():
                logits = self.model(**{k: v for k, v in enc.items()}).logits
                s = _torch.softmax(logits, dim=-1)[:, -1] if logits.shape[-1] > 1 else logits[:, 0]
            scores.extend(s.cpu().numpy().tolist())
        return np.array(scores, dtype=np.float32)

cross_encoder_T = OptimizedONNXCrossEncoder(RERANKER_MODEL_T or "BAAI/bge-reranker-v2-m3")
cross_encoder_U = OptimizedONNXCrossEncoder(RERANKER_MODEL_U or "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chroma: tylko klienci (bez ingestu) ---
try:
    from langchain_chroma import Chroma  # nowszy provider
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
]

