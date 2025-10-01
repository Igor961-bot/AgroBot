from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from openai import OpenAI

from langchain_core.embeddings import Embeddings
try:
    from langchain_chroma import Chroma   
except Exception:
    from langchain_community.vectorstores import Chroma  

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from dotenv import load_dotenv
load_dotenv()

USE_LMSTUDIO: bool = os.getenv("USE_LMSTUDIO", "1").lower() in ("1", "true", "yes")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "bielik-11b-v2.6-instruct")

EMBEDDER_MODEL_T: Optional[str] = os.getenv("EMBEDDER_MODEL_T")
EMBEDDER_MODEL_U: Optional[str] = os.getenv("EMBEDDER_MODEL_U")

RERANKER_MODEL_T: Optional[str] = os.getenv("RERANKER_MODEL_T")  
RERANKER_MODEL_U: Optional[str] = os.getenv("RERANKER_MODEL_U")  
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./chroma_statystyki")  
PERSIST_PATH: str = os.getenv("PERSIST_PATH", "./chroma_ustawa")   


_client_chat = OpenAI(base_url=LLM_BASE_URL, api_key=LMSTUDIO_API_KEY)

class LMStudioPipeline:
    def __init__(self, system: Optional[str] = None, temperature: float = 0.2, max_new_tokens: int = 1024):
        self.system = system
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

    def __call__(self, inputs: str | List[str], **kwargs) -> Dict[str, Any] | List[Dict[str, Any]]:
        if isinstance(inputs, list):
            return [self._one(x, **kwargs) for x in inputs]
        return self._one(inputs, **kwargs)

    def _one(self, prompt: str, **kwargs) -> Dict[str, Any]:
        temp = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.max_new_tokens)))
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": prompt})
        resp = _client_chat.chat.completions.create(
            model=LLM_MODEL_ID, messages=msgs, temperature=temp, max_tokens=max_tokens, stream=False
        )
        return {"generated_text": resp.choices[0].message.content}

def hf_pipeline(task: str = "text-generation", **kwargs) -> LMStudioPipeline:
    assert task in ("text-generation", "text2text-generation", "conversational")
    return LMStudioPipeline(
        system=kwargs.get("system_prompt"),
        temperature=kwargs.get("temperature", 0.2),
        max_new_tokens=kwargs.get("max_new_tokens", 1024),
    )

gen_model = hf_pipeline("text-generation")
tokenizer = None  

def llm_generate(prompt: str, **kw) -> str:
    res = gen_model(prompt, **kw)
    if isinstance(res, list):
        res = res[0]
    return res["generated_text"]

_client_embed = OpenAI(base_url=LLM_BASE_URL, api_key=LMSTUDIO_API_KEY)

class LMStudioEmbeddings(Embeddings):
    def __init__(self, model: str, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def _is_e5(self) -> bool:
        name = (self.model or "").lower()
        return "e5" in name  

    def _prefix(self, texts: List[str], is_query: bool) -> List[str]:
        if not self._is_e5():
            return texts
        p = "query: " if is_query else "passage: "
        return [p + t for t in texts]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = _client_embed.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = self._prefix(texts, is_query=False)
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i+self.batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        text = self._prefix([text], is_query=True)[0]
        return self._embed_batch([text])[0]

def build_embeddings(model_name: Optional[str]) -> Embeddings:
    if not model_name:
        raise ValueError("Brak nazwy modelu embeddings w ENV.")
    if USE_LMSTUDIO:
        try:
            _client_embed.embeddings.create(model=model_name, input=["ping"])
            return LMStudioEmbeddings(model_name)
        except Exception as e:
            print(f"[EMB] LM Studio '{model_name}' niedostępne → fallback lokalny: {e}")
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

class OptimizedONNXCrossEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.fallback_model = None
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
                if logits.shape[-1] == 2:
                    s = _torch.softmax(logits, dim=-1)[:, 1]
                else:
                    s = logits[:, 0]
            scores.extend(s.cpu().numpy().tolist())
        return np.array(scores, dtype=np.float32)

cross_encoder_T = OptimizedONNXCrossEncoder(RERANKER_MODEL_T or "BAAI/bge-reranker-v2-m3")
cross_encoder_U = OptimizedONNXCrossEncoder(RERANKER_MODEL_U or "cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore_T = Chroma(collection_name="statystyki", embedding_function=emb_T, persist_directory=PERSIST_DIR)
vectorstore_U = Chroma(collection_name="ustawa",     embedding_function=emb_U, persist_directory=PERSIST_PATH)

__all__ = [
    "gen_model", "hf_pipeline", "llm_generate", "tokenizer",
    "emb_T", "emb_U", "vectorstore_T", "vectorstore_U",
    "cross_encoder_T", "cross_encoder_U",
    "PERSIST_DIR", "PERSIST_PATH",
]
