# server.py
# pip install fastapi uvicorn pydantic python-dotenv

# --- 0) KILL SWITCH LANGSMITH (musi być ZANIM importujesz LangChain/Twoje moduły) ---
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "0"
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGSMITH_API_KEY", None)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import re
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Twoje moduły – po .env!
from krus_final import ask, want_follow_up, reset_context
from langchain_core.documents import Document
from tabdata_files.interact import answer as answer_tab  # moduł „dane”

# --- 3) Logowanie ---
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "0") in ("1", "true", "True") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")

# --- 4) FastAPI + CORS ---
app = FastAPI(title="KRUS-chat backend")

DEFAULT_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
extra_origins = {o.strip() for o in os.getenv("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()}
allow_origins = sorted(DEFAULT_ORIGINS | extra_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5) Detekcja „dane” ---
WORD_DANE_RE = re.compile(r"\b(dane|statystyczne|statystyki)\b", re.IGNORECASE)
KEY_WORDS = {"dane", "statystyczne", "statystyki"}

def _is_data_query(q: str) -> bool:
    ql = (q or "").lower()
    if any(k in ql for k in KEY_WORDS):
        return True
    return WORD_DANE_RE.search(ql) is not None

# --- 6) Modele odpowiedzi (Pydantic) ---
class DataRow(BaseModel):
    value: Optional[float | int | str] = None
    dataset: Optional[str] = None
    measure: Optional[str] = None
    type: Optional[str] = None
    okres: Optional[str] = None
    region: Optional[str] = None

class AskIn(BaseModel):
    question: str
    reset_memory: bool = False

class Citation(BaseModel):
    id: Optional[str] = None
    chapter: Optional[str] = None
    article: Optional[str] = None
    paragraph: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None

class AskOut(BaseModel):
    answer: str
    citations: List[Citation]
    module: str  # "ustawa" | "dane"
    ask_followup: bool = False
    data_columns: Optional[List[str]] = None
    data_rows: Optional[List[DataRow]] = None

# --- 7) Helpery mapujące dokumenty na JSON ---
def _to_str(v):
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _doc_to_json(d: Document) -> Dict[str, Any]:
    md = d.metadata or {}
    return {
        "id": _to_str(md.get("id")),
        "chapter": _to_str(md.get("rozdzial") or md.get("chapter")),
        "article": _to_str(md.get("artykul") or md.get("article")),
        "paragraph": _to_str(md.get("ust") or md.get("paragraph")),
        "score": _to_float(md.get("rerank_score")),
        "text": d.page_content,
    }

# --- 8) Startup checks (embedder/Chroma) ---
@app.on_event("startup")  # FastAPI deprecates on_event, ale na dev nam wystarczy
def _startup_checks():
    log.info("[STARTUP] cwd=%s", os.getcwd())

    # Spróbuj zalogować podstawowe ścieżki/stan (jeśli eksportujesz z resources.py)
    try:
        from resources import PERSIST_PATH, PERSIST_DIR
        log.info("[STARTUP] PERSIST_PATH=%s", PERSIST_PATH)
        log.info("[STARTUP] PERSIST_DIR=%s", PERSIST_DIR)
    except Exception:
        pass

    # Test embeddera – użyjemy istniejącego emb_U (albo emb_T)
    try:
        from resources import emb_U as _emb
    except Exception:
        try:
            from resources import emb_T as _emb
        except Exception as e:
            log.warning("[STARTUP][EMB] FAIL: %s", e)
            _emb = None

    if _emb is not None:
        try:
            v = _emb.embed_query("ping")
            log.info("[STARTUP][EMB] OK, dim=%s", len(v))
        except Exception as e:
            log.warning("[STARTUP][EMB] FAIL: %s", e)

    # Test wektorowni – najpierw spróbuj globalnego db z krus_final, a jak nie ma, to resources.vectorstore_U
    _db = None
    try:
        from krus_final import db as _db  # krus_final ustawia db = vectorstore_U
    except Exception:
        try:
            from resources import vectorstore_U as _db
        except Exception as e:
            log.warning("[STARTUP][CHROMA] FAIL: %s", e)

    if _db is not None:
        try:
            sample = _db.similarity_search("ustawa", k=1)
            log.info("[STARTUP][CHROMA] sample=%d", len(sample))
        except Exception as e:
            log.warning("[STARTUP][CHROMA] FAIL: %s", e)

# --- 9) Endpointy ---
@app.get("/health")
def health():
    info = {"ok": True, "cwd": os.getcwd()}
    return info

@app.post("/ask", response_model=AskOut)
def ask_endpoint(p: AskIn):
    if p.reset_memory:
        reset_context()

    # Gałąź „dane”
    if _is_data_query(p.question):
        res_tab = answer_tab(p.question)
        if isinstance(res_tab, dict):
            data_txt = (res_tab.get("text") or "").strip()
            rows = res_tab.get("rows") or []
        else:
            data_txt = (res_tab or "").strip()
            rows = []

        cols = ["value", "dataset", "measure", "type", "okres", "region"]
        data_rows = [
            DataRow(**{k: (r.get(k) if isinstance(r, dict) else None) for k in cols})
            for r in rows
        ]

        return AskOut(
            answer=f"Oto znalezione dane tabelaryczne:\n{data_txt}",
            citations=[],
            module="dane",
            ask_followup=False,
            data_columns=cols,
            data_rows=data_rows,
        )

    # Gałąź „ustawa”
    res = ask(p.question)
    answer = res.get("answer", "") if isinstance(res, dict) else str(res)
    src = res.get("source_documents") if isinstance(res, dict) else []
    citations = [_doc_to_json(d) for d in (src or [])]

    return AskOut(
        answer=answer,
        citations=citations,
        module="ustawa",
        ask_followup=True,
    )

@app.post("/followup")
def followup_endpoint():
    want_follow_up()
    return {"ok": True, "message": "Follow-up uzbrojony: następne pytanie będzie dopytaniem."}

@app.post("/reset")
def reset_endpoint():
    reset_context()
    return {"ok": True}

# --- 10) Opcjonalny endpoint debug (ENABLE_DEBUG_ENDPOINT=1 w .env) ---
if os.getenv("ENABLE_DEBUG_ENDPOINT", "0") in ("1", "true", "True"):
    @app.get("/debug")
    def debug():
        out: Dict[str, Any] = {"cwd": os.getcwd()}
        try:
            from resources import emb_U as _emb
            out["embed_dim"] = len(_emb.embed_query("test"))
        except Exception as e:
            out["embed_error"] = str(e)
        try:
            from krus_final import db as _db
            ds = _db.similarity_search("osoba niezdolna do samotnej egzystencji", k=3)
            out["sim_docs"] = len(ds)
            out["sim_ids"] = [getattr(d, "metadata", {}).get("id") for d in ds]
        except Exception as e:
            out["sim_error"] = str(e)
        try:
            from resources import PERSIST_PATH, PERSIST_DIR
            out["PERSIST_PATH"] = PERSIST_PATH
            out["PERSIST_DIR"] = PERSIST_DIR
        except Exception:
            pass
        return out
