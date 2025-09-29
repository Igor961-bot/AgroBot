# server.py
# pip install fastapi uvicorn pydantic

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import re

# === Twoje funkcje (tych importów nie zmieniamy) ===
from krus_final import ask, want_follow_up, reset_context
from langchain_core.documents import Document

# === Dodatkowe importy dla modułu danych ===
from tabdata import answer as answer_tab  # tabelaryczne "dane"

# ---------- FastAPI & CORS ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- detekcja słowa-klucza "dane" ----------
WORD_DANE_RE = re.compile(r"\b(dane|statystyczne|statystyki)\b", re.IGNORECASE)
KEY_WORDS = {"dane", "statystyczne", "statystyki"}

# ---------- modele ----------
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
# ---------- helper ----------
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


def _is_data_query(q: str) -> bool:
    ql = (q or "").lower()
    if any(k in ql for k in KEY_WORDS):
        return True
    return WORD_DANE_RE.search(ql) is not None

# ---------- endpointy ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=AskOut)
def ask_endpoint(p: AskIn):
    # Opcjonalny reset pamięci modułu ustawowego
    if p.reset_memory:
        reset_context()

    # 1) Gałąź "dane" – tylko moduł tabelaryczny
    if _is_data_query(p.question):
        res_tab = answer_tab(p.question)

        if isinstance(res_tab, dict):
            data_txt = (res_tab.get("text") or "").strip()
            rows = res_tab.get("rows") or []
        else:
            # kompatybilność wstecz: gdy tabdata.answer zwróci sam string
            data_txt = (res_tab or "").strip()
            rows = []

        cols = ["value", "dataset", "measure", "type", "okres", "region"]

        return AskOut(
            answer=f"Oto znalezione dane tabelaryczne:\n{data_txt}",
            citations=[],
            module="dane",
            ask_followup=False,
            data_columns=cols,
            data_rows=[DataRow(**{k: (r.get(k) if isinstance(r, dict) else None) for k in cols}) for r in rows],
        )

    # 2) Gałąź "ustawa" – pełny moduł + cytaty
    res = ask(p.question)
    answer = res.get("answer", "") if isinstance(res, dict) else str(res)
    src = res.get("source_documents") if isinstance(res, dict) else []
    citations = [_doc_to_json(d) for d in (src or [])]

    # Po każdej odpowiedzi ustawowej frontend może zapytać usera o dopytanie.
    # Jeśli user kliknie "Tak", frontend powinien zawołać POST /followup
    return AskOut(
        answer=answer,
        citations=citations,  # typowane jako Citation przez Pydantic
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
