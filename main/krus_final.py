
import os
from huggingface_hub import login
import os, re, unicodedata, asyncio
from typing import List, Optional, Dict, Callable
import numpy as np
import torch
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document  

from resources import vectorstore_U
from resources import cross_encoder_U
from langchain_openai import ChatOpenAI
import os


from pydantic import PrivateAttr
  
torch.backends.cuda.matmul.allow_tf32 = True  

sorDEBUG = True
DEBUG = sorDEBUG
RETURN_STRING_WHEN_DEBUG_FALSE = True

RERANK_THRESHOLD = 0.2
K_SIM   = 15
K_FINAL = 5

db = vectorstore_U
model = gen_model
cross_encoder = cross_encoder_U


if "db" not in globals():
    raise RuntimeError("Brak globalnej bazy `db` (Chroma). Zainicjalizuj ją przed załadowaniem skryptu.")

#------------------------------------------------ funkcje pomocnicze ------------------------------------------------

def strip_accents_lower(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

ROLE_CUT_RE = re.compile(
    r"(?i)"                              
    r"(###\s*(?:user|asystent|dokumenty|system)?\s*:?" 
    r"|(?:^|\s)(?:user|asystent|dokumenty|system)\s*:" 
    r"|<\s*(?:user|assistant|docs?|system)\s*>)"      
)

def cut_after_role_markers(s: str) -> str:
    if not s:
        return s
    m = ROLE_CUT_RE.search(s)
    return s[:m.start()].rstrip() if m else s

REF_RE_EXT = re.compile(
    r"(?:art\.?\s*(?P<art>[0-9]+[a-z]?))"
    r"(?:\s*(?:ust(?:\.|ęp)?|ustep)\s*(?P<ust>[0-9]+[a-z]?))?"
    r"(?:\s*(?:pkt\.?)\s*(?P<pkt>[0-9]+[a-z]?))?"
    r"(?:\s*(?:lit\.?)\s*(?P<lit>[a-z]))?",
    re.IGNORECASE
)

def parse_ref_ext(query: str) -> Optional[Dict[str, str]]:
    m = REF_RE_EXT.search(query or "")
    if not m:
        return None
    ref = {}
    if m.group("art"): ref["article"]   = m.group("art").lower()
    if m.group("ust"): ref["paragraph"] = m.group("ust").lower()
    if m.group("pkt"): ref["punkt"]     = m.group("pkt").lower()
    if m.group("lit"): ref["litera"]    = m.group("lit").lower()
    return ref if ref else None


def _build_citations_block(docs: List[Document]) -> str:
    if not docs:
        return "Cytowane ustępy:\n(brak)\n"
    lines = ["Cytowane ustępy:"]
    for d in docs:
        md = d.metadata or {}
        rozdz = md.get("rozdzial", md.get("chapter"))
        art   = md.get("artykul",  md.get("article"))
        ust   = md.get("ust",      md.get("paragraph"))
        pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
        lines.append(f"- [{pid}] Rozdz.{rozdz} Art.{art} Ust.{ust}")
    return "\n".join(lines) + "\n"

def format_docs_for_prompt(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        md = d.metadata or {}
        rozdz = md.get("rozdzial", md.get("chapter"))
        art   = md.get("artykul",  md.get("article"))
        ust   = md.get("ust",      md.get("paragraph"))
        pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
        blocks.append(f"[{pid}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def log_rerank_scores(docs: List[Document], header: str = "DEBUG rerank scores") -> None:
    if not DEBUG: return
    print(header)
    if not docs:
        print("(brak dokumentów)"); return
    for d in docs:
        md = d.metadata or {}
        pid = md.get("id") or f"ch{md.get('chapter')}-art{md.get('article')}-ust{md.get('paragraph')}"
        sc  = md.get("rerank_score")
        print(f"- {pid}: score={sc:.6f}" if isinstance(sc, (int, float)) else f"- {pid}: score=(brak)")


def strip_markdown_bold(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = text.replace("**", "").replace("__", "")
    return text

_SENT_ENDERS = ".?!…"
_CLOSERS = "”»\")]’"

def _rstrip_u(s: str) -> str:
    return s.rstrip(" \t\r\n\u00A0")

_LIST_MARKER_ONLY_RE = re.compile(
    r"""^[\s\u00A0]*(
            [-*+•·—–]
          | (?:\(?\d+[a-z]?\)|\d+[a-z]?[.)])
          | (?:\(?[ivxlcdm]+\)|[ivxlcdm]+[.)])
          | (?:[a-z][.)])
        )[\s\u00A0]*$""",
    re.IGNORECASE | re.VERBOSE
)

def _strip_trailing_empty_list_item(s: str) -> tuple[str, bool]:
    if not s:
        return s, False
    s2 = _rstrip_u(s)
    if not s2:
        return s2, (s2 != s)
    lines = s2.splitlines()
    last = _rstrip_u(lines[-1])
    if _LIST_MARKER_ONLY_RE.match(last or ""):
        return _rstrip_u("\n".join(lines[:-1])), True
    return s2, False

def _ends_with_full_stop(s: str) -> bool:
    return re.search(rf"[{re.escape(_SENT_ENDERS)}][{re.escape(_CLOSERS)}]*[\s\u00A0]*$", s) is not None

_ABBR_SET = {"art","ust","pkt","lit","tj","tzw","np","itd","itp","m.in","prof","dr","nr","poz","cd","al","ul","pl","św","sw"}

def _last_token_before_dot_for_trim(buf: str) -> str:
    m = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿŁŚŻŹĆŃÓÄÖÜĄĘłśżźćńóäöüąę]+)\.\s*$", buf)
    return (m.group(1).lower() if m else "")

def _is_abbreviation_dot(text: str, dot_pos: int) -> bool:
    prev = text[:dot_pos+1]
    tok = _last_token_before_dot_for_trim(prev)
    return tok in _ABBR_SET

def _find_last_safe_boundary(s: str) -> int | None:
    i = len(s) - 1
    while i >= 0 and (s[i].isspace() or s[i] in _CLOSERS or s[i] == "\u00A0"):
        i -= 1
    while i >= 0:
        ch = s[i]
        if ch in _SENT_ENDERS:
            if ch == "." and _is_abbreviation_dot(s[:i+1], i):
                i -= 1
                continue
            return i + 1
        i -= 1
    return None

def trim_incomplete_sentences(text: str) -> str:
    if not text:
        return text
    text = cut_after_role_markers(text)
    s = _rstrip_u(text)
    changed = True
    while changed:
        s, changed = _strip_trailing_empty_list_item(s)
    if s.endswith(":"):
        last_nl = s.rfind("\n")
        s = _rstrip_u(s[:last_nl]) if last_nl != -1 else ""
    if not s:
        return s
    if _ends_with_full_stop(s):
        return s
    cut = _find_last_safe_boundary(s)
    if cut is None:
        return s
    return _rstrip_u(s[:cut])

def _finalize_return(text: str, docs: List[Document], mode: str):
    hint = "\n\n(Jeśli chcesz dopytać, kliknij „Chciałbym dopytać”, a dodam kolejny ustęp do kontekstu.)"
    text_out = text + hint

    debug = [{"id": (d.metadata or {}).get("id"),
              "score": (d.metadata or {}).get("rerank_score")} for d in (docs or [])]
    payload = {"answer": text_out, "source_documents": docs, "debug": {"mode": mode, "rerank": debug}}
    return payload

_SMALLTALK_RULES = [
    (r"^(czesc|cze|hej|heja|hejka|witam|siema|elo|halo|dzien dobry|dobry wieczor)\b",
     "Cześć! W czym mogę pomóc w sprawie KRUS/ustawy?"),
    (r"\b(dzieki|dziekuje|dzieki wielkie|dziekuje bardzo|thx|thanks)\b",
     "Nie ma sprawy! Jeśli chcesz, podaj kolejne pytanie."),
]
def smalltalk_reply(user_q: str) -> Optional[str]:
    qn = strip_accents_lower(user_q)
    for pat, resp in _SMALLTALK_RULES:
        if re.search(pat, qn, flags=re.IGNORECASE):
            return resp
    return None

def _short_doc_label(d: Document) -> str:
    md = d.metadata or {}
    rozdz = md.get("rozdzial", md.get("chapter"))
    art   = md.get("artykul",  md.get("article"))
    ust   = md.get("ust",      md.get("paragraph"))
    pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
    return f"{pid}"

def _update_state_from_docs(docs: List[Document], user_q: str):
    if not docs: return
    STATE.last_docs = docs[:]
    STATE.last_query = user_q
    md0 = docs[0].metadata or {}
    STATE.last_article_num   = (md0.get("artykul")  or md0.get("article"))
    STATE.last_paragraph_num = (md0.get("ust")      or md0.get("paragraph"))
    if DEBUG:
        print(f"[STATE] last_article={STATE.last_article_num} last_paragraph={STATE.last_paragraph_num}")

def rewrite_query(user_q: str) -> str:
    base = (STATE.last_query or "").strip()
    if not base: return user_q
    doc_labels = [_short_doc_label(d) for d in (STATE.last_docs or [])][:6]
    doc_part = f" (odnieś do: {', '.join(doc_labels)})" if doc_labels else ""
    return f"{user_q} — w kontekście poprzedniego pytania: '{base}'{doc_part}"

def route_query(query: str):
    qn = strip_accents_lower(query)
    ref = parse_ref_ext(query)
    if DEBUG:
        print(f"[ROUTER] raw='{query}' | norm='{qn}' | ref={ref}")
    if ref:
        return "EXPLICIT_REF", {"ref": ref}
    return "GENERAL", {"query": query}

#------------------------------------------------ funkcje główne ------------------------------------------------
def retrieve_basic(query: str,
                   k_sim: int = K_SIM,
                   k_final: int = K_FINAL,
                   rerank_threshold: float | None = RERANK_THRESHOLD) -> List[Document]:
    import numpy as _np

    def _doc_pid(md: dict) -> str:
        if md.get("id"): return str(md["id"])
        return f"ch{md.get('rozdzial') or md.get('chapter')}-art{md.get('artykul') or md.get('article')}-ust{md.get('ust') or md.get('paragraph')}"

    def _ce_to_prob(arr: _np.ndarray) -> _np.ndarray:
        arr = _np.asarray(arr, dtype=float)
        # 1D: jeśli już w [0,1] → zostaw; w innym razie potraktuj jako logit i zrób sigmoid
        if arr.ndim == 1:
            if _np.nanmin(arr) >= 0.0 and _np.nanmax(arr) <= 1.0:
                return arr
            return 1.0 / (1.0 + _np.exp(-arr))
        # 2D: klasy (neg,pos) → softmax i bierzemy prawdopodobieństwo ostatniej klasy
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x = arr - arr.max(axis=1, keepdims=True)
            ex = _np.exp(x)
            sm = ex / _np.clip(ex.sum(axis=1, keepdims=True), 1e-9, None)
            return sm[:, -1]
        # awaryjnie: min-max w batchu
        mn, mx = float(_np.nanmin(arr)), float(_np.nanmax(arr))
        if mx - mn < 1e-9:
            return _np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    # --- routing / filtr po referencji (jeśli parser coś znalazł) ---
    ref = parse_ref_ext(query)
    filter_dict: dict | None = None
    if ref:
        f: dict = {}
        if "article" in ref:   f["article"]   = ref["article"]
        if "paragraph" in ref: f["paragraph"] = ref["paragraph"]
        filter_dict = f or None
    if DEBUG:
        print(f"[RET] query={query!r} filter={filter_dict}")

    # --- similarity ---
    docs = db.similarity_search(query, k=k_sim, filter=filter_dict)
    if DEBUG:
        print(f"[RET] similarity → {len(docs)} docs")
        if docs:
            print("[RET] sample:", [_doc_pid((d.metadata or {})) for d in docs[:min(6, len(docs))]])
    if not docs:
        return []

    # --- deduplikacja po PID (często te same ustępy wracają 2×) ---
    uniq: list[Document] = []
    seen: set[str] = set()
    for d in docs:
        pid = _doc_pid(d.metadata or {})
        if pid in seen: 
            continue
        seen.add(pid)
        uniq.append(d)
    docs = uniq
    if DEBUG:
        print(f"[RET] after dedupe → {len(docs)} docs")

    # --- CE scoring ---
    pairs = [(query, d.page_content) for d in docs]
    raw_scores = cross_encoder.predict(pairs, batch_size=32)
    raw_scores = _np.asarray(raw_scores, dtype=float)
    probs = _ce_to_prob(raw_scores)

    if DEBUG:
        try:
            print(f"[RET][CE] raw: min={_np.nanmin(raw_scores):.4f} max={_np.nanmax(raw_scores):.4f} mean={_np.nanmean(raw_scores):.4f}")
        except Exception:
            pass
        print(f"[RET][CE] prob: min={_np.nanmin(probs):.4f} max={_np.nanmax(probs):.4f} mean={_np.nanmean(probs):.4f}")

    # --- próg na PROB + sort ---
    scored_docs = [(d, float(p)) for d, p in zip(docs, probs)]
    if rerank_threshold is not None:
        before = len(scored_docs)
        scored_docs = [(d, s) for d, s in scored_docs if s >= float(rerank_threshold)]
        if DEBUG:
            print(f"[RET][THR] >= {rerank_threshold:.3f} → {len(scored_docs)}/{before} docs")
        if not scored_docs:
            return []

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    scored_docs = scored_docs[:k_final]

    # --- dopnij wynik do metadanych i zwróć ---
    out: List[Document] = []
    for d, s in scored_docs:
        md = dict(d.metadata or {})
        md["rerank_prob"] = float(s)
        md["rerank_score"] = float(s)   # używamy prob jako score
        d.metadata = md
        out.append(d)

    if DEBUG:
        print("[RET] final:", [(_doc_pid(d.metadata or {}), f"{(d.metadata or {}).get('rerank_score'):.3f}") for d in out])
    return out

memory = ConversationBufferWindowMemory(
    k=3, memory_key="chat_history", return_messages=True, output_key="answer"
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "bielik-11b-v2.6-instruct")

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    temperature=0.35,
    max_tokens=512,        # odpowiednik max_new_tokens
)


prompt_base = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "### System:\n"
        "Jesteś ekspertem prawa ubezpieczeń społecznych rolników. "
        "Odpowiadasz WYŁĄCZNIE na podstawie Dokumentów poniżej. "
        "Twoim zadaniem jest odpowiedzenie na pytanie na podstawie Dokumentów. "
        "Jeśli w dokumentach jest BRAK, powiedz, że nie masz wiedzy na ten temat.\n"
        "Formatowanie: bez **…** ani __…__. Limit 6–8 zdań lub 8 punktów.\n"
        "### User:\n{question}\n\n"
        "### Dokumenty:\n{context}\n"
        "### Asystent:\n"
    )
)

prompt_followup = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "### System:\n"
        "Kontynuacja rozmowy. Odpowiadaj WYŁĄCZNIE w oparciu o poniższe dokumenty, "
        "które akumulujemy w toku dopytań użytkownika. "
        "Jeśli dokumenty nie zawierają odpowiedzi, powiedz o tym wprost.\n"
        "Format: bez **…** ani __…__. Zwięźle: 5–7 zdań lub 6 punktów.\n"
        "### User:\n{question}\n\n"
        "### Dokumenty (skumulowane):\n{context}\n"
        "### Asystent:\n"
    )
)

tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

class FunctionRetriever(BaseRetriever):
    k_sim: int
    k_final: int
    rerank_threshold: Optional[float] = None
    _fn: Callable[..., List[Document]] = PrivateAttr()

    def __init__(self, fn: Callable[..., List[Document]], **data):
        super().__init__(**data)
        self._fn = fn

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._fn(query, k_sim=self.k_sim, k_final=self.k_final, rerank_threshold=self.rerank_threshold)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._get_relevant_documents(query))

init_retriever = FunctionRetriever(fn=retrieve_basic, k_sim=K_SIM, k_final=K_FINAL, rerank_threshold=RERANK_THRESHOLD)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=init_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_base},
    return_source_documents=True,
    output_key="answer",
    callback_manager=callback_manager
)

class ConversationState:
    def __init__(self):
        self.last_article_num: Optional[str] = None
        self.last_paragraph_num: Optional[str] = None
        self.last_docs: List[Document] = []
        self.accum_docs: List[Document] = []   
        self.last_query: Optional[str] = None

STATE = ConversationState()
_FOLLOW_UP_NEXT = False

def want_follow_up() -> None:
    global _FOLLOW_UP_NEXT
    _FOLLOW_UP_NEXT = True

def reset_context() -> None:
    STATE.accum_docs.clear()
    STATE.last_docs.clear()
    STATE.last_article_num = None
    STATE.last_paragraph_num = None
    STATE.last_query = None

def _llm_generate(prompt_tmpl: PromptTemplate, **kwargs) -> str:
    text = prompt_tmpl.format(**kwargs)
    out = llm.invoke(text)
    return (out or "").strip()

def answer_from_docs(question: str, docs: List[Document], *, followup: bool):
    ctx = format_docs_for_prompt(docs) if docs else "(brak dokumentów)"
    p = prompt_followup if followup else prompt_base
    answer = _llm_generate(p, question=question, context=ctx)
    answer = trim_incomplete_sentences(strip_markdown_bold(answer)) or answer
    final_text = f"{_build_citations_block(docs)}\nOdpowiedź:\n{answer}"
    return _finalize_return(final_text, docs, mode=("follow_up" if followup else "new_query"))


#main function
def ask(q: str, reset_memory: bool=False):
    """
    Naprawa: nie obcinaj źródeł do jednego dokumentu.
    - New query => STATE.accum_docs = wszystkie docs z retrievera.
    - Follow-up => dobieramy 1 nowy doc (jak było), ale cytujemy całą akumulację.
    """
    if reset_memory:
        try:
            qa_chain.memory.clear()
        except Exception:
            pass
        reset_context()

    st = smalltalk_reply(q)
    if st is not None:
        return _finalize_return(st, [], mode="smalltalk")

    action, payload = route_query(q)
    if action == "EXPLICIT_REF":
        ref = payload["ref"]
        flt = {}
        if "article" in ref:   flt["article"]   = ref["article"]
        if "paragraph" in ref: flt["paragraph"] = ref["paragraph"]
        docs = db.similarity_search("treść przepisu", k=5, filter=flt)
        if DEBUG:
            print("[ROUTER] EXPLICIT_REF → filter", flt, "→ docs:", [ (d.metadata or {}).get("id") for d in docs ])
        if docs:
            _update_state_from_docs(docs, q)
            content = strip_markdown_bold(docs[0].page_content or "")
            content = trim_incomplete_sentences(content) or content
            final_text = f"{_build_citations_block(docs)}\nOdpowiedź (pełny przepis):\n{content}"
            return _finalize_return(final_text, docs, mode="explicit")
        else:
            return _finalize_return("Nie znalazłem takiego artykułu/ustępu.", [], mode="explicit")

    global _FOLLOW_UP_NEXT
    if _FOLLOW_UP_NEXT:
        _FOLLOW_UP_NEXT = False

        # dobieramy 1 nowy dokument (jak wcześniej), ale nie kasujemy poprzednich
        docs_narrow: List[Document] = []
        if STATE.last_article_num:
            flt = {"article": STATE.last_article_num}
            docs_tmp = db.similarity_search(q, k=K_SIM, filter=flt)
            if docs_tmp:
                pairs = [(q, d.page_content) for d in docs_tmp]
                scores = cross_encoder.predict(pairs, batch_size=32)
                scored = sorted([(d, float(s)) for d, s in zip(docs_tmp, np.asarray(scores))],
                                key=lambda x: x[1], reverse=True)[:1]
                docs_narrow = [d for d, _ in scored]

        if not docs_narrow:
            top_global = retrieve_basic(q, k_sim=K_SIM, k_final=1, rerank_threshold=RERANK_THRESHOLD)
            docs_narrow = top_global[:1] if top_global else []

        def _doc_id(d: Document) -> str:
            md = d.metadata or {}
            return md.get("id") or f"ch{md.get('chapter')}-art{md.get('article')}-ust{md.get('paragraph')}"

        seen = {_doc_id(d) for d in STATE.accum_docs}
        for d in docs_narrow:
            if _doc_id(d) not in seen:
                STATE.accum_docs.append(d)

        if STATE.accum_docs:
            _update_state_from_docs([STATE.accum_docs[0]], q)

        # KLUCZOWE: przekazujemy pełną akumulację, więc „Cytowane ustępy” pokaże wszystkie
        return answer_from_docs(q, STATE.accum_docs, followup=True)

    # --- NEW QUERY ---
    res = qa_chain.invoke({"question": q})
    docs = res.get("source_documents", []) or []
    _update_state_from_docs(docs, q)

    # KLUCZOWE: NIE obcinamy do [docs[0]]
    STATE.accum_docs = docs[:] if docs else []

    # Pokaż całą listę w cytowaniach i w kontekście LLM
    return answer_from_docs(q, STATE.accum_docs if STATE.accum_docs else docs, followup=False)
