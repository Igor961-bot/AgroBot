#!/usr/bin/env python
"""
chat_cli_lc.py  –  RAG + LangChain Multi-Query Retriever
--------------------------------------------------------
• build_index.py utworzył kolekcję ChromaDB
• model_loader.load_llm() zwraca (model, tokenizer) – PLLuM-12B-nc-chat
• config.py przechowuje stałe (ścieżki, TOP_K, LONG_TRIGGERS…)
"""

import csv, sys, re, json, numpy as np
from datetime import datetime
from collections import deque

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient

# ---- LangChain ≥ 0.2 ----
from langchain_community.vectorstores import Chroma            # ← OK (warning tylko inform.)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import config
from model_loader import load_llm   

import warnings
from transformers import logging as hf_logging

# 1) ucisz “temperature” + inne INFO/WARNING Transformers
hf_logging.set_verbosity_error()

# 2) ucisz pojedynczy FutureWarning (jeśli Cię razi)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`encoder_attention_mask` is deprecated"
)

# 3) po załadowaniu PLLuM-12B
model, tok = load_llm()
model.generation_config.temperature = None     # usuwa flagę

# 4) zaktualizowane importy LangChain (po pip install -U …)
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma



# --- tuż po importach LangChain -------------------------
from langchain_core.embeddings import Embeddings

class STEmbeddings(Embeddings):
    """Minimalny wrapper aby Chroma dostała embed_query / embed_documents."""
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32).tolist()


# używamy tego samego modelu do parafraz! (krótka generacja ~60 tokenów)
hf_pipe = pipeline("text-generation",
                   model=model,
                   tokenizer=tok,
                   max_length=60,
                   do_sample=False)
para_llm = HuggingFacePipeline(pipeline=hf_pipe)               # jedyny parametr!

# ============ WEKTORY ============ #
client   = PersistentClient(path="chroma_store")
embedder_raw = SentenceTransformer(config.EMBEDDER_NAME, device='cuda')
embedder     = STEmbeddings(embedder_raw)          # <- adapter
vectordb = Chroma(
        client          = client,
        collection_name = config.CHROMA_COLLECTION,
        embedding_function = embedder)             # już OK

from langchain_core.prompts import PromptTemplate

para_prompt = PromptTemplate.from_template(
    "Podaj cztery różne parafrazy pytania – każdą w nowej linii:\n{question}"
)

multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": config.TOP_K}),
        llm=para_llm,
        prompt=para_prompt          # <-- zamiast prompt_template / string
)

cross_encoder = CrossEncoder(config.ENCODER_NAME,
                             max_length=512,
                             device='cuda',
                             tokenizer_kwargs={'truncation': True})

# ============ BUFORY ============ #
conversation_buffer: deque[tuple[str, str]] = deque(maxlen=config.MAX_TURNS)
archived_dialogue: list[str] = []
running_summary: str = ""

# ============ Wczytanie ustawy ============ #
with open(config.JSON_ACT_PATH, encoding="utf-8") as f:
    ACT_DATA = json.load(f)

def get_full_article(ch, art):
    for chapter in ACT_DATA["document"]["chapters"]:
        if str(chapter["number"]) == str(ch):
            for a in chapter["articles"]:
                if str(a["article"]).replace('.', '') == str(art):
                    body = "\n".join(s["content"] for s in a["subsections"] if s.get("content"))
                    return f"Artykuł {art}, Rozdział {ch}:\n{body}"
    return None

def shorten_article_for_ce(txt, tokenizer, max_tokens=510):
    ids = tokenizer(txt)["input_ids"]
    if len(ids) <= max_tokens:
        return txt
    first = tokenizer.decode(ids[:200])
    mid   = tokenizer.decode(ids[len(ids)//2-55: len(ids)//2+55])
    last  = tokenizer.decode(ids[-200:])
    return f"{first}\n…(fragment)…\n{mid}\n…(fragment)…\n{last}"

def summarize_dialogue(text):
    if not text:
        return ""
    prompt = f"Streszcz w maksymalnie trzech zdaniach następującą rozmowę:\n{text}\nStreszczenie:"
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    inp.pop("token_type_ids", None)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=64, repetition_penalty=1.1,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).split("Streszczenie:")[-1].strip()

def infer_verbosity(q): return "long" if any(t in q.lower() for t in config.LONG_TRIGGERS) else "short"
def truncate_sentences(txt, lim): return " ".join(re.split(r"(?<=[.!?])\s+", txt.strip())[:lim])

# ============ LOG CSV ============ #
with open("rag_answers.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Timestamp", "Question", "Context", "Answer"])

print("Wpisz pytanie (lub 'koniec', 'exit'):")
for line in sys.stdin:
    question = line.strip()
    if question.lower() in {"koniec", "exit", "quit"}:
        break
    if not question:
        continue

    # --- bufor dialogu ---
    if len(conversation_buffer) == config.MAX_TURNS:
        u_old, a_old = conversation_buffer.popleft()
        archived_dialogue.append(f"U: {u_old}\nA: {a_old}")
        if len(archived_dialogue) >= config.SUMMARY_TRIGGER:
            running_summary += " " + summarize_dialogue("\n".join(archived_dialogue))
            archived_dialogue.clear()

    # --- Multi-Query retrieval ---
    docs = multi_retriever.get_relevant_documents(question)

    seen, arts_full = set(), []
    for d in docs:
        meta = d.metadata
        key  = (str(meta.get("chapter")), str(meta.get("article")))
        if key not in seen:
            seen.add(key)
            art_full = get_full_article(*key)
            if art_full:
                arts_full.append(art_full)

    if not arts_full:
        print("⛔ Nie znaleziono adekwatnego artykułu.\n")
        continue

    # --- cross-encoder re-rank ---
    ce_pairs  = [(question, shorten_article_for_ce(a, tok)) for a in arts_full]
    scores    = cross_encoder.predict(ce_pairs)
    top_idx   = np.argsort(scores)[::-1][:3]
    context_articles = [arts_full[i] for i in top_idx]
    context   = "\n\n---\n\n".join(context_articles)

    # --- prompt ---
    summary_block = f"Streszczenie starszej rozmowy:\n{running_summary}\n\n" if running_summary else ""
    hist_block = "\n".join(f"Użytkownik: {u}\nOdpowiedziałeś: {a}" for u, a in conversation_buffer)
    if hist_block:
        hist_block = "Ostatnie wymiany:\n" + hist_block + "\n\n"

    prompt = (
        summary_block + hist_block +
        "Pytanie:\n" + question +
        "\n\nKontekst (pełne artykuły powiązane z pytaniem):\n" + context +
        "\n\nInstrukcje dla asystenta:\n"
        "Jesteś asystentem, który ma odpowiadać z zakresu wiedzy o ubezpieczeniach rolniczych, bądź pozytywny "
        "Napisz pełnymi zdaniami, zrozumiałym językiem. "
        "Nie powtarzaj słów „z kontekstu wynika”, „na podstawie kontekstu” ani całych fragmentów ustawy. "
        "Odpowiadaj w punktach. "
        "Jeżeli z kontekstu nie możesz odpowiedzieć na pytanie, odpowiedz: 'nie mam na ten temat wiedzy, spróbuj doprecyzować pytanie.' "
        "Unikaj dygresji i nie cytuj dosłownie kontekstu, chyba że to konieczne.\n"
        "Odpowiedź:"
    )

    verbosity = infer_verbosity(question)
    max_sent  = config.MAX_SENT_LONG if verbosity == "long" else config.MAX_SENT_SHORT

    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=4096).to(model.device)
    inp.pop("token_type_ids", None)

    with torch.inference_mode():
        out = model.generate(**inp,
                             max_new_tokens=256 if verbosity == "short" else 512,
                             repetition_penalty=1.25,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)

    answer = truncate_sentences(tok.decode(out[0], skip_special_tokens=True)
                                .split("Odpowiedź:")[-1].strip(),
                                max_sent)

    print("\nOdpowiedź:\n" + answer + "\n")
    print("\n--- Kontekst użyty ---\n" + context[:800] + ("\n…" if len(context) > 800 else ""))

    conversation_buffer.append((question, answer))
    with open("rag_answers.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), question, context, answer])
