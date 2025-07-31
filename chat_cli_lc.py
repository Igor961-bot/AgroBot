#!/usr/bin/env python
"""
chat_cli_lc.py  –  ta sama logika co poprzednio,
ale retrieval = LangChain Multi-Query (4 parafrazy).
"""
import csv, sys, re, json, numpy as np
from datetime import datetime
from collections import deque

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient

# -------- LangChain --------
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import config
from model_loader import load_llm                         # główny LLM (PLLuM-12B nc-chat)

# ============== Inicjalizacja bazy wektorowej ==========
client          = PersistentClient(path="chroma_store")
collection_raw  = client.get_collection(config.CHROMA_COLLECTION)

# owijamy kolekcję w LangChain-owy VectorStore
embedder        = SentenceTransformer(config.EMBEDDER_NAME, device='cuda')
vectordb        = Chroma(client=client,
                         collection_name=config.CHROMA_COLLECTION,
                         embedding_function=HuggingFaceEmbeddings(model=embedder))

# ============== Mały LLM do parafraz (tani i szybki) =================
# używamy distilroberta-base-polish-paraphrase (przykład) – zmiana na dowolny lekki model PL
PARA_MODEL = "KBLab/robbert-e2e-paraphrase"
para_tok   = AutoTokenizer.from_pretrained(PARA_MODEL)
para_mod   = AutoModelForSeq2SeqLM.from_pretrained(PARA_MODEL).half().to("cuda")
para_llm   = HuggingFacePipeline(
                pipeline=pipeline(
                    "text2text-generation",
                    model=para_mod,
                    tokenizer=para_tok,
                    device=0,
                    max_length=64,
                    do_sample=False))

# LangChain Multi-Query Retriever (4 warianty)
multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": config.TOP_K}),
        llm=para_llm,
        prompt_template="Podaj cztery różne parafrazy tego pytania w jednym wierszu każdą:\n{question}")

# ============== Cross-encoder i główny LLM ===========================
cross_encoder = CrossEncoder(config.ENCODER_NAME, max_length=512,
                             device='cuda', tokenizer_kwargs={'truncation': True})

model, tok = load_llm()                                      # PLLuM-12B nc-chat

# ============== Bufory konwersacji ==========================
conversation_buffer: deque[tuple[str, str]] = deque(maxlen=config.MAX_TURNS)
archived_dialogue: list[str] = []
running_summary: str = ""

# ------------- Wczytaj ustawę do RAM ------------------------
with open(config.JSON_ACT_PATH, encoding="utf-8") as f:
    ACT_DATA = json.load(f)

def get_full_article(ch, art):
    for chapter in ACT_DATA["document"]["chapters"]:
        if str(chapter["number"]) == str(ch):
            for a in chapter["articles"]:
                if str(a["article"]).replace(".", "") == str(art):
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

# ---------- util z poprzedniego kodu ------------------------
def summarize_dialogue(text, model, tokenizer):
    if not text:
        return ""
    prompt = f"Streszcz w maksymalnie trzech zdaniach następującą rozmowę:\n{text}\nStreszczenie:"
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    inp.pop("token_type_ids", None)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=64, repetition_penalty=1.1,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    summ = tokenizer.decode(out[0], skip_special_tokens=True)
    return summ.split("Streszczenie:")[-1].strip()

def infer_verbosity(q): return "long" if any(t in q.lower() for t in config.LONG_TRIGGERS) else "short"
def truncate_sentences(txt, lim): return " ".join(re.split(r"(?<=[.!?])\s+", txt.strip())[:lim])

# --------------------- LOG CSV -----------------------------
with open("rag_answers.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Timestamp", "Question", "Context", "Answer"])

print("Wpisz pytanie (lub 'koniec', 'exit'):")
for line in sys.stdin:
    question = line.strip()
    if question.lower() in {"koniec", "exit", "quit"}:
        break
    if not question:
        continue

    # ---- bufor konwersacji ----
    if len(conversation_buffer) == config.MAX_TURNS:
        u_old, a_old = conversation_buffer.popleft()
        archived_dialogue.append(f"U: {u_old}\nA: {a_old}")
        if len(archived_dialogue) >= config.SUMMARY_TRIGGER:
            running_summary += " " + summarize_dialogue("\n".join(archived_dialogue), model, tok)
            archived_dialogue.clear()

    # ---------------- retrieval (LangChain Multi-Query) ----------------
    results = multi_retriever.get_relevant_documents(question)   # list[Document]
    # deduplikacja po (chapter, article)
    seen, arts_full = set(), []
    for d in results:
        meta = d.metadata
        key  = (str(meta.get("chapter")), str(meta.get("article")))
        if key not in seen:
            seen.add(key)
            full = get_full_article(*key)
            if full:
                arts_full.append(full)

    if not arts_full:
        print("Nie znaleziono adekwatnego artykułu.")
        continue

    # ---- cross-encoder re-rank na pełnych artykułach ----
    ce_pairs   = [(question, shorten_article_for_ce(a, tok)) for a in arts_full]
    ce_scores  = cross_encoder.predict(ce_pairs)
    top_idx    = np.argsort(ce_scores)[::-1][:3]
    context_articles = [arts_full[i] for i in top_idx]
    context    = "\n\n---\n\n".join(context_articles)

    # ---------------- prompt ----------------
    summary_block = f"Streszczenie starszej rozmowy:\n{running_summary}\n\n" if running_summary else ""
    hist_block    = "\n".join(f"Użytkownik: {u}\nOdpowiedziałeś: {a}" for u, a in conversation_buffer)
    if hist_block: hist_block = "Ostatnie wymiany:\n" + hist_block + "\n\n"

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

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        out = model.generate(**inputs,
                             max_new_tokens=256 if verbosity == "short" else 512,
                             repetition_penalty=1.25,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
    raw    = tok.decode(out[0], skip_special_tokens=True)
    answer = truncate_sentences(raw.split("Odpowiedź:")[-1].strip(), max_sent)

    print("\nOdpowiedź:\n" + answer + "\n")
    print("\n--- Kontekst użyty ---\n" + context[:800] + ("\n…" if len(context) > 800 else ""))

    # --- zapisz historię & log ---
    conversation_buffer.append((question, answer))
    with open("rag_answers.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), question, context, answer])
