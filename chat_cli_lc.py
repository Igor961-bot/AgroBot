#!/usr/bin/env python
"""
chat_cli_lc.py  –  RAG + LangChain Multi‑Query Retriever (DEBUG VERSION)
-----------------------------------------------------------------------
Wyświetla pełny prompt, parafrazy MultiQuery oraz finalny kontekst, aby
ułatwić debugowanie.
Pozostała funkcjonalność (bufory, log CSV, cross‑encoder, parafrazy, itd.)
nie została usunięta.
"""

import csv, sys, re, json, numpy as np
from datetime import datetime
from collections import deque

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient

# ---- LangChain ≥ 0.2 ----
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import config
from model_loader import load_llm

import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated"
)

# ---------- LOAD LLM ---------- #
model, tok = load_llm()
model.generation_config.temperature = 0.0  # deterministyczne

from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

class STEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32).tolist()

# ---------- PARAPHRASE LLM ---------- #
hf_pipe = pipeline("text-generation", model=model, tokenizer=tok, max_length=128, do_sample=False)
para_llm = HuggingFacePipeline(pipeline=hf_pipe)

# ---------- VECTORDB ---------- #
client = PersistentClient(path="chroma_store")
embedder_raw = SentenceTransformer(config.EMBEDDER_NAME, device="cuda")
embedder = STEmbeddings(embedder_raw)

vectordb = Chroma(client=client, collection_name=config.CHROMA_COLLECTION, embedding_function=embedder)

from langchain_core.prompts import PromptTemplate
para_prompt = PromptTemplate.from_template("Podaj trzy różne parafrazy pytania – każdą w nowej linii:\n{question}")

multi_retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(search_kwargs={"k": config.TOP_K}), llm=para_llm, prompt=para_prompt)

# ---------- CROSS‑ENCODER ---------- #
cross_encoder = CrossEncoder(config.ENCODER_NAME, max_length=512, device="cuda", tokenizer_kwargs={"truncation": True})

# ---------- BUFFERS ---------- #
conversation_buffer: deque[tuple[str, str]] = deque(maxlen=config.MAX_TURNS)
archived_dialogue: list[str] = []
running_summary: str = ""

# ---------- LOAD ACT ---------- #
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

def summarize_dialogue(text):
    if not text:
        return ""
    prompt = f"Streszcz w maksymalnie trzech zdaniach następującą rozmowę:\n{text}\nStreszczenie:"
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    inp.pop("token_type_ids", None)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).split("Streszczenie:")[-1].strip()

def infer_verbosity(q):
    return "long" if any(t in q.lower() for t in config.LONG_TRIGGERS) else "short"

def truncate_sentences(txt, lim):
    return " ".join(re.split(r"(?<=[.!?])\s+", txt.strip())[:lim])

# ---------- LOG CSV ---------- #
with open("rag_answers.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Timestamp", "Question", "Context", "Answer"])

print("Wpisz pytanie (lub 'koniec', 'exit'):")
for line in sys.stdin:
    question = line.strip()
    if question.lower() in {"koniec", "exit", "quit"}:
        break
    if not question:
        continue

    # ---------- buffers ---------- #
    if len(conversation_buffer) == config.MAX_TURNS:
        u_old, a_old = conversation_buffer.popleft()
        archived_dialogue.append(f"U: {u_old}\nA: {a_old}")
        if len(archived_dialogue) >= config.SUMMARY_TRIGGER:
            running_summary += " " + summarize_dialogue("\n".join(archived_dialogue))
            archived_dialogue.clear()

    # ---------- PARAPHRASES (for debug) ---------- #
    para_prompt_text = para_prompt.format(question=question)
    try:
        paras_raw = para_llm(para_prompt_text)
        paras_text = paras_raw[0]["generated_text"] if isinstance(paras_raw, list) and isinstance(paras_raw[0], dict) and "generated_text" in paras_raw[0] else str(paras_raw)
    except Exception:
        paras_text = "(błąd generowania parafraz)"
    paraphrases = [p.strip() for p in paras_text.split("\n") if p.strip()]

    # ---------- Retrieval ---------- #
    docs = multi_retriever.get_relevant_documents(question)

    seen, arts_full = set(), []
    for d in docs:
        meta = d.metadata
        key = (str(meta.get("chapter")), str(meta.get("article")))
        if key not in seen:
            seen.add(key)
            art_full = get_full_article(*key)
            if art_full:
                arts_full.append(art_full)

    if not arts_full:
        print("⛔ Nie znaleziono adekwatnego artykułu.\n")
        continue

    # ---------- Rerank ---------- #
    ce_pairs = [(question, shorten_article_for_ce(a, tok)) for a in arts_full]
    scores   = cross_encoder.predict(ce_pairs)
    top_idx  = np.argsort(scores)[::-1][:3]
    context_articles = [arts_full[i] for i in top_idx]
    context  = "\n\n---\n\n".join(context_articles)

    # ---------- Prompt build ---------- #
    sys_block = (
        "### System:\n"
        "Jesteś przyjaznym ekspertem KRUS. Odpowiadasz zwięźle, pełnymi zdaniami, w punktach; "
        "nie cytujesz ustawy dosłownie, o ile nie jest to konieczne. Jeśli nie wiesz – napisz ‘nie mam na ten temat wiedzy’.\n"
    )

    summary_block = f"### Streszczenie rozmowy:\n{running_summary}\n\n" if running_summary else ""

    hist_block = ""
    if conversation_buffer:
        joined = "\n".join(f"U: {u}\nA: {a}" for u, a in conversation_buffer)
        hist_block = f"### Ostatnie wymiany:\n{joined}\n\n"

    prompt = (
        sys_block
        + summary_block
        + hist_block
        + f"### User:\n{question}\n\n"
        + f"### Dokumenty:\n{context}\n\n"
        + "### Assistant:\n"
    )

    # ---------- DEBUG PRINTS ---------- #
    print("\n==================== DEBUG INFO ====================")
    print("\nFULL PROMPT:\n" + prompt)
    print("\nPARAPHRASES (MultiQuery):")
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")
    print("\nKONTEKSTY (Top 3 po rerank):")
    for i, art in enumerate(context_articles, 1):
        snippet = art[:800] + ("…" if len(art) > 800 else "")
        print(f"\n--- [{i}] ------------------------------\n{snippet}\n")
    print("===================================================\n")

    # ---------- GENERATE ---------- #
    verbosity = infer_verbosity(question)
    max_sent  = config.MAX_SENT_LONG if verbosity == "long" else config.MAX_SENT_SHORT

    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    inp.pop("token_type_ids", None)
    prompt_len = inp["input_ids"].shape[-1]

    with torch.inference_mode():
        gen_ids = model.generate(**inp, max_new_tokens=400 if verbosity == "long" else 256, repetition_penalty=1.15, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)[0]

    raw_answer = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
    answer = truncate_sentences(raw_answer, max_sent)

    # ---------- OUTPUT ---------- #
    print("Odpowiedź:\n" + answer + "\n")

    conversation_buffer.append((question, answer))
    with open("rag_answers.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), question, context, answer])
