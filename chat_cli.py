#!/usr/bin/env python
"""
  • build_index.py utworzył kolekcję ChromaDB (embedding na podsekcjach)
  • model_loader.load_llm() zwraca (model, tokenizer)
  • config.py zawiera wszystkie stałe
  • embedding/utils.py zawiera summarize_dialogue()
"""
import csv, sys, re, numpy as np, json
from datetime import datetime
from collections import deque

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import CrossEncoder

import config
from model_loader import load_llm

# ----- Inicjalizacja -----
client     = PersistentClient(path="chroma_store")
collection = client.get_collection(config.CHROMA_COLLECTION)
embedder   = SentenceTransformer(config.EMBEDDER_NAME)
model, tok = load_llm()
cross_encoder = CrossEncoder(config.ENCODER_NAME,
                             max_length=512,  
                             device='cuda',  
                             tokenizer_kwargs={'truncation': True})

conversation_buffer: deque[tuple[str, str]] = deque(maxlen=config.MAX_TURNS)
archived_dialogue: list[str] = []
running_summary: str = ""

# ----- Wczytaj ustawę do RAM -----
with open(config.JSON_ACT_PATH, encoding="utf-8") as f:
    ACT_DATA = json.load(f)

def get_full_article(chapter_num, article_num):
    """Zwraca cały artykuł (scalony tekst wszystkich podsekcji) na podstawie numerów."""
    for ch in ACT_DATA["document"]["chapters"]:
        if str(ch["number"]) == str(chapter_num):
            for art in ch["articles"]:
                if str(art["article"]).replace(".", "") == str(article_num):
                    all_subs = [sub["content"] for sub in art["subsections"] if sub.get("content")]
                    return f"Artykuł {article_num}, Rozdział {chapter_num}:\n" + "\n".join(all_subs)
    return None


def shorten_article_for_ce(article, tokenizer, max_tokens=510): 
    tokens = tokenizer(article)["input_ids"]
    if len(tokens) <= max_tokens:
        return article
    first = tokenizer.decode(tokens[:200])
    mid   = tokenizer.decode(tokens[len(tokens)//2 - 55 : len(tokens)//2 + 55])
    last  = tokenizer.decode(tokens[-200:])
    return f"{first}\n... (fragment) ...\n{mid}\n... (fragment) ...\n{last}"





def summarize_dialogue(text: str, model, tokenizer) -> str:
    if not text:
        return ""
    prompt = (
        "Streszcz w maksymalnie trzech zdaniach następującą rozmowę:\n" + text + "\nStreszczenie:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs,
                             max_new_tokens=64,
                             repetition_penalty=1.1,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

    summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if "Streszczenie:" in summary:
        summary = summary.split("Streszczenie:")[-1].strip()
    return summary

def infer_verbosity(q: str) -> str:
    return "long" if any(t in q.lower() for t in config.LONG_TRIGGERS) else "short"

def truncate_sentences(text: str, limit: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:limit]).strip()

def query_rag(question: str):
    q_emb = embedder.encode([question])[0]
    res   = collection.query(
        query_embeddings=[q_emb],
        n_results=config.TOP_K,
        include=["documents", "embeddings", "metadatas"]
    )
    docs = res["documents"][0]
    embs = np.asarray(res["embeddings"][0])
    metas = res["metadatas"][0]
    sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb))
    return docs, sims.tolist(), metas

with open("rag_answers.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Timestamp", "Question", "Context", "Answer"])

print("Wpisz pytanie (lub 'koniec', 'exit'):")
for line in sys.stdin:
    query = line.strip()
    if query.lower() in {"koniec", "exit", "quit"}:
        break
    if not query:
        continue

    # ---------- zarządzanie buforem ----------
    if len(conversation_buffer) == config.MAX_TURNS:
        old_u, old_a = conversation_buffer.popleft()
        archived_dialogue.append(f"U: {old_u}\nA: {old_a}")
        if len(archived_dialogue) >= config.SUMMARY_TRIGGER:
            running_summary += " " + summarize_dialogue("\n".join(archived_dialogue), model, tok)
            archived_dialogue.clear()

    # ---------- retrieval ----------
    docs, sims, metas = query_rag(query)

    print("\n— Konteksty znalezione (sim):")
    for i, (txt, sim, meta) in enumerate(zip(docs, sims, metas), 1):
        snippet = (txt[:120] + "…") if len(txt) > 120 else txt
        print(f"{i}. sim={sim:.3f} | {snippet}")

    # --- Cross-encoding na pełnych artykułach ---
    article_keys = set()
    article_texts = []
    for txt, meta in zip(docs, metas):
        ch_num = meta.get("chapter")
        art_num = meta.get("article")
        key = (str(ch_num), str(art_num))
        if key not in article_keys:
            full_art = get_full_article(ch_num, art_num)
            if full_art:
                article_keys.add(key)
                article_texts.append(full_art)

    short_articles = [shorten_article_for_ce(art, tok, 1024) for art in article_texts]
    ce_pairs = [(query, art) for art in short_articles]
    ce_scores = cross_encoder.predict(ce_pairs)
    topN = 3
    best_idx = np.argsort(ce_scores)[::-1][:topN]
    context_articles = [article_texts[i] for i in best_idx]
    context = "\n\n---\n\n".join(context_articles)


    # ---------- prompt ----------
    summary_block = f"Streszczenie starszej rozmowy między tobą a użytkownikiem:\n{running_summary}\n\n" \
                    if running_summary else ""
    history_block = "\n".join([f"Użytkownik: {u}\nOdpowiedziałeś {a}" for u, a in conversation_buffer])
    if history_block:
        history_block = "Ostatnie wymiany:\n" + history_block + "\n\n"

    prompt = (
        summary_block
        + history_block
        + "Pytanie:\n"
        + query
        + "\n\nKontekst (pełne artykuły powiązane z pytaniem):\n"
        + context
        + "\n\nInstrukcje dla asystenta:\n"
        + "Jesteś asystentem, który ma odpowiadać z zakresu wiedzy o ubezpieczeniach rolniczych, bądź pozytywny"
          "Napisz pełnymi zdaniami, zrozumiałym językiem. "
          "Nie powtarzaj słów „z kontekstu wynika”, „na podstawie kontekstu” ani całych fragmentów ustawy."
          "Odpowiadaj w punktach"
          "Jeżeli z kontekstu nie możesz odpowiedzieć na pytanie, odpowiedz 'nie mam na ten temat wiedzy, spróbuj doprecyzować pytanie.'"
          "Unikaj dygresji i nie cytuj dosłownie kontekstu, chyba że to konieczne.\n"
        + "Odpowiedź:"
    )

    verbosity = infer_verbosity(query)
    max_sent  = config.MAX_SENT_LONG if verbosity == "long" else config.MAX_SENT_SHORT
    inputs    = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=256 if verbosity == "short" else 512,
            repetition_penalty=1.25,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    raw = tok.decode(out[0], skip_special_tokens=True)
    answer = truncate_sentences(raw.split("Odpowiedź:")[-1].strip(), max_sent)
    print("\nOdpowiedź:\n" + answer + "\n")
    print(context)

    conversation_buffer.append((query, answer))
    with open("rag_answers.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), query, context, answer])
