#!/usr/bin/env python
"""
Interaktywny chat RAG dla ustawy KRUS.
Zakładamy:
  • build_index.py utworzył kolekcję Chroma
  • model_loader.load_llm() zwraca (model, tokenizer)
  • config.py zawiera wszystkie stałe
Uruchom:  python chat_cli.py
"""
import csv, sys, re, numpy as np
from datetime import datetime
from collections import deque

from chromadb import Client
from sentence_transformers import SentenceTransformer
import torch

# ---- nasze moduły ----
import config
from model_loader import load_llm
from embedding.utils import summarize_dialogue   # to masz już w projekcie

# ---- inicjalizacja ----
client        = Client()
collection    = client.get_collection(config.CHROMA_COLLECTION)
embedder      = SentenceTransformer(config.EMBEDDER_NAME)
model, tok    = load_llm()

conversation_buffer: deque[tuple[str, str]] = deque(maxlen=config.MAX_TURNS)
archived_dialogue: list[str] = []
running_summary: str = ""

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
        include=["documents", "embeddings"]
    )
    docs = res["documents"][0]
    embs = np.asarray(res["embeddings"][0])
    sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb))
    return docs, sims.tolist()

# ---- plik logu ----
with open("rag_answers.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Timestamp", "Question", "Context", "Answer"])

print("💬  Wpisz pytanie (lub 'koniec', 'exit'):")
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
    docs, sims = query_rag(query)

    print("\n— Konteksty znalezione (sim):")
    for i, (txt, sim) in enumerate(zip(docs, sims), 1):
        snippet = (txt[:120] + "…") if len(txt) > 120 else txt
        print(f"{i}. sim={sim:.3f} | {snippet}")

    context = "\n".join(docs)

    # ---------- prompt ----------
    summary_block = f"### Streszczenie starszej rozmowy między mną a użytkownikiem:\n{running_summary}\n\n" \
                    if running_summary else ""
    history_block = "\n".join([f"Użytkownik: {u}\nOdpowiedziałem: {a}" for u, a in conversation_buffer])
    if history_block:
        history_block = "### Ostatnie wymiany:\n" + history_block + "\n\n"

    prompt = (
        summary_block
        + history_block
        + "Pytanie:\n"
        + query
        + "\n\nKontekst (wyciąg z ustawy):\n"
        + context
        + "\n\nInstrukcje dla asystenta:\n"
        + "Napisz pełnymi zdaniami, zrozumiałym językiem. "
          "Nie powtarzaj słów „z kontekstu wynika”, „na podstawie kontekstu” ani całych fragmentów ustawy. "
          "Jeżeli odpowiedź wymaga kilku punktów, przedstaw je jako listę wypunktowaną. "
          "Unikaj dygresji i nie cytuj dosłownie kontekstu, chyba że to konieczne.\n"
        + "Odpowiedź:"
    )

    # ---------- LLM ----------
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

    # ---------- aktualizacja historii + log ----------
    conversation_buffer.append((query, answer))
    with open("rag_answers.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), query, context, answer])
