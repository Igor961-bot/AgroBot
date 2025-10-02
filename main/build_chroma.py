# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import csv
import shutil
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Chroma
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

# Dokumenty
from langchain_core.documents import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from resources import build_embeddings 

EMBEDDER_MODEL_T: Optional[str] = os.getenv("EMBEDDER_MODEL_T")
EMBEDDER_MODEL_U: Optional[str] = os.getenv("EMBEDDER_MODEL_U")

PERSIST_DIR:  str = os.getenv("PERSIST_DIR",  "./chroma_statystyki")
PERSIST_PATH: str = os.getenv("PERSIST_PATH", "./chroma_ustawa")

DATA_PATH: str = os.getenv("DATA_PATH", "./data/ustawa_with_paragraph_headers.md")
CSV_DIR:  str = os.getenv("CSV_DIR",  "./data/tables/all_data.csv")

CHROMA_WIPE_ON_START: bool = os.getenv("CHROMA_WIPE_ON_START", "false").lower() in ("1", "true", "yes")

def ensure_clean_dir(path: str, wipe: bool) -> None:
    if wipe:
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

_META_RE = re.compile(
    r"<!--\s*chapter\s*:\s*(\d+)\s+article\s*:\s*([0-9a-z]+)\s+paragraph\s*:\s*([0-9a-z]+)\s+id\s*:\s*([^\s>]+)\s*-->",
    re.I
)

def parse_md_ustawa(md_path: str) -> List[Document]:
    docs_raw = TextLoader(md_path, encoding="utf-8").load()
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "ustep")])
    chunks = splitter.split_text(docs_raw[0].page_content)

    docs: List[Document] = []
    for d in chunks:
        md = dict(d.metadata)
        header_text = md.get("ustep", "") or ""
        source_for_meta = header_text + "\n" + d.page_content
        m = _META_RE.search(source_for_meta)
        if m:
            md["chapter"]   = int(m.group(1))
            md["article"]   = m.group(2).lower()
            md["paragraph"] = m.group(3).lower()
            md["id"]        = m.group(4)
            md["rozdzial"]  = md["chapter"]
            md["artykul"]   = md["article"]
            md["ust"]       = md["paragraph"]
        clean_header = _META_RE.sub("", header_text).strip()
        if clean_header:
            md["ustep"] = clean_header
        content = _META_RE.sub("", d.page_content).strip()
        docs.append(Document(page_content=content, metadata=md))
    return docs

def csv_row_to_text(row: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items())

def build_ustawa(emb_model: str, persist_path: str, data_md: str, wipe: bool) -> int:
    ensure_clean_dir(persist_path, wipe)
    emb = build_embeddings(emb_model)   
    if not os.path.isfile(data_md):
        print(f"[USTAWA] Brak pliku: {data_md}")
        return 0
    docs = parse_md_ustawa(data_md)
    if not docs:
        print("[USTAWA] Parser nie zwrócił dokumentów.")
        return 0
    Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=persist_path,
        collection_name="ustawa",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[USTAWA] Zaindeksowano: {len(docs)}")
    return len(docs)

from data_schema import row_to_document  # NEW

def build_statystyki_from_csv(emb_model: str, persist_dir: str, csv_path_or_dir: str, wipe: bool) -> int:
    ensure_clean_dir(persist_dir, wipe)
    emb = build_embeddings(emb_model)

    # zbierz pliki CSV
    csv_files = []
    if os.path.isdir(csv_path_or_dir):
        for fn in os.listdir(csv_path_or_dir):
            if fn.lower().endswith(".csv"):
                csv_files.append(os.path.join(csv_path_or_dir, fn))
    elif os.path.isfile(csv_path_or_dir) and csv_path_or_dir.lower().endswith(".csv"):
        csv_files.append(csv_path_or_dir)

    if not csv_files:
        print(f"[STATYSTYKI] Brak CSV w: {csv_path_or_dir} — tworzę pustą kolekcję.")
        Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=persist_dir)
        return 0

    docs: List[Document] = []
    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"dataset","measure","value","region","period","typ"}
            if not required.issubset(set(reader.fieldnames or [])):
                print(f"[UWAGA] Pomijam '{csv_path}' — brakuje kolumn: {required - set(reader.fieldnames or [])}")
                continue
            for i, row in enumerate(reader):
                docs.append(row_to_document(row, source_file=csv_path, row_index=i))

    if not docs:
        print("[STATYSTYKI] Nie zbudowano żadnego dokumentu.")
        Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=persist_dir)
        return 0

    Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=persist_dir,
        collection_name="statystyki",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[STATYSTYKI] Zaindeksowano wierszy: {len(docs)} (pełne metadata)")
    return len(docs)


# ---------- MAIN ----------
def main() -> None:
    print("[BUILD] Start budowania kolekcji Chroma…")
    try:
        build_ustawa(EMBEDDER_MODEL_U, PERSIST_PATH, DATA_PATH, CHROMA_WIPE_ON_START)
    except Exception as e:
        print(f"[BUILD][USTAWA] Błąd: {e}")
    try:
        build_statystyki_from_csv(EMBEDDER_MODEL_T, PERSIST_DIR, CSV_DIR, CHROMA_WIPE_ON_START)
    except Exception as e:
        print(f"[BUILD][STATYSTYKI] Błąd: {e}")
    print("[BUILD] Zakończono.")

if __name__ == "__main__":
    main()
