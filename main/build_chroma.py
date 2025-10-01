from __future__ import annotations
import os
import re
import csv
import shutil
from typing import List, Dict, Optional

import numpy as np

from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from openai import OpenAI
from langchain_core.embeddings import Embeddings

from dotenv import load_dotenv
load_dotenv()


USE_LMSTUDIO: bool = os.getenv("USE_LMSTUDIO", "1") == "1"
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

EMBEDDER_MODEL_T: Optional[str] = os.getenv("EMBEDDER_MODEL_T")
EMBEDDER_MODEL_U: Optional[str] = os.getenv("EMBEDDER_MODEL_U")

PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./chroma_statystyki")
PERSIST_PATH: str = os.getenv("PERSIST_PATH", "./chroma_ustawa")

DATA_PATH: str = os.getenv("DATA_PATH", "./data/ustawa_with_paragraph_headers.md")
CSV_DIR: str = os.getenv("CSV_DIR", "./data/tables/all_data.csv")

CHROMA_WIPE_ON_START: bool = os.getenv("CHROMA_WIPE_ON_START", "false").lower() in ("1", "true", "yes")

_client_embed = OpenAI(base_url=LLM_BASE_URL, api_key=LMSTUDIO_API_KEY)

class LMStudioEmbeddings(Embeddings):
    def __init__(self, model: str, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = _client_embed.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i+self.batch_size]))
        return out
    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

def build_embeddings(model_name: Optional[str]) -> Embeddings:
    if not model_name:
        raise ValueError("Brak nazwy modelu embeddings w ENV.")
    if USE_LMSTUDIO:
        try:
            _client_embed.embeddings.create(model=model_name, input=["ping"])
            return LMStudioEmbeddings(model_name)
        except Exception as e:
            print(f"[EMB] LM Studio niedostępne dla '{model_name}' → przerwij lub zmień ENV. Szczegóły: {e}")
            raise
    from sentence_transformers import SentenceTransformer
    class ST_Emb(Embeddings):
        def __init__(self, name: str):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(name, device=device)
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
            return v.astype(np.float32).tolist()
        def embed_query(self, text: str) -> List[float]:
            v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
            return v[0].astype(np.float32).tolist()
    return ST_Emb(model_name)

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
    parts = [f"{k}: {v}" for k, v in row.items()]
    return "\n".join(parts)

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

def build_statystyki_from_csv(emb_model: str, persist_dir: str, csv_path: str, wipe: bool) -> int:
    ensure_clean_dir(persist_dir, wipe)
    emb = build_embeddings(emb_model)
    if not os.path.isfile(csv_path):
        print(f"[STATYSTYKI] Brak CSV: {csv_path} — pomijam budowę.")
        Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=persist_dir)
        return 0
    docs: List[Document] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = csv_row_to_text(row)
            md = {"row_index": i, "source": os.path.basename(csv_path)}
            docs.append(Document(page_content=content, metadata=md))
    if not docs:
        print("[STATYSTYKI] CSV puste — kolekcja pozostanie pusta.")
        Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=persist_dir)
        return 0
    Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=persist_dir,
        collection_name="statystyki",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[STATYSTYKI] Zaindeksowano wierszy: {len(docs)}")
    return len(docs)

def main() -> None:
    print("[BUILD] Start budowania kolekcji Chroma…")
    # USTAWA
    try:
        build_ustawa(EMBEDDER_MODEL_U, PERSIST_PATH, DATA_PATH, CHROMA_WIPE_ON_START)
    except Exception as e:
        print(f"[BUILD][USTAWA] Błąd: {e}")
    # STATYSTYKI
    try:
        build_statystyki_from_csv(EMBEDDER_MODEL_T, PERSIST_DIR, CSV_DIR, CHROMA_WIPE_ON_START)
    except Exception as e:
        print(f"[BUILD][STATYSTYKI] Błąd: {e}")
    print("[BUILD] Zakończono.")

if __name__ == "__main__":
    main()
