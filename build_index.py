# DO ODPALENIA TYLKO RAZ, PRZY KAŻDEJ AKTUALIZACJI PLIKU USTAWY

import json, re, argparse
from sentence_transformers import SentenceTransformer
from chromadb import Client
from config import (JSON_ACT_PATH, CHROMA_COLLECTION, EMBEDDER_NAME)

def clean_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.strip())

def prepare_documents(data: dict):
    docs, metas, ids = [], [], []
    for ch in data["document"]["chapters"]:
        ch_no = ch["number"]
        for art in ch["articles"]:
            art_no = art["article"].replace(".", "")
            for sub in art["subsections"]:
                num, txt = sub["number"], sub["content"]
                if not txt or txt == "Treść uchylona":
                    continue
                uid = f"ch{ch_no}_art{art_no}_sub{num}"
                metas.append({"chapter": ch_no, "article": art_no, "sub": num})
                ids.append(uid)
                docs.append(clean_text(txt))
    return docs, metas, ids

def main():
    with open(JSON_ACT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    docs, metas, ids = prepare_documents(data)
    emb = SentenceTransformer(EMBEDDER_NAME)
    client = Client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    col = client.create_collection(CHROMA_COLLECTION,
                                   metadata={"hnsw:space": "cosine"})
    col.add(documents=docs,
            embeddings=emb.encode(docs, batch_size=64, show_progress_bar=True),
            metadatas=metas,
            ids=ids)
    print(f"Zapisano {len(docs)} embeddingów do «{CHROMA_COLLECTION}»")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
