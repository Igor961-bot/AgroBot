# DO ODPALENIA TYLKO RAZ, PRZY KAŻDEJ AKTUALIZACJI PLIKU USTAWY

import json, re, argparse
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from config import (JSON_ACT_PATH, CHROMA_COLLECTION, EMBEDDER_NAME)

def clean_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.strip())


def clean_and_expand_content(content: str) -> str:
    abbreviations = {
        r"\bDz\. U\.": "Dziennik Ustaw",
        r"\bz późn\. zm\.": "z późniejszymi zmianami",
        r"\bKasa\b": "Kasa Rolniczego Ubezpieczenia Społecznego",
        r"\bZakład\b": "Zakład Ubezpieczeń Społecznych",
        r"\bRada Rolników\b": "Rada Ubezpieczenia Społecznego Rolników",
        r"\bPKD\b": "Polska Klasyfikacja Działalności",
        r"\bMonitor Polski\b": "Dziennik Urzędowy Rzeczypospolitej Polskiej Monitor Polski",
    }
    for abbr, full in abbreviations.items():
        content = re.sub(abbr, full, content, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", content.strip())



def prepare_documents(data):
    """
    Tworzy listy: documents, metadatas, ids  – bez rzutowań na int.
    • chapter (int)  – zawsze liczba
    • article (str)  – np. '12', '12a'
    • subsection (str) – np. '1', '1a'  (nie ma None)
    ID:  ch{chapter}_art{article}[_sub{subsection}]
    """
    documents, metadatas, ids, seen = [], [], [], set()

    for chapter in data["document"]["chapters"]:
        ch_num = int(chapter["number"])          # tu dalej liczba

        for article in chapter["articles"]:
            art_num = article["article"].replace('.', '').strip()   # ← str!

            # ----------- tekst główny artykułu -----------
            art_raw = article["content"].strip()
            if art_raw and art_raw != "Treść uchylona":
                uid = f"ch{ch_num}_art{art_num}"
                if uid not in seen:
                    seen.add(uid)
                    documents.append(clean_and_expand_content(art_raw))
                    metadatas.append({"chapter": ch_num, "article": art_num})
                    ids.append(uid)

            # -------------- podpunkty --------------------
            for sub in article["subsections"]:
                sub_num = sub["number"].strip()  # ← str, może być '1a'
                content = sub["content"].strip()
                if not content or content == "Treść uchylona":
                    continue
                uid = f"ch{ch_num}_art{art_num}_sub{sub_num}"
                if uid in seen:
                    continue
                seen.add(uid)
                documents.append(clean_and_expand_content(content))
                metadatas.append(
                    {"chapter": ch_num,
                     "article": art_num,
                     "subsection": sub_num}
                )
                ids.append(uid)

    return documents, metadatas, ids




def main():
    with open(JSON_ACT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    docs, metas, ids = prepare_documents(data)
    emb = SentenceTransformer(EMBEDDER_NAME)
    client = PersistentClient(path="chroma_store")
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
