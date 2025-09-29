import sys, json, os, math, traceback
import numpy as np

# --- 1) import Twoich zasobów ---
# WAŻNE: resources.py nie może w tym momencie usuwać persist_directory,
# bo za każdym importem budujesz indeks od zera. Jeśli w resources.py
# masz `shutil.rmtree(persist_path, ...)` na poziomie modułu – rozważ jego
# komentarz lub kontrolę przez zmienną środowiskową (patrz notka pod kodem).
from resources import db, embedder_u, cross_encoder_ustawa

# --- 2) pomocnicze narzędzia do bezpiecznego dostępu do kolekcji chroma ---
def _get_chroma_collection(db_obj):
    """
    Działa z langchain_chroma i langchain_community:
    zwraca obiekt chromadb.Collection jeśli się da (ma .peek/.count).
    """
    candidates = [
        getattr(db_obj, "_collection", None),
        getattr(getattr(db_obj, "_vectorstore", None), "_collection", None),
        getattr(db_obj, "collection", None),
    ]
    for c in candidates:
        if c is None: 
            continue
        # heurystyka: obiekt z metodą peek lub count
        if hasattr(c, "peek") or hasattr(c, "count"):
            return c
    return None

def _collection_count(coll):
    try:
        return coll.count() if callable(coll.count) else int(coll.count)
    except Exception:
        return None

def _collection_peek(coll, n=3):
    try:
        return coll.peek(n)
    except Exception:
        return None

# --- 3) szybki sanity-check bazy ---
def chroma_sanity():
    print("== CHROMA SANITY ==")
    print("Typ db:", type(db))
    coll = _get_chroma_collection(db)
    if not coll:
        print("! Nie potrafię dostać się do kolekcji Chroma (API różni się między wersjami).")
    else:
        try:
            cnt = _collection_count(coll)
            print("Liczba wektorów w kolekcji:", cnt)
        except Exception as e:
            print("! count() exception:", e)

        pk = _collection_peek(coll, 2)
        if pk:
            # chroma peek zwykle zwraca dict(ids, documents, metadatas, embeddings?)
            keys = list(pk.keys())
            print("peek keys:", keys)
            try:
                print("peek.metadatas[0]:", json.dumps(pk["metadatas"][0], ensure_ascii=False))
                print("peek.doc[0] (fragment):", (pk["documents"][0] or "")[:160].replace("\n"," "))
            except Exception:
                pass

    # spróbuj też przez langchainowy interfejs:
    try:
        docs = db.similarity_search("test", k=1)
        print("similarity_search('test') →", len(docs), "kandydat(ów)")
        if docs:
            print("kandydat[0].meta:", docs[0].metadata)
            print("kandydat[0].text (frag):", docs[0].page_content[:150].replace("\n"," "))
    except Exception as e:
        print("! similarity_search wyjątek:", e)

# --- 4) prosta diagnostyka RAG dla jednego pytania ---
def rag_probe(q, k_sim=10, k_show=5, prob_threshold=None):
    """
    - robi ANN similarity z Chroma
    - liczy CE (ONNX / fallback) -> prawdopodobieństwo
    - pokazuje topy i ostrzega, jeśli wszystko spada poniżej progu
    """
    print("\n== RAG PROBE ==")
    print("Pytanie:", q)
    try:
        docs = db.similarity_search(q, k=k_sim)
    except Exception as e:
        print("! similarity_search wyjątek:", e)
        traceback.print_exc()
        return

    if not docs:
        print("→ Brak kandydatów już na etapie similarity. To wskazuje na: ")
        print("  - pustą kolekcję Chroma, zły persist_directory, różne wersje Chroma,")
        print("  - albo embedder_query ≠ embedder_index (mismatch).")
        return

    print(f"similarity_search zwrócił {len(docs)} kandydatów. Pokażę pierwsze {min(k_show, len(docs))} ID/metadane:")
    for d in docs[:k_show]:
        md = d.metadata or {}
        pid = md.get("id") or f"ch{md.get('chapter')}-art{md.get('article')}-ust{md.get('paragraph')}"
        print(" -", pid, "| art:", md.get("article"), "ust:", md.get("paragraph"))

    # CE
    pairs = [(q, d.page_content) for d in docs]
    try:
        logits_or_probs = cross_encoder_ustawa.predict(pairs, batch_size=16)
    except Exception as e:
        print("! cross_encoder.predict wyjątek:", e)
        traceback.print_exc()
        return

    arr = np.asarray(logits_or_probs, dtype=float)
    # heurystyka: jeśli większość wartości > 1.5 → to pewnie surowe logity → sigmoid
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    if np.nanmedian(arr) > 1.0:
        probs = _sigmoid(arr)
        print("CE wygląda na logity → normalizuję sigmoidem.")
    else:
        probs = arr
        print("CE wygląda na prawdopodobieństwa [0,1].")

    order = np.argsort(-probs)
    print("\nTOP po CE:")
    all_below = True
    for i in order[:min(k_show, len(docs))]:
        d = docs[i]; p = float(probs[i])
        md = d.metadata or {}
        pid = md.get("id") or f"ch{md.get('chapter')}-art{md.get('article')}-ust{md.get('paragraph')}"
        print(f"  • {pid}  prob={p:.4f}  | art={md.get('article')} ust={md.get('paragraph')}")
        if prob_threshold is not None and p >= prob_threshold:
            all_below = False

    if prob_threshold is not None:
        print(f"\nPróg CE: {prob_threshold:.2f}")
        if all_below:
            print("→ UWAGA: wszystkie kandydaty poniżej progu — to tłumaczy 'brak znalezionych ustępów'.")
            print("  Spróbuj obniżyć próg (np. 0.40) albo tymczasowo go wyłączyć (None).")

# --- 5) uruchomienie ---
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or "Kiedy przysługuje renta rolnicza?"
    chroma_sanity()
    rag_probe(query, k_sim=10, k_show=5, prob_threshold=0.75)
