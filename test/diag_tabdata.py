# debug_tabdata.py
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from resources import build_embeddings

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_statystyki")
EMBEDDER_MODEL_T = os.getenv("EMBEDDER_MODEL_T")

print("[CHK] PERSIST_DIR:", os.path.abspath(PERSIST_DIR))
emb = build_embeddings(EMBEDDER_MODEL_T)

vs = Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=PERSIST_DIR)
# Chroma nie ma API „count”, ale można spróbować pobrać topN „*”
try:
    docs = vs.similarity_search("test", k=1)
    print("[CHK] similarity_search działa (nie gwarantuje liczby)")
except Exception as e:
    print("[CHK] Błąd przy similarity_search:", e)

# Jeżeli masz dostęp do klienta niskopoziomowego:
try:
    client = vs._client  # uwaga: to internal
    col = client.get_collection("statystyki")
    print("[CHK] liczba wektorów:", col.count())
except Exception as e:
    print("[CHK] Nie mogę pobrać count():", e)

# Próba na Twoim pytaniu:
q = "podaj dane jakie jest przeciętne świadczenie emerytalne w krus?"
hits = vs.similarity_search(q, k=5)
print("\n[TOP5] trafienia na pytanie:")
for i, d in enumerate(hits, 1):
    print(f"{i}. {d.metadata.get('source','?')} | {str(d.page_content)[:120].replace('\\n',' ')}")
