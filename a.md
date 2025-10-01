# KRUS Chatbot

Interaktywny asystent dla rolników oraz instytucji (ustawy + dane tabelaryczne). Projekt łączy lokalny LLM obsługujący język polski, RAG (wektorowy + BM25/CE), ekstrakcję danych tabelarycznych oraz webowy interfejs czatowy.

---

## Założenia projektu

* **RAG hybrydowy**: ChromaDB (wektory) + BM25; reranking przez **Cross-Encoder (CE)**.
* **Dwa tryby odpowiedzi**:

  1. **Ustawa** – odpowiedzi oparte na treści aktu prawnego z przypiętymi cytatami/artykułami;
  2. **Dane** – szybkie zwroty danych tabelarycznych (CSV/Parquet/SQL) + krótkie źródła do tabeli.
* **Przełączanie trybów** (routing): jeśli w pytaniu pojawi się słowo kluczowe **„dane”**, **„statystyki”** lub **„statystycznie”**, uruchamiany jest moduł **Dane**; w pozostałych przypadkach – moduł **Ustawa**.
* **Front + Backend**:

  * Frontend: React (UI czatu).
  * Backend: Python/FastAPI (endpointy `/ask`, `/followup`).

## Opis repozytorium

* `data/` – pliki źródłowe do budowy kolekcji Chroma (ustawa, tabele).
* `logi/` – logi z działania aplikacji i debug (opcjonalnie).
* `main/` – kod aplikacji:

  * `krus-demo/` – frontend (React);
  * `ask.py` – tryb CLI do zadawania pytań w terminalu;
  * `resources.py` (lub pakiet `resources/`) – wspólne zasoby (modele, embeddery, CE, wektorownie);
  * `krus_final.py` – logika modułu **Ustawa** (RAG + CE + follow-up);
  * `tabdata.py` – logika modułu **Dane** (retrieval + transform + format);
  * `server.py` – backend FastAPI (komunikacja front–back, routing `/ask`).
* `test/` – pliki testowe i przykłady zapytań.
* `.env.sample` – szablon zmiennych środowiskowych.
* `requirements.txt` – lista zależności Pythona.

---

## Uruchamianie aplikacji

W terminalu w katalogu głównym repo (np. `Agrobot/`) zainstaluj zależności:

```bash
pip install -r requirements.txt
```

### Uruchamianie przez CLI

Wejdź do `main/` i uruchom skrypt:

```bash
python ask.py
```

Po załadowaniu możesz zadawać pytania w konsoli. Dla modułu **Ustawa** po pierwszej odpowiedzi możesz skorzystać z trybu **follow-up** (dopytywanie w kontekście poprzedniego pytania) – postępuj zgodnie z instrukcją w terminalu.

### Uruchamianie backendu i frontendu

1. **Backend (FastAPI):**

```bash
cd main
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend (React):**

```bash
cd main/krus-demo
# pierwszy raz
npm i
npm start
# kolejne uruchomienia – wystarczy samo `npm start`
```

W przeglądarce pojawi się UI czatu. Aby skorzystać z trybu **follow-up** po pytaniu o **Ustawę**, użyj przycisku „**Dopytaj**”.

---

```
┌──────────────┐       HTTP/JSON        ┌───────────────┐
│   Frontend   │  ───────────────────▶  │    Backend    │
│   React      │       /ask,/followup   │   FastAPI     │
└─────┬────────┘                         ├─────┬────────┘
      │                                  │     │
      │                                  │     ├─ RAG (Retriever + CE rerank)  ←  backend/krus_final.py
      │                                  │     ├─ KB: ChromaDB / BM25
      │                                  │     ├─ TabData (SQL/CSV/Parquet)    ←  backend/tabdata.py
      │                                  │     └─ LLM (local, 4–8bit)
      │                                  │
      ▼                                  ▼
   UI chat                         Odpowiedź + cytaty + wyniki tabelaryczne
```

┌──────────────┐       HTTP/JSON        ┌───────────────┐
│   Frontend   │  ───────────────────▶  │    Backend    │
│   React      │       /ask,/followup   │   FastAPI     │
└─────┬────────┘                         ├─────┬────────┘
│                                  │     │
│                                  │     ├─ RAG (Retriever + CE rerank)
│                                  │     ├─ KB: ChromaDB / BM25
│                                  │     ├─ TabData (SQL/CSV/Parquet)
│                                  │     └─ LLM (local, 4–8bit)
│                                  │
▼                                  ▼
UI chat                         Odpowiedź + cytaty + wyniki tabelaryczne

```

---

## Struktura katalogów
```

repo/
├─ frontend/
│  ├─ src/
│  │  ├─ App.js
│  │  ├─ assets/
│  │  └─ api.js
│  ├─ package.json
│  └─ .env.example
│
├─ backend/
│  ├─ main.py
│  ├─ api.py
│  ├─ krus_final.py          # Główna logika RAG dla Ustawy KRUS (opis niżej)
│  ├─ tabdata.py             # Silnik wyszukiwania danych tabelarycznych (opis niżej)
│  ├─ rag/
│  │  ├─ retriever.py
│  │  ├─ rerank.py
│  │  └─ loaders/
│  ├─ tabdata/
│  │  ├─ answer.py
│  │  └─ sources/
│  ├─ resources/             # Zewnętrzne zależności i obiekty injektowane
│  │  ├─ db.py               # ChromaDB vectorstore (aliasowane jako `db`/`vectorstore`)
│  │  ├─ vectorstore.py
│  │  ├─ gen_model.py        # Wczytany lokalny LLM (model)
│  │  ├─ tokenizer.py        # Tokenizer do modelu
│  │  ├─ cross_encoder_ustawa.py
│  │  ├─ reranker_ce.py
│  │  ├─ emb.py              # Embedder do fallbacku
│  │  └─ **init**.py         # eksporty: db, vectorstore, gen_model, tokenizer, ...
│  ├─ config.py
│  └─ requirements.txt
│
├─ data/
│  ├─ chroma/
│  ├─ bm25/
│  └─ tables/                # CSV_DIR wskazuje tu
│
├─ scripts/
│  ├─ ingest_docs.py
│  ├─ build_indices.py
│  └─ smoke_test.py
│
├─ docker/
│  ├─ Dockerfile.backend
│  ├─ Dockerfile.frontend
│  └─ docker-compose.yml
│
├─ .env.example
├─ LICENSE
└─ README.md

````

> **Uwaga:** nazwy plików mogą minimalnie różnić się od stanu repo – dopasuj do faktycznej struktury u Ciebie.

---

## Punkt wejścia / uruchamianie

### 1) Backend (FastAPI)
Wymagania: Python 3.10+, wirtualne środowisko, opcjonalnie CUDA + bitsandbytes.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Zmienne środowiskowe (przykład):
# LLM_MODEL_PATH=/models/bielik-11b
# BNB_4BIT=true
# CHROMA_PATH=../data/chroma
# TABDATA_PATH=../data/tables

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
````

Domyślne endpointy:

* `POST /ask` – główne odpowiedzi (tryb ustawa/FAQ; jeśli w pytaniu padnie słowo **„dane”**, backend przełącza się na tryb danych i zwraca tabelaryczny wynik).
* `POST /followup` – kontynuacje dialogu (z zachowaniem kontekstu).

### 2) Frontend (React)

Wymagania: Node.js 18+.

```bash
cd frontend
cp .env.example .env       # ustaw REACT_APP_API_BASE=http://localhost:8000
npm install
npm start                  # lub: npm run dev (Vite)
```

Aplikacja połączy się z backendem ustawionym w `REACT_APP_API_BASE` i pokaże UI czatu.

---

## Zmienne środowiskowe (przykład)

Zobacz `.env.example` w katalogu głównym i w `frontend/.env.example`.

* **Backend (krus_final / RAG)**

  * `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` – opcjonalny tracing.
  * `TOKENIZERS_PARALLELISM` – np. `true`.
  * `LLM_MODEL_PATH` – ścieżka do lokalnego modelu (ładowany w `resources.gen_model`).
  * `CE_MODEL` – identyfikator cross-encodera dla ustawy.
  * Progi: ustawiane w kodzie (`RERANK_THRESHOLD=0.2`, `K_SIM=15`, `K_FINAL=5`).

* **Backend (tabdata / dane)**

  * `CSV_DIR` – ścieżka do katalogu lub pojedynczego pliku CSV (wymagane kolumny: `dataset,measure,value,region,period,typ`).
  * Inne w `resources`: `vectorstore`, `reranker_ce`, `emb` (fallback do kosinusów), `BASE_MODEL_ID` dla `llm_pllum` (opcjonalny LLM do MQ/HyDE).

* **Frontend**

  * `REACT_APP_API_BASE` – URL backendu.

---

## Przepływ zapytań

1. **UI** wysyła `question` do `POST /ask`.
2. **Router (backend/api.py)** sprawdza treść:

   * jeśli zawiera słowo **„dane”** → wywołuje `tabdata.answer()` i zwraca sekcję danych + wiersze;
   * w przeciwnym razie → wywołuje `krus_final.ask()` (RAG dla ustawy/FAQ).
3. **Follow-up**: po kliknięciu „Chciałbym dopytać” backend woła `krus_final.want_follow_up()` i **następne** zapytanie trafia do trybu dobierania dodatkowych ustępów.
4. Odpowiedź JSON: `answer` (tekst), `citations` (lista ustępów) i opcjonalnie `table.rows` dla danych.

---

## Dane i FAQ

* **Ustawa/akty (krus_final.py)**:

  * `retrieve_basic()` → `db.similarity_search` (Chroma) → deduplikacja → **cross-encoder** (`cross_encoder_ustawa.predict`) → sigmoid/softmax → próg `RERANK_THRESHOLD` → top-`K_FINAL`.
  * `ask()` zarządza trybem **new query** vs **follow-up** (przycisk ustawia flagę `want_follow_up()`), akumuluje dokumenty w `STATE.accum_docs` i zawsze wyświetla **pełną listę cytowanych ustępów**.
  * Parser odwołań (`parse_ref_ext`) umożliwia szybkie pobranie konkretnego artykułu/ustępu (tryb `EXPLICIT_REF`).
  * Format odpowiedzi: blok „Cytowane ustępy” + „Odpowiedź: …”; tekst jest czyszczony z markdown bold i przycinany do pełnych zdań.

* **Dane tabelaryczne (tabdata.py)**:

  * Budowanie dokumentów z CSV: `build_documents_from_standard_csv()` wymaga kolumn: `dataset,measure,value,region,period,typ`. Każdy wiersz → `Document(page_content=...)` + metadane (`value`, `okres`, `region`, ...). Słowniki `FIELD_VOCAB` uzupełniane są w locie.
  * Indeksy: dokumenty ładowane do `resources.vectorstore` (Chroma); dodatkowo `BM25Retriever` na pełnym korpusie.
  * **Query understanding**: `parse_query_fields()` (wykrywa okresy: `YYYY`, `YYYY-QN`, regiony z odmianami, `measure`, `typ`, `dataset`), dopasowania miękkie i heurystyki regionów (kraj/województwo/państwo).
  * **Retrieval fusion**: `dense_search` + `MultiQuery` + **HyDE** + `BM25` → **RRF** z wagami `RRF_WEIGHTS` → **reranker_ce** → filtry (narodowe vs zagraniczne, region/okres) → klastrowanie i wybór najświeższych (`pick_latest_per_cluster`).
  * **Odpowiedź**: `answer(question)` zwraca `{"text": "<wartość> (<okres>, <region>).", "rows": [...]}` gdzie `rows` to krótkie źródła do tabeli w UI. Przy braku dopasowań – komunikaty „Brak danych …” + pusty blok [ŹRÓDŁA].

---

## Moduł `resources/` – modele, embedingi, kolekcje Chroma

Ten moduł centralizuje **ładowanie modeli** i **inicjalizację wektorowni** dla dwóch obszarów: *statystyki (tabele)* i *ustawa*.

### Modele generatywne (LLM)

* `BASE_MODEL_ID="speakleash/Bielik-11B-v2.6-Instruct"` (domyślnie; alternatywnie `CYFRAGOVPL/PLLuM-12B-chat`).
* Ładowanie z **bitsandbytes 4-bit (NF4)** i `torch_dtype=bfloat16`, `attn_implementation="sdpa"`.
* Eksportowane obiekty:

  * `tokenizer` – z `AutoTokenizer.from_pretrained(BASE_MODEL_ID)` (fallback do `use_fast=False` gdy trzeba),
  * `gen_model` – `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb, device_map="cuda:0").eval()`.
* **Konsumpcja w modułach**: używaj wyłącznie **`transformers.pipeline`** tworzonego lokalnie w module (HuggingFacePipeline z `gen_model` + `tokenizer`). Dzięki temu inne pliki nie muszą znać szczegółów konfiguracji GPU.

### Tabele (statystyki)

* **Embedder**: własna klasa `MXBAIEmbeddings` (E5-large styl) – pooling uśredniający + L2 normalize; urządzenie: `cuda` jeśli dostępne, inaczej `cpu`.
* **Reranker CE**: `BAAI/bge-reranker-v2-m3` przez `sentence_transformers.CrossEncoder` (urządzenie jak embedder). Eksport: `reranker_ce` (może być `None` jeśli nie udało się załadować – wtedy fallback kosinusowy w `tabdata`).
* **Chroma kolekcja**: `collection_name="statystyki"`, katalog `PERSIST_DIR` (domyślnie `chroma_statystyki`). Eksport: `vectorstore`.
* **Uwaga**: przy starcie wykonywane jest `shutil.rmtree(PERSIST_DIR, ignore_errors=True)` → **czyści kolekcję** i buduje od zera. W produkcji rozważ usunięcie kasowania.

### Ustawa (akty prawne)

* **Embedder**: `HuggingFaceBgeEmbeddings` z `EMBEDDER_MODEL_U="intfloat/multilingual-e5-base"`, `normalize_embeddings=True`.
* **Cross-encoder**: `OptimizedONNXCrossEncoder` owinięty wokół `onnxruntime` (CPU) dla `RERANKER_MODEL_U="radlab/polish-cross-encoder"` z fallbackiem do `sentence_transformers.CrossEncoder`.
* **Przygotowanie korpusu**: plik Markdown `DATA_PATH` (np. `data/ustawa_with_paragraph_headers.md`) z nagłówkami `###` per ustęp oraz komentarzami meta w formie:

  ```
  <!-- chapter:1 article:16a paragraph:3 id:ch1-art16a-ust3 -->
  ```

  Parsowane przez `MarkdownHeaderTextSplitter` i regex `meta_re`; metadane normalizowane do kluczy: `chapter/rozdzial`, `article/artykul`, `paragraph/ust`, `id`.
* **Chroma kolekcja**: tworzona przez `Chroma.from_documents(..., persist_directory=PERSIST_PATH, collection_metadata={"hnsw:space":"cosine"})`. Eksport: `db` (używany w `krus_final`).
* **Uwaga**: analogicznie – start kasuje `PERSIST_PATH` (`shutil.rmtree`).

### Ścieżki i identyfikatory (domyślne)

* `CSV_DIR = "all_dane/all_data (13).csv"` – źródło CSV do indeksowania tabel (w `tabdata`).
* `DATA_PATH = "data/ustawa_with_paragraph_headers.md"` – źródło ustawy.
* `PERSIST_DIR = "chroma_statystyki"` – kolekcja tabel.
* `PERSIST_PATH = "chroma_ustawa"` – kolekcja ustawy.

### Importy w innych modułach

Przykład minimalnego użycia w `krus_final.py` i `tabdata.py`:

```python
# krus_final.py
from resources import db, gen_model, tokenizer, cross_encoder_ustawa
from transformers import pipeline
hf_pipe = pipeline("text-generation", model=gen_model, tokenizer=tokenizer, ...)

# tabdata.py
from resources import vectorstore, reranker_ce, emb, BASE_MODEL_ID
# ewentualnie budowa dodatkowego pipeline'u LLM do MQ/HyDE na bazie BASE_MODEL_ID
```

### Jedna instancja Chroma z dwiema kolekcjami?

Obecny kod tworzy **dwie odrębne bazy** przez dwa `persist_directory` (osobne katalogi). Jeśli chcesz **jedną instancję** z dwiema kolekcjami, ustaw **ten sam `persist_directory`** i różne `collection_name` przy tworzeniu obu store’ów:

```python
PERSIST_ROOT = "./chroma_store"
vectorstore = Chroma(collection_name="statystyki", embedding_function=emb, persist_directory=PERSIST_ROOT)
db          = Chroma(collection_name="ustawa",    embedding_function=embedder_u, persist_directory=PERSIST_ROOT)
```

> Wtedy nie używaj `shutil.rmtree(...)` na root, żeby nie skasować drugiej kolekcji.

---

## API (szkic)

```http
POST /ask
Content-Type: application/json
{
  "question": "tekst pytania",
  "reset_memory": false
}
→ // tryb USTAWA (krus_final.ask)
{
  "answer": "Cytowane ustępy: ...
Odpowiedź:
...",
  "citations": [ {"id": "ch2-art21-ust5", "score": 0.55 }, ... ],
  "debug": { "mode": "new_query|follow_up|explicit", "rerank": [ {"id":"...","score":0.61}, ... ] }
}

→ // tryb DANE (tabdata.answer)
{
  "text": "123 456 (2024-Q2, ogółem).",
  "rows": [ { "value": 123456, "dataset": "...", "measure": "...", "type": "...", "okres": "2024-Q2", "region": "ogółem" }, ... ]
}
```

---

## Budowa indeksów

```bash
# Ingest dokumentów (ustawy/FAQ)
python scripts/ingest_docs.py --src ./raw_docs --out ./data/chroma

# Budowa indeksów BM25 i wektorów
python scripts/build_indices.py --chroma ./data/chroma --bm25 ./data/bm25
```

---


## Rozwiązywanie problemów (FAQ)

* **„Brak danych” w sekcji tabelarycznej**

  * Sprawdź `CSV_DIR` – czy wskazuje na katalog/pliki CSV o **wymaganym schemacie**.
  * W logach `tabdata.py` zobacz, ile dokumentów zbudowano ("Zbudowano dokumentów: N").
  * Upewnij się, że `reranker_ce` jest dostępny; w razie problemów włączy się fallback kosinusowy (embedder `emb`).

* **Za mało cytatów w RAG (ustawa)**

  * Skoryguj `RERANK_THRESHOLD`, `K_SIM`, `K_FINAL` w `krus_final.py`.
  * Zbadaj logi `[RET][CE]` i `[RET][THR]` – czy próg nie odcina wszystkiego.

* **Follow-up nie dodaje nowych ustępów**

  * `want_follow_up()` musi być wywołane **przed** następnym `ask()`.
  * Tryb follow-up dobiera 1 dokument (filtr po `article` gdy możliwe) i **dokleja** do `STATE.accum_docs`.

* **Front nie łączy się z backendem** – `REACT_APP_API_BASE` i CORS w `backend/main.py`.

* **VRAM/perf** – użyj 4-bit BnB dla 11B, zmniejsz `max_new_tokens` w pipeline.
