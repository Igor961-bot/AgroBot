# KRUS chatbot
Krus chatbot to interaktywny asystent dla rolników oraz instytucji (ustawy + dane tabelaryczne). Projekt łączy LLM obsługujący język polski, RAG (wektorowy i BM25), ekstrakcję danych tabelarycznych oraz webowy interfejs czatowy.
## Założenia projektu
* **RAG hybrydowy**: ChromaDB + BM25/Cross-encoder do rerankingu.
* **Dwa tryby odpowiedzi**:
  * **Ustawa** – odpowiedzi z przypisanymi cytatami/artykułami;
  * **Dane** – szybkie zwroty danych tabelarycznych.
Przełączanie pomiędzy modułami aktualnie odbywa się w następujący sposób:
**Dane** należy w zapytaniu użytkownika podać jedno ze słów klucz *dane*, *statystyki*, *statystycznie*, wtedy zostanie użyty moduł do zapytań tabelarycznych.
Tryb ustawy jest domyślnym trybem i nie trzeba wpisywać słów klucz aby go obsługiwać.   
* **Front + Backend**:
  * Frontend w React.
  * Backend w Python/FastAPI.
## Opis repozytorium
* `data/` - folder w którym znajdują się pliki do budowy dwóch instancji chroma 
* `logi/`- - folder do przytrzymywania wyników 
* `test/` - folder do plików testowych 
* `requirements.txt` - plik instalacyjny
* `main/` — kod aplikacji:**
    * `.env` – zmienne środowiskowe (ścieżki, modele, porty).
    * `ask.py` – **CLI** (router: USTAWA ↔ DANE).
    * `build_chroma.py` – **budowa indeksów Chroma** (ustawa + CSV).
    * `data_schema.py` – walidacja i normalizacja wierszy CSV.
    * `krus_final.py` – logika trybu **USTAWA** (retrieve + CE + LLM, follow-up).
    * `resources.py` – modele/embeddi/cross-encodey, klienci Chroma, konfiguracja.
    * `server.py` – **backend FastAPI** endpoiny pod frontend.
    * `krus-demo/` – **frontend React** (UI czatu).
    * `tabdata_files/` – moduł **DANE**:
        * `common.py` – konfiguracja i słowniki.
        * `data_search.py` – wyszukiwanie (Chroma, BM25, MQ, HyDE, CE, filtry).
        * `format.py` – formatowanie odpowiedzi i tabel ze źródłami.
        * `interact.py` – funkcja `answer(question)` — punkt użycia trybu DANE.
        * `transform.py` – pomocnicze (sortowanie okresów, wybór najświeższych).

## Uruchamianie aplikacji 
W terminalu w folderze *Agrobot* należy uruchomić komendę `pip install -r requirements.txt`. 
### Uruchamiania poprzez CLI
Należy przejść do pliku 'ask.py' który trzeba wywołać. Po załadowaniu, można w konsoli zadawać pytania. Jeśli wybrany tryb to odpowiedź na temat ustawy, po zadaniu pierwszego pytania należy postępować zgodnie z instrukcją wyświetloną w terminalu aby użyć trybu `follow_up`, który sprawia, że użytkownik może zadać dopytywać w odniesieniu o poprzednie pytanie. 
### Uruchamianie frontendu oraz backendu
W CMD należy przejść do folderu `.../Agrobot/main` i uruchomić komendę `uvicorn server:app --reload --host 0.0.0.0 --port 8000`. Po załadowaniu backendu w drugim terminalu należy:
**Jeśli odpalane po raz pierwszy** należy po kolei w folderze `.../Agrobot/main/krus-demo` odpalić kompedę `npm i` a następnie, pozaładowaniu `npm start`. 
**Jeśli odpalane po raz kolejny** należy pominąć krok z wpisaniem komendy `npm i`.
Wyświetli się okno, w którym na chacie można zadać pytania. Jeśli użytkownik chce użyć tryby `follow_up` po zadaniu pytania do ustawy może zacisnąc przycisk *dopytaj* (?). 
