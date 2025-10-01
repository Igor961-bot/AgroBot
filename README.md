# KRUS chatbot
Krus chatbot to interaktywny asystent dla rolników oraz instytucji (ustawy + dane tabelaryczne). Projekt łączy LLM obsługujący język polski, RAG (wektorowy i BM25), ekstrakcję danych tabelarycznych oraz webowy interfejs czatowy.
## Założenia projektu
* **RAG hybrydowy**: ChromaDB + BM25/CE do rerankingu.
* **Dwa tryby odpowiedzi**:
  1. **Ustawa** – odpowiedzi z przypisanymi cytatami/artykułami;
  2. **Dane** – szybkie zwroty danych tabelarycznych.
Przełączanie pomiędzy modułami aktualnie odbywa się w następujący sposób:
**Dane** należy w zapytaniu użytkownika podać jedno ze słów klucz *dane*, *statystyki*, *statystycznie*, wtedy 
* **Front + Backend**:
  * Frontend w React.
  * Backend w Python/FastAPI.
## Opis repozytorium
* `data/` - folder w którym znajdują się pliki do budowy dwóch instancji chroma 
* `logi/`- - folder do przytrzymywania wyników 
* `main/` - kod aplikacji
    * `krus-demo` - frontend aplikacji
    * `ask.py` - skrypt do wywołania aplikacji w CLI
    * `resources.py` - plik do przetrzymywania wspólnych dla obu modułów zmiennych
    * `krus_final.py` - logika części ustawowej aplikacji
    * `tabdata.py` - logika części tabelarycznej aplikacji
    * `server.py` - plik do komunikacji backendu z frontendem
* `test/` - folder do plików testowych 
* `.env.sample` - plik do przetrzymywania zmiennych środowiskowych
* `requirements.txt` - plik instalacyjny
## Uruchamianie aplikacji 
W terminalu w folderze *Agrobot* należy uruchomić komendę `pip install requirements.txt`. 
### Uruchamiania poprzez CLI
Należy przejść do pliku 'ask.py' który trzeba wywołać. Po załadowaniu, można w konsoli zadawać pytania. Jeśli wybrany tryb to odpowiedź na temat ustawy, po zadaniu pierwszego pytania należy postępować zgodnie z instrukcją wyświetloną w terminalu aby użyć trybu `follow_up`, który sprawia, że użytkownik może zadać dopytywać w odniesieniu o poprzednie pytanie. 
### Uruchamianie frontendu oraz backendu
W CMD należy przejść do folderu `.../Agrobot/main` i uruchomić komendę `uvicorn server:app --reload --host 0.0.0.0 --port 8000`. Po załadowaniu backendu w drugim terminalu należy:
**Jeśli odpalane po raz pierwszy** należy po kolei w folderze `.../Agrobot/main/krus-demo` odpalić kompedę `npm i` a następnie, pozaładowaniu `npm start`. 
**Jeśli odpalane po raz kolejny** należy pominąć krok z wpisaniem komendy `npm i`.
Wyświetli się okno, w którym na chacie można zadać pytania. Jeśli użytkownik chce użyć tryby `follow_up` po zadaniu pytania do ustawy może zacisnąc przycisk *dopytaj* (?). 
