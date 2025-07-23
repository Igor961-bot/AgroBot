# -------- ogólne ścieżki / nazwy kolekcji --------

JSON_ACT_PATH     = "./data/ustawa_processed.json"     
CHROMA_COLLECTION = "ustawa_1"

# -------- modele --------

MODEL_NAME    = "CYFRAGOVPL/PLLuM-12B-nc-chat" # chatbot
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2" # embedder

# -------- RAG / chat --------
TOP_K            = 5                							# ile powiązanych z zapytaniem fragmetnów ustawy wyciągać z BD?
MAX_TURNS        = 2		    							# ile pytań wstecz do zapamiętania jako surowy tekst?
SUMMARY_TRIGGER  = 3		    							# ile zapytań starszych niż MAX_TURNS w pamięci, aby zacząć podsumowywać kontekst rozmowy?
MAX_SENT_SHORT   = 3		    							# ile maksymalnie zdań w odpowiedzi z kategorii "krótka"
MAX_SENT_LONG    = 8		    							# ile maksymalnie zdań w odpiwiedzi z kategorii "długa"
LONG_TRIGGERS    = {"szczegół", "rozwiń", "dokładn", "przykład", "więcej"}		# trigger-words dla trybu "długiej" odpowiedzi
SIM_THRESHOLD    = 0.30        								# odrzucenie słabych (mało prawdopodobnych) trafień kontekstu w BD
