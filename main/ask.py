import re
from krus_final import ask, want_follow_up, reset_context
from tabdata import answer as answer_tab

WORD_DANE_RE = re.compile(r"\bdane\b", re.IGNORECASE)
key_words = ["dane", "statystyczne", "statystyki"]


print("AgroBot – wpisz pytanie (lub 'exit' aby wyjść)\n")
reset_context()  # start od czystego kontekstu dla modułu ustawowego

while True:
    try:
        user_msg = input("Ty: ").strip()
    except EOFError:
        break

    if not user_msg:
        continue

    if user_msg.lower() in {"exit", "quit", "q"}:
        print("koniec")
        break

    # 1) Gałąź danych: jeśli w treści pada słowo "dane" → tylko moduł tabelaryczny
    if any(kw in user_msg.lower() for kw in key_words) or WORD_DANE_RE.search(user_msg):
        data_txt = (answer_tab(user_msg) or "").strip()
        print("\nOto znalezione dane tabelaryczne:\n" + data_txt + "\n")
        # Uwaga: po danych NIE pytamy o dopytanie (to dotyczy modułu ustawy).
        continue

    # 2) Gałąź ustawowa: pełna logika + pętla dopytań
    res = ask(user_msg)
    ustawa_txt = res["answer"] if isinstance(res, dict) and "answer" in res else str(res)
    print("\n" + ustawa_txt + "\n")

    # 3) Po każdej odpowiedzi ustawowej pytamy o dopytanie
    while True:
        dec = input("Czy chcesz dopytać? [t/n/q]: ").strip().lower()
        if dec in ("t", "tak", "y", "yes"):
            want_follow_up()
            break  # wracamy po kolejne pytanie
        elif dec in ("n", "nie", "no"):
            reset_context()
            break
        elif dec in ("q", "quit", "exit"):
            print("koniec")
            raise SystemExit
        else:
            print("Wpisz: t (tak) / n (nie) / q (wyjście).")



 