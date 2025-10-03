# run_cli.py
import re
import time
from krus_final import ask, want_follow_up, reset_context
from tabdata import answer as answer_tab
#from tabdata_files.interact import answer as answer_tab
WORD_DANE_RE = re.compile(r"\bdane\b", re.IGNORECASE)
KEY_WORDS = {"dane", "statystyczne", "statystyki"}

def print_rows_as_table(rows):
    if not rows:
        print("– brak dodatkowych wierszy źródłowych")
        return
    # minimalna, prosta tabela w ASCII
    cols = ["dataset", "measure", "type", "okres", "region", "value"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    sep = "+ " + " + ".join("-" * widths[c] for c in cols) + " +"

    def _line(row=None, header=False):
        parts = []
        for c in cols:
            s = c if header else str(row.get(c, ""))
            parts.append(s.ljust(widths[c]))
        return "| " + " | ".join(parts) + " |"

    print(sep)
    print(_line(header=True))
    print(sep)
    for r in rows:
        print(_line(row=r))
    print(sep)

def looks_like_data_query(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in KEY_WORDS) or bool(WORD_DANE_RE.search(t))

def fmt_duration(seconds: float) -> str:
    # 850 ms zamiast 0.85 s; powyżej 10 s z miejscem po przecinku
    if seconds < 1:
        return f"{int(round(seconds * 1000))} ms"
    if seconds < 10:
        return f"{seconds:.2f} s"
    return f"{seconds:.1f} s"

if __name__ == "__main__":
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

        # 1) Gałąź danych: jeśli w treści pada słowo „dane” → tylko moduł tabelaryczny
        if looks_like_data_query(user_msg):
            t0 = time.perf_counter()
            try:
                res = answer_tab(user_msg)
            except Exception as e:
                print(f"\n[BŁĄD][DANE] {e}\n")
                continue

            if isinstance(res, dict):
                print("\nOto znalezione dane tabelaryczne:")
                print(res.get("text", ""))
                rows = res.get("rows", [])
                if rows:
                    print()
                    print_rows_as_table(rows)
                print()
            else:
                # fallback, gdyby kiedyś answer() zwrócił string
                print("\nOto znalezione dane tabelaryczne:\n" + str(res) + "\n")

            t1 = time.perf_counter()
            print(f"[CZAS][DANE] {fmt_duration(t1 - t0)}\n")
            # po danych NIE pytamy o dopytanie (to dotyczy modułu ustawy).
            continue

        # 2) Gałąź ustawowa: pełna logika + (ew.) pętla dopytań
        t0 = time.perf_counter()
        try:
            res = ask(user_msg)
        except Exception as e:
            print(f"\n[BŁĄD][USTAWA] {e}\n")
            continue

        ustawa_txt = res["answer"] if isinstance(res, dict) and "answer" in res else str(res)
        print("\n" + ustawa_txt + "\n")
        t1 = time.perf_counter()
        print(f"[CZAS][USTAWA] {fmt_duration(t1 - t0)}\n")

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
