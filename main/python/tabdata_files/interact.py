# -*- coding: utf-8 -*-
#interact.py
from typing import Dict, Any

from .common import timer, dbg, _is_country_name 
from .data_search import retrieve, parse_query_fields, derive_preferences, _choose_best_doc , valid_value 
from .transform import period_key 
from .format import build_sources_rows, _pl_number, _unit_from_measure
from data_schema import F_DATASET, F_MEASURE, F_VALUE, F_REGION, F_OKRES, has_core_fields  # tylko jeśli logujesz pola


def answer(question: str, k_ctx: int = 8):
    with timer("answer_total"):
        docs = retrieve(question, k_final=max(12, k_ctx))
        dbg("ANSWER_docs", n=len(docs), head=[
            (d.metadata.get(F_DATASET), d.metadata.get(F_MEASURE),
             d.metadata.get(F_OKRES), d.metadata.get(F_REGION), d.metadata.get(F_VALUE))
            for d in docs[:3]
        ])

        if not docs:
            parsed = parse_query_fields(question)
            msgs = []
            if parsed.get("period"):
                msgs.append(f"dla okresu „{parsed['period']}”")
            if _is_country_name(parsed.get("region")):
                msgs.append(f"dla państwa „{parsed['region']}”")
            return {"text": ("Brak danych " + " i ".join(msgs) + ".") if msgs else "Brak danych pasujących do pytania.",
                    "rows": []}

        parsed = parse_query_fields(question)
        prefs  = derive_preferences(question, parsed)

        docs_sorted = sorted(docs, key=lambda d: period_key(d.metadata.get(F_OKRES)), reverse=True)
        best = _choose_best_doc(docs_sorted, question, parsed, prefs)
        if not best:
            return {"text": "Brak danych w dostarczonej dokumentacji.", "rows": []}

        # jeśli wybrany ma brak wartości/opisu – podmień na pierwszy sensowny
        if (not has_core_fields(best)) or (not valid_value(best)):
            repl = next((d for d in docs_sorted if has_core_fields(d) and valid_value(d)), None)
            if repl is not None:
                best = repl

        mv = best.metadata or {}
        val = mv.get(F_VALUE)
        measure = mv.get(F_MEASURE) or ""
        okres = mv.get(F_OKRES) or "-"
        region = mv.get(F_REGION) or "ogółem"
        unit = _unit_from_measure(measure)
        value_text = _pl_number(val) + unit if val is not None else "-"

        docs_top = [best] + [d for d in docs_sorted if d is not best][:5]
        rows = build_sources_rows(docs_top)

        text = f"{value_text} ({okres}, {region})."
        dbg("ANSWER_best", value=value_text, okres=okres, region=region)
        return {"text": text, "rows": rows}


# ---------------------- Self-test ----------------------
if __name__ == "__main__":
    from pprint import pprint
    q = "podaj dane jakie jest przeciętne świadczenie emerytalne w krus?"
    print("Self-test TABDATA…")
    out = answer(q)
    pprint(out.get("text"))
    print("Rows:")
    for r in out.get("rows", [])[:5]:
        print(r)
