# test_embeddings.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, traceback, random
from typing import List

from dotenv import load_dotenv
load_dotenv()

def hr():
    print("-" * 72)

def fmt_ms(t: float) -> str:
    return f"{t*1000:.0f} ms"

def show_env():
    print("[ENV] USE_LMSTUDIO =", os.getenv("USE_LMSTUDIO"))
    print("[ENV] LLM_BASE_URL =", os.getenv("LLM_BASE_URL"))
    print("[ENV] LMSTUDIO_API_KEY =", os.getenv("LMSTUDIO_API_KEY"))
    print("[ENV] EMBEDDER_MODEL_U =", os.getenv("EMBEDDER_MODEL_U"))
    print("[ENV] EMBEDDER_MODEL_T =", os.getenv("EMBEDDER_MODEL_T"))
    print("[ENV] EMBED_BATCH =", os.getenv("EMBED_BATCH", "(default)"))
    print("[ENV] EMBED_MAX_CHARS =", os.getenv("EMBED_MAX_CHARS", "(default)"))
    print("[ENV] EMBED_CONNECT_TIMEOUT_S =", os.getenv("EMBED_CONNECT_TIMEOUT_S", "(default)"))
    print("[ENV] EMBED_READ_TIMEOUT_S =", os.getenv("EMBED_READ_TIMEOUT_S", "(default)"))
    print("[ENV] EMBED_WRITE_TIMEOUT_S =", os.getenv("EMBED_WRITE_TIMEOUT_S", "(default)"))

def raw_openai_ping():
    """
    Ping bezpośrednio na /v1/embeddings, żeby zweryfikować sieć/timeouty
    niezależnie od LangChain.
    """
    print("\n[RAW] Bezpośredni ping na /v1/embeddings …")
    try:
        import httpx
        from openai import OpenAI

        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        api_key  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        model_id = os.getenv("EMBEDDER_MODEL_U") or os.getenv("EMBEDDER_MODEL_T")

        if not model_id:
            print("[RAW] Brak EMBEDDER_MODEL_U/T w .env")
            return

        connect_s = float(os.getenv("EMBED_CONNECT_TIMEOUT_S", "5"))
        read_s    = float(os.getenv("EMBED_READ_TIMEOUT_S",  "60"))
        write_s   = float(os.getenv("EMBED_WRITE_TIMEOUT_S", "60"))

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                timeout=httpx.Timeout(connect=connect_s, read=read_s, write=write_s, pool=None),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ),
        )
        t0 = time.perf_counter()
        r = client.embeddings.create(model=model_id, input=["ping"])
        dt = time.perf_counter() - t0
        dim = len(r.data[0].embedding)
        print(f"[RAW] OK: dim={dim}, time={fmt_ms(dt)}")
    except Exception as e:
        print("[RAW] ERROR:", repr(e))
        traceback.print_exc()

def build_embedder():
    print("\n[LC] Tworzę embedder przez resources.build_embeddings …")
    try:
        from resources import build_embeddings
        model_id = os.getenv("EMBEDDER_MODEL_U") or os.getenv("EMBEDDER_MODEL_T")
        if not model_id:
            print("[LC] Brak EMBEDDER_MODEL_U/T w .env")
            return None
        t0 = time.perf_counter()
        emb = build_embeddings(model_id)
        dt = time.perf_counter() - t0
        print(f"[LC] OK: build_embeddings('{model_id}') w {fmt_ms(dt)}")
        return emb
    except Exception as e:
        print("[LC] ERROR przy build_embeddings:", repr(e))
        traceback.print_exc()
        return None

def lorem(n_chars: int) -> str:
    base = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed non risus. Suspendisse lectus tortor, dignissim sit amet, "
            "adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. ")
    s = []
    while len("".join(s)) < n_chars:
        s.append(base)
    return ("".join(s))[:n_chars]

def test_embed_query(emb) -> None:
    print("\n[TEST] embed_query('ping') …")
    try:
        t0 = time.perf_counter()
        v = emb.embed_query("ping")
        dt = time.perf_counter() - t0
        print(f"[TEST] OK: dim={len(v)}, time={fmt_ms(dt)}")
    except Exception as e:
        print("[TEST] embed_query ERROR:", repr(e))
        traceback.print_exc()

def test_embed_documents(emb) -> None:
    print("\n[TEST] embed_documents – batch & size sanity …")
    # przygotuj paczkę: 3× krótkie, 3× średnie, 3× długie
    docs: List[str] = []
    random.seed(42)
    for _ in range(3):
        docs.append("krótki tekst " + str(random.random()))
    for _ in range(3):
        docs.append(lorem(800))     # ~800 znaków
    for _ in range(3):
        docs.append(lorem(3000))    # ~3000 znaków (przytnie się, jeśli masz EMBED_MAX_CHARS)

    # wymuś kilka batchy, powielając listę
    factor = 4  # 9*4=36 dokumentów → przy EMBED_BATCH=8 będzie 5 batchy
    payload = docs * factor
    print(f"[TEST] items={len(payload)}")

    try:
        t0 = time.perf_counter()
        V = emb.embed_documents(payload)
        dt = time.perf_counter() - t0
        print(f"[TEST] OK: got {len(V)} vectors, time={fmt_ms(dt)} (~{fmt_ms(dt/len(payload))}/item)")
    except Exception as e:
        print("[TEST] embed_documents ERROR:", repr(e))
        traceback.print_exc()

def main():
    hr(); show_env(); hr()
    raw_openai_ping()
    emb = build_embedder()
    if emb is None:
        return
    test_embed_query(emb)
    test_embed_documents(emb)
    hr()
    print("Done.")

if __name__ == "__main__":
    main()
