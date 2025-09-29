 #resources

# wczytac bielik i w modułach zostawic tylko hf_pipeline
# wczytac embedder i crossencoder z tabdata
# stworzyc dwie kolekcje w jednej instacni chormaDB
# 
import os, re, unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import torch

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
except Exception:
    pass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

import os, re, shutil, unicodedata, asyncio
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
import torch
from langchain_core.documents import Document  
import torch
import os
from langchain_core.documents import Document
from typing import List, Optional
from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from transformers import pipeline as hf_pipeline
import shutil, re
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import CrossEncoder
from transformers import pipeline as hf_pipeline
from langchain_community.vectorstores import Chroma
try:
    from langchain_chroma import Chroma
    _CHROMA_NEW = True
except ImportError:
    from langchain_community.vectorstores import Chroma
    _CHROMA_NEW = False
from transformers import AutoTokenizer as HF_AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
_HAS_RERANK = True
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except Exception:
    _HAS_RERANK = False


BASE_MODEL_ID    = "speakleash/Bielik-11B-v2.6-Instruct"
# "speakleash/Bielik-11B-v2.6-Instruct"
# "CYFRAGOVPL/PLLuM-12B-chat"

# tabele
RERANKER_MODEL_T = "BAAI/bge-reranker-v2-m3"
EMBEDDER_MODEL_T = "intfloat/multilingual-e5-large"

# ustawa
# RERANKER_MODEL_U = "BAAI/bge-reranker-v2-m3"
# EMBEDDER_MODEL_U = "intfloat/multilingual-e5-large"
EMBEDDER_MODEL_U   = "intfloat/multilingual-e5-base"
RERANKER_MODEL_U   = "radlab/polish-cross-encoder"

CSV_DIR = "C:/Users/admin/Desktop/KRUS-data-/all_dane/all_data (13).csv"
DATA_PATH = "C:/Users/admin/Desktop/krus-chatbot/AgroBot/data/ustawa_with_paragraph_headers.md"
PERSIST_PATH = "chroma_ustawa"
PERSIST_DIR = "chroma_statystyki"

# Bielik/Pllum
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=False)
except Exception as e:
    print("Fast tokenizer fail → slow:", e)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 
)


gen_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,      
    device_map="cuda:0",               
    low_cpu_mem_usage=False,
    attn_implementation="sdpa",      
    trust_remote_code=True,
).eval()

#----------------------------------------- resources tabelaryczne -----------------------------------------
#EMB TABELI

device = "cpu"

class MXBAIEmbeddings(Embeddings):
    def __init__(self, model_id: str = EMBEDDER_MODEL_T, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.net = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32
        ).to(self.device)
        self.net.eval()

    @torch.inference_mode()
    def _encode(self, texts: List[str], is_query=False, batch_size=32) -> List[List[float]]:
        prefix = "query: " if is_query else "passage: "
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i+batch_size]]
            enc = self.tok(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            last = self.net(**enc).last_hidden_state
            attn = enc["attention_mask"].unsqueeze(-1)
            emb = (last * attn).sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.extend(emb.detach().cpu().tolist())
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]

emb = MXBAIEmbeddings()

persist_dir = PERSIST_DIR
shutil.rmtree(persist_dir, ignore_errors=True)
os.makedirs(persist_dir, exist_ok=True)

vectorstore = Chroma(
    collection_name="statystyki",
    embedding_function=emb,
    persist_directory=PERSIST_DIR
)

#CE TABELE
CE_MODEL_ID = RERANKER_MODEL_T

reranker_ce = None
try:
    reranker_ce = CrossEncoder(CE_MODEL_ID, device=emb.device)
except Exception:
    reranker_ce = None


#----------------------------------------- resources ustawowe -----------------------------------------

class OptimizedONNXCrossEncoder:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        providers = ["CPUExecutionProvider"]
        
        try:
            self.model = ORTModelForSequenceClassification.from_pretrained(
                model_name,
                provider=providers[0],
                export=True
            )
            print("ONNX Cross Encoder załadowany na CPU!")
        except Exception as e:
            print(f"Fallback do zwykłego CrossEncoder na CPU: {e}")
            from sentence_transformers import CrossEncoder
            self.fallback_model = CrossEncoder(model_name, device="cpu")
            self.model = None
    
    def predict(self, pairs, batch_size=32, **kwargs):
        if self.model is None:
            return self.fallback_model.predict(pairs, batch_size=batch_size)
        
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            texts_1 = [p[0] for p in batch]
            texts_2 = [p[1] for p in batch]
            
            inputs = self.tokenizer(
                texts_1, texts_2,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if outputs.logits.shape[-1] == 2:
                    scores = torch.softmax(outputs.logits, dim=-1)[:, 1]
                else:
                    scores = outputs.logits[:, 0]
                scores = scores.cpu().numpy()
                all_scores.extend(scores.tolist())
        
        return np.array(all_scores)

#CE USTAWA
cross_encoder_ustawa = OptimizedONNXCrossEncoder(RERANKER_MODEL_U, device=device)
from sentence_transformers import CrossEncoder

# EMB USTAWA
embedder_u = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDER_MODEL_U,
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# chromaDB dla ustawy
persist_path = PERSIST_PATH

shutil.rmtree(persist_path, ignore_errors=True)
os.makedirs(persist_path, exist_ok=True)

docs_raw = TextLoader(DATA_PATH, encoding="utf-8").load()
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "ustep")])
chunks = header_splitter.split_text(docs_raw[0].page_content)

# meta: <!-- chapter:1 article:16a paragraph:3 id:ch1-art16a-ust3 -->
meta_re = re.compile(
    r"<!--\s*chapter\s*:\s*(\d+)\s+article\s*:\s*([0-9a-z]+)\s+paragraph\s*:\s*([0-9a-z]+)\s+id\s*:\s*([^\s>]+)\s*-->",
    re.I
)

normed: List[Document] = []
for d in chunks:
    md = dict(d.metadata)
    header_text = md.get("ustep", "") or ""
    source_for_meta = header_text + "\n" + d.page_content
    m = meta_re.search(source_for_meta)
    if m:
        md["chapter"]   = int(m.group(1))
        md["article"]   = m.group(2).lower()
        md["paragraph"] = m.group(3).lower()
        md["id"]        = m.group(4)
        md["rozdzial"]  = md["chapter"]
        md["artykul"]   = md["article"]
        md["ust"]       = md["paragraph"]

    clean_header = meta_re.sub("", header_text).strip()
    if clean_header:
        md["ustep"] = clean_header

    content = meta_re.sub("", d.page_content).strip()
    normed.append(Document(page_content=content, metadata=md))

print(f"[CHROMA] Liczba ustępów: {len(normed)}")

db = Chroma.from_documents(
    documents=normed,
    embedding=embedder_u,
    persist_directory=persist_path,
    collection_metadata={"hnsw:space": "cosine"}
)

