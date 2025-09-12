import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from config import MODEL_NAME

def load_llm():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok
