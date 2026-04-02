from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_device(preferred: str = "mps") -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_seq2seq_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model
