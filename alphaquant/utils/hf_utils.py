from __future__ import annotations
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_causal_lm(name_or_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=getattr(torch, dtype), device_map="auto")
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok