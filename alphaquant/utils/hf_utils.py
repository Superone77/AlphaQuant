from __future__ import annotations
from typing import Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def try_import_transformers() -> Tuple[Any, Any]:  # (transformers, torch)
    try:
        import transformers  # type: ignore
    except Exception as exc:
        raise RuntimeError("transformers is required for model loading; install it or run with --dry-run") from exc
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for model loading; install it or run with --dry-run") from exc
    return transformers, torch


def load_hf_causal_lm(model_id: str, device: str = "cpu", dtype: str = "fp32") -> Any:
    transformers, torch = try_import_transformers()
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device if device != "cpu" else None,
    )
    return model


def list_module_names(model: Any) -> list:
    return [name for name, _ in model.named_modules()]