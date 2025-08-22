
import os, re, csv, math, time, logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False
    print("Transformers not available. Please `pip install transformers`.")

import pandas as pd

def setup_logging(level: str = "INFO"):
    level = level.upper()
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.info(f"Log level set to {level}")

def torch_dtype_from_str(s: str):
    s = s.lower()
    if s == "float32":
        return torch.float32
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "auto":
        return None
    raise ValueError(f"Unknown dtype: {s}")

# @title Alpha-Hill computation (robust version)
from typing import List

@torch.no_grad()
def alpha_hill_from_weight(
    W: torch.Tensor,
    k: Optional[int] = None,
    k_frac: float = 0.1,
    eps: float = 1e-12,
    use_lowrank: bool = False,
    q_mult: float = 2.0,
    force_cpu_svd: bool = True,
    use_eig: bool = False,
) -> Tuple[float, int, int, str]:
    """Compute PL_Alpha_Hill for a weight tensor W.

    Returns:
        alpha: float (nan if not computable)
        k_used: int
        n_eigs_used: int
        method: str  in {"svd_full", "svd_lowrank", "eig_full", "svd_full_fallback"}
    """
    # Ensure dense & 2D
    if W.is_sparse:
        W = W.to_dense()
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)

    m, n = W.shape
    min_dim = min(m, n)
    if min_dim < 2:
        return float("nan"), 1, min_dim, "degenerate"

    # Prefer SVD; compute in float32 on CPU for stability by default
    if use_eig:
        X = (W.to(dtype=torch.float32, device="cpu").T @ W.to(dtype=torch.float32, device="cpu"))
        X = (X + X.T) * 0.5
        try:
            lam = torch.linalg.eigvalsh(X)
            method = "eig_full"
        except Exception:
            s = torch.linalg.svdvals(W.to(dtype=torch.float32, device="cpu"))
            lam = (s ** 2)
            method = "svd_full_fallback"
    else:
        method = "svd_full"
        W_ = W.to(dtype=torch.float32, device="cpu") if force_cpu_svd else W
        try:
            if use_lowrank:
                q_guess = max(32, int(min_dim * max(k_frac, 0.05) * q_mult))
                q_guess = min(q_guess, min_dim - 1)
                if q_guess < 2:
                    s = torch.linalg.svdvals(W.to(dtype=torch.float32, device="cpu"))
                    method = "svd_full"
                else:
                    try:
                        s = torch.svd_lowrank(W_.to(dtype=torch.float32, device="cpu"), q=q_guess)[1]
                        method = "svd_lowrank"
                    except Exception:
                        s = torch.linalg.svdvals(W.to(dtype=torch.float32, device="cpu"))
                        method = "svd_full_fallback"
            else:
                s = torch.linalg.svdvals(W_.to(dtype=torch.float32, device="cpu"))
                method = "svd_full"
        except Exception:
            s = torch.linalg.svdvals(W.to(dtype=torch.float32, device="cpu"))
            method = "svd_full_fallback"

        lam = (s ** 2)

    lam, _ = torch.sort(lam)
    n_eigs = lam.numel()
    if n_eigs < 2:
        return float("nan"), 1, n_eigs, method

    if k is None:
        k_used = max(10, int(n_eigs * k_frac))
    else:
        k_used = k
    k_used = max(1, min(k_used, n_eigs - 1))

    eps_t = torch.tensor(eps, dtype=lam.dtype, device=lam.device)
    lam_ref = torch.clamp(lam[-k_used-1], min=eps_t)  # λ_{n-k}
    top = lam[-k_used:]
    denom = torch.log(top / lam_ref).sum().clamp_min(eps_t)
    alpha = float(1.0 + (k_used / float(denom)))
    return alpha, k_used, n_eigs, method


def categorize(name: str) -> str:
    low = name.lower()
    if any(x in low for x in [".q_proj", "q_proj", ".q.", "query_proj"]):
        return "attn_q"
    if any(x in low for x in [".k_proj", "k_proj", ".k.", "key_proj"]):
        return "attn_k"
    if any(x in low for x in [".v_proj", "v_proj", ".v.", "value_proj"]):
        return "attn_v"
    if any(x in low for x in [".o_proj", "o_proj", "out_proj", "proj_out"]):
        return "attn_o"

    if any(x in low for x in ["up_proj", "w1", "fc1", "dense_h_to_4h", "mlp_up"]):
        return "mlp_up"
    if any(x in low for x in ["gate_proj", "mlp_gate"]):
        return "mlp_gate"
    if any(x in low for x in ["down_proj", "w2", "fc2", "dense_4h_to_h", "mlp_down"]):
        return "mlp_down"

    if any(x in low for x in ["router", "route", "gate", "gating"]):
        return "router_or_gate"

    if "expert" in low or "experts" in low:
        return "expert_linear"

    return "other_linear"


def iter_linear_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def safe_get_in_out_features(module: nn.Linear):
    try:
        return int(module.out_features), int(module.in_features)
    except Exception:
        w = module.weight
        return int(w.shape[0]), int(w.shape[1])


def alpha_hill_from_model(
    W: torch.Tensor,
    k: Optional[int] = None,
    k_frac: float = 0.1,
    eps: float = 1e-12,
    use_lowrank: bool = False,
    q_mult: float = 2.0,
    force_cpu_svd: bool = True,
    use_eig: bool = False,
) -> Tuple[float, int, int, str]:
    """Compute PL_Alpha_Hill for all weights in a model.

    Returns:
        alpha: float (nan if not computable)
        k_used: int
        n_eigs_used: int
        method: str  in {"svd_full", "svd_lowrank", "eig_full", "svd_full_fallback"}
    """

    



if __name__ == '__main__':

    # @title Parameters
    MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"  # @param {type:"string"}
    CACHE_DIR = "/local/mnt/workspace/wanqi/tmp/" # @param {type:"string"}
    REVISION = None                         # @param {type:"string"}
    DEVICE = "cuda"                          # @param ["cpu", "cuda"]
    DTYPE = "auto"                          # @param ["auto", "float32", "float16", "bfloat16"]
    OUT_CSV = "/local/mnt/workspace/wanqi/tmp/Alpha_Hill/pl_alpha_hill.csv"           # @param {type:"string"}

    K = None                                # @param {type:"integer"}
    K_FRAC = 0.10                           # @param {type:"number"}
    FORCE_CPU_SVD = True                    # @param {type:"boolean"}
    LOWRANK = False                         # @param {type:"boolean"}
    Q_MULT = 2.0                            # @param {type:"number"}
    USE_EIG = False                         # @param {type:"boolean"}
    FILTER_REGEX = None                     # @param {type:"string"}
    LOG_LEVEL = "INFO"                      # @param ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]

    setup_logging(LOG_LEVEL)
    print(f"MODEL_ID={MODEL_ID}, DEVICE={DEVICE}, DTYPE={DTYPE}, OUT_CSV={OUT_CSV}")

    # @title Load model
    assert HF_AVAILABLE, "Transformers is not available. Please install it."

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    logging.info(f"Loaded config: {cfg.__class__.__name__}")

    dtype = torch_dtype_from_str(DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
        cache_dir=CACHE_DIR
    )
    model.eval()
    if DEVICE == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        logging.info("Model moved to CUDA.")
    else:
        model.to("cpu")
        logging.info("Model on CPU.")

    # @title Run: compute PL_Alpha_Hill for every nn.Linear and save CSV
    regex = re.compile(FILTER_REGEX) if FILTER_REGEX else None
    rows = []

    header = [
        "name","category","shape","out_features","in_features","numel",
        "dtype","device","alpha_hill","k_used","k_frac","n_eigs","method",
        "elapsed_ms","error"
    ]

    total = ok = failed = 0
    t_start = time.perf_counter()

    for name, mod in iter_linear_modules(model):
        if regex and not regex.search(name):
            continue
        total += 1
        w = mod.weight.detach() if hasattr(mod, "weight") and mod.weight is not None else None
        if w is None:
            rows.append([name, categorize(name), "NA","NA","NA","NA","NA","NA","nan","NA",K_FRAC,"NA","NA","0.0","no_weight"])
            failed += 1
            continue

        w = w.contiguous()
        out_f, in_f = safe_get_in_out_features(mod)
        shape_str = f"[{out_f}, {in_f}]"
        numel = int(w.numel())
        dtype_str = str(w.dtype).replace("torch.", "")
        device_str = str(w.device)

        t0 = time.perf_counter()
        error = ""
        try:
            alpha, k_used, n_eigs, method = alpha_hill_from_weight(
                w, k=K, k_frac=K_FRAC, eps=1e-12,
                use_lowrank=LOWRANK, q_mult=Q_MULT,
                force_cpu_svd=FORCE_CPU_SVD, use_eig=USE_EIG,
            )
            ok += 1
        except Exception as e:
            alpha, k_used, n_eigs, method = float("nan"), K or -1, -1, "failed"
            error = f"{type(e).__name__}: {e}"
            failed += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append([
            name, categorize(name), shape_str, out_f, in_f, numel,
            dtype_str, device_str,
            f"{alpha:.6f}" if isinstance(alpha, float) and math.isfinite(alpha) else str(alpha),
            k_used, K_FRAC, n_eigs, method, f"{elapsed_ms:.3f}", error
        ])

    elapsed_total = (time.perf_counter() - t_start)
    print(f"Done. total={total}, ok={ok}, failed={failed}, time={elapsed_total:.2f}s")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(OUT_CSV, index=False)
    print(f"CSV saved to: {os.path.abspath(OUT_CSV)}")

    # Show a preview
    df.head(20)

    # @title Optional: summarize by category
    import pandas as pd
    if 'df' in locals():
        summary = (
            df[df['alpha_hill'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
            .assign(alpha_hill=lambda d: d['alpha_hill'].astype(float))
            .groupby('category')['alpha_hill']
            .agg(['count','mean','median','min','max'])
            .sort_values('mean')
        )
        summary
    else:
        print("No dataframe available.")