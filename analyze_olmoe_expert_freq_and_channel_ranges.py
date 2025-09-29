#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------- 全局缓存（按你的习惯保持接口一致） ----------
otc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # output channel (xmax, xmin)
icc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # input channel (xmax, xmin)

# --------- 钩子函数：统计 per-channel 的 max/min ----------
def layer_omax_hook(m, i, o):
    n = m.name
    if not isinstance(o, torch.Tensor):
        return
    if o.ndim == 3:       # (B, T, D)
        xmax = torch.amax(o, dim=(0, 1))
        xmin = torch.amin(o, dim=(0, 1))
    elif o.ndim == 2:     # (B, D)
        xmax = torch.amax(o, dim=0)
        xmin = torch.amin(o, dim=0)
    else:
        return
    if n not in otc:
        otc[n] = (xmax.detach().cpu(), xmin.detach().cpu())
    else:
        otc[n] = (torch.max(otc[n][0], xmax).cpu(), torch.min(otc[n][1], xmin).cpu())

def layer_i0max_hook(m, i, o):
    n = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    x = i[0]
    if x.ndim == 3:
        xmax = torch.amax(x, dim=(0, 1))
        xmin = torch.amin(x, dim=(0, 1))
    elif x.ndim == 2:
        xmax = torch.amax(x, dim=0)
        xmin = torch.amin(x, dim=0)
    else:
        return
    if n not in icc:
        icc[n] = (xmax.detach().cpu(), xmin.detach().cpu())
    else:
        icc[n] = (torch.max(icc[n][0], xmax).cpu(), torch.min(icc[n][1], xmin).cpu())

# --------- 简单的激活采样器（可扩展为缓存样本矩阵） ----------
class ActSampler:
    def __init__(self, cap_rows: int = 32768):
        self.cap_rows = cap_rows
        self.data: Dict[str, torch.Tensor] = {}
    def _append(self, name: str, out: torch.Tensor):
        if not isinstance(out, torch.Tensor):
            return
        if out.ndim == 3:
            B, T, D = out.shape
            mat = out.reshape(B * T, D)
        elif out.ndim == 2:
            mat = out
        else:
            return
        mat = mat.detach().to(torch.float32).cpu()
        cur = self.data.get(name)
        if cur is None:
            self.data[name] = mat[: self.cap_rows]
        else:
            remain = max(0, self.cap_rows - cur.shape[0])
            if remain > 0:
                self.data[name] = torch.cat([cur, mat[:remain]], dim=0)
    def hook(self, m, i, o):
        self._append(m.name, o)

# --------- 遍历一个 Transformer block 内的关键子模块 ----------
def iter_block_modules(block: nn.Module) -> Dict[str, nn.Module]:
    t = {}
    # 下面的字段名称与 OLMoE 的实现接近，若后续版本略有差异，可 print(block) 自查字段
    t['input_layernorm'] = block.input_layernorm
    t['post_attention_layernorm'] = block.post_attention_layernorm
    attn = block.self_attn
    t['self_attn.q_proj'] = attn.q_proj
    t['self_attn.k_proj'] = attn.k_proj
    t['self_attn.v_proj'] = attn.v_proj
    t['self_attn.o_proj'] = attn.o_proj
    mlp = block.mlp
    # 在 OLMoE 中，mlp 是 MoE 结构，包含 router 与 experts；这里只抓常见线性层
    if hasattr(mlp, "gate_proj"): t['mlp.gate_proj'] = mlp.gate_proj
    if hasattr(mlp, "up_proj"):   t['mlp.up_proj']   = mlp.up_proj
    if hasattr(mlp, "down_proj"): t['mlp.down_proj'] = mlp.down_proj
    return t

# --------- 设备/数据类型 ----------
@torch.no_grad()
def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)

def str2dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ['float32', 'fp32']:   return torch.float32
    if s in ['float16', 'fp16']:   return torch.float16
    if s in ['bfloat16', 'bf16']:  return torch.bfloat16
    return torch.float32

# --------- 读取 EBSS 生成的 jsonl 数据，切分成等长片段 ----------
@torch.no_grad()
def load_jsonl_calib(tokenizer: AutoTokenizer, jsonl_path: str, seq_len: int, max_samples: int) -> List[torch.Tensor]:
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("text", "")
            if t:
                texts.append(t)
    # 拼接编码，然后切块
    joined = "\n\n".join(texts)
    toks = tokenizer(joined, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    chunks = []
    i = 0
    while i + seq_len <= toks.numel() and len(chunks) < max_samples:
        chunks.append(toks[i:i + seq_len].clone())
        i += seq_len
    return chunks

# --------- 统计专家激活频率（soft/hard） ----------
@torch.no_grad()
def accumulate_expert_freq(router_probs_layers: List[torch.Tensor],
                           freq_soft: List[torch.Tensor],
                           freq_hard: List[torch.Tensor]):
    """
    router_probs_layers: list over MoE layers; each tensor shape (B, T, E)
    freq_soft[f_l]: (E,), 累加概率和
    freq_hard[f_l]: (E,), 对每个位置取 argmax 专家计数+1
    """
    for l, probs in enumerate(router_probs_layers):
        # (B, T, E)
        p = probs  # already detached outside
        B, T, E = p.shape
        # soft 累积
        freq_soft[l] += p.sum(dim=(0, 1)).cpu()
        # hard 累积
        argmax_ids = torch.argmax(p, dim=-1)  # (B, T)
        binc = torch.bincount(argmax_ids.view(-1).cpu(), minlength=E)
        freq_hard[l] += binc

# --------- 可视化 ----------
def plot_expert_freq(freq: torch.Tensor, title: str, outpath: str):
    """
    freq: (E,)
    """
    plt.figure(figsize=(10, 3.5))
    xs = list(range(freq.numel()))
    ys = freq.float().numpy()
    plt.bar(xs, ys)
    plt.xlabel("Expert ID")
    plt.ylabel("Counts")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_channel_ranges(xmax: torch.Tensor, xmin: torch.Tensor, title: str, outpath: str):
    """
    xmax/xmin: (D,)
    """
    D = xmax.numel()
    xs = list(range(D))
    plt.figure(figsize=(11, 3.5))
    plt.plot(xs, xmax.numpy(), label="max")
    plt.plot(xs, xmin.numpy(), label="min")
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# --------- 主流程：单层（示例：第 4 个 block，layer_idx=3） ----------
@torch.no_grad()
def analyze_one_layer(model_id: str,
                      ebss_jsonl: str,
                      out_dir: str,
                      device: torch.device,
                      dtype: torch.dtype,
                      seq_len: int,
                      max_samples: int,
                      batch_size: int,
                      layer_idx: int = 3,
                      soft_or_hard: str = "both"):

    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    # 让模型返回 router 概率
    model.config.output_router_probs = True
    model.config.add_router_probs = True

    # 读取 EBSS jsonl 数据，切成 (seq_len) 的片段
    chunks = load_jsonl_calib(tok, ebss_jsonl, seq_len=seq_len, max_samples=max_samples)
    if len(chunks) == 0:
        raise RuntimeError("No chunks produced from the jsonl file; check seq_len/max_samples or file content.")

    # 预分配专家频率容器（以第一批前向得到的 MoE 层数和专家数为准）
    # 先跑一个小样，确定 MoE 层数与专家数
    warm = tok("Hello", return_tensors="pt")["input_ids"].to(device)
    wout = model(warm, output_router_probs=True)
    router_layers_warm = [rp.detach().cpu() for rp in wout.router_probs]  # list of (1, L, E)
    num_moe_layers = len(router_layers_warm)
    num_experts = router_layers_warm[0].shape[-1] if num_moe_layers > 0 else getattr(model.config, "num_experts", 0)

    if num_moe_layers == 0 or num_experts == 0:
        raise RuntimeError("This model didn't return router_probs; ensure it's an MoE variant and config enables router probs.")

    freq_soft = [torch.zeros(num_experts, dtype=torch.float32) for _ in range(num_moe_layers)]
    freq_hard = [torch.zeros(num_experts, dtype=torch.float32) for _ in range(num_moe_layers)]

    # ---- 针对指定 block 注册钩子，统计 per-channel min/max ----
    global otc, icc
    otc.clear(); icc.clear()

    block = model.model.layers[layer_idx]
    modules = iter_block_modules(block)
    sampler = ActSampler(cap_rows=32768)  # 如需保存样本，可用 sampler.data

    handles = []
    for name, mod in modules.items():
        layer_name = f"layer_{layer_idx}_{name}"
        mod.name = layer_name
        def combined_hook(m, i, o):
            layer_omax_hook(m, i, o)
            layer_i0max_hook(m, i, o)
            sampler.hook(m, i, o)
        handles.append(mod.register_forward_hook(combined_hook))

    # ---- 批处理跑数据，统计专家频率 + 钩子收集范围 ----
    for i in range(0, len(chunks), batch_size):
        batch_ids = chunks[i:i + batch_size]
        max_len = max(x.numel() for x in batch_ids)
        batch = torch.stack([
            torch.nn.functional.pad(x, (0, max_len - x.numel()), value=tok.pad_token_id)
            for x in batch_ids
        ], dim=0).to(device)

        out = model(batch, output_router_probs=True)
        router_layers = [rp.detach() for rp in out.router_probs]  # list of (B, L, E)
        accumulate_expert_freq(router_layers, freq_soft, freq_hard)

        # 显存紧张时可以：
        del out
        torch.cuda.empty_cache() if device.type == "cuda" else None

    for h in handles:
        h.remove()

    # ---- 保存专家频率与通道范围 ----
    torch.save({
        "freq_soft": [t.cpu() for t in freq_soft],
        "freq_hard": [t.cpu() for t in freq_hard],
        "oc_stats": {k: (v[0].cpu(), v[1].cpu()) for k, v in otc.items()},
        "ic_stats": {k: (v[0].cpu(), v[1].cpu()) for k, v in icc.items()},
        "num_moe_layers": num_moe_layers,
        "num_experts": num_experts,
        "layer_idx": layer_idx,
        "modules": list(modules.keys())
    }, os.path.join(out_dir, f"layer{layer_idx}_stats.pt"))

    # ---- 可视化（当前示例：只画本层） ----
    # 专家频率（soft/hard）
    if soft_or_hard in ("both", "soft"):
        for l in range(num_moe_layers):
            plot_expert_freq(freq_soft[l], title=f"Layer-{l} Expert Freq (soft)", 
                             outpath=os.path.join(out_dir, f"experts_layer{l}_soft.png"))
    if soft_or_hard in ("both", "hard"):
        for l in range(num_moe_layers):
            plot_expert_freq(freq_hard[l], title=f"Layer-{l} Expert Freq (hard)",
                             outpath=os.path.join(out_dir, f"experts_layer{l}_hard.png"))

    # 针对本 block 的各个子模块，画 per-channel min/max
    for name, (xmax, xmin) in otc.items():
        plot_channel_ranges(xmax, xmin,
                            title=f"{name} OUTPUT channel ranges",
                            outpath=os.path.join(out_dir, f"{name.replace('.', '_')}_out_range.png"))
    for name, (xmax, xmin) in icc.items():
        plot_channel_ranges(xmax, xmin,
                            title=f"{name} INPUT channel ranges",
                            outpath=os.path.join(out_dir, f"{name.replace('.', '_')}_in_range.png"))

    print(f"[Done] Saved results & plots to: {out_dir}")

# --------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125")
    p.add_argument("--ebss_jsonl", type=str, required=True, help="EBSS 生成的数据集 jsonl 路径")
    p.add_argument("--out", type=str, default="olmoe_stats_vis")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--layer_idx", type=int, default=3, help="要分析的 block 索引（示例：3 即第4层）")
    p.add_argument("--soft_or_hard", type=str, default="both", choices=["soft", "hard", "both"])
    return p.parse_args()

def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = str2dtype(args.dtype)
    analyze_one_layer(
        model_id=args.model,
        ebss_jsonl=args.ebss_jsonl,
        out_dir=args.out,
        device=device,
        dtype=dtype,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        layer_idx=args.layer_idx,
        soft_or_hard=args.soft_or_hard
    )

if __name__ == "__main__":
    main()
