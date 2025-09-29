#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EBSS-style balanced self-sampling for OLMoE-1B-7B-0125
- Generates a calibration dataset that is (a) low-perplexity (probability-guided)
  and (b) balanced across experts (penalize stddev of expert-usage counts).
- Implements a lightweight beam-style expansion with an expert-balance term.

Recommended GPU + torch.bfloat16/float16.

References:
- OLMoE model doc (router probs / num_experts): https://huggingface.co/docs/transformers/main/model_doc/olmoe
- Model card: allenai/OLMoE-1B-7B-0125 on Hugging Face
"""

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class BeamState:
    input_ids: torch.Tensor           # (1, L)
    past_key_values: Dict | None      # cache
    logprob_sum: float                # sum of log p
    length: int
    # copy of global expert counts after taking this beam's path
    expert_counts: torch.Tensor       # (num_experts,)

def std_penalty(expert_counts: torch.Tensor) -> float:
    # sigma = stddev over experts (scalar)
    return float(torch.std(expert_counts.float()).item())

def update_counts_with_router_probs(
    counts: torch.Tensor,
    router_probs_per_layer: List[torch.Tensor],
    topk: int = 1,
) -> torch.Tensor:
    """
    Update expert-usage counts with the current token's router probabilities.

    router_probs_per_layer: list of tensors with shape (1, 1, num_experts) for this step
    Strategy:
      - For each MoE layer, we add the *expected* usage: probs if you want soft,
        or add 1 to the argmax expert if you want hard. Here we use soft-add (expected),
        which is smoother and aligns with balancing.
    """
    new_counts = counts.clone()
    for probs in router_probs_per_layer:
        # probs: (batch=1, seq=1, num_experts)
        p = probs[0, 0]  # (num_experts,)
        new_counts[: len(p)] += p.detach().cpu()
    return new_counts

def choose_device(dtype: str):
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------------
# Core EBSS Sampler
# -----------------------------

class EBSSSampler:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        attn_impl: str = "sdpa",
        tau: float = 1.2,
        beam_size: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 42,
        device: str = None,
    ):
        set_seed(seed)

        self.device = choose_device(dtype) if device is None else torch.device(device)
        self.tau = tau
        self.beam_size = beam_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Some OLMoE cards don't define BOS; seed with a space by default
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        torch_dtype = torch.bfloat16 if dtype in ["bf16", "bfloat16"] else torch.float16
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation=attn_impl,
            quantization_config=quantization_config,
        )
        self.model.eval()

        # ensure router probs are returned
        self.model.config.output_router_probs = True
        self.model.config.add_router_probs = True

        # read num_experts from config
        self.num_experts = getattr(self.model.config, "num_experts", 64)

        # global expert counts across the dataset we are building
        self.global_counts = torch.zeros(self.num_experts, dtype=torch.float32)

    @torch.inference_mode()
    def step_forward(self, input_ids: torch.Tensor, past_key_values=None):
        """
        Run one forward step on the last token to get next-token logits
        and router probs for this step (list over MoE layers).
        """
        out = self.model(
            input_ids=input_ids.to(self.model.device),
            use_cache=True,
            past_key_values=past_key_values,
            output_router_probs=True,
        )

        logits = out.logits[:, -1, :]  # (1, vocab)
        # router_probs: tuple(len = num_layers) of (1, seq_len, num_experts)
        router_probs_layers = out.router_probs if hasattr(out, "router_probs") else None
        # keep only probs for the *last step* (seq_len-1 -> last)
        step_router_probs = []
        if router_probs_layers is not None:
            for rp in router_probs_layers:
                # rp: (1, seq_len, num_experts)
                step_router_probs.append(rp[:, -1:, :].detach())
        return logits, out.past_key_values, step_router_probs

    def expand_beam(
        self,
        beam: BeamState,
        k: int,
    ) -> List[BeamState]:
        """
        Expand one beam by top-k sampling with temperature/top_p, then score with:
            score = -avg_nll  +  (sigma(global_counts_after)/tau) - (sigma(global_counts_before)/tau)
        Equivalent to penalizing increase in imbalance.
        """
        logits, pkv, step_router_probs = self.step_forward(beam.input_ids, beam.past_key_values)

        # nucleus sampling candidates (top_p) + temperature
        probs = torch.softmax(logits / max(1e-6, self.temperature), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumsum <= self.top_p).sum().item()
        cutoff = max(cutoff, k)  # ensure at least k
        candidate_ids = sorted_ids[:, :cutoff][0]

        # pick top-k by probability for expansion (pre-score filter)
        topk_ids = candidate_ids[:k]

        new_beams: List[BeamState] = []
        for tok_id in topk_ids:
            tok_id = tok_id.view(1, 1)
            logp = float(torch.log(probs[0, tok_id.item()] + 1e-12))

            new_input_ids = torch.cat([beam.input_ids.to(self.model.device), tok_id.to(self.model.device)], dim=1)
            new_len = beam.length + 1
            new_logprob_sum = beam.logprob_sum + logp

            # update expert counts (soft add by router probs of this step)
            new_counts = update_counts_with_router_probs(beam.expert_counts, step_router_probs, topk=1)

            # compute marginal imbalance penalty (only the increment)
            before_sigma = std_penalty(beam.expert_counts)
            after_sigma  = std_penalty(new_counts)
            imbalance_penalty = (after_sigma - before_sigma) / max(self.tau, 1e-6)

            # average NLL term
            avg_nll = - new_logprob_sum / new_len
            # final score: lower is better
            total_score = avg_nll + imbalance_penalty

            new_beams.append(
                BeamState(
                    input_ids=new_input_ids,
                    past_key_values=pkv,  # reuse pkv from the same forward (speed)
                    logprob_sum=new_logprob_sum,
                    length=new_len,
                    expert_counts=new_counts,
                )
            )
        # return k beams with best (lowest) score
        new_beams.sort(key=lambda b: (-b.logprob_sum / b.length) + (std_penalty(b.expert_counts) - std_penalty(beam.expert_counts)) / max(self.tau, 1e-6))
        return new_beams[:k]

    @torch.inference_mode()
    def generate_one(self, prompt_ids: torch.Tensor) -> str:
        """
        Generate one EBSS sample from a prompt.
        """
        # init beam
        init_counts = self.global_counts.clone()
        # warmup one forward to build cache for prompt
        logits, pkv, rp_layers = self.step_forward(prompt_ids)
        # we don't add counts for the prompt token routing (optional); keep counts as global

        init_beam = BeamState(
            input_ids=prompt_ids.to(self.model.device),
            past_key_values=pkv,
            logprob_sum=0.0,
            length=0,
            expert_counts=init_counts,
        )
        beams = [init_beam]

        for _ in range(self.max_new_tokens):
            candidates: List[BeamState] = []
            for b in beams:
                candidates.extend(self.expand_beam(b, k=self.beam_size))
            # keep best beams
            # score = avg_nll + sigma_increment/tau
            candidates.sort(key=lambda b: (-b.logprob_sum / max(1, b.length)) + (std_penalty(b.expert_counts) - std_penalty(self.global_counts)) / max(self.tau, 1e-6))
            beams = candidates[: self.beam_size]

            # early stop if EOS dominates the best beam
            best = beams[0]
            if best.input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break

        best = beams[0]
        # commit best sample & update global expert counts with its final counts diff
        self.global_counts = best.expert_counts

        text = self.tokenizer.decode(best.input_ids[0], skip_special_tokens=True)
        return text.strip()

    def make_prompts(self, n: int) -> List[torch.Tensor]:
        """
        Build simple neutral prompts to start self-sampling.
        If BOS token is undefined, we seed with a single space or lightweight stems.
        """
        seeds = [
            " ", "In a recent discussion,", "The following story begins with",
            "Let’s consider the idea that", "Meanwhile,", "A simple definition is that",
            "Researchers observed that", "As a reminder,", "In summary,"
        ]
        prompts = random.sample(seeds, k=min(n, len(seeds))) + [random.choice(seeds) for _ in range(max(0, n - len(seeds)))]
        tensors = []
        for s in prompts:
            toks = self.tokenizer(s, return_tensors="pt").input_ids.to(self.model.device)
            tensors.append(toks)
        return tensors

    def build_dataset(
        self,
        out_path: str,
        num_samples: int = 256,
    ):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        prompts = self.make_prompts(num_samples)

        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(num_samples):
                text = self.generate_one(prompts[i])
                # basic cleaning
                text = " ".join(text.split())
                record = {"id": i, "text": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"[EBSS] generated {i+1}/{num_samples} samples | sigma={std_penalty(self.global_counts):.4f}")

        print(f"[EBSS] Done. Saved to: {out_path}")
        print(f"[EBSS] Final expert-balance sigma: {std_penalty(self.global_counts):.6f}")
        # Optional: show top/bottom experts to inspect balance
        counts = self.global_counts.numpy().tolist()
        print(f"[EBSS] Expert counts (sum over layers/steps, soft): {counts}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125",
                   help="HF model id")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--load_in_4bit", action="store_true", help="optional weight-only 4bit for memory")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--tau", type=float, default=1.2, help="temperature for balance penalty")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="ebss_olmoe_0125.jsonl")
    return p.parse_args()

def main():
    args = parse_args()
    sampler = EBSSSampler(
        model_name=args.model,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        attn_impl=args.attn_impl,
        tau=args.tau,
        beam_size=args.beam_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    sampler.build_dataset(out_path=args.out, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
