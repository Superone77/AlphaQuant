#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict

from alphaquant.utils.replacement import load_layer_config, plan_model_layer_schemes, summarize_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize model with layer-wise configuration")
    parser.add_argument("--model", type=str, default=None, help="HF model id or local path (optional in --dry-run)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    parser.add_argument("--layer-config", type=str, required=True, help="Path to JSON defining layer-wise schemes")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan without loading the model")
    parser.add_argument("--save-plan", type=str, default=None, help="Optional path to save the expanded plan as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg: Dict[str, Any] = load_layer_config(args.layer_config)
    print(summarize_config(cfg))

    if args.dry_run and not args.model:
        print("[dry-run] Skipping model load and layer expansion.")
        if args.save_plan:
            with open(args.save_plan, "w") as f:
                json.dump({"summary": summarize_config(cfg)}, f, indent=2)
        return

    # Load model only if requested / needed
    if args.model is None:
        raise SystemExit("--model must be provided unless --dry-run is used")

    # Optional imports to avoid hard deps when dry-running
    from alphaquant.utils.hf_utils import load_hf_causal_lm

    print(f"Loading model: {args.model} (device={args.device}, dtype={args.dtype})")
    model = load_hf_causal_lm(args.model, device=args.device, dtype=args.dtype)

    plan = plan_model_layer_schemes(model, cfg)
    print(f"Planned {len(plan)} target modules.")
    for name, scheme in plan[:50]:
        print(f"  {name}: {scheme}")
    if len(plan) > 50:
        print(f"  ... ({len(plan) - 50} more)")

    if args.save_plan:
        os.makedirs(os.path.dirname(args.save_plan), exist_ok=True) if os.path.dirname(args.save_plan) else None
        with open(args.save_plan, "w") as f:
            json.dump({name: scheme for name, scheme in plan}, f, indent=2)
        print(f"Saved plan to {args.save_plan}")


if __name__ == "__main__":
    main()