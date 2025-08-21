#!/usr/bin/env python
from __future__ import annotations
import argparse, json
import torch
from spinquant_mini import (
    replace_linear_with_quant, QuantLinearConfig,
    MXFP4Quantizer, MXFP4Config,
    MXFP8Quantizer, MXFP8Config,
    Calibrator,
)
from spinquant_mini.utils.hf_utils import load_hf_causal_lm
from datasets import load_dataset

QCLS = {
    'mxfp4': (MXFP4Quantizer, MXFP4Config),
    'mxfp8': (MXFP8Quantizer, MXFP8Config),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--include', nargs='*', default=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--wq', choices=QCLS.keys(), default='mxfp4')
    ap.add_argument('--aq', choices=QCLS.keys(), default='mxfp8')
    ap.add_argument('--group', type=int, default=64)
    ap.add_argument('--calib_ds', default='wikitext', help='HF dataset path')
    ap.add_argument('--calib_split', default='wikitext-2-raw-v1')
    ap.add_argument('--calib_batches', type=int, default=16)
    ap.add_argument('--save', default=None, help='save dir for quantized model (optional, safetensors)')
    args = ap.parse_args()

    model, tok = load_hf_causal_lm(args.model, args.device, args.dtype)

    WQ, WCfg = QCLS[args.wq]
    AQ, ACfg = QCLS[args.aq]
    wcfg = WCfg(group_size=args.group, dtype=args.dtype)
    acfg = ACfg(group_size=args.group, dtype=args.dtype)

    qcfg = QuantLinearConfig(weight_quantizer_cls=WQ, weight_quantizer_cfg=wcfg,
                             act_quantizer_cls=AQ, act_quantizer_cfg=acfg, bias=True)

    replaced = replace_linear_with_quant(model, qcfg, include=args.include, exclude=args.exclude)
    print(f"Replaced {len(replaced)} linear layers. Examples: {replaced[:8]}")

    # calibration dataloader
    ds = load_dataset(args.calib_ds, args.calib_split)
    texts = ds['train']['text'][:args.calib_batches*8]
    toks = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(dict(input_ids=toks['input_ids'], attention_mask=toks['attention_mask']), batch_size=8)

    Calibrator(model).collect(dl, num_batches=args.calib_batches, device=args.device)
    print("Calibration finished.")

    if args.save:
        model.save_pretrained(args.save, safe_serialization=True)
        tok.save_pretrained(args.save)
        print(f"Saved quantized model to {args.save}")

if __name__ == '__main__':
    main()