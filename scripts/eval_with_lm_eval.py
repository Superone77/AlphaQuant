#!/usr/bin/env python
from __future__ import annotations
import argparse
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from spinquant_mini import (
    replace_linear_with_quant, QuantLinearConfig,
    MXFP4Quantizer, MXFP4Config,
    MXFP8Quantizer, MXFP8Config,
)

QCLS = {
    'mxfp4': (MXFP4Quantizer, MXFP4Config),
    'mxfp8': (MXFP8Quantizer, MXFP8Config),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--tasks', default='hellaswag')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--include', nargs='*', default=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--wq', choices=QCLS.keys(), default='mxfp4')
    ap.add_argument('--aq', choices=QCLS.keys(), default='mxfp8')
    ap.add_argument('--group', type=int, default=64)
    args = ap.parse_args()

    # Build HFLM first
    lm = HFLM(pretrained=args.pretrained, device=args.device, dtype=args.dtype, batch_size=args.batch_size)

    # Access underlying HF model and inject QuantLinear
    model = lm.model  # HFLM exposes the underlying transformers model
    WQ, WCfg = QCLS[args.wq]
    AQ, ACfg = QCLS[args.aq]
    qcfg = QuantLinearConfig(weight_quantizer_cls=WQ, weight_quantizer_cfg=WCfg(group_size=args.group, dtype=args.dtype),
                             act_quantizer_cls=AQ, act_quantizer_cfg=ACfg(group_size=args.group, dtype=args.dtype))

    replaced = replace_linear_with_quant(model, qcfg, include=args.include, exclude=args.exclude)
    print(f"[lm-eval] Replaced {len(replaced)} Linear layers.")

    # No separate calibration here; for fair eval you may want to run quantize_model.py first and save weights

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
    )
    print(results)

if __name__ == '__main__':
    main()