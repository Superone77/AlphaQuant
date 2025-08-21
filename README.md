# SpinQuant-Mini

### 1) Quantize & calibrate

```bash
python scripts/quantize_model.py \
  --model meta-llama/Llama-3.2-1B \
  --wq mxfp4 --aq mxfp8 --group 64 \
  --include q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
  --calib_ds wikitext --calib_split wikitext-2-raw-v1 --calib_batches 16 \
  --save ./llama32_1b_mx48_g64
```

### 2) Evaluate with lm-eval
```bash
python scripts/eval_with_lm_eval.py \
  --pretrained ./llama32_1b_mx48_g64 \
  --tasks hellaswag,arc_easy,winogrande \
  --batch_size 8
```