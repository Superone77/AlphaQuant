# 1) 安装依赖
# pip install -U transformers accelerate bitsandbytes torch --index-url https://download.pytorch.org/whl/cu121

# 2) 运行（建议 BF16 / A100 等显存>=24GB；或加 --load_in_4bit）
# python ebss_build_olmoe_dataset.py \
#   --model allenai/OLMoE-1B-7B-0125 \
#   --dtype bfloat16 \
#   --beam_size 2 \
#   --tau 1.2 \
#   --max_new_tokens 32 \
#   --num_samples 512 \
#   --out ebss_olmoe_0125.jsonl

  python analyze_olmoe_expert_freq_and_channel_ranges.py \
  --model allenai/OLMoE-1B-7B-0125 \
  --ebss_jsonl ebss_olmoe_0125.jsonl \
  --dtype fp16 \
  --seq_len 256 \
  --max_samples 1024 \
  --batch_size 4 \
  --layer_idx 3 \
  --out vis_layer3 \
  --soft_or_hard both