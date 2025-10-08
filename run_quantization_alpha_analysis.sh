#!/bin/bash
# Run quantization-based Alpha-Hill analysis for Llama 3.1 8B

set -e

MODEL="meta-llama/Llama-3.1-8B"
DEVICE="cpu"  # 使用 cuda，如果没有GPU改为 cpu
DTYPE="bf16"   # 使用 bf16 加载，更快且节省内存
QUANT_FORMATS="bf16,mxfp8,mxfp4,fp8,fp4,int8,int6,int4"
OUTPUT_DIR="./results/llama31_8b_quantization_alpha"
K_FRAC=0.1

echo "=================================================="
echo "Quantization Alpha-Hill Analysis"
echo "=================================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Quantization formats: $QUANT_FORMATS"
echo "Output: $OUTPUT_DIR"
echo "=================================================="
echo ""

# 设置 CUDA 设备（如果需要）
export CUDA_VISIBLE_DEVICES=0

# 运行分析
python scripts/analyze_quantization_alpha_hill.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --quant-formats "$QUANT_FORMATS" \
  --k-frac $K_FRAC \
  --output-dir "$OUTPUT_DIR" \
  --log-level INFO

echo ""
echo "=================================================="
echo "分析完成！"
echo "=================================================="
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo "  - CSV: quantization_alpha_results.csv"
echo "  - 图表: plots/ 目录"
echo ""
echo "查看结果："
echo "  cat $OUTPUT_DIR/quantization_alpha_results.csv"
echo "  ls $OUTPUT_DIR/plots/"
echo ""
echo "=================================================="

