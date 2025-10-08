#!/bin/bash
# 从 CSV 生成量化 Alpha-Hill 可视化

set -e

# 配置
CSV_FILE="./results/llama31_8b_quantization_alpha/quantization_alpha_results.csv"
OUTPUT_DIR="./results/llama31_8b_quantization_alpha"
QUANT_FORMATS="bf16,mxfp8,mxfp4,fp8,fp4,int8,int6,int4,int3,int2"

echo "=================================================="
echo "从 CSV 生成可视化"
echo "=================================================="
echo "CSV 文件: $CSV_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "=================================================="
echo ""

# 检查 CSV 文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "错误: CSV 文件不存在: $CSV_FILE"
    echo "请先运行 run_quantization_alpha_analysis.sh 生成数据"
    exit 1
fi

# 运行可视化脚本
python scripts/visualize_quantization_alpha.py \
  --csv "$CSV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --quant-formats "$QUANT_FORMATS"

echo ""
echo "=================================================="
echo "可视化完成！"
echo "=================================================="
echo ""
echo "生成的图表:"
echo "  - alpha_distribution.png      (整体分布箱线图)"
echo "  - alpha_by_category.png       (按类别对比)"
echo "  - alpha_heatmap.png           (热力图)"
echo "  - alpha_variance_analysis.png (方差分析)"
echo "  - plots/                      (单层柱状图)"
echo ""
echo "查看图表："
echo "  open $OUTPUT_DIR/alpha_distribution.png"
echo ""
echo "=================================================="

