#!/bin/bash

# Example script to demonstrate group quantization
# This script shows how to use the new group quantization feature

echo "======================================================================"
echo "Group Quantization Example Script"
echo "======================================================================"
echo ""
echo "This script demonstrates the new group quantization feature that has"
echo "been added to INT and FP quantizers (excluding MX types)."
echo ""
echo "Group quantization divides tensors into groups and computes separate"
echo "quantization parameters for each group, improving accuracy for weights"
echo "with varying magnitudes across different regions."
echo ""
echo "Key parameters:"
echo "  - use_group_quant: Enable/disable group quantization (default: False)"
echo "  - group_size: Number of elements per group (default: 128)"
echo ""
echo "======================================================================"
echo ""

# Run the Python example script
python example_group_quantization.py

echo ""
echo "======================================================================"
echo "Example completed!"
echo ""
echo "You can also integrate group quantization into your own scripts:"
echo ""
echo "Python example:"
echo "  from alphaquant.quantizers.int_quantizers import INT4Config, INT4Quantizer"
echo ""
echo "  # Enable group quantization with custom group size"
echo "  cfg = INT4Config(use_group_quant=True, group_size=256)"
echo "  quantizer = INT4Quantizer(cfg)"
echo "  quantized_weight = quantizer.quantize_weight(weight_tensor)"
echo ""
echo "======================================================================"

