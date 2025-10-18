#!/usr/bin/env python3
"""
自适应量化搜索算法

基于 Alpha_Hill 重要性的量化阈值搜索，自动平衡模型性能和压缩率。
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquant.alpha_hill.utils import alpha_hill_from_model, setup_logging
from alphaquant.utils.hf_utils import load_hf_causal_lm
from scripts.calculate_avg_bits import calculate_average_bits


class AdaptiveQuantizationSearch:
    """自适应量化搜索器"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cpu",
                 dtype: str = "bf16",
                 ppl_threshold: float = 0.1,
                 target_avg_bits: float = 8.0,
                 step_size: float = 0.5,
                 batch_size: int = 1):
        """
        初始化搜索器
        
        Args:
            model_path: 模型路径或ID
            device: 设备类型
            dtype: 数据类型
            ppl_threshold: PPL升高的可接受百分比 (0.1 = 10%)
            target_avg_bits: 目标平均bit数
            step_size: 阈值搜索步长
            batch_size: 评估时的batch size
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.ppl_threshold = ppl_threshold
        self.target_avg_bits = target_avg_bits
        self.step_size = step_size
        self.batch_size = batch_size
        
        self.model = None
        self.alpha_results = None
        self.baseline_ppl = None
        self.search_history = []
        
        # 量化格式对应的bit数
        self.bit_mapping = {
            "mxfp4": 4.25,
            "mxfp8": 8.25,
            "bf16": 16.0
        }
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        self.model = load_hf_causal_lm(self.model_path, device=self.device, dtype=self.dtype)
        self.model.eval()
        print("模型加载完成")
    
    def compute_alpha_hill(self):
        """计算所有层的 Alpha_Hill 值"""
        print("计算 Alpha_Hill 值...")
        self.alpha_results = alpha_hill_from_model(
            self.model,
            k_frac=0.1,
            force_cpu_svd=True,
            use_lowrank=False
        )
        print(f"计算完成，共 {len(self.alpha_results)} 层")
        
        # 过滤有效结果并排序
        valid_results = {}
        for name, result in self.alpha_results.items():
            if isinstance(result.get('alpha'), (int, float)) and np.isfinite(result['alpha']):
                valid_results[name] = result
        
        # 按 alpha 值降序排序
        sorted_layers = sorted(valid_results.items(), 
                              key=lambda x: x[1]['alpha'], reverse=True)
        
        return sorted_layers
    
    def evaluate_ppl(self, config: Dict[str, Any]) -> float:
        """评估模型在 wikitext 任务上的 PPL"""
        try:
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
            
            # 创建量化配置
            quant_config = self.create_quantization_config(config)
            
            # 应用量化
            from alphaquant.utils.replacement import apply_layer_wise_quantization
            quantized_model = apply_layer_wise_quantization(self.model, quant_config, self.dtype)
            
            # 检查量化后的模型
            if quantized_model is None:
                print("警告: 量化后的模型为 None，使用原始模型")
                quantized_model = self.model
            
            # 确保模型在正确的设备上
            if hasattr(quantized_model, 'to'):
                quantized_model = quantized_model.to(self.device)
            
            # 获取模型的 tokenizer
            tokenizer = None
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'tokenizer'):
                tokenizer = self.model.config.tokenizer
            elif hasattr(self.model, 'tokenizer'):
                tokenizer = self.model.tokenizer
            
            # 创建 HFLM 包装器
            try:
                lm = HFLM(
                    pretrained=None, 
                    model=quantized_model, 
                    tokenizer=tokenizer,
                    dtype=self.dtype, 
                    batch_size=self.batch_size
                )
            except Exception as hflm_error:
                print(f"HFLM 初始化失败: {hflm_error}")
                # 尝试使用更简单的初始化方式
                lm = HFLM(
                    pretrained=None, 
                    model=quantized_model, 
                    batch_size=self.batch_size
                )
            
            # 运行 wikitext 评估
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=['wikitext'],
                batch_size=self.batch_size,
                device=self.device
            )
            
            # 提取 PPL
            ppl = results['results']['wikitext']['word_perplexity']
            return ppl
            
        except Exception as e:
            print(f"PPL 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def create_quantization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建量化配置"""
        quant_config = {}
        
        for layer_name, layer_config in config.items():
            if "lm_head" in layer_name or "mlp.gate" in layer_name:
                # 保持 bf16
                quant_config[layer_name] = {"skip": True}
            else:
                quant_config[layer_name] = layer_config
        
        return quant_config
    
    def calculate_current_avg_bits(self, config: Dict[str, Any]) -> float:
        """计算当前配置的平均 bit 数"""
        # 获取模型参数数量
        model_params = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_params = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    weight_params += module.bias.numel()
                model_params[name] = weight_params
        
        # 计算平均 bit 数
        results = calculate_average_bits(config, model_params)
        return results['average_bits']
    
    def search_mxfp4_threshold(self, sorted_layers: List[Tuple[str, Dict]], 
                               baseline_ppl: float) -> Tuple[float, Dict[str, Any]]:
        """搜索 mxfp4 阈值"""
        print("\n开始搜索 mxfp4 阈值...")
        
        max_alpha = max(layer[1]['alpha'] for layer in sorted_layers)
        current_threshold = max_alpha
        
        while current_threshold > 0:
            # 创建当前配置
            config = self.create_search_config(sorted_layers, current_threshold, None)
            
            # 计算平均 bit 数
            avg_bits = self.calculate_current_avg_bits(config)
            
            # 评估 PPL
            current_ppl = self.evaluate_ppl(config)
            ppl_increase = (current_ppl - baseline_ppl) / baseline_ppl
            
            print(f"mxfp4 阈值: {current_threshold:.3f}, "
                  f"平均 bit: {avg_bits:.2f}, "
                  f"PPL: {current_ppl:.3f}, "
                  f"PPL 增加: {ppl_increase:.1%}")
            
            # 记录搜索历史
            self.search_history.append({
                'stage': 'mxfp4_search',
                'threshold': current_threshold,
                'avg_bits': avg_bits,
                'ppl': current_ppl,
                'ppl_increase': ppl_increase,
                'config': config.copy()
            })
            
            # 检查是否达到目标
            if avg_bits <= self.target_avg_bits:
                print(f"达到目标平均 bit 数: {avg_bits:.2f}")
                return current_threshold, config
            
            # 检查 PPL 是否超出阈值
            if ppl_increase > self.ppl_threshold:
                print(f"PPL 增加超出阈值: {ppl_increase:.1%} > {self.ppl_threshold:.1%}")
                return current_threshold + self.step_size, config
            
            # 降低阈值
            current_threshold -= self.step_size
        
        # 如果阈值降到 0 以下，返回 0
        return 0.0, self.create_search_config(sorted_layers, 0.0, None)
    
    def search_mxfp8_threshold(self, sorted_layers: List[Tuple[str, Dict]], 
                               mxfp4_threshold: float,
                               baseline_ppl: float) -> Tuple[float, Dict[str, Any]]:
        """搜索 mxfp8 阈值"""
        print(f"\n开始搜索 mxfp8 阈值 (mxfp4 阈值: {mxfp4_threshold:.3f})...")
        
        max_alpha = max(layer[1]['alpha'] for layer in sorted_layers)
        current_threshold = max_alpha
        
        while current_threshold > 0:
            # 创建当前配置
            config = self.create_search_config(sorted_layers, mxfp4_threshold, current_threshold)
            
            # 计算平均 bit 数
            avg_bits = self.calculate_current_avg_bits(config)
            
            # 评估 PPL
            current_ppl = self.evaluate_ppl(config)
            ppl_increase = (current_ppl - baseline_ppl) / baseline_ppl
            
            print(f"mxfp8 阈值: {current_threshold:.3f}, "
                  f"平均 bit: {avg_bits:.2f}, "
                  f"PPL: {current_ppl:.3f}, "
                  f"PPL 增加: {ppl_increase:.1%}")
            
            # 记录搜索历史
            self.search_history.append({
                'stage': 'mxfp8_search',
                'threshold': current_threshold,
                'avg_bits': avg_bits,
                'ppl': current_ppl,
                'ppl_increase': ppl_increase,
                'config': config.copy()
            })
            
            # 检查是否达到目标
            if avg_bits <= self.target_avg_bits:
                print(f"达到目标平均 bit 数: {avg_bits:.2f}")
                return current_threshold, config
            
            # 检查 PPL 是否超出阈值
            if ppl_increase > self.ppl_threshold:
                print(f"PPL 增加超出阈值: {ppl_increase:.1%} > {self.ppl_threshold:.1%}")
                return current_threshold + self.step_size, config
            
            # 检查是否低于最小 alpha 值
            min_alpha = min(layer[1]['alpha'] for layer in sorted_layers)
            if current_threshold <= min_alpha:
                print(f"阈值已低于最小 alpha 值: {current_threshold:.3f} <= {min_alpha:.3f}")
                return current_threshold, config
            
            # 降低阈值
            current_threshold -= self.step_size
        
        # 如果阈值降到 0 以下，返回 0
        return 0.0, self.create_search_config(sorted_layers, mxfp4_threshold, 0.0)
    
    def create_search_config(self, sorted_layers: List[Tuple[str, Dict]], 
                           mxfp4_threshold: float, 
                           mxfp8_threshold: Optional[float]) -> Dict[str, Any]:
        """创建搜索配置"""
        config = {}
        
        for layer_name, layer_info in sorted_layers:
            alpha = layer_info['alpha']
            
            if alpha >= mxfp4_threshold:
                config[layer_name] = {
                    "wq": "mxfp4",
                    "aq": "mxfp4",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                }
            elif mxfp8_threshold is not None and alpha >= mxfp8_threshold:
                config[layer_name] = {
                    "wq": "mxfp8",
                    "aq": "mxfp8",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                }
            else:
                # 使用 bf16
                config[layer_name] = {
                    "wq": "bf16",
                    "aq": "bf16",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                }
        
        return config
    
    def create_final_config(self, mxfp4_threshold: float, 
                           mxfp8_threshold: float) -> Dict[str, Any]:
        """创建最终的量化配置"""
        config = {
            "default": {
                "wq": "bf16",
                "aq": "bf16",
                "group_size": 32
            },
            "overrides": []
        }
        
        for layer_name, layer_info in self.alpha_results.items():
            if not isinstance(layer_info.get('alpha'), (int, float)) or not np.isfinite(layer_info['alpha']):
                continue
            
            alpha = layer_info['alpha']
            
            if "lm_head" in layer_name or "mlp.gate" in layer_name:
                # 跳过量化
                config["overrides"].append({
                    "pattern": layer_name,
                    "skip": True,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                })
            elif alpha >= mxfp4_threshold:
                # 使用 mxfp4
                config["overrides"].append({
                    "pattern": layer_name,
                    "wq": "mxfp4",
                    "aq": "mxfp4",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                })
            elif alpha >= mxfp8_threshold:
                # 使用 mxfp8
                config["overrides"].append({
                    "pattern": layer_name,
                    "wq": "mxfp8",
                    "aq": "mxfp8",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                })
            else:
                # 使用 bf16
                config["overrides"].append({
                    "pattern": layer_name,
                    "wq": "bf16",
                    "aq": "bf16",
                    "group_size": 32,
                    "alpha_hill": alpha,
                    "category": layer_info.get('category', 'unknown')
                })
        
        # 添加搜索摘要
        config["search_summary"] = {
            "mxfp4_threshold": mxfp4_threshold,
            "mxfp8_threshold": mxfp8_threshold,
            "ppl_threshold": self.ppl_threshold,
            "target_avg_bits": self.target_avg_bits,
            "step_size": self.step_size,
            "search_history": self.search_history
        }
        
        return config
    
    def run_search(self) -> Dict[str, Any]:
        """运行完整的搜索过程"""
        print("开始自适应量化搜索")
        print("=" * 60)
        print(f"PPL 阈值: {self.ppl_threshold:.1%}")
        print(f"目标平均 bit 数: {self.target_avg_bits}")
        print(f"搜索步长: {self.step_size}")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 计算 Alpha_Hill 值
        sorted_layers = self.compute_alpha_hill()
        
        # 3. 评估基线 PPL
        print("\n评估基线 PPL...")
        baseline_config = self.create_search_config(sorted_layers, 0, 0)  # 全部 bf16
        self.baseline_ppl = self.evaluate_ppl(baseline_config)
        print(f"基线 PPL: {self.baseline_ppl:.3f}")
        
        # 4. 搜索 mxfp4 阈值
        mxfp4_threshold, mxfp4_config = self.search_mxfp4_threshold(sorted_layers, self.baseline_ppl)
        print(f"\nmxfp4 阈值确定: {mxfp4_threshold:.3f}")
        
        # 5. 搜索 mxfp8 阈值
        mxfp8_threshold, mxfp8_config = self.search_mxfp8_threshold(sorted_layers, mxfp4_threshold, self.baseline_ppl)
        print(f"\nmxfp8 阈值确定: {mxfp8_threshold:.3f}")
        
        # 6. 创建最终配置
        final_config = self.create_final_config(mxfp4_threshold, mxfp8_threshold)
        
        # 7. 计算最终结果
        final_avg_bits = self.calculate_current_avg_bits(final_config)
        final_ppl = self.evaluate_ppl(final_config)
        final_ppl_increase = (final_ppl - self.baseline_ppl) / self.baseline_ppl
        
        print("\n" + "=" * 60)
        print("搜索完成！")
        print("=" * 60)
        print(f"mxfp4 阈值: {mxfp4_threshold:.3f}")
        print(f"mxfp8 阈值: {mxfp8_threshold:.3f}")
        print(f"最终平均 bit 数: {final_avg_bits:.2f}")
        print(f"最终 PPL: {final_ppl:.3f}")
        print(f"PPL 增加: {final_ppl_increase:.1%}")
        
        return final_config


def main():
    parser = argparse.ArgumentParser(description="自适应量化搜索")
    parser.add_argument("--model", required=True, help="模型路径或ID")
    parser.add_argument("--device", default="cpu", help="设备类型 (cpu/cuda)")
    parser.add_argument("--dtype", default="bf16", help="数据类型 (bf16/fp16/fp32)")
    parser.add_argument("--ppl-threshold", type=float, default=0.1, 
                       help="PPL升高的可接受百分比 (0.1 = 10%)")
    parser.add_argument("--target-avg-bits", type=float, default=8.0, 
                       help="目标平均bit数")
    parser.add_argument("--step-size", type=float, default=0.5, 
                       help="阈值搜索步长")
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="评估时的batch size")
    parser.add_argument("--output-config", required=True, 
                       help="输出配置文件路径")
    parser.add_argument("--output-plan", required=True, 
                       help="输出量化计划文件路径")
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = AdaptiveQuantizationSearch(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        ppl_threshold=args.ppl_threshold,
        target_avg_bits=args.target_avg_bits,
        step_size=args.step_size,
        batch_size=args.batch_size
    )
    
    # 运行搜索
    final_config = searcher.run_search()
    
    # 保存配置
    os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
    with open(args.output_config, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"\n配置文件已保存到: {args.output_config}")
    
    # 保存量化计划
    os.makedirs(os.path.dirname(args.output_plan), exist_ok=True)
    with open(args.output_plan, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"量化计划已保存到: {args.output_plan}")


if __name__ == "__main__":
    main() 