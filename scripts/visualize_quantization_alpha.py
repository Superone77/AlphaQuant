#!/usr/bin/env python
"""
从 CSV 数据生成量化 Alpha-Hill 可视化
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_layer_bar_chart(df_row: pd.Series, quant_formats: List[str], output_path: str):
    """为单个层生成柱状图"""
    layer_name = df_row['layer_name']
    alphas = [df_row[f'alpha_{fmt}'] for fmt in quant_formats]
    
    # 限制柱子高度不超过 10，但保留原始值用于显示
    alphas_clamped = [min(a, 10.0) if not np.isnan(a) else a for a in alphas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quant_formats)))
    bars = ax.bar(quant_formats, alphas_clamped, color=colors, alpha=0.8, edgecolor='black')
    
    # 在柱子上添加数值（显示原始值，但位置在 min(alpha, 10)）
    for bar, alpha_original, alpha_clamped in zip(bars, alphas, alphas_clamped):
        if not np.isnan(alpha_clamped):
            # 文本位置在柱子顶部（最高 10）
            text_y = alpha_clamped
            # 显示原始值
            ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                   f'{alpha_original:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Quantization Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Alpha-Hill: {layer_name}', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 10)  # 设置 y 轴上限为 10
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overall_distribution(df: pd.DataFrame, quant_formats: List[str], output_path: str):
    """生成整体分布箱线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = []
    for fmt in quant_formats:
        col = f'alpha_{fmt}'
        if col in df.columns:
            data_to_plot.append(df[col].dropna().values)
    
    bp = ax.boxplot(data_to_plot, labels=quant_formats, patch_artist=True)
    
    # 自定义颜色
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quant_formats)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Quantization Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
    ax.set_title('Alpha-Hill Distribution Across Quantization Formats', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)  # 设置 y 轴上限为 10
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_category_comparison(df: pd.DataFrame, quant_formats: List[str], output_path: str):
    """按类别分组对比"""
    categories = df['category'].unique()
    n_categories = len(categories)
    n_formats = len(quant_formats)
    
    fig, axes = plt.subplots(2, (n_categories + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_df = df[df['category'] == category]
        
        means = [cat_df[f'alpha_{fmt}'].mean() for fmt in quant_formats]
        stds = [cat_df[f'alpha_{fmt}'].std() for fmt in quant_formats]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_formats))
        bars = ax.bar(range(n_formats), means, yerr=stds, capsize=5,
                     color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xticks(range(n_formats))
        ax.set_xticklabels(quant_formats, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Alpha-Hill', fontsize=10)
        ax.set_title(f'{category} ({len(cat_df)} layers)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 10)  # 设置 y 轴上限为 10
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_categories, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Alpha-Hill by Category and Quantization Format', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(df: pd.DataFrame, quant_formats: List[str], output_path: str):
    """生成热力图"""
    # 准备数据矩阵
    layer_names = df['layer_name'].values
    alpha_matrix = np.array([[df.iloc[i][f'alpha_{fmt}'] 
                             for fmt in quant_formats] 
                             for i in range(len(df))])
    
    # 如果层太多，只显示部分
    max_layers_show = 50
    if len(layer_names) > max_layers_show:
        step = len(layer_names) // max_layers_show
        indices = list(range(0, len(layer_names), step))
        layer_names = layer_names[indices]
        alpha_matrix = alpha_matrix[indices, :]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(layer_names) * 0.2)))
    
    im = ax.imshow(alpha_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(range(len(quant_formats)))
    ax.set_xticklabels(quant_formats, rotation=45, ha='right')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([name.split('.')[-1] if len(name) > 30 else name 
                        for name in layer_names], fontsize=6)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Alpha-Hill Value', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_xlabel('Quantization Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title('Alpha-Hill Heatmap Across Layers and Formats', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_variance_analysis(df: pd.DataFrame, quant_formats: List[str], output_path: str):
    """分析量化格式间的方差"""
    # 计算每层的方差
    alpha_cols = [f'alpha_{fmt}' for fmt in quant_formats]
    df_temp = df[alpha_cols].copy()
    df_temp['variance'] = df_temp.var(axis=1)
    df_temp['std'] = df_temp.std(axis=1)
    df_temp['range'] = df_temp.max(axis=1) - df_temp.min(axis=1)
    df_temp['layer_name'] = df['layer_name']
    df_temp['category'] = df['category']
    
    # 按方差排序
    df_sorted = df_temp.sort_values('std', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：标准差柱状图（前30层）
    top_n = min(30, len(df_sorted))
    top_layers = df_sorted.head(top_n)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    bars = ax1.barh(range(top_n), top_layers['std'].values, color=colors, alpha=0.8)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([name.split('.')[-1] if len(name) > 40 else name 
                         for name in top_layers['layer_name'].values], fontsize=8)
    ax1.set_xlabel('Standard Deviation of Alpha', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Layers with Highest Variance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 右图：按类别的平均方差
    category_var = df_temp.groupby('category').agg({
        'std': 'mean',
        'range': 'mean',
        'variance': 'mean'
    }).sort_values('std', ascending=False)
    
    categories = category_var.index
    x = np.arange(len(categories))
    width = 0.6
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(categories)))
    bars = ax2.bar(x, category_var['std'].values, width, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_ylabel('Mean Std Dev of Alpha', fontsize=11, fontweight='bold')
    ax2.set_title('Variance by Layer Category', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="从 CSV 生成量化 Alpha-Hill 可视化")
    parser.add_argument("--csv", type=str, required=True, help="CSV 文件路径")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="输出目录 (默认: CSV 所在目录)")
    parser.add_argument("--quant-formats", type=str,
                       default="bf16,mxfp8,mxfp4,fp8,fp4,int8,int6,int4,int3,int2",
                       help="量化格式列表")
    parser.add_argument("--skip-individual", action='store_true',
                       help="跳过单层图表生成")
    
    args = parser.parse_args()
    
    # 读取 CSV
    print(f"\n读取 CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"加载了 {len(df)} 层")
    
    # 解析量化格式
    quant_formats = [f.strip() for f in args.quant_formats.split(',')]
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.csv).parent
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}\n")
    
    # 1. 生成单层柱状图
    if not args.skip_individual:
        print("生成单层柱状图...")
        for idx, row in df.iterrows():
            layer_name = row['layer_name']
            plot_name = f"layer_{idx+1:03d}_{layer_name.replace('.', '_')}.png"
            plot_path = plots_dir / plot_name
            plot_layer_bar_chart(row, quant_formats, str(plot_path))
            if (idx + 1) % 10 == 0:
                print(f"  已生成 {idx+1}/{len(df)} 个图表")
        print(f"✓ 完成 {len(df)} 个单层图表\n")
    
    # 2. 整体分布箱线图
    print("生成整体分布图...")
    plot_overall_distribution(df, quant_formats, 
                             str(output_dir / "alpha_distribution.png"))
    print("✓ 完成\n")
    
    # 3. 按类别对比
    print("生成类别对比图...")
    plot_category_comparison(df, quant_formats,
                            str(output_dir / "alpha_by_category.png"))
    print("✓ 完成\n")
    
    # 4. 热力图
    print("生成热力图...")
    plot_heatmap(df, quant_formats, str(output_dir / "alpha_heatmap.png"))
    print("✓ 完成\n")
    
    # 5. 方差分析
    print("生成方差分析图...")
    plot_variance_analysis(df, quant_formats,
                          str(output_dir / "alpha_variance_analysis.png"))
    print("✓ 完成\n")
    
    # 统计信息
    print("="*60)
    print("统计摘要")
    print("="*60)
    for fmt in quant_formats:
        col = f'alpha_{fmt}'
        if col in df.columns:
            values = df[col].dropna()
            print(f"{fmt:>6s}: mean={values.mean():.4f}, "
                  f"std={values.std():.4f}, "
                  f"min={values.min():.4f}, "
                  f"max={values.max():.4f}")
    
    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  - {output_dir / 'alpha_distribution.png'}")
    print(f"  - {output_dir / 'alpha_by_category.png'}")
    print(f"  - {output_dir / 'alpha_heatmap.png'}")
    print(f"  - {output_dir / 'alpha_variance_analysis.png'}")
    if not args.skip_individual:
        print(f"  - {plots_dir}/ (共 {len(df)} 个单层图表)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

