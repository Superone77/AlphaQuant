# -*- coding: utf-8 -*-
"""
从 CSV 加载 (name, alpha, k, xmin, MSE_INT3..MSE_INT7) 并绘制曲线：
- 每个矩阵一条曲线，横轴=比特(自动识别 CSV 内所有 MSE_INT* 列)，纵轴=MSE
- 颜色按 alpha 从小到大映射到 colormap（小 alpha 更深）
- 带 colorbar，修复了借位轴问题（使用 fig.colorbar(..., ax=ax)）
"""

import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境时使用；如需交互可注释掉
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def load_csv(path):
    """
    读取 CSV，返回：
      items: list of dict {name, alpha:float, k:int, xmin:float, mse:dict{bit->value}}
      bits: sorted list of detected INT bits
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 自动发现 MSE_INT* 列
        bit_cols = []
        bit_re = re.compile(r"^MSE_INT(\d+)$")
        for col in reader.fieldnames:
            m = bit_re.match(col)
            if m:
                bit_cols.append(int(m.group(1)))
        if not bit_cols:
            raise ValueError("CSV 中未发现列名形如 MSE_INT* 的列。")

        bit_cols = sorted(bit_cols)
        # 逐行解析
        for row in reader:
            try:
                name = row["name"]
                alpha = float(row["alpha"])
                k = int(row.get("k", 0)) if row.get("k", "").strip() != "" else 0
                xmin = float(row.get("xmin", "0.0"))
            except Exception as e:
                raise ValueError(f"解析基础字段失败：{e}\n行内容：{row}")

            mse_map = {}
            for b in bit_cols:
                key = f"MSE_INT{b}"
                try:
                    val = float(row[key])
                except Exception:
                    val = np.nan
                mse_map[b] = val
            items.append({"name": name, "alpha": alpha, "k": k, "xmin": xmin, "mse": mse_map})

    return items, bit_cols

def plot_curves_from_items(items, bits, save_png=None, cmap_name="viridis",
                           ylog=True, title=None):
    """
    根据 items 绘制曲线。
    - items: [{name, alpha, mse:{bit->value}}]
    - bits: [3,4,5,6,7,...]
    - cmap_name: 选择色谱（如 "viridis","magma_r","cividis" 等）
    - ylog: 是否对 y 轴取对数
    """
    if not items:
        raise ValueError("空数据：items 为空。")

    # 收集 alpha
    alphas = np.array([it["alpha"] for it in items], dtype=np.float64)
    a_min, a_max = float(np.nanmin(alphas)), float(np.nanmax(alphas))
    if not np.isfinite(a_min) or not np.isfinite(a_max):
        raise ValueError("alpha 含有非有限值（NaN/Inf）。")
    if a_max == a_min:
        a_max = a_min + 1e-6  # 避免归一化分母为零

    # 颜色映射：小 alpha -> 深色（顺序色谱小值本来就更深）
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=2, vmax=3)

    fig, ax = plt.subplots(figsize=(14, 9))

    # 画所有曲线
    for it in items:
        alpha = it["alpha"]
        color = cmap(norm(alpha))
        ys = [it["mse"].get(b, np.nan) for b in bits]
        # 过滤非数
        ys = [np.nan if (y is None or not np.isfinite(y)) else y for y in ys]
        ax.plot(bits, ys, color=color, linewidth=1.0, alpha=0.9)

    # 配置坐标轴
    ax.set_xlabel("Bits (INT)")
    ax.set_ylabel("Per-tensor Quantization MSE")
    if title is None:
        title = "INT MSE vs. Bits (color = PL_Alpha_Hill)"
    ax.set_title(title)
    ax.set_xticks(bits)
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    # colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("PL_Alpha_Hill (α)")

    fig.tight_layout()
    if save_png:
        fig.savefig(save_png, dpi=200)
        print(f"[Saved] {save_png}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 路径（由前一脚本导出）")
    parser.add_argument("--save_png", type=str, default="alpha_int_mse_from_csv.png")
    parser.add_argument("--cmap", type=str, default="inferno", help="色谱名称，如 viridis/magma_r/cividis")
    parser.add_argument("--ylog", action="store_true", help="y 轴用 log 尺度（推荐）")
    parser.add_argument("--no-ylog", dest="ylog", action="store_false")
    parser.set_defaults(ylog=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    items, bits = load_csv(args.csv)
    print(f"Loaded {len(items)} items; Bits detected: {bits}")
    plot_curves_from_items(
        items,
        bits,
        save_png=args.save_png,
        cmap_name=args.cmap,
        ylog=args.ylog,
        title=args.title
    )

if __name__ == "__main__":
    main()
