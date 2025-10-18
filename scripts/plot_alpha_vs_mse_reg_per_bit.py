# -*- coding: utf-8 -*-
"""
从 CSV 加载 (alpha, MSE_INT*)，对每个 bit 画一张图：
- 散点：alpha vs MSE
- 回归曲线：在 log10(MSE) 上做回归（linear / poly2 / loess），再还原到线性域绘制
- 输出每个 bit 的回归参数与 R^2，并保存图像

用法示例：
  python plot_alpha_vs_mse_reg_per_bit.py \
      --csv olmoe_layer3_moe_int_mse_hadamard.csv \
      --out_prefix alpha_mse_reg \
      --fit linear --ylog
"""

import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境下使用；交互可注释掉
import matplotlib.pyplot as plt

def load_csv_alpha_mse(csv_path):
    """
    读取 CSV，返回：
      alpha: (N,) float
      mse_by_bit: dict{bit: np.ndarray shape (N,)}
      bits: sorted list of bits
      names: list of matrix names (可用于调试)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # 找出 MSE_INT* 列
        bit_cols = []
        pat = re.compile(r"^MSE_INT(\d+)$")
        for col in fieldnames:
            m = pat.match(col)
            if m:
                bit_cols.append(int(m.group(1)))
        if not bit_cols:
            raise ValueError("CSV 中未发现形如 MSE_INT* 的列。")
        bit_cols = sorted(bit_cols)

        alphas = []
        names = []
        mse_arrays = {b: [] for b in bit_cols}

        for row in reader:
            # 可能存在空行/非法值
            try:
                a = float(row["alpha"])
            except Exception:
                continue

            alphas.append(a)
            names.append(row.get("name", ""))

            for b in bit_cols:
                key = f"MSE_INT{b}"
                try:
                    v = float(row[key])
                except Exception:
                    v = np.nan
                mse_arrays[b].append(v)

        alpha = np.array(alphas, dtype=float)
        mse_by_bit = {b: np.array(mse_arrays[b], dtype=float) for b in bit_cols}

    return alpha, mse_by_bit, bit_cols, names

# ---------------- 回归方法 ---------------- #

def fit_linear_log10(alpha, mse):
    """
    log10(MSE) = a + b * alpha
    返回：a, b, y_pred_on_grid, grid_alpha, R^2
    """
    mask = np.isfinite(alpha) & np.isfinite(mse) & (mse > 0)
    x = alpha[mask]
    y = np.log10(mse[mask])
    if x.size < 3:
        return None

    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)  # [a, b]
    a, b = coef

    # R^2
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # 画曲线的网格
    grid = np.linspace(np.min(x), np.max(x), 200)
    y_grid = a + b * grid
    y_grid_lin = 10 ** y_grid
    return dict(a=a, b=b, r2=r2, grid=grid, y_grid_lin=y_grid_lin)

def fit_poly2_log10(alpha, mse):
    """
    log10(MSE) = a + b * alpha + c * alpha^2
    """
    mask = np.isfinite(alpha) & np.isfinite(mse) & (mse > 0)
    x = alpha[mask]
    y = np.log10(mse[mask])
    if x.size < 5:
        return None

    # 构造 [1, x, x^2]
    X = np.vstack([np.ones_like(x), x, x**2]).T
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)  # [a, b, c]
    a, b, c = coef

    y_hat = a + b * x + c * x**2
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    grid = np.linspace(np.min(x), np.max(x), 200)
    y_grid = a + b * grid + c * grid**2
    y_grid_lin = 10 ** y_grid
    return dict(a=a, b=b, c=c, r2=r2, grid=grid, y_grid_lin=y_grid_lin)

def fit_loess_log10(alpha, mse, frac=0.3):
    """
    需要 statsmodels：pip install statsmodels
    返回平滑曲线（无显式参数、仅给 y_grid）
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        return None
    mask = np.isfinite(alpha) & np.isfinite(mse) & (mse > 0)
    x = alpha[mask]
    y = np.log10(mse[mask])
    if x.size < 10:
        return None
    # 排序后平滑
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
    grid = smoothed[:, 0]
    y_grid = smoothed[:, 1]
    y_grid_lin = 10 ** y_grid
    # 计算一个简化版 R^2（相对均值）
    y_hat = np.interp(x, grid, y_grid)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return dict(r2=r2, grid=grid, y_grid_lin=y_grid_lin)

# ---------------- 画图 ---------------- #

def plot_one_bit(alpha, mse, bit, out_prefix, fit="linear", ylog=True):
    """
    alpha, mse: 1D arrays
    保存文件名：{out_prefix}_bit{bit}.png
    """
    mask = np.isfinite(alpha) & np.isfinite(mse) & (mse > 0)
    a = alpha[mask]
    y = mse[mask]
    if a.size == 0:
        print(f"[bit {bit}] 有效数据为 0，跳过。")
        return

    # 拟合
    fit_res = None
    if fit == "linear":
        fit_res = fit_linear_log10(a, y)
    elif fit == "poly2":
        fit_res = fit_poly2_log10(a, y)
    elif fit == "loess":
        fit_res = fit_loess_log10(a, y, frac=0.3)
    else:
        raise ValueError(f"未知拟合类型：{fit}")

    # 画图
    plt.figure(figsize=(8, 6))
    plt.scatter(a, y, s=14, color="#1f77b4", alpha=0.7, edgecolors="none", label="data")

    title = f"Alpha vs MSE @ INT{bit}  (fit={fit})"
    if fit_res is not None:
        plt.plot(fit_res["grid"], fit_res["y_grid_lin"], color="#d62728", lw=2.0, label=f"fit curve")
        if "a" in fit_res:
            # 线性或二次
            if "c" in fit_res:
                print(f"[bit {bit}] poly2: log10(MSE)= {fit_res['a']:.4f} + {fit_res['b']:.4f}*alpha + {fit_res['c']:.4f}*alpha^2 | R^2={fit_res['r2']:.4f}")
            else:
                print(f"[bit {bit}] linear: log10(MSE)= {fit_res['a']:.4f} + {fit_res['b']:.4f}*alpha | R^2={fit_res['r2']:.4f}")
            title += f"\n$R^2$={fit_res['r2']:.3f}"
        else:
            # loess
            print(f"[bit {bit}] loess: R^2={fit_res['r2']:.4f}")
            title += f"\nLOESS $R^2$={fit_res['r2']:.3f}"
    else:
        print(f"[bit {bit}] 样本不足，未绘制回归曲线。")

    plt.title(title)
    plt.xlabel("PL_Alpha_Hill (α)")
    plt.ylabel("Per-tensor Quantization MSE")
    if ylog:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    out_path = f"{out_prefix}_bit{bit}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="输入 CSV 路径")
    ap.add_argument("--out_prefix", type=str, default="alpha_mse_reg_nolog")
    ap.add_argument("--fit", type=str, default="linear", choices=["linear", "poly2", "loess"])
    ap.add_argument("--ylog", action="store_true", help="y 轴对数显示（推荐）")
    ap.add_argument("--no-ylog", dest="ylog", action="store_false")
    ap.set_defaults(ylog=True)
    args = ap.parse_args()

    alpha, mse_by_bit, bits, _ = load_csv_alpha_mse(args.csv)
    print(f"Loaded {alpha.size} samples. Bits detected: {bits}. Fit={args.fit}")

    for b in bits:
        plot_one_bit(alpha, mse_by_bit[b], b, args.out_prefix, fit=args.fit, ylog=args.ylog)

if __name__ == "__main__":
    main()
