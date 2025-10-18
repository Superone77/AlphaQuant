# -*- coding: utf-8 -*-
"""
从 CSV (name, alpha, k, xmin, MSE_INT3..MSE_INT7) 读取数据，
对每个 bit 在 [alpha, log10(MSE)] 空间用 KMeans(k=2) 聚类，
列出每个簇包含的层名称、专家 id、子层类型（up/gate/down/…），
并导出带有每-bit 簇标签的 CSV，以及一个跨 bit 多数投票的稳定簇标签。

用法:
  python find_layer_split_from_csv.py \
      --csv olmoe_layer3_moe_int_mse_hadamard.csv \
      --out_csv clustered_layers.csv \
      --bits 3 4 5 6 7
"""

import argparse
import csv
import re
import numpy as np
from collections import Counter, defaultdict

# 可选：如无 sklearn，可 pip install scikit-learn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------ 解析 CSV ------------------ #

BIT_RE = re.compile(r"^MSE_INT(\d+)$")
EXPERT_RE = re.compile(r"experts\.(\d+)")

# 常见子层名关键词
SUB_NAMES = ["up_proj", "gate_proj", "down_proj", "w1", "w2", "w3", "proj", "fc1", "fc2"]

def load_csv(path):
    items = []  # list of dict
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        bit_list = []
        for col in fieldnames:
            m = BIT_RE.match(col)
            if m:
                bit_list.append(int(m.group(1)))
        bit_list = sorted(bit_list)

        for row in reader:
            try:
                name = row["name"]
                alpha = float(row["alpha"])
            except Exception:
                continue
            mse = {}
            for b in bit_list:
                key = f"MSE_INT{b}"
                try:
                    mse[b] = float(row[key])
                except Exception:
                    mse[b] = np.nan
            # 解析 expert id 与子层名
            m = EXPERT_RE.search(name)
            expert_id = int(m.group(1)) if m else -1
            sub = ""
            for s in SUB_NAMES:
                if s in name:
                    sub = s
                    break
            items.append({
                "name": name,
                "alpha": alpha,
                "expert": expert_id,
                "sub": sub,
                "mse": mse
            })
    return items, bit_list

# ------------------ 聚类 & 汇总 ------------------ #

def cluster_one_bit(items, bit, random_state=0):
    """
    在 [alpha, log10(MSE_INT{bit})] 空间做 KMeans(k=2)
    返回: labels, summary
    """
    xs, idxs = [], []
    for i, it in enumerate(items):
        a = it["alpha"]
        y = it["mse"].get(bit, np.nan)
        if np.isfinite(a) and np.isfinite(y) and y > 0:
            xs.append([a, np.log10(y)])
            idxs.append(i)

    if len(xs) < 10:
        return None, None

    X = np.asarray(xs, dtype=float)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    labels = km.fit_predict(Xn)
    sil = silhouette_score(Xn, labels) if len(set(labels)) == 2 else np.nan

    # 簇统计
    cluster_stats = {}
    for c in [0, 1]:
        mask = (labels == c)
        n = int(mask.sum())
        mu = X[mask].mean(axis=0) if n > 0 else np.array([np.nan, np.nan])
        cluster_stats[c] = {
            "count": n,
            "alpha_mean": float(mu[0]),
            "log10mse_mean": float(mu[1]),
        }

    summary = {
        "bit": bit,
        "silhouette": float(sil),
        "cluster_stats": cluster_stats,
        "idxs": idxs,      # 原 items 索引
        "labels": labels.tolist()
    }
    return labels, summary

def majority_vote(labels_per_bit):
    """
    labels_per_bit: dict{bit: list (len=N or None)}
    对每个样本在多个 bit 上的簇标签做多数投票；若平票/缺失则返回 -1
    返回: voted_labels (len=N)
    """
    # 找到 N
    N = None
    for b, lab in labels_per_bit.items():
        if lab is not None:
            N = len(lab)
            break
    if N is None:
        return None

    # 逐样本收集标签
    votes = [Counter() for _ in range(N)]
    for b, lab in labels_per_bit.items():
        if lab is None:
            continue
        for i, v in enumerate(lab):
            if v is not None and v != -2:  # 用 -2 表示该样本在该 bit 无效（未参与）
                votes[i][v] += 1

    out = []
    for vc in votes:
        if not vc:
            out.append(-1)
        else:
            common = vc.most_common()
            if len(common) == 1 or common[0][1] > common[1][1]:
                out.append(common[0][0])
            else:
                out.append(-1)  # 平票
    return out

# ------------------ 主流程 ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="输入 CSV（含 alpha 与 MSE_INT* 列）")
    ap.add_argument("--out_csv", type=str, default="clustered_layers.csv", help="输出：带聚类标签的 CSV")
    ap.add_argument("--bits", type=int, nargs="*", default=None, help="只针对这些 bit；默认自动识别全部")
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    items, bits_all = load_csv(args.csv)
    bits = args.bits if args.bits else bits_all
    bits = [b for b in bits if b in bits_all]
    print(f"Loaded {len(items)} layers. Bits: {bits}")

    # 每个 bit 做一次 KMeans
    labels_per_bit = {}        # bit -> list(len=N or None)
    summaries = []             # 保存统计

    # 初始化 label 容器（-2 表示该样本在该 bit 未参与）
    N = len(items)
    for b in bits:
        labels_per_bit[b] = [-2] * N

    for b in bits:
        labels, summary = cluster_one_bit(items, b, random_state=args.random_state)
        if labels is None:
            print(f"[INT{b}] 有效点不足，跳过聚类。")
            continue
        idxs = summary["idxs"]
        for loc, lbl in zip(idxs, labels):
            labels_per_bit[b][loc] = int(lbl)
        summaries.append(summary)

        # 打印统计与示例
        cs = summary["cluster_stats"]
        print(f"\n=== INT{b} ===  silhouette={summary['silhouette']:.3f}")
        for c in [0, 1]:
            print(f"  cluster {c}: n={cs[c]['count']}, "
                  f"mean alpha={cs[c]['alpha_mean']:.3f}, "
                  f"mean log10(MSE)={cs[c]['log10mse_mean']:.3f}")

        # 列举每簇的前若干名字（按子层分组统计）
        groups = defaultdict(list)
        for i, lbl in zip(summary["idxs"], summary["labels"]):
            it = items[i]
            groups[lbl].append((it["name"], it["expert"], it["sub"]))
        for c in [0, 1]:
            subs = Counter([g[2] for g in groups[c]])
            print(f"  cluster {c} sub-layer counts: {dict(subs)}")

    # 跨 bit 多数投票得到稳定簇
    voted = majority_vote(labels_per_bit)
    if voted is not None:
        print("\n=== Majority vote across bits ===")
        counts = Counter([v for v in voted if v in [0, 1]])
        print(f"  counts: {dict(counts)} ( -1 表示平票/缺失 )")

    # 导出带标签的 CSV
    out_cols = ["name", "expert", "sub", "alpha"]
    out_cols += [f"MSE_INT{b}" for b in bits_all]
    out_cols += [f"cluster_INT{b}" for b in bits]
    out_cols += ["cluster_majority"]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_cols)
        for i, it in enumerate(items):
            row = [it["name"], it["expert"], it["sub"], f"{it['alpha']:.6f}"]
            # 原始 MSE
            for b in bits_all:
                val = it["mse"].get(b, np.nan)
                row.append(f"{val:.8e}" if np.isfinite(val) else "")
            # 此次选择的 bit 标签
            for b in bits:
                lab = labels_per_bit[b][i]
                row.append(lab if lab in [0,1] else "")
            # 多数投票
            mv = voted[i] if voted is not None else -1
            row.append(mv if mv in [0,1] else "")
            writer.writerow(row)

    print(f"\n[Saved] {args.out_csv}")

    # 额外提示：哪些层“稳定属于同一簇”
    if voted is not None:
        stable0 = [items[i]["name"] for i,v in enumerate(voted) if v==0]
        stable1 = [items[i]["name"] for i,v in enumerate(voted) if v==1]
        print(f"\nStable cluster 0: {len(stable0)} layers (examples)")
        for s in stable0[:10]:
            print("  ", s)
        print(f"Stable cluster 1: {len(stable1)} layers (examples)")
        for s in stable1[:10]:
            print("  ", s)

if __name__ == "__main__":
    main()
