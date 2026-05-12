#!/usr/bin/env python3
"""Insert figures into the EN and ZH article 07 second-order methods."""
import os, sys

BASE = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/07-second-order-methods"

EN_PATH = "/root/chenk-hugo/content/en/optimization-theory/07-second-order-methods.md"
ZH_PATH = "/root/chenk-hugo/content/zh/optimization-theory/07-second-order-methods.md"

def fig_md(n, alt, caption):
    return (
        f"\n![{alt}]({BASE}/fig{n}.png)\n"
        f"*{caption}*\n"
    )

# (anchor heading line, figure markdown to insert BEFORE that heading)
EN_INSERTS = [
    ("### 1.2 Quadratic local convergence",
     fig_md(1, "Newton's method local quadratic model",
            "Figure 1. Newton's method approximates f by the local quadratic at x_k and jumps to that quadratic's minimum. If f is itself quadratic, one step suffices.")),
    ("### 1.3 The catch: globalization",
     fig_md(2, "Convergence rates: GD vs BFGS vs Newton",
            "Figure 2. Error vs iteration on a log scale: gradient descent decays linearly (constant rate), BFGS achieves superlinear decay, and Newton doubles the number of correct digits per step (quadratic).")),
    ("### 1.4 When the Hessian is indefinite",
     fig_md(3, "Damped Newton backtracking line search",
            "Figure 3. Damped Newton: starting from x_k, the pure step (alpha=1) overshoots; backtracking halves alpha until the Armijo sufficient-decrease bound (dotted) is satisfied.")),
    ("### 3.2 Where the two-loop comes from",
     fig_md(4, "L-BFGS two-loop recursion",
            "Figure 4. The L-BFGS two-loop recursion. The backward loop sweeps the m history pairs to produce alpha_i and an updated q; H_k^0 is applied; the forward loop sweeps in reverse, yielding r = H_k g in O(mn) without ever forming H_k.")),
    ("### 4.3 Convergence",
     fig_md(5, "Trust region dogleg path",
            "Figure 5. Trust-region subproblem in 2D: contours of the quadratic model m(d), the trust ball ||d||<=Delta (dashed), the Cauchy direction d_SD, the Newton step d_N, and the dogleg broken line. The dogleg solution is the intersection of the path with the trust boundary.")),
]

def zh_fig_md(n, alt, caption):
    return (
        f"\n![{alt}]({BASE}/fig{n}.png)\n"
        f"*{caption}*\n"
    )

ZH_INSERTS = [
    ("### 1.2 局部二次收敛性",
     zh_fig_md(1, "牛顿法局部二次模型",
               "图 1. 牛顿法在 x_k 处用局部二次模型近似 f，并直接跳跃到该二次模型的极小点；若 f 本身就是二次函数，则一步收敛。")),
    ("### 1.3 关键难点：全局化（Globalization）",
     zh_fig_md(2, "收敛速率对比：梯度下降 vs BFGS vs 牛顿",
               "图 2. 误差随迭代次数的对数曲线：梯度下降以线性速率衰减（每步乘一个常数），BFGS 实现超线性，牛顿法每步将有效数字位数翻倍（二次收敛）。")),
    ("### 1.4 当 Hessian 矩阵不定时",
     zh_fig_md(3, "阻尼牛顿法回溯线搜索",
               "图 3. 阻尼牛顿法：从 x_k 出发，纯牛顿步（alpha=1）越过了局部最优；回溯逐次将 alpha 减半，直至满足 Armijo 充分下降条件（虚线为上界）。")),
    ("### 3.2 双循环递推的来源",
     zh_fig_md(4, "L-BFGS 双循环递推",
               "图 4. L-BFGS 双循环递推示意：第一轮反向遍历 m 对历史，得到 alpha_i 与更新后的 q；中间施加初始缩放 H_k^0；第二轮正向遍历，最终输出 r = H_k g，全程复杂度 O(mn)、无需构造 H_k。")),
    ("### 4.3 收敛性",
     zh_fig_md(5, "信赖域狗腿法路径",
               "图 5. 二维信赖域子问题：二次模型 m(d) 的等高线、信赖域 ||d||<=Delta（虚线圆）、最速下降方向 d_SD、牛顿步 d_N，以及由 0 → d_SD → d_N 构成的狗腿折线。狗腿解为路径与信赖域边界的交点。")),
]

def apply(path, inserts):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for anchor, block in inserts:
        if anchor not in text:
            print(f"  ! anchor not found in {path}: {anchor!r}")
            continue
        # ensure idempotent: skip if image for this anchor already inserted
        # check if the figure URL right before the anchor is already there; simple approach:
        # we insert only if the figure URL fragment (the figN.png referenced) isn't already
        # present immediately before the anchor.
        # Find anchor position
        idx = text.index(anchor)
        # detect if any of the figure URLs in this block are already in the file
        # to avoid duplicates if rerun
        url_marker = block.strip().split("(")[1].split(")")[0]  # e.g. https://.../figN.png
        if url_marker in text:
            print(f"  - already inserted ({url_marker}) in {os.path.basename(path)}, skipping")
            continue
        text = text[:idx] + block + "\n" + text[idx:]
        print(f"  + inserted before {anchor!r} in {os.path.basename(path)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

print("EN:")
apply(EN_PATH, EN_INSERTS)
print("ZH:")
apply(ZH_PATH, ZH_INSERTS)
print("done")
