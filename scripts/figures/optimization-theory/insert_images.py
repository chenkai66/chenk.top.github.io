import re

BASE_URL = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations"

# (file, [(anchor_after_text, image_markdown)])
inserts = {
    "content/en/optimization-theory/01-convex-analysis-foundations.md": [
        # fig 1: after the "invertible matrices not convex" paragraph (end of 1.1), before "### 1.2"
        ("A surprising one: the set of *invertible* matrices is **not** convex. Consider $X = I$ and $Y = -I$; their midpoint is the zero matrix.",
         f"\n\n![Convex set vs non-convex set: a set is convex iff every line segment between two of its points stays inside.]({BASE_URL}/fig1_convex_set.png)"),
        # fig 2: after the projection theorem geometric reading paragraph, before "---" / section 2
        ("The projection theorem has a beautiful geometric reading: $\\pi_C(y)$ is the point of $C$ where the segment $y \\to z$ meets $C$ at an angle of at most $90^\\circ$ to every other direction inside $C$.",
         f"\n\n![Projection of $y$ onto a closed convex set $C$: $z = \\pi_C(y)$ is the unique closest point, and the residual $y - z$ makes a non-acute angle with every direction $x - z$ pointing into $C$.]({BASE_URL}/fig2_projection.png)"),
        # fig 3: after "(F2) => (D): integrate twice..." block, before "### 2.2"
        ("(F2) $\\Rightarrow$ (D): integrate twice. Specifically, for the line $g(\\lambda) = f((1 - \\lambda) x + \\lambda y)$, $g''(\\lambda) = (y - x)^\\top \\nabla^2 f((1 - \\lambda) x + \\lambda y) (y - x) \\geq 0$, so $g$ is convex on $[0, 1]$, which is exactly (D).",
         f"\n\n![Two equivalent views of convexity: (left) the first-order condition says the tangent at any point is a global lower bound on $f$; (right) the epigraph $\\mathrm{{epi}}(f)$ is itself a convex set in $\\mathbb{{R}}^{{n+1}}$.]({BASE_URL}/fig3_convex_function.png)"),
        # fig 4: after the geometric reading paragraph in 3.1, before "### 3.2"
        ("For a fixed slope $y$, $f^*(y)$ is the largest value of $\\langle y, x \\rangle - f(x)$. Equivalently, the affine function $x \\mapsto \\langle y, x \\rangle - f^*(y)$ is the highest affine minorant of $f$ with slope $y$. So $f^*$ tracks, for each slope, how far the supporting hyperplane sits below the graph.",
         f"\n\n![Geometric meaning of the conjugate: for slope $y$, the highest affine minorant of $f$ with that slope is $x \\mapsto y\\,x - f^*(y)$; the value $f^*(y)$ measures the vertical drop from the origin to where this line crosses the $y$-axis.]({BASE_URL}/fig4_conjugate.png)"),
        # fig 5: after Example 1 sentence ending "...lies below |x| everywhere.", before "**Example 2:..."
        ("At $x = 0$, every slope between $-1$ and $1$ defines a tangent line that lies below $|x|$ everywhere.",
         f"\n\n![Subgradients of $|x|$ at the kink: every slope in $[-1, 1]$ produces a line through the origin that stays below $|x|$, so $\\partial f(0) = [-1, 1]$.]({BASE_URL}/fig5_subgradient.png)"),
    ],
    "content/zh/optimization-theory/01-convex-analysis-foundations.md": [
        # ZH fig 1: anchor on the invertible-matrix sentence
        ("一个反直觉的例子：所有**可逆矩阵**构成的集合**不是凸集**。取 $X = I$ 与 $Y = -I$，其连线中点为零矩阵（不可逆）。",
         f"\n\n![凸集与非凸集对比：集合是凸集，当且仅当其中任意两点之间的线段都包含在集合内。]({BASE_URL}/fig1_convex_set.png)"),
        # ZH fig 2: anchor on the projection geometric reading
        ("投影定理具有优美的几何解释：$\\pi_C(y)$ 是 $C$ 中使得线段 $y \\to z$ 与 $C$ 内任一方向夹角均不超过 $90^\\circ$ 的唯一点。",
         f"\n\n![点 $y$ 到闭凸集 $C$ 的投影：$z = \\pi_C(y)$ 是唯一的最近点，残量 $y - z$ 与指向 $C$ 内部的任意方向 $x - z$ 的夹角均不小于 $90^\\circ$。]({BASE_URL}/fig2_projection.png)"),
        # ZH fig 3: anchor on the (F2) => (D) closing sentence
        ("故 $g$ 在 $[0,1]$ 上是凸函数，这正是 (D)。",
         f"\n\n![凸性的两种等价视角：（左）一阶条件——任意点处的切线都是 $f$ 的全局下界；（右）上镜图 $\\mathrm{{epi}}(f)$ 本身是 $\\mathbb{{R}}^{{n+1}}$ 中的凸集。]({BASE_URL}/fig3_convex_function.png)"),
        # ZH fig 4: anchor on geometric reading sentence
        ("对固定斜率 $y$，$f^*(y)$ 表示 $\\langle y, x \\rangle - f(x)$ 关于 $x$ 能取到的最大值。等价地，仿射函数 $x \\mapsto \\langle y, x \\rangle - f^*(y)$ 是斜率为 $y$、且位于 $f$ 下方的最高仿射下界（affine minorant）。因此，$f^*$ 刻画了：对每个可能的斜率 $y$，对应的支持超平面（supporting hyperplane）距离函数图像下方有多远。",
         f"\n\n![共轭函数的几何含义：对斜率 $y$，$f$ 下方斜率为 $y$ 的最高仿射下界为 $x \\mapsto y\\,x - f^*(y)$；$f^*(y)$ 即该直线与 $y$ 轴交点到原点的纵向距离。]({BASE_URL}/fig4_conjugate.png)"),
        # ZH fig 5: anchor on the Example 1 closing sentence
        ("在 $x = 0$ 处，任意斜率介于 $-1$ 与 $1$ 之间的直线均为 $|x|$ 的支撑线，且整体位于其下方。",
         f"\n\n![$|x|$ 在尖点 $x=0$ 处的次梯度集合：$[-1, 1]$ 中的每个斜率都对应一条过原点且位于 $|x|$ 下方的支撑直线，故 $\\partial f(0) = [-1, 1]$。]({BASE_URL}/fig5_subgradient.png)"),
    ]
}

# Read both files, perform the inserts
for fpath, items in inserts.items():
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    original_len = len(content)
    for anchor, ins in items:
        if ins is None:
            continue
        if anchor in content:
            content = content.replace(anchor, anchor + ins, 1)
            print(f"  inserted in {fpath}: anchor matched ({anchor[:50]}...)")
        else:
            print(f"  WARNING in {fpath}: anchor NOT FOUND ({anchor[:80]}...)")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"done {fpath}: {original_len} -> {len(content)} bytes\n")
