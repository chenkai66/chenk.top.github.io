"""Insert figure references into EN and ZH versions of article 10."""
import re

URL = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction"

# Each entry: (anchor_substring, insertion_text)
# Anchor must be a unique line; we insert AFTER the matched block.

EN_INSERTIONS = [
    # fig1 — after the SGD framework axioms paragraph (end of section 1)
    (
        "These two assumptions (unbiased + bounded variance) are the SGD axioms. The strength of the resulting bounds depends on what additional structure $f$ has.",
        "\n\n![SGD vs Full GD trajectories on an ill-conditioned 2D quadratic](" + URL + "/fig1.png)\n*Full GD follows a smooth, deterministic descent path; SGD takes a noisy zigzag in expectation around the same direction. The variance of each SGD step is what the noise budget $\\sigma^2$ controls.*",
    ),
    # fig4 — end of section 4
    (
        "This works only up to a \"critical batch size\" beyond which the noise is no longer the bottleneck (McCandlish et al., 2018).",
        "\n\n![Mini-batch variance and the critical batch size](" + URL + "/fig4.png)\n*Left: gradient variance falls as $\\sigma^2/B$ on a log-log plot. Right: the linear scaling rule lets effective step size grow with $B$ — but only up to a critical batch $B^\\star$, beyond which speedup saturates because the gradient signal, not noise, becomes the bottleneck.*",
    ),
    # fig3 — end of section 5.1
    (
        "This is what gives the linear convergence rate.",
        "\n\n![SGD vs SVRG gradient samples around a fixed point](" + URL + "/fig3.png)\n*Each light arrow is one stochastic gradient sample; the bold blue arrow is the true $\\nabla f(x)$. SGD samples (orange) scatter widely around the mean; SVRG samples (green) cluster tightly because the control variate $-\\nabla f_{i_t}(\\tilde w_s) + \\tilde g_s$ cancels most of the variance.*",
    ),
    # fig2 — end of section 5.3 (after the comparison list)
    (
        "For $n \\approx \\kappa$ (typical regularized ML), SVRG is $\\sim \\kappa \\times$ faster than full GD and $\\sim \\kappa^2 / (\\kappa \\log(1/\\epsilon))$ faster than SGD for small $\\epsilon$.",
        "\n\n![Convergence rates: SGD, Full GD, SVRG, Katyusha](" + URL + "/fig2.png)\n*Suboptimality vs total gradient evaluations on a log-log axis. SGD's $1/\\sqrt{T}$ rate is the slowest curve; Full GD is geometric but the per-step cost is $n$. SVRG and Katyusha are linear in the number of epochs, eventually beating both.*",
    ),
    # fig5 — end of section 6 (after Katyusha is optimal)
    (
        "So Katyusha is **optimal** for the strongly convex finite-sum setting.",
        "\n\n![Total gradient evaluations needed to reach high precision](" + URL + "/fig5.png)\n*With $n=10^4$ and $\\kappa=10^3$, Full GD needs $\\sim 10^{11}$ gradients to reach $\\epsilon=10^{-4}$; SGD's $O(\\kappa^2/\\epsilon)$ scaling requires $\\sim 10^{10}$. SVRG drops it to $\\sim 10^{5}$, and Katyusha shaves another factor of $\\sqrt{n/\\kappa}$.*",
    ),
]

ZH_INSERTIONS = [
    # fig1
    (
        "这两条假设（无偏性 + 有界方差）构成了 SGD 的公理基础。由此导出的收敛界强度，取决于目标函数 $f$ 所具备的额外结构（如凸性、强凸性、光滑性等）。",
        "\n\n![病态二次函数上 SGD 与全梯度下降的轨迹对比](" + URL + "/fig1.png)\n*全梯度下降沿确定性路径平稳下降；SGD 则在同方向上呈现噪声锯齿轨迹。每步 SGD 的扰动幅度正由噪声预算 $\\sigma^2$ 所控制。*",
    ),
    # fig4
    (
        "但该规律仅在「临界批大小」（critical batch size）以内成立；超过该阈值后，噪声不再是最主要瓶颈（McCandlish 等，2018）。",
        "\n\n![小批量方差与临界批大小](" + URL + "/fig4.png)\n*左：梯度方差按 $\\sigma^2/B$ 的速率随批大小线性下降（log-log 坐标）。右：线性缩放律允许有效步长随 $B$ 同比例增长——但仅到临界批大小 $B^\\star$ 为止，超过后加速比饱和，因为此时梯度信号本身（而非噪声）成为瓶颈。*",
    ),
    # fig3
    (
        "正是这一特性赋予了 SVRG 线性收敛速率。",
        "\n\n![同一点处 SGD 与 SVRG 的随机梯度样本对比](" + URL + "/fig3.png)\n*每条浅色箭头是一次随机梯度采样，加粗蓝色箭头表示真实梯度 $\\nabla f(x)$。SGD（橙色）样本在均值附近大幅散布；SVRG（绿色）样本紧密聚集——控制变量 $-\\nabla f_{i_t}(\\tilde w_s) + \\tilde g_s$ 抵消了大部分方差。*",
    ),
    # fig2
    (
        "当 $n \\approx \\kappa$（典型正则化机器学习场景）时，SVRG 比全梯度下降快约 $\\kappa$ 倍；相比 SGD，在小 $\\epsilon$ 下快约 $\\kappa^2 / (\\kappa \\log(1/\\epsilon)) = \\kappa / \\log(1/\\epsilon)$ 倍。",
        "\n\n![SGD、全梯度下降、SVRG 与 Katyusha 的收敛速率](" + URL + "/fig2.png)\n*log-log 坐标下，次优间隙 $f(x_T) - f^\\star$ 随梯度计算次数 $T$ 的变化。SGD 的 $1/\\sqrt{T}$ 速率最为缓慢；全梯度下降几何收敛但每步消耗 $n$ 次梯度计算；SVRG 与 Katyusha 在 epoch 数上呈线性收敛，最终全面胜出。*",
    ),
    # fig5
    (
        "因此，Katyusha 在强凸有限和问题中是**最优的**。",
        "\n\n![达到给定精度所需的总梯度计算次数](" + URL + "/fig5.png)\n*在 $n=10^4$、$\\kappa=10^3$ 的设定下，全梯度下降需约 $10^{11}$ 次梯度计算才能达到 $\\epsilon=10^{-4}$；SGD 的 $O(\\kappa^2/\\epsilon)$ 复杂度约需 $10^{10}$ 次。SVRG 将其降至 $\\sim 10^5$，Katyusha 进一步省下 $\\sqrt{n/\\kappa}$ 倍。*",
    ),
]


def patch(path, insertions):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for anchor, addition in insertions:
        if anchor not in text:
            raise SystemExit(f"ANCHOR NOT FOUND in {path}: {anchor[:80]!r}")
        # Avoid double-insert if URL already there in subsequent paragraph
        # Insert immediately after the anchor
        text = text.replace(anchor, anchor + addition, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patched {path}")


patch("/root/chenk-hugo/content/en/optimization-theory/10-stochastic-variance-reduction.md",
      EN_INSERTIONS)
patch("/root/chenk-hugo/content/zh/optimization-theory/10-stochastic-variance-reduction.md",
      ZH_INSERTIONS)

print("All patched.")
