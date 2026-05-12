ZH = "content/zh/optimization-theory/02-smoothness-strong-convexity-nesterov.md"

URL6 = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig6_trajectories.png"
URL7 = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig7_stepsize.png"

with open(ZH) as f:
    text = f.read()

# Anchor 1: end of §1.2 (the descent-lemma corollary line)
zh_anchor_step = '只要 $\\eta \\le 1/L$，括号项 $\\ge 1/2$，**每一步都严格下降**，且下降量被梯度范数平方控制。这就是"步长不要超过 $1/L$"的硬来源——不是经验之谈，而是 descent lemma 的直接推论。'
zh_block_step = (
    f"![一维二次函数上 GD 在三种步长下的行为：eta=0.8/L 收敛、eta=1/L 一步到位、eta=2.2/L 发散]({URL7})\n\n"
    "同一条二次函数 $f(x) = \\tfrac{L}{2}x^2$（这里 $L=4$）下的三种情形。"
    "左：小步长（$\\eta = 0.8/L$）单调爬下来；"
    "中：经典 $\\eta = 1/L$ 从 $x_0=1$ 一步直达最优——不过冲的最快配置；"
    "右：一旦越过 $2/L$ 这条天花板，迭代点就会爆炸，descent lemma 失效。"
    "区间 $[1/L, 2/L]$ 正是\"既最快又安全\"的临界带；低于它浪费迭代，高于它直接发散。"
)
if zh_block_step.split("\n")[0] not in text:
    if zh_anchor_step not in text:
        raise SystemExit("ZH anchor 1 not found")
    text = text.replace(zh_anchor_step, zh_anchor_step + "\n\n" + zh_block_step, 1)
    print("ZH patched: fig7")
else:
    print("ZH fig7 already present")

# Anchor 2: end of Theorem 7 statement
zh_anchor_traj = "则 $f(x_t) - f^\\star \\le \\big(1 - \\sqrt{1/\\kappa}\\big)^t (f(x_0) - f^\\star)$，对应 $t = \\mathcal O(\\sqrt{\\kappa}\\log(1/\\varepsilon))$。"
zh_block_traj = (
    f"![旋转后的病态二次型上的迭代轨迹：GD 沿陡方向来回震荡，Nesterov 沿低谷顺势滑行]({URL6})\n\n"
    "左图：在 $\\kappa = 30$ 的旋转二次型上跑 80 代。"
    "GD（蓝色，为放大效果取 $\\eta = 1.9/L$）在陡方向上反复过冲，沿狭长山谷曲折前行，沿平方向几乎挪不动；"
    "Nesterov（紫色，$\\eta = 1/L$）则沿山谷积累动量，几乎不激发陡模。"
    "右图：到 $x^\\star$ 距离的对数图——两条曲线斜率的差距，正是定理 5 与定理 7 中 $\\kappa$ vs $\\sqrt{\\kappa}$ 速率差的几何投影。"
)
if zh_block_traj.split("\n")[0] not in text:
    if zh_anchor_traj not in text:
        raise SystemExit("ZH anchor 2 not found")
    text = text.replace(zh_anchor_traj, zh_anchor_traj + "\n\n" + zh_block_traj, 1)
    print("ZH patched: fig6")
else:
    print("ZH fig6 already present")

with open(ZH, "w") as f:
    f.write(text)
print("ZH file updated")
