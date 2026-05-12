#!/usr/bin/env python3
"""Insert 4 figures into EN and ZH discrete-global-optimization article."""
import sys

EN_PATH = "/root/chenk-hugo/content/en/optimization-theory/12-discrete-global-optimization.md"
ZH_PATH = "/root/chenk-hugo/content/zh/optimization-theory/12-discrete-global-optimization.md"

BASE_URL = "https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/12-discrete-global-optimization"

EN_EDITS = [
    (
        "In the best case, branch-and-bound prunes most of the tree and finds the optimum after solving polynomially many LPs. In the worst case it still enumerates all $2^p$ leaves — confirming that the algorithm is exponential in the worst case.\n\n### A.4 Cutting planes",
        "In the best case, branch-and-bound prunes most of the tree and finds the optimum after solving polynomially many LPs. In the worst case it still enumerates all $2^p$ leaves — confirming that the algorithm is exponential in the worst case.\n\n![Branch-and-bound search tree with three pruning rules](" + BASE_URL + "/fig_bnb_tree.png)\n\nThe figure above traces a small B&B run with two integer variables. The root LP gives a fractional solution $(2.4, 1.7)$ and a lower bound of $8.5$. We branch on $z_1$, then on $z_2$ inside each child. Node P4 is integer-feasible and becomes the incumbent with $\\text{UB} = 9.6$; node P5 is also integer-feasible but its objective $9.7$ is worse than the incumbent so it is pruned by bound; node P6 is LP-infeasible and pruned. Whole sub-trees never need to be solved — this is what makes B&B fast in practice even though its worst case is still exponential.\n\n### A.4 Cutting planes",
    ),
    (
        "Modern MILP solvers (Gurobi, CPLEX, SCIP) use 10+ kinds of cuts: Gomory, mixed-integer rounding, lift-and-project, clique cuts, flow cover cuts. The modern algorithm is **branch-and-cut**: at each node, try to add violated cuts before branching.\n\n### A.5 What you can solve in practice",
        "Modern MILP solvers (Gurobi, CPLEX, SCIP) use 10+ kinds of cuts: Gomory, mixed-integer rounding, lift-and-project, clique cuts, flow cover cuts. The modern algorithm is **branch-and-cut**: at each node, try to add violated cuts before branching.\n\n![LP relaxation polytope and the effect of a cutting plane](" + BASE_URL + "/fig_lp_relaxation.png)\n\nThe left panel shows the geometric content of LP relaxation: the LP optimum (amber star) sits at a fractional vertex of the polytope, while the IP optimum (purple disk) is one of the green integer lattice points strictly inside the relaxation. The right panel adds one cutting plane $z_1 + z_2 \\le 4$ (orange line). The original fractional vertex is now infeasible (purple cross), every green integer point is preserved, and the new LP vertex happens to be integer — the LP relaxation alone now solves the IP. In real solvers cuts rarely close the gap in one shot, but each one tightens the bound and makes B&B prune more aggressively.\n\n### A.5 What you can solve in practice",
    ),
    (
        "| Hybrid          | Memetic algorithms, large-neighborhood search | Combine local and global search             |\n\n**Simulated annealing (SA)**:",
        "| Hybrid          | Memetic algorithms, large-neighborhood search | Combine local and global search             |\n\n![Heuristic taxonomy: trajectory, population, constructive, hybrid](" + BASE_URL + "/fig_heuristic_taxonomy.png)\n\nThe four families above are the standard cuts in the metaheuristic literature. Trajectory methods (single-state) are cheap and effective on combinatorial problems; population methods carry many candidates in parallel and are the natural fit for continuous multi-modal landscapes; constructive methods build a solution piece by piece and dominate problems with strong sequential structure; hybrid methods combine an outer global search with an inner local-search or exact solver.\n\n**Simulated annealing (SA)**:",
    ),
    (
        "The dominant lesson from the literature: no heuristic uniformly dominates. Choosing the right one for a problem is craftsmanship. Worth tuning for a few weeks if the problem will be solved repeatedly; otherwise just use whatever is in your favorite library.\n\n---\n\n## Part C:",
        "The dominant lesson from the literature: no heuristic uniformly dominates. Choosing the right one for a problem is craftsmanship. Worth tuning for a few weeks if the problem will be solved repeatedly; otherwise just use whatever is in your favorite library.\n\n![SA, GA, and PSO on a multi-modal 1-D landscape](" + BASE_URL + "/fig_heuristic_convergence.png)\n\nA toy illustration on a 1-D multi-modal objective. The left panel shows the landscape and where each method ends up after a fixed budget: SA (blue) and GA (green population) reach the global basin near the amber star, while PSO (purple diamond) is trapped in a shallow local minimum on the left because all particles converged toward an early-found basin. The right panel plots best-so-far value over iterations: SA and GA approach the global optimum (dashed amber); PSO plateaus above it. The point is not that PSO is bad — with a different seed and inertia schedule it would also reach the global optimum — but that no method dominates universally and seed-to-seed variance is real. This is why production runs use 30+ independent restarts.\n\n---\n\n## Part C:",
    ),
]

ZH_EDITS = [
    (
        "在最好情形下，分支定界法能剪去绝大部分搜索树，仅需求解多项式数量级的 LP 即可找到最优解；而在最坏情形下，它仍需遍历全部 $2^p$ 个叶节点——这印证了该算法在最坏情况下的指数时间复杂度。\n\n### A.4 割平面法",
        "在最好情形下，分支定界法能剪去绝大部分搜索树，仅需求解多项式数量级的 LP 即可找到最优解；而在最坏情形下，它仍需遍历全部 $2^p$ 个叶节点——这印证了该算法在最坏情况下的指数时间复杂度。\n\n![分支定界搜索树与三种剪枝规则](" + BASE_URL + "/fig_bnb_tree.png)\n\n上图展示了一个包含两个整数变量的小型 B&B 运行。根节点 LP 的最优解为分数 $(2.4, 1.7)$，下界 $\\text{LB}=8.5$。我们先在 $z_1$ 上分支，再在每个子节点中对 $z_2$ 分支。节点 P4 是整数可行解，成为当前最优解（incumbent），上界更新为 $\\text{UB}=9.6$；节点 P5 也是整数可行解，但其目标值 $9.7$ 劣于当前最优解，被界剪枝；节点 P6 的 LP 不可行，被不可行性剪枝。诸多整个子树根本无需求解——这正是 B&B 在实践中快速高效的原因，尽管其最坏情况仍为指数。\n\n### A.4 割平面法",
    ),
    (
        "现代 MILP 求解器（如 Gurobi、CPLEX、SCIP）综合运用十余类割平面：Gomory 割、混合整数舍入割（MIR）、提升与投影割（lift-and-project）、团割（clique cut）、流覆盖割（flow cover cut）等。当前主流算法为**分支割平面法**（branch-and-cut）：在每个搜索节点处，先尝试生成并添加被违反的割平面，再决定是否分支。\n\n### A.5 实践中可求解的问题规模",
        "现代 MILP 求解器（如 Gurobi、CPLEX、SCIP）综合运用十余类割平面：Gomory 割、混合整数舍入割（MIR）、提升与投影割（lift-and-project）、团割（clique cut）、流覆盖割（flow cover cut）等。当前主流算法为**分支割平面法**（branch-and-cut）：在每个搜索节点处，先尝试生成并添加被违反的割平面，再决定是否分支。\n\n![LP 松弛多面体与割平面的作用](" + BASE_URL + "/fig_lp_relaxation.png)\n\n左图展示 LP 松弛的几何含义：LP 最优解（琰色五角星）位于多面体的某个分数顶点上，而整数规划（IP）的最优解（紫色圆点）是松弛区域内部某个绿色整数格点。右图加入一条割平面 $z_1+z_2\\le 4$（橙色直线）：原本的分数顶点被切掉（紫色 ×），所有绿色整数点仍被保留，而新的 LP 顶点恰好是整数点——仅凭 LP 松弛就已解出了原 IP。实际求解器中单一割平面很少能一步到位，但每一条割都会收紧下界，使 B&B 能更激进地剪枝。\n\n### A.5 实践中可求解的问题规模",
    ),
    (
        "| 混合类（Hybrid）       | 模因算法（Memetic algorithms）、大规模邻域搜索（LNS）         | 融合局部搜索与全局搜索能力                         |\n\n**模拟退火",
        "| 混合类（Hybrid）       | 模因算法（Memetic algorithms）、大规模邻域搜索（LNS）         | 融合局部搜索与全局搜索能力                         |\n\n![启发式算法分类：轨迹、种群、构造与混合](" + BASE_URL + "/fig_heuristic_taxonomy.png)\n\n上图是启发式（metaheuristic）文献中的标准分类。轨迹类（单状态）开销低，适合组合优化问题；种群类并行携带多个候选解，是连续多峰问题的天然选择；构造类逐步拼接可行解，在序列结构强的问题上占优；混合类则将外层全局搜索与内层的局部搜索或精确求解器结合。\n\n**模拟退火",
    ),
    (
        "文献中的核心共识是：**不存在一种启发式方法能在所有问题上全面胜出**。为特定问题选择合适的方法是一门技艺（craftsmanship）。若该问题需反复求解，值得投入数周时间调参与定制；否则，直接采用你最熟悉的优化库（如 `scipy.optimize`、`DEAP` 或 `pyswarms`）中已实现的成熟版本即可。\n\n---\n\n## 第三部分 C：",
        "文献中的核心共识是：**不存在一种启发式方法能在所有问题上全面胜出**。为特定问题选择合适的方法是一门技艺（craftsmanship）。若该问题需反复求解，值得投入数周时间调参与定制；否则，直接采用你最熟悉的优化库（如 `scipy.optimize`、`DEAP` 或 `pyswarms`）中已实现的成熟版本即可。\n\n![SA、GA 与 PSO 在一维多峰函数上的表现](" + BASE_URL + "/fig_heuristic_convergence.png)\n\n一维多峰目标函数上的玩具示例。左图展示函数地貌与各方法在固定预算后的最终位置：SA（蓝色）与 GA（绿色种群）都接近了琰色星所在的全局低谷，而 PSO（紫色菱形）被左侧一个浅局部最优困住，因为所有粒子都过早向初期发现的那个盆地收敛。右图绘出迭代过程中的历史最优值：SA 与 GA 逐渐逼近全局最优值（虚线琰色），PSO 则停滞在一个较差的值上。重点不在于 PSO 本身差——换个随机种子与惯性调度后，它同样能找到全局最优——而是没有任何方法能全面胜出，且随机种子间的方差不容忽视。这正是为什么生产环境中总是运行 30 次以上独立重启。\n\n---\n\n## 第三部分 C：",
    ),
]


def apply_edits(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for i, (old, new) in enumerate(edits, 1):
        if old not in text:
            print(f"FAIL: edit {i} anchor not found in {path}")
            sys.exit(1)
        if text.count(old) > 1:
            print(f"FAIL: edit {i} anchor not unique in {path}, count={text.count(old)}")
            sys.exit(1)
        text = text.replace(old, new, 1)
        print(f"OK: edit {i} applied to {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


apply_edits(EN_PATH, EN_EDITS)
apply_edits(ZH_PATH, ZH_EDITS)
print("DONE")
