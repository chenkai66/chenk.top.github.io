---
title: 'Optimization (12): Discrete and Global Optimization'
date: 2022-09-30 09:00:00
tags:
  - Optimization
  - Discrete Optimization
  - Portfolio Optimization
  - Metaheuristics
categories: Algorithm
lang: en
mathjax: true
description: "When variables are integer-valued or the problem is non-convex with multiple basins, classical convex methods stop working. This article surveys what does work: integer programming via branch-and-bound, LP relaxation gap analysis, the heuristic taxonomy (PSO, GA, SOA, simulated annealing), and a deep walk-through of the Spiral Optimization Algorithm applied to constrained mean-variance portfolio optimization."
disableNunjucks: true
translationKey: "optim-12"
series: optimization-theory
series_order: 12
aliases:
  - /en/standalone/solving-constrained-mean-variance-portfolio-optimization-pro/
---

The first eleven articles in this series tackled **continuous convex** problems (or convex relaxations of non-convex ones). This final article addresses two harder regimes:

- **Discrete optimization**: variables take integer or combinatorial values. The feasible set is a finite (but exponentially large) collection of points. Linear and convex tools no longer apply directly because there are no derivatives across the integer lattice.
- **Global non-convex optimization**: variables are continuous but the function has many local minima, and we want the *global* one. Methods like Newton and L-BFGS only find local minima.

Both regimes share a key feature: **provably optimal algorithms are exponential** in the worst case. Practical solutions come from (a) exact algorithms with smart pruning (branch-and-bound) and (b) heuristics that find good (but not optimal) solutions in polynomial time.

This article:

1. Reviews integer programming and the branch-and-bound algorithm with the LP relaxation as the bounding mechanism.
2. Surveys the heuristic landscape — particle swarm, genetic algorithms, simulated annealing, spiral optimization — and where each fits.
3. Walks through a substantial case study: applying the Spiral Optimization Algorithm to a constrained mean-variance portfolio problem (a mixed-integer nonlinear program, MINLP).

## What You Will Learn

- The integer programming formulation and how branch-and-bound prunes the search tree using LP relaxation bounds.
- Cutting planes, Gomory cuts, and the modern branch-and-cut framework that powers commercial MIP solvers.
- The heuristic taxonomy: trajectory-based (SA, tabu) vs population-based (GA, PSO, SOA), and a decision rubric for choosing among them.
- How a quadratic program turns into a mixed-integer nonlinear program when you add cardinality and buy-in constraints.
- The SOA update rule, why a rotation matrix plus geometric radius decay produces a useful exploration-exploitation schedule, and how it differs from PSO and GA.
- How to handle integer + box constraints with quadratic penalties (and why the penalty weight $\rho$ is the most subtle hyperparameter).
- Concrete numerical results from a published benchmark and an out-of-sample backtest, with commentary on what the evidence does and does not support.

## Prerequisites

[Articles 08](../08-lagrangian-duality-kkt/) (Lagrangian duality), 09 (interior-point) for the LP/QP solver mentions. Otherwise self-contained.

---

## Integer Programming and Branch-and-Bound

### A.1 The integer programming problem

A general **mixed-integer linear program** (MILP) is
$$
\begin{aligned}
\min_{x, z} \quad & c^\top x + d^\top z \\
\text{s.t.} \quad & A x + B z \leq b \\
& x \in \mathbb{R}^n_+, \quad z \in \mathbb{Z}^p_+.
\end{aligned}
$$
The integer variables $z$ make this NP-hard: even feasibility of pure 0-1 integer programs is NP-complete (it encodes 3SAT). The continuous variables $x$ can model production levels, prices, etc., while integer variables model decisions ("open this facility?", "use this route?").

The naive approach — enumerate all $2^p$ values of $z$, solve an LP for each — is hopeless for $p > 30$. Branch-and-bound is what makes practical MILPs solvable.

### A.2 LP relaxation

Drop the integrality constraint to get the **LP relaxation**:
$$
\min \, c^\top x + d^\top z \quad \text{s.t.} \quad A x + B z \leq b, \ x, z \geq 0.
$$
This LP gives a **lower bound** on the MILP optimum (any MILP-feasible solution is also LP-feasible). Solving it in polynomial time (interior-point methods, article 09) is the building block of all serious MILP solvers.

If the LP relaxation happens to have integer-valued $z^\star_{LP}$, we are done. Otherwise, pick a fractional $z^\star_{LP, j} \notin \mathbb{Z}$ and **branch**.

### A.3 Branch-and-bound

```yaml
Algorithm: Branch-and-Bound for MILP
Initialize: incumbent UB = +∞, problem queue = {original LP relaxation}
while queue not empty:
    Pop a sub-problem P from the queue
    Solve P's LP relaxation, get LB_P
    if LB_P ≥ UB: prune (this branch can't beat incumbent)
    if LB_P solution is integer-feasible:
        if c^T sol < UB: UB ← c^T sol, save incumbent
    else:
        Pick fractional variable z_j with value z_j^*
        Branch: queue += {P + (z_j ≤ ⌊z_j^*⌋), P + (z_j ≥ ⌈z_j^*⌉)}
return incumbent
```

Three key concepts:

- **Pruning by bound**: if the LP relaxation of a sub-problem already exceeds the best known integer solution, that entire sub-tree is discarded.
- **Pruning by infeasibility**: if a sub-problem's LP is infeasible, the sub-tree is empty.
- **Pruning by integrality**: if the LP relaxation happens to be integer-feasible, that's a leaf.

In the best case, branch-and-bound prunes most of the tree and finds the optimum after solving polynomially many LPs. In the worst case it still enumerates all $2^p$ leaves — confirming that the algorithm is exponential in the worst case.

![Branch-and-bound search tree with three pruning rules](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/12-discrete-global-optimization/fig_bnb_tree.png)

The figure above traces a small B&B run with two integer variables. The root LP gives a fractional solution $(2.4, 1.7)$ and a lower bound of $8.5$. We branch on $z_1$, then on $z_2$ inside each child. Node P4 is integer-feasible and becomes the incumbent with $\text{UB} = 9.6$; node P5 is also integer-feasible but its objective $9.7$ is worse than the incumbent so it is pruned by bound; node P6 is LP-infeasible and pruned. Whole sub-trees never need to be solved — this is what makes B&B fast in practice even though its worst case is still exponential.

### A.4 Cutting planes

A **cutting plane** is an inequality $\alpha^\top z \leq \beta$ that is satisfied by every integer-feasible $z$ but violated by the current LP relaxation optimum. Adding cuts tightens the LP relaxation, raising the lower bound and pruning more aggressively.

**Gomory cut**. From the simplex tableau of the LP optimum, if a basic variable $z_j$ has value $\bar z_j = \lfloor \bar z_j \rfloor + f_j$ with $0 < f_j < 1$, then
$$
\sum_k \mathrm{frac}(\bar a_{jk}) \cdot z_k \geq f_j
$$
holds for every integer $z$ but not for the fractional LP optimum. (Here $\bar a_{jk}$ are the tableau entries and $\mathrm{frac}(x) = x - \lfloor x \rfloor$.)

Modern MILP solvers (Gurobi, CPLEX, SCIP) use 10+ kinds of cuts: Gomory, mixed-integer rounding, lift-and-project, clique cuts, flow cover cuts. The modern algorithm is **branch-and-cut**: at each node, try to add violated cuts before branching.

![LP relaxation polytope and the effect of a cutting plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/12-discrete-global-optimization/fig_lp_relaxation.png)

The left panel shows the geometric content of LP relaxation: the LP optimum (amber star) sits at a fractional vertex of the polytope, while the IP optimum (purple disk) is one of the green integer lattice points strictly inside the relaxation. The right panel adds one cutting plane $z_1 + z_2 \le 4$ (orange line). The original fractional vertex is now infeasible (purple cross), every green integer point is preserved, and the new LP vertex happens to be integer — the LP relaxation alone now solves the IP. In real solvers cuts rarely close the gap in one shot, but each one tightens the bound and makes B&B prune more aggressively.

### A.5 What you can solve in practice

| Problem size            | Realistic with modern solvers? |
| ----------------------- | ------------------------------ |
| 0--1 IP, 100 vars       | Yes, seconds to minutes.       |
| 0--1 IP, 1000 vars      | Often yes, minutes to hours.   |
| Mixed IP, 10K vars      | Sometimes, depends on structure. |
| Mixed IP, $\geq 100$K   | Usually requires decomposition (Benders, column generation). |

The frontier moves: problems considered intractable in 1990 are routine today, due to a $10^{12}$ × speedup from algorithm improvements (cuts, presolve) and a $10^6$ × speedup from hardware.

---

## Heuristics for Hard Problems

When branch-and-bound is too slow or the problem is non-linear (MINLP), heuristics are the fallback. They sacrifice optimality guarantees for tractable runtime.

### B.1 Taxonomy

| Family          | Examples                          | Strengths                                  |
| --------------- | --------------------------------- | ------------------------------------------ |
| Trajectory      | Hill climbing, simulated annealing, tabu search | Simple, low-memory, good for combinatorial |
| Population      | Genetic algorithms, PSO, SOA, differential evolution | Parallelizable, escapes local minima       |
| Constructive    | Greedy, ant colony, GRASP         | Build solutions step by step                |
| Hybrid          | Memetic algorithms, large-neighborhood search | Combine local and global search             |

![Heuristic taxonomy: trajectory, population, constructive, hybrid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/12-discrete-global-optimization/fig_heuristic_taxonomy.png)

The four families above are the standard cuts in the metaheuristic literature. Trajectory methods (single-state) are cheap and effective on combinatorial problems; population methods carry many candidates in parallel and are the natural fit for continuous multi-modal landscapes; constructive methods build a solution piece by piece and dominate problems with strong sequential structure; hybrid methods combine an outer global search with an inner local-search or exact solver.

**Simulated annealing (SA)**: random walk on the solution space, accepting moves that improve and accepting worse moves with probability $e^{-\Delta f / T}$. The temperature $T$ decays over time — high $T$ explores, low $T$ exploits. Theoretically converges to the global optimum if $T$ decays slowly enough; practically requires careful cooling schedules.

**Genetic algorithm (GA)**: maintain a population of solutions; combine pairs via crossover and mutation; keep the best. Handles binary and categorical decision variables naturally. Notoriously sensitive to operator design.

**Particle swarm (PSO)**: $N$ particles each with position $x_i$ and velocity $v_i$ in continuous space; velocity updates pull each particle toward the swarm-best and its own best:
$$
v_i \leftarrow w v_i + c_1 r_1 (p_i^{\text{best}} - x_i) + c_2 r_2 (p^{\text{global best}} - x_i).
$$
Strong on continuous problems with multiple basins.

**Spiral optimization (SOA)**: similar to PSO but each particle follows a logarithmic spiral toward the current incumbent, with the spiral's radius shrinking geometrically. We treat this in detail in the case study below.

### B.2 Choosing among heuristics

| If your problem is...                                   | Try...                          |
| ------------------------------------------------------- | ------------------------------- |
| Binary / combinatorial (TSP, scheduling)                | SA, tabu, GA                    |
| Continuous with multiple basins                         | PSO, DE, CMA-ES, SOA            |
| Mixed integer with continuous variables                 | Hybrid (GA outer + LP inner)    |
| Convex but very large                                   | Don't use heuristics; use IPM/SGD |
| Black-box (no gradient, expensive function evaluations) | Bayesian optimization, GA       |

The dominant lesson from the literature: no heuristic uniformly dominates. Choosing the right one for a problem is craftsmanship. Worth tuning for a few weeks if the problem will be solved repeatedly; otherwise just use whatever is in your favorite library.

![SA, GA, and PSO on a multi-modal 1-D landscape](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/12-discrete-global-optimization/fig_heuristic_convergence.png)

A toy illustration on a 1-D multi-modal objective. The left panel shows the landscape and where each method ends up after a fixed budget: SA (blue) and GA (green population) reach the global basin near the amber star, while PSO (purple diamond) is trapped in a shallow local minimum on the left because all particles converged toward an early-found basin. The right panel plots best-so-far value over iterations: SA and GA approach the global optimum (dashed amber); PSO plateaus above it. The point is not that PSO is bad — with a different seed and inertia schedule it would also reach the global optimum — but that no method dominates universally and seed-to-seed variance is real. This is why production runs use 30+ independent restarts.

---

## Case Study — The Spiral Optimization Algorithm on a Mean-Variance Portfolio

The remainder of this article is a deep walk-through of applying SOA to a real MINLP from finance. The running example is constrained mean-variance portfolio selection.

Markowitz's mean-variance model is elegant until you add real trading constraints: "if you buy a stock at all, hold at least 5% of it" and "pick exactly 10 names from the S&P 500." The closed-form quadratic program quietly mutates into a *mixed-integer nonlinear program* (MINLP), and the standard solver chain (Lagrange multipliers, KKT conditions, interior-point methods) stops working. The case study reviewed here applies the **Spiral Optimization Algorithm** (SOA), a population-based metaheuristic, to this problem and shows it can find competitive feasible solutions where gradient methods fail outright.

## From Quadratic Program to MINLP

### The classical mean-variance problem

Let $\mathbf{y} \in \mathbb{R}^n$ be the vector of capital fractions, $\overline{\mathbf{r}} \in \mathbb{R}^n$ the vector of expected asset returns, and $Q \in \mathbb{R}^{n \times n}$ the positive semidefinite covariance matrix of returns. For a target portfolio return $R_p$, the *long-only* mean-variance problem is
$$
\begin{aligned}
\min_{\mathbf{y}} \quad & V(\mathbf{y}) = \mathbf{y}^\top Q \mathbf{y} \\
\text{s.t.} \quad & \overline{\mathbf{r}}^\top \mathbf{y} = R_p, \\
& \mathbf{e}^\top \mathbf{y} = 1, \\
& y_i \geq 0, \quad i = 1, \dots, n,
\end{aligned}
$$
where $\mathbf{e}$ is the all-ones vector. This is a convex quadratic program. Sweep $R_p$ across an interval and you trace the *efficient frontier*: the locus of portfolios that minimise variance for each return level.

![Mean-variance frontier with cardinality constraint](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/spiral-portfolio/fig1_efficient_frontier.png)

The figure above shows the geometry on a five-asset universe. The cloud of dots is 5,000 random portfolios sampled uniformly from the simplex (coloured by Sharpe-like ratio). The purple curve is the unconstrained efficient frontier (shorting permitted), and the dashed blue curve is the *cardinality-constrained* frontier with $K=3$. Two observations are immediate: (i) the cardinality-constrained frontier sits to the right of the unconstrained one at every return level (less choice means less diversification, which means more risk), and (ii) the gap between the two is *not* uniform in $R_p$. At extreme returns the gap widens because only a few combinations can hit the target at all.

### Adding the buy-in threshold

Real desks rarely hold a 0.3% position in a stock. The buy-in threshold says: if you hold asset $i$ at all, hold at least $l_i$. Introduce a binary indicator $z_i \in \{0, 1\}$ for inclusion and link it to $y_i$ via box constraints:
$$l_i z_i \leq y_i \leq u_i z_i, \qquad 0 < l_i < u_i \leq 1, \qquad z_i \in \{0, 1\}.$$
When $z_i = 0$ the entire row collapses to $y_i = 0$. When $z_i = 1$ the weight is forced into $[l_i, u_i]$. This is the precise mathematical instant at which the problem becomes mixed-integer: the feasible set is now a finite union of polytopes (one per choice of $\mathbf{z}$), and convexity is gone.

### Adding the cardinality constraint

The cardinality constraint pins the portfolio to exactly $K$ assets:
$$\sum_{i=1}^{n} z_i = K.$$
Combining everything yields the full MINLP studied in the paper:
$$
\begin{aligned}
\min_{\mathbf{y}, \mathbf{z}} \quad & V(\mathbf{y}) = \mathbf{y}^\top Q \mathbf{y} \\
\text{s.t.} \quad & \overline{\mathbf{r}}^\top \mathbf{y} = R_p, \\
& \mathbf{e}^\top \mathbf{y} = 1, \\
& \sum_{i=1}^{n} z_i = K, \\
& l_i z_i \leq y_i \leq u_i z_i, \\
& z_i \in \{0, 1\}, \quad i = 1, \dots, n.
\end{aligned}
$$
This object has $\binom{n}{K}$ combinatorial branches. Even at $n = 100, K = 10$ that is $1.7 \times 10^{13}$ subsets, well outside what brute-force enumeration can touch. Branch-and-bound MINLP solvers (BARON, SCIP, Bonmin) can attack it but their wall clock grows quickly; this is the scale at which metaheuristics become attractive.

## The Spiral Optimization Algorithm

### Update rule

SOA, introduced by Tamura and Yasuda (2011), is a population-based metaheuristic inspired by the logarithmic spirals seen in plant phyllotaxis and galactic arms. At iteration $k$, each candidate $\mathbf{x}_k^{(j)}$ is updated toward the current best $\mathbf{x}^*$ via
$$\mathbf{x}_{k+1}^{(j)} \;=\; \mathbf{x}^* \;+\; r \cdot R(\theta) \, \big(\mathbf{x}_k^{(j)} - \mathbf{x}^*\big),$$
where $R(\theta)$ is a $d$-dimensional rotation matrix with angle $\theta$, and $r \in (0, 1)$ is a contraction factor. The composition of rotation and contraction traces a logarithmic spiral inward.

![SOA spiral trajectory and radius schedule](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/spiral-portfolio/fig2_spiral_trajectory.png)

The left panel shows the trajectories of five candidates initialised in the four quadrants of a non-convex landscape. The amber star marks the current best (which happens to be the global minimum here). Each candidate spirals inward, sampling the loss along the way. The right panel makes the *exploration vs exploitation* trade-off explicit: the geometric envelope $r^k$ governs how fast the spiral collapses. A slow shrink ($r = 0.95$) keeps candidates wandering far from $\mathbf{x}^*$ for many iterations (more exploration), while a fast shrink ($r = 0.85$) collapses them quickly onto the incumbent (more exploitation).

### Why the spiral specifically?

Compared to other metaheuristics:

- **Particle swarm optimization (PSO)** updates each particle by a velocity that mixes its personal best and the global best, plus stochastic noise. The velocity has no built-in shrinkage; you typically need a separate inertia decay schedule and tuning of the cognitive / social weights.
- **Genetic algorithms (GA)** use crossover and mutation. They handle integer variables naturally but the schema theorem is weak in continuous spaces; convergence is often slow.
- **Simulated annealing (SA)** uses a single trajectory with temperature decay. No population means no parallelism in the search.

SOA's selling point is that the rotation $R(\theta)$ guarantees the candidate cycles around the incumbent (so it samples *different* directions deterministically), while the contraction $r$ guarantees eventual convergence. The exploration-exploitation balance reduces to a single hyperparameter: the spectral radius of $r R(\theta)$.

### Updating the incumbent

After each candidate is moved, the population is re-evaluated and $\mathbf{x}^*$ is updated to the best point seen so far. This is the only stochastic element in classical SOA: the initial sampling. Some variants (including the one in the paper) inject random perturbations on candidates that have stagnated, to escape the basin of the current incumbent.

## Constraint Handling

### Quadratic penalty

The paper handles all constraints with a quadratic penalty:
$$\min_{\mathbf{y}, \mathbf{z}} \; F(\mathbf{y}, \mathbf{z}) = V(\mathbf{y}) + \rho \cdot P(\mathbf{y}, \mathbf{z}),$$
where $P$ measures total constraint violation:
$$
P = \big(\overline{\mathbf{r}}^\top \mathbf{y} - R_p\big)^2
  \,+\, \big(\mathbf{e}^\top \mathbf{y} - 1\big)^2
  \,+\, \sum_{i=1}^{n} \max(0, l_i z_i - y_i)^2
  \,+\, \sum_{i=1}^{n} \max(0, y_i - u_i z_i)^2
  \,+\, \Big(\sum_i z_i - K\Big)^2.
  $$
The integer constraint $z_i \in \{0,1\}$ is enforced by *rounding*: SOA searches over $z_i \in [0,1]$ continuously and rounds to the nearest integer when evaluating $P$.

![Penalty pulls the optimum into the feasible band, plus 2-D feasibility map](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/spiral-portfolio/fig3_constraint_handling.png)

The left panel shows what the penalty does on a 1-D weight slice: the raw variance $V(y)$ (grey dashed) has its minimum sitting in the infeasible region (purple dot, below the buy-in threshold). Adding $\rho \cdot P(y)$ produces sharp parabolic walls outside the feasible band $[l, u]$; the resulting penalised objective (solid blue) has its optimum pulled into the green strip (amber diamond). The right panel shows a 2-D feasibility map for two assets with cardinality $K=1$: only the green region is feasible. Crosses are infeasible candidates, dots are feasible ones.

### The penalty weight $\rho$ is subtle

This is the most-fiddled hyperparameter in metaheuristic-with-penalty literature, and it is genuinely tricky:

- **Too small** and the algorithm finds beautifully low-variance portfolios that are infeasible (don't sum to 1, or hold sub-threshold positions). The penalty is a soft suggestion, not a barrier.
- **Too large** and $V$ becomes a rounding error inside $\rho P$. Numerical precision suffers; the search effectively optimises feasibility alone, ignoring variance.
- **Just right** is problem-specific. The paper uses $\rho = 10^4$ for the five-asset case, which works because $V$ is $O(1)$ and $P$ is small but non-zero.

A more robust alternative is the *augmented Lagrangian* approach, which adapts $\rho$ over iterations based on observed violations. The paper does not use this, so re-tuning $\rho$ is part of the cost when porting the method to a new universe.

### Repair operators

After each spiral update, candidates can drift outside the unit box. The paper applies a simple *repair*: clip $y_i$ to $[0, 1]$ and renormalise so $\mathbf{e}^\top \mathbf{y} = 1$. This is cheap and keeps every candidate trivially feasible with respect to the budget constraint, leaving only target return, buy-in, and cardinality to the penalty.

## Numerical Results

### The benchmark

Following Bartholomew-Biggs and Kane (2009), the paper uses a five-asset universe with mean return vector
$$\overline{\mathbf{r}} = (0.10, 0.13, 0.085, 0.155, 0.07)^\top$$
and a $5 \times 5$ positive semidefinite covariance matrix (the precise values are in the paper). Target return $R_p = 0.05$, buy-in $l_i = 0.05$, cardinality $K = 5$ (all assets active), penalty $\rho = 10^4$, 50 iterations.

### Convergence comparison

![Convergence vs Quasi-Newton, DIRECT, and PSO](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/spiral-portfolio/fig4_convergence.png)

The figure compares best-so-far variance over iterations for SOA-MINLP, Quasi-Newton, DIRECT, and a PSO baseline I added for context. The shaded blue band is the 10-90 percentile of 30 independent SOA runs; the solid blue curve is the median.

Two things are worth noticing. First, the final values rank consistently with what the paper reports: SOA-MINLP at $V = 0.6969$, Quasi-Newton at $0.7123$, DIRECT at $0.7458$, with PSO landing in between at $0.7250$. Quasi-Newton converges fast in iteration count but to a worse local optimum because the penalty surface is non-smooth and gradient-based methods get stuck. DIRECT (a deterministic Lipschitz-based partition method) is more thorough but pays for it in iterations. Second, the SOA band is narrow by iteration 60 — the run-to-run variability is small in this regime, which is reassuring for a stochastic method.

The catch: this is a five-asset problem. Every claim about SOA's relative ranking should be re-checked at scale.

### An out-of-sample backtest

To pressure-test the *portfolio* (not just the *solver*), I simulated three years of daily returns from the multivariate Gaussian implied by $\overline{\mathbf{r}}$ and $Q$ and compared three rules: equal weight, unconstrained mean-variance at target return 11%, and an SOA-MINLP-style portfolio (long-only, $K=3$, buy-in $0.10$).

![Out-of-sample backtest equity and drawdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/spiral-portfolio/fig5_backtest.png)

The unconstrained MV portfolio has the highest in-model Sharpe but it shorts assets and concentrates aggressively, which translates into a worse drawdown (bottom panel) than the SOA-MINLP portfolio. Equal weight is the most defensive but leaves return on the table. The SOA-MINLP rule hits a sweet spot: the cardinality and buy-in constraints regularise the portfolio, giving up a little expected return for materially better drawdown behaviour. This is the practical case for cardinality constraints: not theoretical optimality, but *risk diversification you can implement on a real desk*.

## When to Use SOA (and When Not To)

**Use SOA when:**

- The problem is non-convex with discrete or combinatorial structure (cardinality, sector limits, integer lot sizes).
- The asset universe is small to medium ($n \lesssim 100$).
- You can afford 30+ independent runs to assess solution stability.
- You don't need an optimality certificate, just a good feasible solution.

**Skip SOA when:**

- The problem is convex or near-convex (use a QP solver: OSQP, CVXOPT, MOSEK).
- $n > 1000$ (use commercial MINLP: Gurobi, CPLEX, BARON; or specialised methods like the lasso-style relaxations of Bertsimas et al.).
- You need real-time rebalancing under tight latency budgets.
- The constraints are simple bounds (just project; no fancy solver needed).

## Hyperparameter Tuning, in Practice

| Hyperparameter | Typical range | Effect |
| -------------- | ------------- | ------ |
| Population size $N$ | 30 - 100 | Larger $N$: better exploration, linear cost per iteration |
| Max iterations | 50 - 500 | Read off from a convergence plot; stop when median plateau is flat |
| Spiral angle $\theta$ | $\pi / 6$ to $\pi / 3$ | Larger angle: more circumferential exploration |
| Contraction $r$ | 0.85 - 0.95 | Smaller $r$: faster convergence, higher local-optimum risk |
| Penalty weight $\rho$ | $10^2$ - $10^6$ | Tune until violations are zero on the median run |

**Rule of thumb to start with:** $N = 50$, max_iter $= 100$, $\theta = \pi / 4$, $r = 0.92$, $\rho = 10^4$. Run 30 trials and inspect the convergence band. If the band stays wide at iteration 100, increase $N$. If violations persist, scale $\rho$ up by an order of magnitude.

## Limitations and Honest Caveats

- **No optimality certificate.** SOA is a metaheuristic. You can show it found a good feasible point; you cannot show it found *the* optimum. For regulated capital (Basel, Solvency II), this matters.
- **Stochastic outputs.** Two runs from different seeds give different portfolios. Operationally, you need a tie-breaking rule (lowest variance? closest to a benchmark? ensemble?).
- **Non-stationarity.** $Q$ and $\overline{\mathbf{r}}$ are estimated from finite history and are noisy. A solver that finds the variance-minimising portfolio under a *wrong* covariance matrix is not obviously better than one that finds an approximate solution. Robust optimization (worst-case over a covariance ambiguity set) is more important than solver perfection.
- **Penalty $\rho$ requires re-tuning.** Every new universe means another sweep. Augmented Lagrangian variants help.
- **Rotation matrix in high dimensions.** Constructing a meaningful rotation in $\mathbb{R}^{500}$ is non-trivial; the original SOA paper proposes Householder-based constructions, but the empirical performance degrades past a few hundred dimensions.

## Summary

The paper makes a focused, defensible claim: a modified SOA with quadratic penalties handles cardinality and buy-in constraints competitively against Quasi-Newton and DIRECT on a small benchmark. The figure I would draw from this is more cautious. SOA is not a replacement for commercial MINLP solvers at scale, but it is a useful tool in the band where the universe is too small to warrant Gurobi licences and too constrained for vanilla quadratic programming. The cardinality and buy-in constraints, in turn, are not academic curiosities: they materially regularise out-of-sample risk, as the backtest above shows. The methodological lesson is that the *constraints* often matter more than the *solver*: a good portfolio with the right constraints, found by an okay solver, will usually beat a "perfect" portfolio with the wrong constraints.

## References

1. Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77-91.
2. Tamura, K., & Yasuda, K. (2011). *Spiral Dynamics Inspired Optimization*. Journal of Advanced Computational Intelligence and Intelligent Informatics, 15(8), 1116-1122.
3. Kania, A., & Sidarto, K. A. (2016). *Solving Mixed Integer Nonlinear Programming Using Spiral Dynamics Optimization Algorithm*. AIP Conference Proceedings, 1716.
4. Bartholomew-Biggs, M., & Kane, S. J. (2009). *A Global Optimization Problem in Portfolio Selection*. Computational Management Science, 6(3), 329-345.
5. Bertsimas, D., & Cory-Wright, R. (2022). *A Scalable Algorithm for Sparse Portfolio Selection*. INFORMS Journal on Computing.
