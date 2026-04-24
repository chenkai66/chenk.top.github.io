"""
Figure generation script for ML Math Derivations Part 15:
Hidden Markov Models.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single concept so the underlying mathematics becomes
visible at a glance.

Figures:
    fig1_graphical_model        Plate / unrolled graphical model showing the
                                latent Markov chain i_1 -> i_2 -> ... -> i_T
                                emitting observations o_1, ..., o_T.
    fig2_transition_diagram     Three-state weather automaton (Sunny / Rainy /
                                Cloudy) with annotated transition probabilities
                                drawn as a directed graph with self-loops.
    fig3_forward_trellis        Forward algorithm trellis: alpha values flow
                                left to right, with a sum (Sigma) at every
                                node showing how partial probabilities are
                                accumulated.
    fig4_backward_trellis       Backward algorithm trellis: beta values flow
                                right to left from the boundary beta_T = 1.
    fig5_viterbi_path           Viterbi trellis with the single most-likely
                                path highlighted; back-pointers visualised by
                                arrow weight.
    fig6_baum_welch             Baum-Welch / EM monotonic log-likelihood
                                ascent over iterations on a synthetic HMM,
                                with a small inset showing parameter recovery.
    fig7_pos_tagging            POS tagging application: the Viterbi tags for
                                "I love natural language processing" with a
                                small per-token emission heatmap.

Usage:
    python3 scripts/figures/ml-math-derivations/15-hmm.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so that the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "15-Hidden-Markov-Models"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "15-隐马尔可夫模型"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _node(ax, xy, label, color, *, radius=0.32, text_color="white", fs=12):
    c = Circle(xy, radius, facecolor=color, edgecolor=C_DARK, lw=1.4, zorder=3)
    ax.add_patch(c)
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fs,
        fontweight="bold",
        zorder=4,
    )


def _arrow(ax, p0, p1, *, color=C_DARK, lw=1.3, style="-|>", mut=14, alpha=1.0,
           rad=0.0, ls="-"):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=mut,
        color=color,
        lw=lw,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Graphical model (unrolled)
# ---------------------------------------------------------------------------
def fig1_graphical_model() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.4))

    T = 5
    xs = np.linspace(0.6, 9.4, T)
    y_hidden = 2.4
    y_obs = 0.7

    # Hidden states (top row)
    for t, x in enumerate(xs, start=1):
        _node(ax, (x, y_hidden), f"$i_{t}$", C_PURPLE)
    # Observations (bottom row)
    for t, x in enumerate(xs, start=1):
        _node(ax, (x, y_obs), f"$o_{t}$", C_AMBER)

    # Transition arrows (hidden -> hidden)
    for t in range(T - 1):
        _arrow(
            ax,
            (xs[t] + 0.32, y_hidden),
            (xs[t + 1] - 0.32, y_hidden),
            color=C_PURPLE,
            lw=1.7,
        )
    # Emission arrows (hidden -> observation)
    for t in range(T):
        _arrow(
            ax,
            (xs[t], y_hidden - 0.32),
            (xs[t], y_obs + 0.32),
            color=C_AMBER,
            lw=1.5,
        )

    # Labels
    ax.text(
        xs[0] - 0.7,
        y_hidden,
        "Hidden:",
        ha="right",
        va="center",
        fontsize=11,
        color=C_PURPLE,
        fontweight="bold",
    )
    ax.text(
        xs[0] - 0.7,
        y_obs,
        "Observed:",
        ha="right",
        va="center",
        fontsize=11,
        color=C_AMBER,
        fontweight="bold",
    )

    # Annotation for transition / emission
    ax.annotate(
        r"$a_{ij}=P(i_{t+1}=j\,|\,i_t=i)$",
        xy=((xs[1] + xs[2]) / 2, y_hidden + 0.05),
        xytext=((xs[1] + xs[2]) / 2, 3.55),
        ha="center",
        fontsize=10.5,
        color=C_PURPLE,
        arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=0.8),
    )
    ax.annotate(
        r"$b_j(k)=P(o_t=v_k\,|\,i_t=j)$",
        xy=(xs[3], (y_hidden + y_obs) / 2),
        xytext=(xs[3] + 1.3, 1.55),
        ha="left",
        fontsize=10.5,
        color=C_AMBER,
        arrowprops=dict(arrowstyle="-", color=C_AMBER, lw=0.8),
    )

    ax.set_xlim(-0.6, 11.2)
    ax.set_ylim(0.0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "HMM as a Graphical Model: Markov Chain of Hidden States Emits Observations",
        fontsize=12.5,
        pad=10,
    )

    _save(fig, "fig1_graphical_model")


# ---------------------------------------------------------------------------
# Figure 2: 3-state weather transition diagram
# ---------------------------------------------------------------------------
def fig2_transition_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    # Place three states on an equilateral triangle
    centers = {
        "Sunny": np.array([0.0, 1.6]),
        "Rainy": np.array([-1.6, -1.0]),
        "Cloudy": np.array([1.6, -1.0]),
    }
    colors = {"Sunny": C_AMBER, "Rainy": C_BLUE, "Cloudy": C_PURPLE}
    R = 0.55

    # Transition matrix used for arrow labels
    A = {
        ("Sunny", "Sunny"): 0.70,
        ("Sunny", "Rainy"): 0.10,
        ("Sunny", "Cloudy"): 0.20,
        ("Rainy", "Sunny"): 0.30,
        ("Rainy", "Rainy"): 0.50,
        ("Rainy", "Cloudy"): 0.20,
        ("Cloudy", "Sunny"): 0.30,
        ("Cloudy", "Rainy"): 0.30,
        ("Cloudy", "Cloudy"): 0.40,
    }

    # Draw nodes
    for name, c in centers.items():
        circ = Circle(c, R, facecolor=colors[name], edgecolor=C_DARK,
                      lw=1.6, zorder=3)
        ax.add_patch(circ)
        ax.text(
            c[0],
            c[1],
            name,
            ha="center",
            va="center",
            color="white",
            fontsize=11.5,
            fontweight="bold",
            zorder=4,
        )

    def edge_pts(a, b, off=0.0):
        ca, cb = centers[a], centers[b]
        v = cb - ca
        n = np.array([-v[1], v[0]])
        n = n / np.linalg.norm(n)
        ca2 = ca + (cb - ca) / np.linalg.norm(cb - ca) * R + n * off
        cb2 = cb - (cb - ca) / np.linalg.norm(cb - ca) * R + n * off
        return ca2, cb2

    # Directed edges between distinct states (curved both ways)
    pairs = [
        ("Sunny", "Rainy"),
        ("Rainy", "Sunny"),
        ("Sunny", "Cloudy"),
        ("Cloudy", "Sunny"),
        ("Rainy", "Cloudy"),
        ("Cloudy", "Rainy"),
    ]
    for a, b in pairs:
        rad = 0.22
        p0, p1 = edge_pts(a, b)
        _arrow(ax, p0, p1, color=C_DARK, lw=1.3, rad=rad, mut=14)
        # Label near the midpoint, offset along the curve normal
        mid = (np.array(p0) + np.array(p1)) / 2
        v = np.array(p1) - np.array(p0)
        n = np.array([-v[1], v[0]])
        n = n / np.linalg.norm(n)
        lp = mid + n * 0.28
        ax.text(
            lp[0],
            lp[1],
            f"{A[(a, b)]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=C_DARK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85),
        )

    # Self-loops
    loop_offsets = {
        "Sunny": np.array([0.0, 0.95]),
        "Rainy": np.array([-0.85, -0.5]),
        "Cloudy": np.array([0.85, -0.5]),
    }
    for name, off in loop_offsets.items():
        c = centers[name]
        loop_c = c + off
        loop = Circle(loop_c, 0.28, fill=False, edgecolor=C_DARK, lw=1.3,
                      zorder=2)
        ax.add_patch(loop)
        # Arrowhead on the loop
        ang = np.deg2rad(20)
        head_pt = loop_c + 0.28 * np.array([np.cos(ang), np.sin(ang)])
        tail_pt = loop_c + 0.28 * np.array([np.cos(ang + 0.3),
                                            np.sin(ang + 0.3)])
        _arrow(ax, tail_pt, head_pt, color=C_DARK, lw=1.3, mut=10)
        ax.text(
            loop_c[0],
            loop_c[1] + 0.45,
            f"{A[(name, name)]:.2f}",
            ha="center",
            fontsize=10,
            color=C_DARK,
            fontweight="bold",
        )

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.4, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Three-State Weather Markov Chain (rows of $A$ sum to 1)",
        fontsize=12.5,
        pad=10,
    )

    _save(fig, "fig2_transition_diagram")


# ---------------------------------------------------------------------------
# Helper: tiny HMM used for trellis figures
# ---------------------------------------------------------------------------
def _toy_hmm():
    """Returns (pi, A, B, obs, state_names, obs_names) for a 3-state, T=5 HMM."""
    rng = np.random.default_rng(7)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.65, 0.20],
            [0.20, 0.25, 0.55],
        ]
    )
    B = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ]
    )
    obs = np.array([0, 1, 2, 1, 0])
    return pi, A, B, obs, ["S1", "S2", "S3"], ["v1", "v2", "v3"]


def _forward(pi, A, B, obs):
    T = len(obs)
    N = len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = (alpha[t - 1] @ A[:, j]) * B[j, obs[t]]
    return alpha


def _backward(pi, A, B, obs):
    T = len(obs)
    N = len(pi)
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs[t + 1]] * beta[t + 1])
    return beta


def _viterbi(pi, A, B, obs):
    T = len(obs)
    N = len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            tmp = delta[t - 1] * A[:, j]
            psi[t, j] = int(np.argmax(tmp))
            delta[t, j] = tmp[psi[t, j]] * B[j, obs[t]]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, delta, psi


def _trellis(ax, values, obs_names, state_names, *, color, title,
             highlight_path=None, value_fmt="{:.3f}", direction="forward",
             show_letter=r"\alpha"):
    T, N = values.shape
    xs = np.arange(T)
    ys = np.arange(N)[::-1]  # top-down

    # Background grid
    for x in xs:
        ax.axvline(x, color=C_LIGHT, lw=0.7, zorder=0)
    for y in ys:
        ax.axhline(y, color=C_LIGHT, lw=0.7, zorder=0)

    # Edges (faded)
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                if direction == "forward":
                    p0, p1 = (xs[t], ys[i]), (xs[t + 1], ys[j])
                else:
                    p0, p1 = (xs[t + 1], ys[j]), (xs[t], ys[i])
                _arrow(ax, p0, p1, color=C_GRAY, lw=0.6, mut=8, alpha=0.35)

    # Highlight path
    if highlight_path is not None:
        for t in range(T - 1):
            i, j = highlight_path[t], highlight_path[t + 1]
            _arrow(
                ax,
                (xs[t], ys[i]),
                (xs[t + 1], ys[j]),
                color=C_GREEN,
                lw=2.6,
                mut=16,
            )

    # Nodes
    vmax = values.max() + 1e-12
    for t in range(T):
        for s in range(N):
            v = values[t, s]
            alpha = 0.25 + 0.75 * (v / vmax)
            face = color
            if highlight_path is not None and highlight_path[t] == s:
                face = C_GREEN
                alpha = 1.0
            circ = Circle(
                (xs[t], ys[s]),
                0.27,
                facecolor=face,
                edgecolor=C_DARK,
                lw=1.2,
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(circ)
            ax.text(
                xs[t],
                ys[s],
                value_fmt.format(v),
                ha="center",
                va="center",
                fontsize=8.6,
                color="white" if alpha > 0.6 else C_DARK,
                fontweight="bold",
                zorder=4,
            )

    # Axis labels
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"$t={t+1}$\n$o_t={obs_names[t]}$" for t in range(T)],
        fontsize=10,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(state_names, fontsize=10)
    ax.set_xlim(-0.6, T - 0.4)
    ax.set_ylim(-0.7, N - 0.3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12.5, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Figure 3: Forward trellis
# ---------------------------------------------------------------------------
def fig3_forward_trellis() -> None:
    pi, A, B, obs, state_names, obs_names = _toy_hmm()
    obs_chars = [obs_names[k] for k in obs]
    alpha = _forward(pi, A, B, obs)

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    _trellis(
        ax,
        alpha,
        obs_chars,
        state_names,
        color=C_BLUE,
        title=r"Forward Algorithm Trellis:  $\alpha_t(j)=\left[\sum_i \alpha_{t-1}(i)\,a_{ij}\right]\,b_j(o_t)$",
        direction="forward",
    )
    # Annotate sum operation on a sample node
    ax.annotate(
        "Sum over\nincoming paths",
        xy=(2, 1),
        xytext=(2.9, 2.6),
        fontsize=9.5,
        color=C_BLUE,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.0),
    )
    p_obs = alpha[-1].sum()
    ax.text(
        4.45,
        -0.55,
        rf"$P(\mathbf{{O}}|\lambda)=\sum_i \alpha_T(i) = {p_obs:.4f}$",
        ha="right",
        fontsize=10.5,
        color=C_DARK,
        bbox=dict(facecolor="white", edgecolor=C_BLUE, pad=4),
    )
    fig.tight_layout()
    _save(fig, "fig3_forward_trellis")


# ---------------------------------------------------------------------------
# Figure 4: Backward trellis
# ---------------------------------------------------------------------------
def fig4_backward_trellis() -> None:
    pi, A, B, obs, state_names, obs_names = _toy_hmm()
    obs_chars = [obs_names[k] for k in obs]
    beta = _backward(pi, A, B, obs)

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    _trellis(
        ax,
        beta,
        obs_chars,
        state_names,
        color=C_PURPLE,
        title=r"Backward Algorithm Trellis: $\beta_t(i)=\sum_j a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)$",
        direction="backward",
    )
    ax.annotate(
        r"Boundary: $\beta_T(i)=1$",
        xy=(4, 1),
        xytext=(3.0, 2.6),
        fontsize=9.5,
        color=C_PURPLE,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=1.0),
    )
    fig.tight_layout()
    _save(fig, "fig4_backward_trellis")


# ---------------------------------------------------------------------------
# Figure 5: Viterbi most-likely path
# ---------------------------------------------------------------------------
def fig5_viterbi_path() -> None:
    pi, A, B, obs, state_names, obs_names = _toy_hmm()
    obs_chars = [obs_names[k] for k in obs]
    path, delta, _ = _viterbi(pi, A, B, obs)

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    _trellis(
        ax,
        delta,
        obs_chars,
        state_names,
        color=C_AMBER,
        title=r"Viterbi Trellis: $\delta_t(j)=\max_i\,\delta_{t-1}(i)\,a_{ij}\,b_j(o_t)$  (max replaces sum)",
        highlight_path=path,
        direction="forward",
    )
    best = " -> ".join(state_names[s] for s in path)
    ax.text(
        4.45,
        -0.55,
        f"Most-likely state sequence:  {best}    "
        rf"$\log P^* = {np.log(delta[-1, path[-1]]):.3f}$",
        ha="right",
        fontsize=10.5,
        color=C_DARK,
        bbox=dict(facecolor="white", edgecolor=C_GREEN, pad=4),
    )
    fig.tight_layout()
    _save(fig, "fig5_viterbi_path")


# ---------------------------------------------------------------------------
# Figure 6: Baum-Welch convergence
# ---------------------------------------------------------------------------
def fig6_baum_welch() -> None:
    rng = np.random.default_rng(2026)
    N, M, T = 3, 4, 400

    # True parameters
    pi_t = np.array([0.5, 0.3, 0.2])
    A_t = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
            [0.20, 0.20, 0.60],
        ]
    )
    B_t = np.array(
        [
            [0.50, 0.20, 0.20, 0.10],
            [0.10, 0.50, 0.30, 0.10],
            [0.10, 0.20, 0.30, 0.40],
        ]
    )

    # Generate sequence
    states = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)
    states[0] = rng.choice(N, p=pi_t)
    obs[0] = rng.choice(M, p=B_t[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(N, p=A_t[states[t - 1]])
        obs[t] = rng.choice(M, p=B_t[states[t]])

    # Initialise Baum-Welch from random parameters
    def rand_row(n):
        x = rng.random(n) + 0.1
        return x / x.sum()

    pi = rand_row(N)
    A = np.array([rand_row(N) for _ in range(N)])
    B = np.array([rand_row(M) for _ in range(N)])

    log_liks = []
    n_iter = 40
    for _ in range(n_iter):
        # Forward (with scaling)
        alpha = np.zeros((T, N))
        c = np.zeros(T)
        alpha[0] = pi * B[:, obs[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = (alpha[t - 1] @ A[:, j]) * B[j, obs[t]]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        ll = np.sum(np.log(c))
        log_liks.append(ll)

        # Backward (with same scaling)
        beta = np.zeros((T, N))
        beta[-1] = 1.0 / c[-1]
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i] * B[:, obs[t + 1]] * beta[t + 1])
            beta[t] /= c[t]

        # Posteriors
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = 0.0
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (
                        alpha[t, i] * A[i, j] * B[j, obs[t + 1]] * beta[t + 1, j]
                    )
                    denom += xi[t, i, j]
            xi[t] /= denom

        # M-step
        pi = gamma[0]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        B_new = np.zeros_like(B)
        for k in range(M):
            mask = obs == k
            B_new[:, k] = gamma[mask].sum(axis=0)
        B = B_new / gamma.sum(axis=0)[:, None]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    iters = np.arange(1, n_iter + 1)
    ax.plot(iters, log_liks, "-o", color=C_BLUE, lw=2.0, ms=4.5,
            label="Baum-Welch")
    ax.fill_between(iters, log_liks, log_liks[0], color=C_BLUE, alpha=0.08)
    ax.axhline(log_liks[-1], color=C_GREEN, ls="--", lw=1.2,
               label=f"Final log-lik = {log_liks[-1]:.1f}")
    ax.set_xlabel("EM iteration", fontsize=11)
    ax.set_ylabel(r"Log-likelihood  $\log P(\mathbf{O}\,|\,\lambda^{(k)})$",
                  fontsize=11)
    ax.set_title("Baum-Welch Monotonically Improves the Likelihood",
                 fontsize=12.5)
    ax.legend(loc="lower right", frameon=True)

    # Right: heatmap of recovered transition matrix vs. true
    ax2 = axes[1]
    # Permute the recovered states to best match true states by row-similarity
    from itertools import permutations

    best_perm = None
    best_score = -np.inf
    for p in permutations(range(N)):
        score = -np.linalg.norm(A_t - A[list(p)][:, list(p)])
        if score > best_score:
            best_score = score
            best_perm = list(p)
    A_aligned = A[best_perm][:, best_perm]

    diff = np.abs(A_t - A_aligned)
    im = ax2.imshow(A_aligned, cmap="Purples", vmin=0, vmax=1)
    for i in range(N):
        for j in range(N):
            ax2.text(
                j,
                i,
                f"{A_aligned[i, j]:.2f}\n({A_t[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if A_aligned[i, j] > 0.45 else C_DARK,
            )
    ax2.set_title(f"Recovered $\\hat A$  (true in parens)\nMax|err|={diff.max():.2f}",
                  fontsize=11.5)
    ax2.set_xticks(range(N))
    ax2.set_yticks(range(N))
    ax2.set_xticklabels([f"to {i+1}" for i in range(N)], fontsize=9)
    ax2.set_yticklabels([f"from {i+1}" for i in range(N)], fontsize=9)
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, "fig6_baum_welch")


# ---------------------------------------------------------------------------
# Figure 7: POS tagging application
# ---------------------------------------------------------------------------
def fig7_pos_tagging() -> None:
    states = ["PRON", "VERB", "ADJ", "NOUN"]
    words = ["I", "love", "natural", "language", "processing"]

    pi = np.array([0.55, 0.05, 0.10, 0.30])
    A = np.array(
        [
            [0.05, 0.70, 0.05, 0.20],   # PRON ->
            [0.10, 0.05, 0.30, 0.55],   # VERB ->
            [0.02, 0.05, 0.13, 0.80],   # ADJ  ->
            [0.05, 0.40, 0.05, 0.50],   # NOUN ->
        ]
    )
    # Hand-crafted emission rows over the 5 vocab tokens
    B = np.array(
        [
            [0.95, 0.01, 0.01, 0.02, 0.01],  # PRON
            [0.01, 0.85, 0.02, 0.05, 0.07],  # VERB
            [0.01, 0.02, 0.85, 0.07, 0.05],  # ADJ
            [0.01, 0.05, 0.10, 0.42, 0.42],  # NOUN
        ]
    )
    obs = np.arange(len(words))
    path, _, _ = _viterbi(pi, A, B, obs)
    tags = [states[i] for i in path]

    fig = plt.figure(figsize=(11.2, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.5], hspace=0.45)

    # --- Top: tagged sentence as a row of boxes ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, len(words))
    ax1.set_ylim(0, 2.4)
    ax1.axis("off")
    ax1.set_title("POS Tagging via Viterbi: Hidden Tag Sequence Maximises $P(\\mathbf{T}\\,|\\,\\mathbf{W})$",
                  fontsize=12.5)

    palette = {"PRON": C_BLUE, "VERB": C_PURPLE, "ADJ": C_AMBER, "NOUN": C_GREEN}
    for t, (w, tag) in enumerate(zip(words, tags)):
        # Word box
        box_w = FancyBboxPatch(
            (t + 0.08, 1.25),
            0.84,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="white",
            edgecolor=C_DARK,
            lw=1.2,
        )
        ax1.add_patch(box_w)
        ax1.text(t + 0.5, 1.52, w, ha="center", va="center",
                 fontsize=12, fontweight="bold", color=C_DARK)

        # Tag box
        box_t = FancyBboxPatch(
            (t + 0.08, 0.35),
            0.84,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=palette[tag],
            edgecolor=C_DARK,
            lw=1.2,
        )
        ax1.add_patch(box_t)
        ax1.text(t + 0.5, 0.62, tag, ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")

        # Connector
        _arrow(ax1, (t + 0.5, 1.22), (t + 0.5, 0.92), color=C_GRAY,
               lw=1.0, mut=10)

        if t < len(words) - 1:
            _arrow(ax1, (t + 0.92, 0.62), (t + 1.08, 0.62),
                   color=palette[tag], lw=1.4, mut=12)

    # --- Bottom: emission heatmap ---
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(B, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(words)))
    ax2.set_xticklabels(words, fontsize=10.5)
    ax2.set_yticks(range(len(states)))
    ax2.set_yticklabels(states, fontsize=10.5)
    ax2.set_title("Emission probabilities $b_j(w)$  (rows are tags, columns are words)",
                  fontsize=11.5)
    for i in range(len(states)):
        for j in range(len(words)):
            ax2.text(
                j,
                i,
                f"{B[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if B[i, j] > 0.45 else C_DARK,
            )

    # Highlight the chosen (tag, word) cells
    for t, j in enumerate(path):
        ax2.add_patch(
            Rectangle(
                (t - 0.5, j - 0.5),
                1,
                1,
                fill=False,
                edgecolor=C_AMBER,
                lw=2.2,
            )
        )
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)

    _save(fig, "fig7_pos_tagging")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 15 (HMM) figures ...")
    fig1_graphical_model()
    fig2_transition_diagram()
    fig3_forward_trellis()
    fig4_backward_trellis()
    fig5_viterbi_path()
    fig6_baum_welch()
    fig7_pos_tagging()
    print("Done.")


if __name__ == "__main__":
    main()
