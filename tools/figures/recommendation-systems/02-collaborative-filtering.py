"""
Figure generation for Recommendation Systems (02): Collaborative Filtering
and Matrix Factorization.

Outputs six figures into BOTH the EN and ZH asset folders:
  fig1_user_user_vs_item_item.png
  fig2_user_similarity_heatmap.png
  fig3_cosine_vs_pearson.png
  fig4_matrix_factorization.png
  fig5_sgd_vs_als_convergence.png
  fig6_latent_space.png

Run:
    python 02-collaborative-filtering.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.6,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GREY = "#64748b"

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source/_posts/en/recommendation-systems/02-collaborative-filtering"
ZH_DIR = REPO_ROOT / "source/_posts/zh/recommendation-systems/02-协同过滤与矩阵分解"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        fig.savefig(folder / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — User-User vs Item-Item CF schematic
# ---------------------------------------------------------------------------

def fig1_user_user_vs_item_item() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    def draw_panel(ax, title, mode):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, pad=12)

        users = ["U1", "U2", "U3", "Target"]
        items = ["A", "B", "C", "D"]
        u_y = [5.5, 4.3, 3.1, 1.5]
        i_y = [5.5, 4.3, 3.1, 1.9]

        # User column
        for name, y in zip(users, u_y):
            color = ORANGE if name == "Target" else BLUE
            box = FancyBboxPatch(
                (0.6, y - 0.35), 1.7, 0.7,
                boxstyle="round,pad=0.05",
                linewidth=1.2, edgecolor=color, facecolor="white",
            )
            ax.add_patch(box)
            ax.text(1.45, y, name, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

        # Item column
        for name, y in zip(items, i_y):
            box = FancyBboxPatch(
                (7.7, y - 0.35), 1.7, 0.7,
                boxstyle="round,pad=0.05",
                linewidth=1.2, edgecolor=PURPLE, facecolor="white",
            )
            ax.add_patch(box)
            ax.text(8.55, y, name, ha="center", va="center",
                    color=PURPLE, fontsize=11, fontweight="bold")

        # Known interactions (light grey)
        known = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0)]
        for ui, ii in known:
            ax.annotate(
                "", xy=(7.7, i_y[ii]), xytext=(2.3, u_y[ui]),
                arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.9),
            )

        if mode == "user":
            # Find similar users to Target via shared items, then recommend.
            ax.annotate(
                "", xy=(2.3, u_y[0]) if False else (1.45, u_y[0] - 0.35),
                xytext=(1.45, u_y[3] + 0.35),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=2.0,
                                connectionstyle="arc3,rad=-0.4"),
            )
            ax.text(0.05, 3.5, "similar\nusers", color=GREEN,
                    fontsize=10, fontweight="bold", ha="left")
            ax.annotate(
                "", xy=(7.7, i_y[2]), xytext=(2.3, u_y[3]),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.2),
            )
            ax.text(5.0, 1.0, "recommend C\n(neighbours liked it)",
                    color=ORANGE, fontsize=10, fontweight="bold", ha="center")
        else:
            # Item-item: Target liked A, find items similar to A, recommend.
            ax.annotate(
                "", xy=(8.55, i_y[2] - 0.35), xytext=(8.55, i_y[0] + 0.35),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=2.0,
                                connectionstyle="arc3,rad=0.5"),
            )
            ax.text(9.95, 4.3, "similar\nitems", color=GREEN,
                    fontsize=10, fontweight="bold", ha="left")
            ax.annotate(
                "", xy=(7.7, i_y[2]), xytext=(2.3, u_y[3]),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.2),
            )
            ax.text(5.0, 1.0, "recommend C\n(similar to A you liked)",
                    color=ORANGE, fontsize=10, fontweight="bold", ha="center")

        ax.text(1.45, 6.5, "Users", ha="center", color="#1f2937",
                fontsize=12, fontweight="bold")
        ax.text(8.55, 6.5, "Items", ha="center", color="#1f2937",
                fontsize=12, fontweight="bold")

    draw_panel(axes[0], "User-Based CF: find similar users", "user")
    draw_panel(axes[1], "Item-Based CF: find similar items", "item")

    fig.suptitle("Two views of collaborative filtering",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig1_user_user_vs_item_item.png")


# ---------------------------------------------------------------------------
# Figure 2 — User similarity heatmap (Pearson) on the toy rating matrix
# ---------------------------------------------------------------------------

def fig2_user_similarity_heatmap() -> None:
    users = ["Alice", "Bob", "Carol", "David", "Eve"]
    # Rows: users, columns: items (Shawshank, Forrest, Pursuit, Titanic, Godfather)
    R = np.array(
        [
            [5, 5, np.nan, 3, 4],
            [5, 5, 4, 2, np.nan],
            [4, 4, 3, 5, 5],
            [2, 1, np.nan, 4, 3],
            [np.nan, np.nan, 2, 3, 4],
        ],
        dtype=float,
    )

    n = len(users)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mask = ~np.isnan(R[i]) & ~np.isnan(R[j])
            if mask.sum() < 2:
                sim[i, j] = np.nan
                continue
            ri = R[i][mask] - np.mean(R[i][mask])
            rj = R[j][mask] - np.mean(R[j][mask])
            denom = np.sqrt((ri ** 2).sum() * (rj ** 2).sum())
            sim[i, j] = (ri * rj).sum() / denom if denom else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(users)
    ax.set_yticklabels(users)
    ax.set_title("Pearson similarity between users", pad=14)

    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="#475569")
            else:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "#1f2937",
                        fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("similarity", rotation=270, labelpad=14)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    save(fig, "fig2_user_similarity_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Cosine vs Pearson similarity comparison
# ---------------------------------------------------------------------------

def fig3_cosine_vs_pearson() -> None:
    # Two users with identical *shape* but different *level*.
    items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
    u1 = np.array([5, 4, 5, 3, 4])
    u2 = np.array([3, 2, 3, 1, 2])  # u1 - 2

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_v = cosine(u1, u2)
    pear_v = pearson(u1, u2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left: raw ratings — cosine sees them as "high" vs "low"
    ax = axes[0]
    x = np.arange(len(items))
    ax.plot(x, u1, marker="o", color=BLUE, lw=2.2, label="User A (generous rater)")
    ax.plot(x, u2, marker="s", color=ORANGE, lw=2.2, label="User B (strict rater)")
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(0, 6)
    ax.set_ylabel("rating")
    ax.set_title(f"Raw ratings — cosine = {cos_v:.2f}", pad=10)
    ax.legend(loc="upper right", framealpha=0.95)

    # Right: mean-centred ratings — Pearson sees them as identical taste
    ax = axes[1]
    u1c = u1 - u1.mean()
    u2c = u2 - u2.mean()
    ax.plot(x, u1c, marker="o", color=BLUE, lw=2.2, label="User A (centred)")
    ax.plot(x, u2c, marker="s", color=ORANGE, lw=2.2,
            label="User B (centred)", linestyle="--")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(-2.2, 2.2)
    ax.set_ylabel("rating − user mean")
    ax.set_title(f"Mean-centred — Pearson = {pear_v:.2f}", pad=10)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Cosine sees scale, Pearson sees taste",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig3_cosine_vs_pearson.png")


# ---------------------------------------------------------------------------
# Figure 4 — Matrix factorization decomposition R ≈ P · Qᵀ
# ---------------------------------------------------------------------------

def fig4_matrix_factorization() -> None:
    rng = np.random.default_rng(7)
    m, n, k = 6, 8, 3
    P_true = rng.normal(0, 1, (m, k))
    Q_true = rng.normal(0, 1, (n, k))
    R = P_true @ Q_true.T
    # mask out ~40% to show sparsity
    mask = rng.random((m, n)) < 0.4
    R_obs = R.copy()
    R_obs[mask] = np.nan

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[3.2, 0.4, 1.2, 0.4, 0.4, 0.4, 2.4],
                          wspace=0.15)

    ax_r = fig.add_subplot(gs[0, 0])
    ax_eq = fig.add_subplot(gs[0, 1]); ax_eq.axis("off")
    ax_p = fig.add_subplot(gs[0, 2])
    ax_dot = fig.add_subplot(gs[0, 3]); ax_dot.axis("off")
    ax_q = fig.add_subplot(gs[0, 6])

    vmax = max(np.nanmax(np.abs(R_obs)), 1)
    im = ax_r.imshow(R_obs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     aspect="auto")
    ax_r.set_title(f"R  ({m} users × {n} items)", pad=8)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    # mark missing entries
    for i in range(m):
        for j in range(n):
            if np.isnan(R_obs[i, j]):
                ax_r.text(j, i, "?", ha="center", va="center",
                          color="#1f2937", fontsize=10, fontweight="bold")

    ax_eq.text(0.5, 0.5, "≈", ha="center", va="center",
               fontsize=28, color="#1f2937")

    ax_p.imshow(P_true, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax_p.set_title(f"P  ({m} × k)", pad=8, color=BLUE)
    ax_p.set_xticks([]); ax_p.set_yticks([])

    ax_dot.text(0.5, 0.5, "·", ha="center", va="center",
                fontsize=36, color="#1f2937")

    ax_q.imshow(Q_true.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax_q.set_title(f"Qᵀ  (k × {n})", pad=8, color=PURPLE)
    ax_q.set_xticks([]); ax_q.set_yticks([])

    fig.suptitle("Matrix factorization: a sparse rating matrix as the "
                 "product of two thin matrices",
                 fontsize=13, fontweight="bold", y=1.04)

    cbar = fig.colorbar(im, ax=[ax_r, ax_p, ax_q], shrink=0.85, pad=0.02)
    cbar.set_label("value")

    save(fig, "fig4_matrix_factorization.png")


# ---------------------------------------------------------------------------
# Figure 5 — SGD vs ALS convergence
# ---------------------------------------------------------------------------

def fig5_sgd_vs_als_convergence() -> None:
    rng = np.random.default_rng(0)
    m, n, k_true, k = 80, 60, 4, 8
    P_t = rng.normal(0, 1, (m, k_true))
    Q_t = rng.normal(0, 1, (n, k_true))
    R = P_t @ Q_t.T + rng.normal(0, 0.2, (m, n))
    mask = rng.random((m, n)) < 0.5  # observed entries

    obs = np.argwhere(mask)
    y = R[mask]

    # SGD
    rng2 = np.random.default_rng(1)
    P = rng2.normal(0, 0.1, (m, k))
    Q = rng2.normal(0, 0.1, (n, k))
    lr, reg = 0.02, 0.05
    sgd_rmse = []
    for epoch in range(40):
        idx = rng2.permutation(len(obs))
        for t in idx:
            i, j = obs[t]
            err = R[i, j] - P[i] @ Q[j]
            pi = P[i].copy()
            P[i] += lr * (err * Q[j] - reg * P[i])
            Q[j] += lr * (err * pi - reg * Q[j])
        pred = (P @ Q.T)[mask]
        sgd_rmse.append(float(np.sqrt(((pred - y) ** 2).mean())))

    # ALS
    rng3 = np.random.default_rng(2)
    P = rng3.normal(0, 0.1, (m, k))
    Q = rng3.normal(0, 0.1, (n, k))
    reg_als = 0.5
    eye = reg_als * np.eye(k)
    als_rmse = []
    for epoch in range(40):
        for i in range(m):
            js = np.where(mask[i])[0]
            if len(js) == 0:
                continue
            Qi = Q[js]
            P[i] = np.linalg.solve(Qi.T @ Qi + eye, Qi.T @ R[i, js])
        for j in range(n):
            is_ = np.where(mask[:, j])[0]
            if len(is_) == 0:
                continue
            Pj = P[is_]
            Q[j] = np.linalg.solve(Pj.T @ Pj + eye, Pj.T @ R[is_, j])
        pred = (P @ Q.T)[mask]
        als_rmse.append(float(np.sqrt(((pred - y) ** 2).mean())))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    epochs = np.arange(1, 41)
    ax.plot(epochs, sgd_rmse, color=BLUE, lw=2.3, marker="o", ms=4,
            label="SGD (per-sample updates)")
    ax.plot(epochs, als_rmse, color=PURPLE, lw=2.3, marker="s", ms=4,
            label="ALS (closed-form per epoch)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training RMSE")
    ax.set_title("Convergence on a synthetic 80×60 rating matrix", pad=10)
    ax.legend(framealpha=0.95)
    ax.set_ylim(bottom=0)

    save(fig, "fig5_sgd_vs_als_convergence.png")


# ---------------------------------------------------------------------------
# Figure 6 — Latent factor space (users + items in 2D)
# ---------------------------------------------------------------------------

def fig6_latent_space() -> None:
    # Hand-picked 2D embeddings for a tiny movie scenario.
    # Axis 1: serious / dramatic (right) vs light / fun (left)
    # Axis 2: classic (up) vs modern blockbuster (down)
    users = {
        "Alice":   ( 0.85,  0.55),
        "Bob":     ( 0.55, -0.20),
        "Carol":   ( 0.10,  0.80),
        "David":   (-0.70, -0.10),
        "Eve":     (-0.30,  0.45),
    }
    items = {
        "Shawshank":   ( 0.90,  0.70),
        "Forrest":     ( 0.65,  0.40),
        "Godfather":   ( 0.50,  0.85),
        "Pursuit":     ( 0.40, -0.10),
        "Titanic":     (-0.10,  0.20),
        "Toy Story":   (-0.65,  0.55),
        "Avengers":    (-0.55, -0.50),
        "Fast&Furious":(-0.85, -0.60),
    }

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    ax.axhline(0, color="#cbd5e1", lw=0.8)
    ax.axvline(0, color="#cbd5e1", lw=0.8)

    for name, (x, y) in users.items():
        ax.scatter(x, y, s=180, color=BLUE, edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=BLUE)

    for name, (x, y) in items.items():
        ax.scatter(x, y, s=180, marker="^", color=PURPLE,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(8, -12), textcoords="offset points",
                    fontsize=9.5, color=PURPLE)

    # Highlight Alice ↔ Shawshank match (large dot product)
    ax_x, ax_y = users["Alice"]
    sx, sy = items["Shawshank"]
    ax.annotate(
        "", xy=(sx, sy), xytext=(ax_x, ax_y),
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0,
                        connectionstyle="arc3,rad=0.15"),
    )
    ax.text(0.92, 0.62, "high p·q\n→ recommend",
            color=GREEN, fontsize=10, fontweight="bold")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.95, 1.05)
    ax.set_xlabel("latent factor 1  (light  ←   →  serious)")
    ax.set_ylabel("latent factor 2  (modern  ↓   ↑  classic)")
    ax.set_title("Users (●) and items (▲) in a learned 2D latent space",
                 pad=12)

    save(fig, "fig6_latent_space.png")


def main() -> None:
    fig1_user_user_vs_item_item()
    fig2_user_similarity_heatmap()
    fig3_cosine_vs_pearson()
    fig4_matrix_factorization()
    fig5_sgd_vs_als_convergence()
    fig6_latent_space()
    print(f"Wrote 6 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
