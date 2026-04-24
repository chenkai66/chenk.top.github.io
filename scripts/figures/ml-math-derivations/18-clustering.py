"""
Figure generation script for ML Math Derivations Part 18:
Clustering Algorithms.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a specific intuition behind one clustering family so that
the math can be tied back to a picture.

Figures:
    fig1_kmeans_steps           Four-step Lloyd iteration on a 3-blob set:
                                random init -> assign -> update -> converged.
                                Voronoi cells, centroid trajectories, and
                                WCSS values are annotated per panel.
    fig2_dbscan_density         DBSCAN on two interleaved moons. Core,
                                border, and noise points are colored
                                differently; one core point's epsilon ball
                                is highlighted to show the density rule.
    fig3_dendrogram             Agglomerative (Ward) dendrogram on a small
                                2D dataset, with a horizontal cut line that
                                produces three clusters. Cluster colors are
                                propagated to the scatter inset.
    fig4_silhouette_curve       Silhouette score vs K on a 4-cluster blob
                                problem: the optimal K is annotated and the
                                "wrong K" regions are shaded.
    fig5_elbow                  Within-cluster sum of squares vs K (the
                                classic elbow plot) on the same data, with
                                the elbow point highlighted.
    fig6_gmm_vs_kmeans          GMM vs K-means on an elliptical/anisotropic
                                blob. K-means draws straight (Voronoi)
                                boundaries and mis-assigns; GMM recovers the
                                tilted ellipses (covariance contours shown).
    fig7_algo_comparison        3 algorithms x 3 dataset shapes grid:
                                K-means, DBSCAN, Spectral on Blobs, Moons,
                                Circles. Shows where each algorithm shines
                                or fails.

Usage:
    python3 scripts/figures/ml-math-derivations/18-clustering.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

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
C_RED = "#dc2626"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "ml-math-derivations"
    / "18-Clustering-Algorithms"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "ml-math-derivations"
    / "18-聚类算法"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lloyd helpers (so we can grab intermediate states)
# ---------------------------------------------------------------------------
def _lloyd_step(X, centroids):
    """Single Lloyd iteration: assignment + centroid update. Returns
    (labels, new_centroids, wcss)."""
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    new_centroids = np.array([
        X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
        for k in range(centroids.shape[0])
    ])
    wcss = float(np.sum(
        [np.sum((X[labels == k] - centroids[k]) ** 2)
         for k in range(centroids.shape[0])]
    ))
    return labels, new_centroids, wcss


# ---------------------------------------------------------------------------
# Figure 1: Lloyd iteration -- four panels
# ---------------------------------------------------------------------------
def fig1_kmeans_steps():
    rng = np.random.default_rng(7)
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.9,
                      center_box=(-5, 5), random_state=11)

    # Deliberately bad initialization to show movement.
    init = np.array([[-3.5, -3.0], [-3.0, -2.0], [3.0, -3.5]])

    states = [(None, init.copy(), None)]
    centroids = init.copy()
    for _ in range(8):
        labels, new_c, wcss = _lloyd_step(X, centroids)
        states.append((labels, new_c, wcss))
        if np.allclose(new_c, centroids):
            break
        centroids = new_c

    # Pick four characteristic states.
    panels = [
        ("Step 0: Initialization", None, init, None),
        ("Step 1: First assignment + update", *states[1]),
        ("Step 2: Centroids drift", *states[2]),
        ("Converged: stable assignment", *states[-1]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for ax, (title, labels, cents, wcss) in zip(axes, panels):
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], s=14, c=C_GRAY, alpha=0.7,
                       edgecolor="none")
        else:
            for k in range(3):
                m = labels == k
                ax.scatter(X[m, 0], X[m, 1], s=16, c=PALETTE[k],
                           alpha=0.75, edgecolor="none")

        # Trail of centroid trajectories up to current step.
        idx = panels.index((title, labels, cents, wcss))
        if idx > 0:
            traj = np.array(
                [s[1] for s in states[:idx + 1]]
            )  # shape (T, K, 2)
            for k in range(3):
                ax.plot(traj[:, k, 0], traj[:, k, 1],
                        color=C_DARK, lw=0.9, alpha=0.5, zorder=3)

        ax.scatter(cents[:, 0], cents[:, 1], s=260, marker="*",
                   c=[PALETTE[k] for k in range(3)], edgecolor=C_DARK,
                   linewidth=1.6, zorder=5)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if wcss is not None:
            ax.text(0.02, 0.97, f"J = {wcss:.1f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold", color=C_DARK,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white",
                              edgecolor=C_LIGHT))

    fig.suptitle(
        "Lloyd's Algorithm: each iteration strictly decreases WCSS J",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig1_kmeans_steps")


# ---------------------------------------------------------------------------
# Figure 2: DBSCAN density -- core / border / noise
# ---------------------------------------------------------------------------
def fig2_dbscan_density():
    rng = np.random.default_rng(0)
    X, _ = make_moons(n_samples=260, noise=0.08, random_state=3)
    # Sprinkle noise points.
    noise = rng.uniform(-1.5, 2.5, size=(25, 2))
    noise[:, 1] = rng.uniform(-1.0, 1.5, size=25)
    X = np.vstack([X, noise])

    eps = 0.20
    min_samples = 5
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Identify core / border / noise.
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[db.core_sample_indices_] = True
    noise_mask = labels == -1
    border_mask = ~core_mask & ~noise_mask

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # Color clusters; noise = gray.
    cluster_ids = sorted(set(labels) - {-1})
    cluster_colors = {cid: PALETTE[i % len(PALETTE)]
                      for i, cid in enumerate(cluster_ids)}

    # Plot border first (smaller, ringed), then core (larger).
    for cid in cluster_ids:
        m_border = border_mask & (labels == cid)
        ax.scatter(X[m_border, 0], X[m_border, 1], s=42,
                   facecolor="white", edgecolor=cluster_colors[cid],
                   linewidth=1.6, label=None, zorder=3)
        m_core = core_mask & (labels == cid)
        ax.scatter(X[m_core, 0], X[m_core, 1], s=46,
                   c=cluster_colors[cid], edgecolor=C_DARK, linewidth=0.6,
                   zorder=4)

    ax.scatter(X[noise_mask, 0], X[noise_mask, 1], s=46, marker="x",
               c=C_GRAY, linewidths=1.5, label="Noise", zorder=2)

    # Highlight one core point's epsilon-neighborhood.
    pick_idx = db.core_sample_indices_[
        np.argmax(X[db.core_sample_indices_, 0])
    ]
    cx, cy = X[pick_idx]
    ax.add_patch(Circle((cx, cy), eps, facecolor=C_AMBER, alpha=0.18,
                        edgecolor=C_AMBER, linewidth=1.6, zorder=1))
    ax.scatter([cx], [cy], s=140, marker="*", color=C_AMBER,
               edgecolor=C_DARK, linewidth=1.2, zorder=5)
    ax.annotate(
        f"core point\n|N_eps(x)| >= MinPts={min_samples}",
        xy=(cx, cy), xytext=(cx + 0.45, cy + 0.55),
        fontsize=10, fontweight="bold", color=C_DARK,
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_AMBER),
    )

    # Custom legend.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C_BLUE, markeredgecolor=C_DARK,
               markersize=9, label="Core"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor=C_BLUE,
               markeredgewidth=1.6, markersize=9, label="Border"),
        Line2D([0], [0], marker="x", color=C_GRAY, lw=0,
               markersize=10, markeredgewidth=1.8, label="Noise"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    ax.set_title(
        f"DBSCAN: density-based clusters on noisy moons (eps={eps}, "
        f"MinPts={min_samples})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig2_dbscan_density")


# ---------------------------------------------------------------------------
# Figure 3: Hierarchical dendrogram + scatter inset
# ---------------------------------------------------------------------------
def fig3_dendrogram():
    X, y = make_blobs(n_samples=24, centers=3, cluster_std=0.55,
                      center_box=(-3, 3), random_state=2)
    Z = linkage(X, method="ward")

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.18)
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_pts = fig.add_subplot(gs[0, 1])

    cut = 4.5  # produces 3 clusters
    # Color threshold passes the threshold for cluster coloring.
    dn = dendrogram(
        Z,
        ax=ax_dend,
        color_threshold=cut,
        above_threshold_color=C_GRAY,
        leaf_font_size=8,
    )
    ax_dend.axhline(cut, color=C_RED, lw=1.5, ls="--",
                    label=f"cut at d = {cut}")
    ax_dend.set_title(
        "Agglomerative (Ward) Dendrogram",
        fontsize=12, fontweight="bold",
    )
    ax_dend.set_xlabel("sample index", fontsize=10)
    ax_dend.set_ylabel("merge distance", fontsize=10)
    ax_dend.legend(loc="upper right", framealpha=0.95)

    # Recover the cluster labels via fcluster equivalent.
    from scipy.cluster.hierarchy import fcluster
    labels = fcluster(Z, t=cut, criterion="distance")

    # Map cluster id -> color (keep aligned with what dendrogram drew).
    uniq = sorted(set(labels))
    color_map = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(uniq)}

    for cid in uniq:
        m = labels == cid
        ax_pts.scatter(X[m, 0], X[m, 1], s=80, c=color_map[cid],
                       edgecolor=C_DARK, linewidth=0.8,
                       label=f"cluster {cid}")
    ax_pts.set_title("Resulting 3 clusters",
                     fontsize=12, fontweight="bold")
    ax_pts.set_xticks([])
    ax_pts.set_yticks([])
    ax_pts.set_aspect("equal")
    ax_pts.legend(loc="best", framealpha=0.95, fontsize=9)

    fig.suptitle(
        "Hierarchical clustering: a horizontal cut maps the tree to flat "
        "clusters",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_dendrogram")


# ---------------------------------------------------------------------------
# Figure 4: Silhouette curve
# ---------------------------------------------------------------------------
def _build_eval_data():
    return make_blobs(n_samples=600, centers=4, cluster_std=0.85,
                      center_box=(-6, 6), random_state=42)


def fig4_silhouette_curve():
    X, _ = _build_eval_data()
    Ks = list(range(2, 11))
    scores = []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        scores.append(silhouette_score(X, km.labels_))

    best_k = Ks[int(np.argmax(scores))]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(Ks, scores, "-o", color=C_BLUE, lw=2.0, markersize=8,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.axvline(best_k, color=C_GREEN, lw=1.4, ls="--",
               label=f"optimal K = {best_k}")
    ax.scatter([best_k], [max(scores)], s=180, marker="*",
               color=C_GREEN, edgecolor=C_DARK, linewidth=1.2, zorder=5)

    # Shade "wrong K" regions.
    ax.axvspan(Ks[0] - 0.4, best_k - 0.5, color=C_AMBER, alpha=0.08)
    ax.axvspan(best_k + 0.5, Ks[-1] + 0.4, color=C_PURPLE, alpha=0.08)
    ax.text(2.2, max(scores) * 0.55, "underfit\n(too few)",
            fontsize=9.5, color=C_AMBER, fontweight="bold")
    ax.text(8.6, max(scores) * 0.55, "overfit\n(too many)",
            fontsize=9.5, color=C_PURPLE, fontweight="bold", ha="right")

    ax.set_xlabel("Number of clusters K", fontsize=11)
    ax.set_ylabel("Mean silhouette score", fontsize=11)
    ax.set_title(
        "Silhouette analysis: pick K that maximizes (b - a) / max(a, b)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(Ks)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlim(Ks[0] - 0.4, Ks[-1] + 0.4)
    fig.tight_layout()
    _save(fig, "fig4_silhouette_curve")


# ---------------------------------------------------------------------------
# Figure 5: Elbow plot
# ---------------------------------------------------------------------------
def fig5_elbow():
    X, _ = _build_eval_data()
    Ks = list(range(1, 11))
    inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)

    # Find the elbow via maximum distance to the line connecting endpoints.
    p1 = np.array([Ks[0], inertias[0]])
    p2 = np.array([Ks[-1], inertias[-1]])
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)
    dists = []
    for k, j in zip(Ks, inertias):
        p = np.array([k, j])
        # Perpendicular distance from p to line p1-p2.
        d = np.abs(np.cross(line_vec, p - p1)) / line_norm
        dists.append(d)
    elbow_idx = int(np.argmax(dists))
    elbow_k = Ks[elbow_idx]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(Ks, inertias, "-o", color=C_PURPLE, lw=2.0, markersize=8,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.plot([Ks[0], Ks[-1]], [inertias[0], inertias[-1]],
            ls=":", color=C_GRAY, lw=1.2, label="endpoint reference line")
    ax.scatter([elbow_k], [inertias[elbow_idx]], s=220, marker="*",
               color=C_AMBER, edgecolor=C_DARK, linewidth=1.4, zorder=5,
               label=f"elbow at K = {elbow_k}")
    ax.annotate(
        "rapid drop ends here",
        xy=(elbow_k, inertias[elbow_idx]),
        xytext=(elbow_k + 1.6, inertias[elbow_idx] + 0.18 *
                (inertias[0] - inertias[-1])),
        fontsize=10, color=C_DARK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_AMBER),
    )

    ax.set_xlabel("Number of clusters K", fontsize=11)
    ax.set_ylabel("Within-cluster sum of squares (inertia)", fontsize=11)
    ax.set_title(
        "Elbow method: diminishing returns mark the right K",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(Ks)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig5_elbow")


# ---------------------------------------------------------------------------
# Figure 6: GMM vs K-means on anisotropic blobs
# ---------------------------------------------------------------------------
def _draw_ellipse(ax, mean, cov, color, n_std=2.0, alpha=0.18):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color,
                  linewidth=1.6)
    ax.add_patch(ell)


def fig6_gmm_vs_kmeans():
    rng = np.random.default_rng(1)
    # Anisotropic blobs: stretch + rotate.
    X, y = make_blobs(n_samples=500, centers=3, cluster_std=0.7,
                      center_box=(-4, 4), random_state=4)
    transform = np.array([[0.6, -0.7], [-0.4, 0.85]])
    X = X @ transform

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=0).fit(X)
    km_labels = km.labels_
    gmm_labels = gmm.predict(X)

    # Align color order across the two panels by matching centers.
    def _match(src_centers, tgt_centers):
        order = []
        used = set()
        for sc in src_centers:
            d = np.linalg.norm(tgt_centers - sc, axis=1)
            for i in np.argsort(d):
                if i not in used:
                    order.append(i)
                    used.add(i)
                    break
        return order

    order = _match(km.cluster_centers_, gmm.means_)
    remap = {old: new for new, old in enumerate(order)}
    gmm_labels_aligned = np.array([remap[i] for i in gmm_labels])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # --- K-means panel ---
    ax = axes[0]
    for k in range(3):
        m = km_labels == k
        ax.scatter(X[m, 0], X[m, 1], s=18, c=PALETTE[k], alpha=0.75,
                   edgecolor="none")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               s=240, marker="*", c=[PALETTE[k] for k in range(3)],
               edgecolor=C_DARK, linewidth=1.4, zorder=5)
    # Voronoi (decision) boundary.
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = km.predict(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=np.arange(-0.5, 3.5, 1),
               colors=C_DARK, linewidths=1.0, alpha=0.6)
    ax.set_title("K-means: straight (Voronoi) boundaries",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # --- GMM panel ---
    ax = axes[1]
    for k in range(3):
        m = gmm_labels_aligned == k
        ax.scatter(X[m, 0], X[m, 1], s=18, c=PALETTE[k], alpha=0.75,
                   edgecolor="none")
    for new_idx, old_idx in enumerate(order):
        _draw_ellipse(ax, gmm.means_[old_idx], gmm.covariances_[old_idx],
                      PALETTE[new_idx])
        _draw_ellipse(ax, gmm.means_[old_idx], gmm.covariances_[old_idx],
                      PALETTE[new_idx], n_std=1.0, alpha=0.30)
    ax.scatter(gmm.means_[order, 0], gmm.means_[order, 1],
               s=240, marker="*", c=[PALETTE[k] for k in range(3)],
               edgecolor=C_DARK, linewidth=1.4, zorder=5)
    ax.set_title("GMM: tilted Gaussian ellipses",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.suptitle(
        "Anisotropic blobs: K-means cuts in straight lines, "
        "GMM follows the covariance",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_gmm_vs_kmeans")


# ---------------------------------------------------------------------------
# Figure 7: 3 algorithms x 3 dataset shapes
# ---------------------------------------------------------------------------
def fig7_algo_comparison():
    n = 300
    datasets = [
        ("Blobs",
         make_blobs(n_samples=n, centers=3, cluster_std=0.7,
                    random_state=0)[0]),
        ("Moons", make_moons(n_samples=n, noise=0.06, random_state=0)[0]),
        ("Circles",
         make_circles(n_samples=n, factor=0.5, noise=0.05,
                      random_state=0)[0]),
    ]

    # Per-dataset hyperparameters tuned for clarity.
    algo_specs = [
        ("K-means",
         lambda K: KMeans(n_clusters=K, n_init=10, random_state=0)),
        ("DBSCAN", lambda K: DBSCAN(eps=0.18, min_samples=5)),
        ("Spectral",
         lambda K: SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                      n_neighbors=10,
                                      assign_labels="kmeans",
                                      random_state=0)),
    ]
    K_per_dataset = {"Blobs": 3, "Moons": 2, "Circles": 2}
    eps_per_dataset = {"Blobs": 0.7, "Moons": 0.18, "Circles": 0.18}

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.18})

    for i, (dname, X) in enumerate(datasets):
        K = K_per_dataset[dname]
        for j, (aname, factory) in enumerate(algo_specs):
            ax = axes[i, j]
            if aname == "DBSCAN":
                model = DBSCAN(eps=eps_per_dataset[dname], min_samples=5)
            else:
                model = factory(K)
            labels = model.fit_predict(X)

            uniq = sorted(set(labels))
            for li, lab in enumerate(uniq):
                m = labels == lab
                if lab == -1:
                    ax.scatter(X[m, 0], X[m, 1], s=18, marker="x",
                               color=C_GRAY, linewidths=1.0)
                else:
                    ax.scatter(
                        X[m, 0], X[m, 1], s=18,
                        c=PALETTE[li % len(PALETTE)], alpha=0.85,
                        edgecolor="none",
                    )
            if i == 0:
                ax.set_title(aname, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(dname, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    fig.suptitle(
        "Same data, three lenses: each algorithm encodes a different prior",
        fontsize=13, fontweight="bold", y=0.995,
    )
    _save(fig, "fig7_algo_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating Part 18 (Clustering) figures...")
    fig1_kmeans_steps()
    fig2_dbscan_density()
    fig3_dendrogram()
    fig4_silhouette_curve()
    fig5_elbow()
    fig6_gmm_vs_kmeans()
    fig7_algo_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
