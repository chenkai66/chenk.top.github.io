"""Generate 84 differential geometry figures for chenk.top blog v2.

12 articles x 7 figs each. 3D figures use real surfaces/quivers/curves.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse, Polygon, Wedge, Arc
from matplotlib.patheffects import withSimplePatchShadow
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

BG = "#fdfcf9"
C = {
    "red": "#e85d4a", "amber": "#f5a834", "purple": "#8b5cf6",
    "blue": "#3b82f6", "green": "#10b981", "gray": "#6b7280", "dark": "#1f2937",
    "light": "#f3f4f6", "pink": "#ec4899", "teal": "#14b8a6",
}
DPI = 180
shadow = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.15)

OUT_BASE = "/tmp/dg_figs_v2"

SLUGS = [
    "01-curves-in-space", "02-surfaces-first-form", "03-second-form-curvature",
    "04-intrinsic-geometry", "05-gauss-bonnet", "06-smooth-manifolds",
    "07-vector-fields-flows", "08-differential-forms", "09-integration-stokes",
    "10-riemannian-geometry", "11-curvature-on-manifolds", "12-bundles-and-physics",
]


def setup_fig(figsize=(8, 6), n3d=False):
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=DPI)
    if n3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(BG)
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
    return fig, ax


def setup_3d(ax, elev=22, azim=-55, title=""):
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=13, color=C["dark"], pad=10)
    ax.set_xlabel("x", color=C["gray"], fontsize=9)
    ax.set_ylabel("y", color=C["gray"], fontsize=9)
    ax.set_zlabel("z", color=C["gray"], fontsize=9)
    ax.tick_params(colors=C["gray"], labelsize=8)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
        ax.yaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
        ax.zaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
    except Exception:
        pass
    ax.grid(True, alpha=0.2)


def title_2d(ax, txt):
    ax.set_title(txt, fontsize=13, color=C["dark"], pad=10)


def save(fig, slug, name):
    folder = os.path.join(OUT_BASE, slug)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def card(ax, x, y, w, h, color, text, fontsize=11, textcolor="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         facecolor=color, edgecolor="none")
    box.set_path_effects([shadow])
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold")


# ============================================================
# 01 - Curves in Space
# ============================================================
def fig_01_1():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    t = np.linspace(0, 4 * np.pi, 200)
    x, y, z = np.cos(t), np.sin(t), t / 4
    ax.plot(x, y, z, color=C["blue"], lw=2.5, label="helix")
    # Frenet frames at sample points
    for ti in np.linspace(0, 4 * np.pi, 6)[1:-1]:
        p = np.array([np.cos(ti), np.sin(ti), ti / 4])
        T = np.array([-np.sin(ti), np.cos(ti), 0.25]); T /= np.linalg.norm(T)
        N = np.array([-np.cos(ti), -np.sin(ti), 0]); N /= np.linalg.norm(N)
        B = np.cross(T, N); B /= np.linalg.norm(B)
        L = 0.4
        ax.quiver(*p, *(T * L), color=C["red"], lw=2)
        ax.quiver(*p, *(N * L), color=C["green"], lw=2)
        ax.quiver(*p, *(B * L), color=C["purple"], lw=2)
    setup_3d(ax, elev=18, azim=-60, title="Helix with Frenet frame (T, N, B)")
    legend_lines = [
        Line2D([0], [0], color=C["red"], lw=2, label="T (tangent)"),
        Line2D([0], [0], color=C["green"], lw=2, label="N (normal)"),
        Line2D([0], [0], color=C["purple"], lw=2, label="B (binormal)"),
    ]
    ax.legend(handles=legend_lines, loc="upper left", fontsize=9)
    return save(fig, SLUGS[0], "dg_v2_01_1_helix_frenet")


def fig_01_2():
    fig, ax = setup_fig(figsize=(9, 6))
    t = np.linspace(-1.5, 1.5, 200)
    x = t
    y = 0.5 * t ** 2
    ax.plot(x, y, color=C["blue"], lw=2.5, label="curve $\\gamma$")
    # Osculating circle at t=0.5
    t0 = 0.5
    p = np.array([t0, 0.5 * t0 ** 2])
    dp = np.array([1, t0])
    ddp = np.array([0, 1])
    speed = np.linalg.norm(dp)
    kappa = abs(dp[0] * ddp[1] - dp[1] * ddp[0]) / speed ** 3
    R = 1 / kappa
    n = np.array([-dp[1], dp[0]]) / speed
    ctr = p + R * n
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ctr[0] + R * np.cos(th), ctr[1] + R * np.sin(th),
            "--", color=C["red"], lw=1.8, label=f"osculating circle, $R=1/\\kappa$")
    ax.plot(*p, "o", color=C["dark"], markersize=8)
    ax.plot(*ctr, "x", color=C["red"], markersize=10, mew=2)
    ax.annotate(f"$\\kappa = {kappa:.2f}$", xy=p, xytext=(p[0] + 0.3, p[1] - 0.5),
                fontsize=12, color=C["dark"])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    title_2d(ax, "Curvature $\\kappa$ — radius of the osculating circle")
    return save(fig, SLUGS[0], "dg_v2_01_2_curvature_geom")


def fig_01_3():
    fig = plt.figure(figsize=(11, 5.5), facecolor=BG, dpi=DPI)
    ax1 = fig.add_subplot(121, projection='3d'); ax1.set_facecolor(BG)
    ax2 = fig.add_subplot(122, projection='3d'); ax2.set_facecolor(BG)
    # Circle: constant curvature
    t = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(t), np.sin(t), np.zeros_like(t), color=C["green"], lw=2.5)
    ax1.set_title("Circle: $\\kappa = 1/R$ const, $\\tau = 0$", fontsize=11, color=C["dark"])
    setup_3d(ax1, elev=25, azim=-50)
    # Helix: constant curvature and torsion
    s = np.linspace(0, 4 * np.pi, 200)
    ax2.plot(np.cos(s), np.sin(s), s / 4, color=C["blue"], lw=2.5)
    ax2.set_title("Helix: $\\kappa$ const, $\\tau$ const", fontsize=11, color=C["dark"])
    setup_3d(ax2, elev=18, azim=-60)
    fig.suptitle("Circle vs helix — torsion measures non-planarity", fontsize=13, color=C["dark"])
    return save(fig, SLUGS[0], "dg_v2_01_3_circle_helix")


def fig_01_4():
    fig, ax = setup_fig(figsize=(9, 6))
    t = np.linspace(0, 1.4, 80)
    x = t
    y = np.sin(np.pi * t)
    ax.plot(x, y, color=C["blue"], lw=2.5, label="curve $\\gamma(t)$")
    # Mark equal arc-length intervals
    s = np.cumsum(np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2))
    s_total = s[-1]
    targets = np.linspace(0, s_total, 9)
    for st in targets:
        idx = np.argmin(np.abs(s - st))
        ax.plot(x[idx], y[idx], "o", color=C["red"], markersize=7, zorder=5)
    ax.text(0.4, -0.35, "equal arc-length spacing → arc-length parameter $s$",
            fontsize=11, color=C["dark"], ha="center")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_aspect("equal")
    title_2d(ax, "Arc-length parameterization $\\gamma(s)$")
    return save(fig, SLUGS[0], "dg_v2_01_4_arc_length")


def fig_01_5():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    t = np.linspace(0, 4 * np.pi, 200)
    x, y, z = np.cos(t), np.sin(t), t / 4
    ax.plot(x, y, z, color=C["blue"], lw=2.2)
    # Osculating plane at t0
    t0 = 2.5
    p = np.array([np.cos(t0), np.sin(t0), t0 / 4])
    T = np.array([-np.sin(t0), np.cos(t0), 0.25]); T /= np.linalg.norm(T)
    N = np.array([-np.cos(t0), -np.sin(t0), 0]); N /= np.linalg.norm(N)
    # Plane spanned by T and N
    u, v = np.meshgrid(np.linspace(-0.7, 0.7, 12), np.linspace(-0.7, 0.7, 12))
    PX = p[0] + u * T[0] + v * N[0]
    PY = p[1] + u * T[1] + v * N[1]
    PZ = p[2] + u * T[2] + v * N[2]
    ax.plot_surface(PX, PY, PZ, color=C["amber"], alpha=0.35, edgecolor="none")
    ax.quiver(*p, *(T * 0.5), color=C["red"], lw=2)
    ax.quiver(*p, *(N * 0.5), color=C["green"], lw=2)
    ax.scatter([p[0]], [p[1]], [p[2]], color=C["dark"], s=40, zorder=10)
    setup_3d(ax, elev=20, azim=-55, title="Osculating plane: span of T and N")
    return save(fig, SLUGS[0], "dg_v2_01_5_osculating")


def fig_01_6():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Frenet–Serret formulas (arc-length s)", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    formulas = [
        ("$T'(s) = \\kappa\\, N$", C["red"]),
        ("$N'(s) = -\\kappa\\, T + \\tau\\, B$", C["green"]),
        ("$B'(s) = -\\tau\\, N$", C["purple"]),
    ]
    for i, (f, col) in enumerate(formulas):
        y = 4.0 - i * 1.1
        card(ax, 1.5, y - 0.35, 7, 0.7, col, f, fontsize=14)
    ax.text(5, 0.4, "Three vectors, two scalars ($\\kappa$, $\\tau$) — full local description",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[0], "dg_v2_01_6_frenet_serret")


def fig_01_7():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG, dpi=DPI)
    for a in axs: a.set_facecolor(BG); a.set_aspect("equal")
    # cardioid
    t = np.linspace(0, 2 * np.pi, 300)
    r = 1 - np.cos(t)
    axs[0].plot(r * np.cos(t), r * np.sin(t), color=C["red"], lw=2.5)
    axs[0].set_title("Cardioid $r = 1 - \\cos\\theta$", fontsize=11, color=C["dark"])
    # lemniscate
    t = np.linspace(-np.pi / 4, np.pi / 4, 100)
    r2 = np.cos(2 * t)
    r = np.sqrt(np.maximum(r2, 0))
    x1 = r * np.cos(t); y1 = r * np.sin(t)
    x2 = -x1
    axs[1].plot(x1, y1, color=C["green"], lw=2.5)
    axs[1].plot(x2, y1, color=C["green"], lw=2.5)
    axs[1].set_title("Lemniscate $r^2 = \\cos 2\\theta$", fontsize=11, color=C["dark"])
    # spiral
    t = np.linspace(0, 6 * np.pi, 400)
    r = 0.1 * t
    axs[2].plot(r * np.cos(t), r * np.sin(t), color=C["purple"], lw=2.5)
    axs[2].set_title("Archimedean spiral $r = a\\theta$", fontsize=11, color=C["dark"])
    for a in axs:
        a.grid(True, alpha=0.25); a.tick_params(labelsize=8, colors=C["gray"])
    fig.suptitle("Classical plane curves", fontsize=13, color=C["dark"])
    fig.tight_layout()
    return save(fig, SLUGS[0], "dg_v2_01_7_classical_curves")


# ============================================================
# 02 - Surfaces & First Form
# ============================================================
def fig_02_1():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)
    ax.plot_surface(X, Y, Z, color=C["blue"], alpha=0.55, edgecolor=C["dark"], linewidth=0.3)
    # Highlight one parameter curve
    v0 = np.pi / 2.5
    ax.plot(np.cos(u) * np.sin(v0), np.sin(u) * np.sin(v0), np.cos(v0) * np.ones_like(u),
            color=C["red"], lw=2.5, label="$u$-curve")
    u0 = 1.2
    ax.plot(np.cos(u0) * np.sin(v), np.sin(u0) * np.sin(v), np.cos(v),
            color=C["green"], lw=2.5, label="$v$-curve")
    setup_3d(ax, elev=20, azim=-50, title="Parameterized surface $\\mathbf{r}(u,v)$")
    ax.legend(fontsize=10)
    return save(fig, SLUGS[1], "dg_v2_02_1_surface_patch")


def fig_02_2():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(-1.2, 1.2, 30)
    v = np.linspace(-1.2, 1.2, 30)
    U, V = np.meshgrid(u, v)
    Z = 0.4 * (U ** 2 - V ** 2)
    ax.plot_surface(U, V, Z, color=C["amber"], alpha=0.5, edgecolor="none")
    # Tangent plane at (0.4, 0.3)
    p = np.array([0.4, 0.3, 0.4 * (0.4 ** 2 - 0.3 ** 2)])
    ru = np.array([1, 0, 0.8 * 0.4])
    rv = np.array([0, 1, -0.8 * 0.3])
    n = np.cross(ru, rv); n /= np.linalg.norm(n)
    s, t = np.meshgrid(np.linspace(-0.6, 0.6, 8), np.linspace(-0.6, 0.6, 8))
    PX = p[0] + s * ru[0] + t * rv[0]
    PY = p[1] + s * ru[1] + t * rv[1]
    PZ = p[2] + s * ru[2] + t * rv[2]
    ax.plot_surface(PX, PY, PZ, color=C["blue"], alpha=0.4, edgecolor="none")
    ax.quiver(*p, *(ru * 0.5), color=C["red"], lw=2.5)
    ax.quiver(*p, *(rv * 0.5), color=C["green"], lw=2.5)
    ax.quiver(*p, *(n * 0.7), color=C["purple"], lw=2.5)
    ax.scatter([p[0]], [p[1]], [p[2]], color=C["dark"], s=40)
    setup_3d(ax, elev=25, azim=-55, title="Tangent plane $T_p S$ and normal $\\mathbf{n}$")
    return save(fig, SLUGS[1], "dg_v2_02_2_tangent_plane")


def fig_02_3():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "First fundamental form $I = E\\,du^2 + 2F\\,du\\,dv + G\\,dv^2$",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    items = [
        ("$E = \\langle\\mathbf{r}_u, \\mathbf{r}_u\\rangle$", C["red"], "length of $u$-curve"),
        ("$F = \\langle\\mathbf{r}_u, \\mathbf{r}_v\\rangle$", C["amber"], "angle between curves"),
        ("$G = \\langle\\mathbf{r}_v, \\mathbf{r}_v\\rangle$", C["green"], "length of $v$-curve"),
    ]
    for i, (f, col, desc) in enumerate(items):
        y = 3.6 - i * 1.0
        card(ax, 0.5, y - 0.3, 4.0, 0.65, col, f, fontsize=12)
        ax.text(5.0, y, desc, fontsize=11, color=C["dark"], va="center")
    ax.text(5, 0.3, "Encodes intrinsic geometry — lengths, angles, areas",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[1], "dg_v2_02_3_first_form")


def fig_02_4():
    fig, ax = setup_fig(figsize=(9, 6))
    # parameter domain with curve
    t = np.linspace(0, 1, 100)
    u = t; v = 0.4 * np.sin(2 * np.pi * t)
    ax.plot(u, v, color=C["blue"], lw=2.5, label="curve $(u(t), v(t))$ in domain")
    ax.fill_between(u, v - 0.02, v + 0.02, color=C["blue"], alpha=0.1)
    for ti in np.linspace(0, 1, 8):
        ui = ti; vi = 0.4 * np.sin(2 * np.pi * ti)
        ax.plot(ui, vi, "o", color=C["red"], markersize=5)
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.text(0.5, -0.6,
            "$L = \\int_a^b \\sqrt{E\\,\\dot u^2 + 2F\\,\\dot u\\,\\dot v + G\\,\\dot v^2}\\,dt$",
            fontsize=13, color=C["dark"], ha="center")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.7, 0.6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    title_2d(ax, "Arc length on a surface — uses first form")
    return save(fig, SLUGS[1], "dg_v2_02_4_arc_length_surf")


def fig_02_5():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show parallelogram of du dv with sides ru, rv
    O = np.array([0.5, 0.5])
    ru = np.array([1.5, 0.4])
    rv = np.array([0.3, 1.2])
    pts = np.array([O, O + ru, O + ru + rv, O + rv])
    poly = Polygon(pts, closed=True, facecolor=C["amber"], alpha=0.4, edgecolor=C["dark"], lw=1.5)
    ax.add_patch(poly)
    ax.annotate("", xy=O + ru, xytext=O,
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.5))
    ax.annotate("", xy=O + rv, xytext=O,
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=2.5))
    ax.text(*(O + ru / 2 + np.array([0.0, -0.15])), "$\\mathbf{r}_u\\,du$",
            color=C["red"], fontsize=12, ha="center")
    ax.text(*(O + rv / 2 + np.array([-0.2, 0])), "$\\mathbf{r}_v\\,dv$",
            color=C["green"], fontsize=12, ha="center")
    ax.text(O[0] + (ru[0] + rv[0]) / 2, O[1] + (ru[1] + rv[1]) / 2,
            "$dA = \\sqrt{EG - F^2}\\,du\\,dv$", fontsize=12, color=C["dark"], ha="center")
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 2.5); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    title_2d(ax, "Area element on a surface")
    return save(fig, SLUGS[1], "dg_v2_02_5_area_element")


def fig_02_6():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show conformal map preserving angles
    u = np.linspace(-1, 1, 12)
    for ui in u:
        ax.plot([ui, ui], [-1, 1], color=C["blue"], lw=0.8, alpha=0.7)
        ax.plot([-1, 1], [ui, ui], color=C["red"], lw=0.8, alpha=0.7)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.text(0, -1.35, "Isothermal coordinates: $E = G,\\, F = 0$ → $I = \\lambda^2(du^2 + dv^2)$",
            fontsize=11, color=C["dark"], ha="center")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    title_2d(ax, "Isothermal (conformal) coordinates — preserve angles")
    return save(fig, SLUGS[1], "dg_v2_02_6_isothermal")


def fig_02_7():
    fig = plt.figure(figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    # Sphere
    ax = fig.add_subplot(131, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.6, edgecolor="none")
    ax.set_title("Sphere $S^2$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Cylinder
    ax = fig.add_subplot(132, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(-1, 1, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U), np.sin(U), V, color=C["green"], alpha=0.6, edgecolor="none")
    ax.set_title("Cylinder", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Torus
    ax = fig.add_subplot(133, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    R, r = 1.2, 0.4
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, color=C["amber"], alpha=0.6, edgecolor="none")
    ax.set_title("Torus $T^2$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=25, azim=-50)
    fig.suptitle("Classical surfaces", fontsize=13, color=C["dark"])
    fig.tight_layout()
    return save(fig, SLUGS[1], "dg_v2_02_7_classical_surfaces")


# ============================================================
# 03 - Second Form & Curvature
# ============================================================
def fig_03_1():
    fig = plt.figure(figsize=(11, 5.5), facecolor=BG, dpi=DPI)
    ax1 = fig.add_subplot(121, projection='3d'); ax1.set_facecolor(BG)
    ax2 = fig.add_subplot(122, projection='3d'); ax2.set_facecolor(BG)
    # Surface
    u = np.linspace(-1.2, 1.2, 25); v = np.linspace(-1.2, 1.2, 25)
    U, V = np.meshgrid(u, v)
    Z = 0.4 * (U ** 2 + V ** 2 * 0.5)
    ax1.plot_surface(U, V, Z, color=C["amber"], alpha=0.55, edgecolor="none")
    # Normals
    for ui in [-0.8, 0, 0.8]:
        for vi in [-0.8, 0, 0.8]:
            zi = 0.4 * (ui ** 2 + 0.5 * vi ** 2)
            zu = 0.8 * ui; zv = 0.4 * vi
            n = np.array([-zu, -zv, 1]); n /= np.linalg.norm(n)
            ax1.quiver(ui, vi, zi, *n * 0.4, color=C["red"], lw=1.8)
    ax1.set_title("Surface $S$ with normals", fontsize=11, color=C["dark"])
    setup_3d(ax1, elev=25, azim=-50)
    # Unit sphere with corresponding normals plotted
    th = np.linspace(0, 2 * np.pi, 40); ph = np.linspace(0, np.pi, 20)
    TH, PH = np.meshgrid(th, ph)
    ax2.plot_surface(np.cos(TH) * np.sin(PH), np.sin(TH) * np.sin(PH), np.cos(PH),
                     color=C["blue"], alpha=0.3, edgecolor="none")
    for ui in [-0.8, 0, 0.8]:
        for vi in [-0.8, 0, 0.8]:
            zu = 0.8 * ui; zv = 0.4 * vi
            n = np.array([-zu, -zv, 1]); n /= np.linalg.norm(n)
            ax2.scatter(*n, color=C["red"], s=40, zorder=10)
    ax2.set_title("Gauss map image on $S^2$", fontsize=11, color=C["dark"])
    setup_3d(ax2, elev=20, azim=-50)
    fig.suptitle("Gauss map $N: S \\to S^2$", fontsize=13, color=C["dark"])
    return save(fig, SLUGS[2], "dg_v2_03_1_gauss_map")


def fig_03_2():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show shape operator action: tangent plane + eigenvectors
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), color=C["dark"], lw=1.2, alpha=0.6)
    # Two principal directions
    e1 = np.array([1.5, 0]); e2 = np.array([0, 0.7])
    ax.annotate("", xy=e1, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.5))
    ax.annotate("", xy=e2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=2.5))
    ax.text(*e1 + np.array([0.05, -0.15]), "$\\mathbf{e}_1$ ($\\kappa_1$)",
            color=C["red"], fontsize=12)
    ax.text(*e2 + np.array([0.1, 0.02]), "$\\mathbf{e}_2$ ($\\kappa_2$)",
            color=C["green"], fontsize=12)
    # Eigen-ellipse (image under shape operator)
    th2 = np.linspace(0, 2 * np.pi, 100)
    ax.plot(1.5 * np.cos(th2), 0.7 * np.sin(th2), "--", color=C["purple"], lw=1.8,
            label="image of unit circle under $S_p$")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=10)
    title_2d(ax, "Shape operator $S_p$ and principal directions")
    return save(fig, SLUGS[2], "dg_v2_03_2_shape_operator")


def fig_03_3():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(-1.2, 1.2, 30); v = np.linspace(-1.2, 1.2, 30)
    U, V = np.meshgrid(u, v)
    Z = 0.5 * U ** 2 - 0.3 * V ** 2  # saddle for clear principal directions
    ax.plot_surface(U, V, Z, color=C["amber"], alpha=0.55, edgecolor="none")
    # At origin principal directions are x and y
    ax.quiver(0, 0, 0, 1, 0, 0, color=C["red"], lw=3, label="$\\mathbf{e}_1$")
    ax.quiver(0, 0, 0, 0, 1, 0, color=C["green"], lw=3, label="$\\mathbf{e}_2$")
    ax.scatter([0], [0], [0], color=C["dark"], s=60)
    setup_3d(ax, elev=22, azim=-50, title="Principal curvature directions on saddle")
    ax.legend(fontsize=10)
    return save(fig, SLUGS[2], "dg_v2_03_3_principal_curvatures")


def fig_03_4():
    fig = plt.figure(figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    # Sphere K > 0
    ax = fig.add_subplot(131, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["red"], alpha=0.7, edgecolor="none")
    ax.set_title("Sphere: $K > 0$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Cylinder K = 0
    ax = fig.add_subplot(132, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(-1, 1, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U), np.sin(U), V, color=C["amber"], alpha=0.7, edgecolor="none")
    ax.set_title("Cylinder: $K = 0$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Saddle K < 0
    ax = fig.add_subplot(133, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(-1.2, 1.2, 30); v = np.linspace(-1.2, 1.2, 30)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(U, V, U ** 2 - V ** 2, color=C["purple"], alpha=0.7, edgecolor="none")
    ax.set_title("Saddle: $K < 0$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=22, azim=-50)
    fig.suptitle("Three signs of Gaussian curvature $K = \\kappa_1\\kappa_2$",
                 fontsize=13, color=C["dark"])
    return save(fig, SLUGS[2], "dg_v2_03_4_gauss_K_three")


def fig_03_5():
    fig, ax = setup_fig(figsize=(9, 6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "Mean curvature  $H = \\dfrac{\\kappa_1 + \\kappa_2}{2}$",
            fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    items = [
        ("$\\kappa_1, \\kappa_2 > 0$", "convex bump", C["red"]),
        ("$\\kappa_1 = -\\kappa_2$", "minimal surface ($H = 0$)", C["purple"]),
        ("Soap films minimize area", "→ $H = 0$ everywhere", C["blue"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.0 - i * 1.1
        card(ax, 1.0, y - 0.3, 3.5, 0.65, col, a, fontsize=12)
        ax.text(5.0, y, "→ " + b, fontsize=12, color=C["dark"], va="center")
    return save(fig, SLUGS[2], "dg_v2_03_5_mean_curvature")


def fig_03_6():
    fig, ax = setup_fig(figsize=(10, 6))
    types = ["Sphere\n$R=1$", "Cylinder\n$R=1$", "Saddle"]
    K = [1.0, 0.0, -0.5]
    H = [1.0, 0.5, 0.0]
    x = np.arange(len(types))
    w = 0.35
    ax.bar(x - w / 2, K, w, color=C["blue"], label="$K$")
    ax.bar(x + w / 2, H, w, color=C["amber"], label="$H$")
    ax.axhline(0, color=C["dark"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("curvature", color=C["dark"])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.25, axis="y")
    title_2d(ax, "Comparison: Gauss vs. mean curvature on three surfaces")
    return save(fig, SLUGS[2], "dg_v2_03_6_curvature_compare")


def fig_03_7():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    # Helicoid: x = u cos v, y = u sin v, z = c v
    u = np.linspace(-1, 1, 30)
    v = np.linspace(0, 3 * np.pi, 60)
    U, V = np.meshgrid(u, v)
    X = U * np.cos(V); Y = U * np.sin(V); Z = 0.3 * V
    ax.plot_surface(X, Y, Z, color=C["teal"], alpha=0.7, edgecolor="none")
    setup_3d(ax, elev=18, azim=-55, title="Helicoid — minimal surface ($H = 0$)")
    return save(fig, SLUGS[2], "dg_v2_03_7_minimal_surfaces")


# ============================================================
# 04 - Intrinsic Geometry
# ============================================================
def fig_04_1():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "Christoffel symbols $\\Gamma^k_{ij}$ — connection from $g_{ij}$",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    eqn = "$\\Gamma^k_{ij} = \\dfrac{1}{2} g^{k\\ell} (\\partial_i g_{j\\ell} + \\partial_j g_{i\\ell} - \\partial_\\ell g_{ij})$"
    card(ax, 1.0, 3.0, 8.0, 1.2, C["blue"], eqn, fontsize=14)
    ax.text(5, 1.8, "Determines covariant derivative $\\nabla_i V^j = \\partial_i V^j + \\Gamma^j_{ik} V^k$",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.2, "Built only from the metric — purely intrinsic",
            fontsize=11, color=C["green"], ha="center", style="italic")
    return save(fig, SLUGS[3], "dg_v2_04_1_christoffel")


def fig_04_2():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 4.9, "Geodesic equation", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eqn = "$\\ddot\\gamma^k + \\Gamma^k_{ij}\\,\\dot\\gamma^i\\,\\dot\\gamma^j = 0$"
    card(ax, 1.5, 2.8, 7.0, 1.2, C["red"], eqn, fontsize=16)
    ax.text(5, 1.7, "Locally length-minimizing curves — generalize straight lines",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.1, "On Euclidean space: $\\Gamma = 0$ → $\\ddot\\gamma = 0$ (lines)",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[3], "dg_v2_04_2_geodesic_eq")


def fig_04_3():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 25)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.4, edgecolor="none")
    # Great circle in equator
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color=C["red"], lw=3, label="great circle")
    # Tilted great circle
    n = np.array([0.3, 0.4, 1]); n /= np.linalg.norm(n)
    e1 = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), n) * n; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    pts = np.array([np.cos(ti) * e1 + np.sin(ti) * e2 for ti in t])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=C["green"], lw=3, label="tilted great circle")
    setup_3d(ax, elev=22, azim=-55, title="Geodesics on $S^2$ — great circles")
    ax.legend(fontsize=10)
    return save(fig, SLUGS[3], "dg_v2_04_3_sphere_geodesic")


def fig_04_4():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.5, "Theorema Egregium (Gauss, 1827)", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    card(ax, 1.0, 3.5, 8.0, 1.2, C["purple"],
         "$K$ depends only on the metric $g_{ij}$", fontsize=14)
    ax.text(5, 2.7, "→ $K$ is intrinsic: invariant under isometry",
            fontsize=12, color=C["dark"], ha="center")
    ax.text(5, 2.0, "→ Cannot flatten a sphere onto a plane preserving distances",
            fontsize=11, color=C["red"], ha="center")
    ax.text(5, 1.3, "(why all flat maps of Earth distort)",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[3], "dg_v2_04_4_egregium")


def fig_04_5():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=BG, dpi=DPI)
    for a in axs: a.set_facecolor(BG); a.set_aspect("equal")
    # plane region
    pts = np.array([[0, 0], [1, 0], [1.3, 0.8], [0.3, 1]])
    axs[0].add_patch(Polygon(pts, facecolor=C["blue"], alpha=0.4, edgecolor=C["dark"]))
    axs[0].set_xlim(-0.3, 1.7); axs[0].set_ylim(-0.3, 1.3)
    axs[0].set_title("Region $A$ — $K = 0$", fontsize=11, color=C["dark"])
    # rotated/translated copy
    th = 0.6
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts2 = pts @ R.T + np.array([0.4, 0.2])
    axs[1].add_patch(Polygon(pts2, facecolor=C["red"], alpha=0.4, edgecolor=C["dark"]))
    axs[1].set_xlim(-0.3, 1.7); axs[1].set_ylim(-0.3, 1.5)
    axs[1].set_title("Isometric image — same $K$", fontsize=11, color=C["dark"])
    # arrow
    fig.suptitle("Isometry preserves Gaussian curvature", fontsize=13, color=C["dark"])
    fig.tight_layout()
    return save(fig, SLUGS[3], "dg_v2_04_5_isometry")


def fig_04_6():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    R, r = 1.5, 0.5
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, color=C["amber"], alpha=0.4, edgecolor="none")
    # A geodesic-like curve - meridian (closed)
    t = np.linspace(0, 2 * np.pi, 100)
    Xm = (R + r * np.cos(t)) * np.cos(0)
    Ym = (R + r * np.cos(t)) * np.sin(0)
    Zm = r * np.sin(t)
    ax.plot(Xm, Ym, Zm, color=C["red"], lw=3, label="meridian (closed)")
    # winding curve
    s = np.linspace(0, 6 * np.pi, 300)
    Xw = (R + r * np.cos(3 * s)) * np.cos(s)
    Yw = (R + r * np.cos(3 * s)) * np.sin(s)
    Zw = r * np.sin(3 * s)
    ax.plot(Xw, Yw, Zw, color=C["blue"], lw=2, label="winding curve")
    setup_3d(ax, elev=30, azim=-60, title="Geodesics on torus — closed and dense")
    ax.legend(fontsize=10)
    return save(fig, SLUGS[3], "dg_v2_04_6_torus_geodesic")


def fig_04_7():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.0, "Intrinsic vs Extrinsic geometry", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    card(ax, 0.5, 1.8, 4.5, 2.5, C["green"], "", fontsize=10)
    ax.text(2.75, 3.7, "Intrinsic", fontsize=13, color="white", ha="center", fontweight="bold")
    ax.text(2.75, 3.0, "lengths, angles, $K$\nfirst form $g_{ij}$", fontsize=11,
            color="white", ha="center")
    ax.text(2.75, 2.2, "ant on the surface", fontsize=10, color="white", ha="center", style="italic")
    card(ax, 6.0, 1.8, 4.5, 2.5, C["red"], "", fontsize=10)
    ax.text(8.25, 3.7, "Extrinsic", fontsize=13, color="white", ha="center", fontweight="bold")
    ax.text(8.25, 3.0, "embedding, $H$, normals\nsecond form", fontsize=11,
            color="white", ha="center")
    ax.text(8.25, 2.2, "viewer in ambient $\\mathbb{R}^3$", fontsize=10, color="white",
            ha="center", style="italic")
    ax.text(5.5, 0.7, "Egregium: $K$ is intrinsic but $H$ is not",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[3], "dg_v2_04_7_intrinsic_extrinsic")


# ============================================================
# 05 - Gauss-Bonnet
# ============================================================
def fig_05_1():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, np.pi, 30); v = np.linspace(0, np.pi / 1.2, 25)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.4, edgecolor="none")
    # Geodesic triangle
    A = np.array([0, 0, 1])
    B = np.array([1, 0, 0])
    Cp = np.array([0, 1, 0])
    def arc(p, q, n=30):
        ts = np.linspace(0, 1, n)
        pts = []
        for t in ts:
            v = (1 - t) * p + t * q
            pts.append(v / np.linalg.norm(v))
        return np.array(pts)
    for p, q in [(A, B), (B, Cp), (Cp, A)]:
        a = arc(p, q)
        ax.plot(a[:, 0], a[:, 1], a[:, 2], color=C["red"], lw=3)
    setup_3d(ax, elev=25, azim=-60, title="Geodesic triangle on sphere")
    return save(fig, SLUGS[4], "dg_v2_05_1_local_gb")


def fig_05_2():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 25)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.3, edgecolor="none")
    # Triangle with 3 right-angle-ish vertices
    A = np.array([0, 0, 1])
    B = np.array([1, 0, 0])
    Cp = np.array([0, 1, 0])
    def arc(p, q, n=40):
        ts = np.linspace(0, 1, n); pts = []
        for t in ts:
            v = (1 - t) * p + t * q; pts.append(v / np.linalg.norm(v))
        return np.array(pts)
    poly_pts = []
    for p, q in [(A, B), (B, Cp), (Cp, A)]:
        a = arc(p, q)
        ax.plot(a[:, 0], a[:, 1], a[:, 2], color=C["red"], lw=3)
        poly_pts.extend(a.tolist())
    poly = Poly3DCollection([poly_pts], color=C["red"], alpha=0.25)
    ax.add_collection3d(poly)
    ax.text2D(0.03, 0.93, "Angle sum = $3\\pi/2 > \\pi$  on $S^2$",
              transform=ax.transAxes, fontsize=12, color=C["dark"], fontweight="bold")
    setup_3d(ax, elev=25, azim=-50, title="Spherical triangle — angle excess")
    return save(fig, SLUGS[4], "dg_v2_05_2_geodesic_triangle_sphere")


def fig_05_3():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Global Gauss-Bonnet", fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    eqn = "$\\int_M K\\,dA = 2\\pi\\,\\chi(M)$"
    card(ax, 2.0, 2.5, 6.0, 1.5, C["purple"], eqn, fontsize=18)
    ax.text(5, 1.5, "Total curvature = $2\\pi$ × Euler characteristic",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 0.9, "Geometry constrains topology",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[4], "dg_v2_05_3_global_gb")


def fig_05_4():
    fig = plt.figure(figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    # sphere chi=2
    ax = fig.add_subplot(131, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["red"], alpha=0.7, edgecolor="none")
    ax.set_title("Sphere $\\chi = 2$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # torus chi=0
    ax = fig.add_subplot(132, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, 2 * np.pi, 25)
    U, V = np.meshgrid(u, v)
    R, r = 1.2, 0.4
    ax.plot_surface((R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V),
                    color=C["amber"], alpha=0.7, edgecolor="none")
    ax.set_title("Torus $\\chi = 0$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=30, azim=-50)
    # Genus 2 (double torus) - approximate with two tori joined
    ax = fig.add_subplot(133, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, 2 * np.pi, 20)
    U, V = np.meshgrid(u, v)
    R, r = 0.7, 0.3
    X1 = -1 + (R + r * np.cos(V)) * np.cos(U); Y1 = (R + r * np.cos(V)) * np.sin(U); Z1 = r * np.sin(V)
    X2 = 1 + (R + r * np.cos(V)) * np.cos(U); Y2 = (R + r * np.cos(V)) * np.sin(U); Z2 = r * np.sin(V)
    ax.plot_surface(X1, Y1, Z1, color=C["purple"], alpha=0.7, edgecolor="none")
    ax.plot_surface(X2, Y2, Z2, color=C["purple"], alpha=0.7, edgecolor="none")
    ax.set_title("Double torus $\\chi = -2$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=25, azim=-50)
    fig.suptitle("Euler characteristics", fontsize=13, color=C["dark"])
    return save(fig, SLUGS[4], "dg_v2_05_4_euler_char")


def fig_05_5():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    # Triangulation of a sphere using icosahedron-like triangles
    from itertools import combinations
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1, s2 in [(1, phi), (-1, phi), (1, -phi), (-1, -phi)]:
        verts.append([0, s1, s2])
        verts.append([s1, s2, 0])
        verts.append([s2, 0, s1])
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts[0])
    # crude triangles connecting nearest neighbors
    tris = []
    for i, j, k in combinations(range(len(verts)), 3):
        d1 = np.linalg.norm(verts[i] - verts[j])
        d2 = np.linalg.norm(verts[j] - verts[k])
        d3 = np.linalg.norm(verts[i] - verts[k])
        if max(d1, d2, d3) < 1.2:
            tris.append([verts[i], verts[j], verts[k]])
    poly = Poly3DCollection(tris, alpha=0.5, facecolor=C["amber"], edgecolor=C["dark"], linewidth=0.7)
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    setup_3d(ax, elev=25, azim=-50, title="Triangulation of a closed surface")
    return save(fig, SLUGS[4], "dg_v2_05_5_triangulation")


def fig_05_6():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Topology constrains geometry", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    rows = [
        ("Sphere", "$\\chi = 2$", "$\\int K = 4\\pi$ → some $K > 0$", C["red"]),
        ("Torus", "$\\chi = 0$", "$\\int K = 0$ → $K$ averages to 0", C["amber"]),
        ("Double torus", "$\\chi = -2$", "$\\int K = -4\\pi$ → some $K < 0$", C["purple"]),
    ]
    for i, (a, b, c, col) in enumerate(rows):
        y = 4.0 - i * 1.05
        card(ax, 0.5, y - 0.3, 2.0, 0.65, col, a, fontsize=11)
        ax.text(3.0, y, b, fontsize=11, color=C["dark"], va="center")
        ax.text(4.5, y, "→ " + c, fontsize=11, color=C["dark"], va="center")
    return save(fig, SLUGS[4], "dg_v2_05_6_topology_constrains")


def fig_05_7():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Applications of Gauss-Bonnet", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    items = [
        ("Hairy ball theorem", "Vector fields on $S^2$ must vanish", C["red"]),
        ("No flat torus in $\\mathbb{R}^3$", "Embedded torus has positive & negative $K$", C["amber"]),
        ("Polyhedron formula", "$V - E + F = \\chi$", C["green"]),
        ("Index theorem ancestor", "Generalizes to higher dimensions", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.1 - i * 0.85
        card(ax, 0.5, y - 0.25, 3.5, 0.55, col, a, fontsize=10)
        ax.text(4.5, y, b, fontsize=10, color=C["dark"], va="center")
    return save(fig, SLUGS[4], "dg_v2_05_7_examples")


# ============================================================
# 06 - Smooth Manifolds
# ============================================================
def fig_06_1():
    fig, ax = setup_fig(figsize=(10, 6))
    # Manifold blob with two charts
    th = np.linspace(0, 2 * np.pi, 100)
    rb = 1.5 + 0.2 * np.sin(3 * th)
    ax.plot(rb * np.cos(th), rb * np.sin(th), color=C["dark"], lw=2)
    ax.fill(rb * np.cos(th), rb * np.sin(th), color=C["light"], alpha=0.5)
    # Chart regions
    c1 = Circle((-0.7, 0.3), 0.7, color=C["red"], alpha=0.35)
    c2 = Circle((0.6, -0.2), 0.8, color=C["blue"], alpha=0.35)
    ax.add_patch(c1); ax.add_patch(c2)
    ax.text(-0.7, 0.3, "$U_1$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.text(0.6, -0.2, "$U_2$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.text(0, -2.0, "$M = \\bigcup_\\alpha U_\\alpha$",
            fontsize=14, color=C["dark"], ha="center")
    # Charts to R^n shown as squares to right
    ax.add_patch(Rectangle((2.5, 0.8), 1.2, 1.2, facecolor=C["red"], alpha=0.35, edgecolor=C["dark"]))
    ax.text(3.1, 1.4, "$\\varphi_1(U_1)$", fontsize=11, ha="center", va="center")
    ax.add_patch(Rectangle((2.5, -1.5), 1.2, 1.2, facecolor=C["blue"], alpha=0.35, edgecolor=C["dark"]))
    ax.text(3.1, -0.9, "$\\varphi_2(U_2)$", fontsize=11, ha="center", va="center")
    ax.annotate("", xy=(2.5, 1.4), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5))
    ax.annotate("", xy=(2.5, -0.9), xytext=(1.0, -0.2),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5))
    ax.set_xlim(-2.5, 4.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
    ax.axis("off")
    title_2d(ax, "Charts and atlas — locally Euclidean")
    return save(fig, SLUGS[5], "dg_v2_06_1_chart_atlas")


def fig_06_2():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # Two squares
    ax.add_patch(Rectangle((0.5, 1.5), 3, 3, facecolor=C["red"], alpha=0.3, edgecolor=C["dark"]))
    ax.text(2, 4.7, "$\\varphi_\\alpha(U_\\alpha \\cap U_\\beta)$", fontsize=11, ha="center")
    ax.add_patch(Rectangle((6.5, 1.5), 3, 3, facecolor=C["blue"], alpha=0.3, edgecolor=C["dark"]))
    ax.text(8, 4.7, "$\\varphi_\\beta(U_\\alpha \\cap U_\\beta)$", fontsize=11, ha="center")
    ax.annotate("", xy=(6.5, 3), xytext=(3.5, 3),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=2.5))
    ax.text(5, 3.4, "$\\varphi_\\beta \\circ \\varphi_\\alpha^{-1}$",
            fontsize=13, color=C["purple"], ha="center", fontweight="bold")
    ax.text(5, 0.8, "Smooth ($C^\\infty$) → smooth manifold structure",
            fontsize=11, color=C["dark"], ha="center")
    title_2d(ax, "Transition maps")
    return save(fig, SLUGS[5], "dg_v2_06_2_transition")


def fig_06_3():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 50); v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V); Y = np.sin(U) * np.sin(V); Z = np.cos(V)
    ax.plot_surface(X, Y, Z, color=C["light"], alpha=0.4, edgecolor="none")
    # Highlight the 'north hemisphere minus south pole' chart and vice versa
    mask_n = Z > -0.95
    Xn = np.where(mask_n, X, np.nan); Yn = np.where(mask_n, Y, np.nan); Zn = np.where(mask_n, Z, np.nan)
    ax.plot_surface(Xn, Yn, Zn, color=C["red"], alpha=0.5, edgecolor="none")
    ax.scatter([0], [0], [-1], color=C["dark"], s=80, label="south pole excluded")
    ax.scatter([0], [0], [1], color=C["blue"], s=80, label="north pole")
    setup_3d(ax, elev=20, azim=-50, title="$S^2$: stereographic projection — two charts")
    ax.legend(fontsize=9)
    return save(fig, SLUGS[5], "dg_v2_06_3_sphere_charts")


def fig_06_4():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show fundamental polygon with identification
    ax.add_patch(Rectangle((1, 1), 4, 4, facecolor=C["amber"], alpha=0.3, edgecolor=C["dark"], lw=1.5))
    # Arrows on edges
    ax.annotate("", xy=(5, 1), xytext=(1, 1),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.5))
    ax.annotate("", xy=(5, 5), xytext=(1, 5),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.5))
    ax.annotate("", xy=(1, 5), xytext=(1, 1),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=2.5))
    ax.annotate("", xy=(5, 5), xytext=(5, 1),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=2.5))
    ax.text(3, 0.6, "identify horizontal edges", fontsize=10, color=C["red"], ha="center")
    ax.text(0.5, 3, "identify vertical", fontsize=10, color=C["blue"], ha="center", rotation=90)
    ax.text(3, 5.5, "torus = $\\mathbb{R}^2 / \\mathbb{Z}^2$", fontsize=12, color=C["dark"],
            ha="center", fontweight="bold")
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_aspect("equal"); ax.axis("off")
    title_2d(ax, "Torus — fundamental polygon and atlas")
    return save(fig, SLUGS[5], "dg_v2_06_4_torus_charts")


def fig_06_5():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 25)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.4, edgecolor="none")
    p = np.array([np.cos(0.7) * np.sin(1.0), np.sin(0.7) * np.sin(1.0), np.cos(1.0)])
    # Tangent plane at p
    e1 = np.cross([0, 0, 1], p); e1 /= np.linalg.norm(e1)
    e2 = np.cross(p, e1); e2 /= np.linalg.norm(e2)
    s, t = np.meshgrid(np.linspace(-0.4, 0.4, 10), np.linspace(-0.4, 0.4, 10))
    PX = p[0] + s * e1[0] + t * e2[0]
    PY = p[1] + s * e1[1] + t * e2[1]
    PZ = p[2] + s * e1[2] + t * e2[2]
    ax.plot_surface(PX, PY, PZ, color=C["red"], alpha=0.4, edgecolor="none")
    ax.quiver(*p, *(e1 * 0.5), color=C["red"], lw=2.5)
    ax.quiver(*p, *(e2 * 0.5), color=C["green"], lw=2.5)
    ax.scatter(*p, color=C["dark"], s=50)
    setup_3d(ax, elev=25, azim=-50, title="Tangent space $T_p M$")
    return save(fig, SLUGS[5], "dg_v2_06_5_tangent_space")


def fig_06_6():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    # Two manifolds
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(0.5 + np.cos(th), 3 + np.sin(th), color=C["dark"], lw=2)
    ax.fill(0.5 + np.cos(th), 3 + np.sin(th), color=C["red"], alpha=0.3)
    ax.text(0.5, 3, "$M$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.plot(8 + 1.2 * np.cos(th), 3 + 1.0 * np.sin(th), color=C["dark"], lw=2)
    ax.fill(8 + 1.2 * np.cos(th), 3 + 1.0 * np.sin(th), color=C["blue"], alpha=0.3)
    ax.text(8, 3, "$N$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(6.7, 3), xytext=(1.6, 3),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=3))
    ax.text(4.2, 3.4, "$f: M \\to N$", fontsize=14, color=C["purple"], ha="center", fontweight="bold")
    ax.text(4.2, 1.5, "smooth iff $\\psi \\circ f \\circ \\varphi^{-1}$ is $C^\\infty$ in coordinates",
            fontsize=11, color=C["dark"], ha="center")
    title_2d(ax, "Smooth maps between manifolds")
    return save(fig, SLUGS[5], "dg_v2_06_6_smooth_map")


def fig_06_7():
    fig = plt.figure(figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    # Sphere
    ax = fig.add_subplot(131, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["red"], alpha=0.7, edgecolor="none")
    ax.set_title("$S^2$ orientable", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Torus
    ax = fig.add_subplot(132, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, 2 * np.pi, 25)
    U, V = np.meshgrid(u, v)
    R, r = 1.2, 0.4
    ax.plot_surface((R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V),
                    color=C["amber"], alpha=0.7, edgecolor="none")
    ax.set_title("$T^2$ orientable", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=25, azim=-50)
    # Klein bottle (immersed)
    ax = fig.add_subplot(133, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    # Standard immersion of Klein bottle
    a = 2
    X = (a + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.cos(U)
    Y = (a + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.sin(U)
    Z = np.sin(U / 2) * np.sin(V) + np.cos(U / 2) * np.sin(2 * V)
    ax.plot_surface(X, Y, Z, color=C["purple"], alpha=0.6, edgecolor="none")
    ax.set_title("Klein bottle (non-orientable)", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=25, azim=-60)
    fig.suptitle("Classical manifolds", fontsize=13, color=C["dark"])
    return save(fig, SLUGS[5], "dg_v2_06_7_classical_manifolds")


# ============================================================
# 07 - Vector Fields & Flows
# ============================================================
def fig_07_1():
    fig, ax = setup_fig(figsize=(9, 6))
    Y, X = np.mgrid[-2:2:15j, -2:2:15j]
    U = -Y; V = X  # rotation field
    ax.quiver(X, Y, U, V, color=C["blue"], alpha=0.85, scale=30, width=0.004)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    title_2d(ax, "Planar vector field $X = -y\\,\\partial_x + x\\,\\partial_y$")
    return save(fig, SLUGS[6], "dg_v2_07_1_vector_field_2d")


def fig_07_2():
    fig, ax = setup_fig(figsize=(9, 6))
    Y, X = np.mgrid[-2:2:80j, -2:2:80j]
    U = -Y - 0.3 * X
    V = X - 0.3 * Y
    ax.streamplot(X, Y, U, V, color=C["purple"], density=1.6, linewidth=1.2)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    title_2d(ax, "Integral curves of a vector field (streamlines)")
    return save(fig, SLUGS[6], "dg_v2_07_2_integral_curves")


def fig_07_3():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show flow phi_t at three times via streamplot + sample point trajectories
    Y, X = np.mgrid[-2:2:60j, -2:2:60j]
    U = -Y; V = X
    ax.streamplot(X, Y, U, V, color=C["light"], density=1.2, linewidth=0.8)
    # Trajectory of point starting at (1.5, 0)
    t = np.linspace(0, 2 * np.pi * 0.6, 80)
    x0 = 1.5
    xs = x0 * np.cos(t); ys = x0 * np.sin(t)
    ax.plot(xs, ys, color=C["blue"], lw=2.5)
    for ti, col, lab in [(0, C["red"], "$t=0$"), (40, C["amber"], "$t=t_1$"), (79, C["green"], "$t=t_2$")]:
        ax.plot(xs[ti], ys[ti], "o", color=col, markersize=10, zorder=5)
        ax.annotate(lab, (xs[ti], ys[ti]), xytext=(xs[ti] + 0.15, ys[ti] + 0.15),
                    fontsize=10, color=col, fontweight="bold")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    title_2d(ax, "Flow $\\varphi_t$: $\\dot p = X(p)$")
    return save(fig, SLUGS[6], "dg_v2_07_3_flow_phi_t")


def fig_07_4():
    fig, ax = setup_fig(figsize=(9, 6))
    # Lie bracket: noncommutativity of flows -> closing the rectangle gap
    O = np.array([0, 0])
    e1 = np.array([1.5, 0]); e2 = np.array([0.4, 1.4])
    # First X then Y
    p = O.copy()
    pts1 = [p.copy()]; p = p + e1; pts1.append(p.copy()); p = p + e2; pts1.append(p.copy())
    p = O.copy()
    pts2 = [p.copy()]; p = p + e2; pts2.append(p.copy()); p = p + e1 + np.array([0.2, 0.1]); pts2.append(p.copy())
    pts1 = np.array(pts1); pts2 = np.array(pts2)
    ax.plot(pts1[:, 0], pts1[:, 1], "-o", color=C["blue"], lw=2.5, label="$\\varphi_t^Y \\circ \\varphi_t^X$")
    ax.plot(pts2[:, 0], pts2[:, 1], "-o", color=C["red"], lw=2.5, label="$\\varphi_t^X \\circ \\varphi_t^Y$")
    ax.annotate("", xy=pts1[-1], xytext=pts2[-1],
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=2.5))
    ax.text(1.2, 1.7, "$[X,Y]$ — gap measures non-commutativity",
            fontsize=11, color=C["purple"])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=10)
    title_2d(ax, "Lie bracket geometric meaning")
    return save(fig, SLUGS[6], "dg_v2_07_4_lie_bracket")


def fig_07_5():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Lie derivative $\\mathcal{L}_X$", fontsize=14,
            color=C["dark"], ha="center", fontweight="bold")
    eq = "$\\mathcal{L}_X T = \\dfrac{d}{dt}|_{t=0} (\\varphi_t^*\\, T)$"
    card(ax, 1.0, 3.2, 8.0, 1.2, C["blue"], eq, fontsize=14)
    ax.text(5, 2.4, "Rate of change along the flow $\\varphi_t$ of $X$",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.7, "Functions: $\\mathcal{L}_X f = X(f)$", fontsize=11, color=C["green"], ha="center")
    ax.text(5, 1.1, "Vector fields: $\\mathcal{L}_X Y = [X, Y]$", fontsize=11, color=C["red"], ha="center")
    return save(fig, SLUGS[6], "dg_v2_07_5_lie_derivative")


def fig_07_6():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    Y, X = np.mgrid[-2:2:60j, -2:2:60j]
    cases = [
        ("Rotation $(-y, x)$", -Y, X, C["blue"]),
        ("Source $(x, y)$", X, Y, C["red"]),
        ("Sink $(-x, -y)$", -X, -Y, C["purple"]),
    ]
    for ax, (title, U, V, col) in zip(axs, cases):
        ax.set_facecolor(BG)
        ax.streamplot(X, Y, U, V, color=col, density=1.3, linewidth=1.0)
        ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_title(title, fontsize=11, color=C["dark"])
        ax.grid(True, alpha=0.25)
    fig.suptitle("Vector field examples", fontsize=13, color=C["dark"])
    fig.tight_layout()
    return save(fig, SLUGS[6], "dg_v2_07_6_field_examples")


def fig_07_7():
    fig, ax = setup_fig(figsize=(9, 6))
    # Phase portrait of nonlinear oscillator
    Y, X = np.mgrid[-3:3:60j, -3:3:60j]
    U = Y
    V = -np.sin(X) - 0.2 * Y  # damped pendulum
    ax.streamplot(X, Y, U, V, color=C["teal"], density=1.6, linewidth=1.0)
    # Equilibria
    ax.plot([0], [0], "o", color=C["green"], markersize=10, label="stable equilibrium")
    ax.plot([np.pi, -np.pi], [0, 0], "x", color=C["red"], markersize=12, mew=2, label="saddle")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=10)
    title_2d(ax, "Phase portrait: damped pendulum")
    return save(fig, SLUGS[6], "dg_v2_07_7_phase_portrait")


# ============================================================
# 08 - Differential Forms
# ============================================================
def fig_08_1():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show a 1-form as level surfaces (parallel lines) and a vector
    for c in np.linspace(-3, 3, 13):
        ax.plot([-3, 3], [c, c], color=C["blue"], lw=1, alpha=0.6)
    # vector
    ax.annotate("", xy=(1.5, 1.3), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=3))
    ax.text(1.6, 1.4, "$v$", fontsize=14, color=C["red"])
    # number of lines crossed
    ax.text(0.2, -2.5, "$\\omega(v)$ = number of level lines crossed",
            fontsize=11, color=C["dark"])
    ax.text(0.2, -2.9, "1-form $\\omega = dy$ here; $\\omega(v) = v_y$",
            fontsize=10, color=C["gray"])
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    title_2d(ax, "1-form as a linear functional on tangent vectors")
    return save(fig, SLUGS[7], "dg_v2_08_1_one_form")


def fig_08_2():
    fig, ax = setup_fig(figsize=(9, 6))
    # 2-form: oriented parallelogram
    O = np.array([0.5, 0.5])
    v = np.array([1.5, 0.3])
    w = np.array([0.4, 1.2])
    pts = np.array([O, O + v, O + v + w, O + w])
    ax.add_patch(Polygon(pts, closed=True, facecolor=C["amber"], alpha=0.5,
                         edgecolor=C["dark"], lw=1.5))
    ax.annotate("", xy=O + v, xytext=O,
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.5))
    ax.annotate("", xy=O + w, xytext=O,
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=2.5))
    ax.text(*(O + v + np.array([0.05, -0.15])), "$v$", fontsize=13, color=C["red"])
    ax.text(*(O + w + np.array([-0.15, 0.05])), "$w$", fontsize=13, color=C["green"])
    # Curved arrow showing orientation
    ax.annotate("", xy=O + 0.5 * v + np.array([-0.05, 0.4]),
                xytext=O + 0.4 * w + np.array([0.4, -0.05]),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=2,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(O[0] + 0.5, O[1] + 0.3, "$\\omega(v, w)$", fontsize=13, color=C["dark"], ha="center")
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    title_2d(ax, "2-form: signed area on $(v, w)$ pair")
    return save(fig, SLUGS[7], "dg_v2_08_2_two_form")


def fig_08_3():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Wedge product $\\wedge$ — antisymmetric",
            fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    formulas = [
        "$dx \\wedge dy = -\\,dy \\wedge dx$",
        "$dx \\wedge dx = 0$",
        "$(\\alpha \\wedge \\beta)(v, w) = \\alpha(v)\\beta(w) - \\alpha(w)\\beta(v)$",
    ]
    for i, f in enumerate(formulas):
        y = 3.8 - i * 0.8
        card(ax, 1.5, y - 0.25, 7, 0.55, [C["red"], C["amber"], C["purple"]][i], f, fontsize=12)
    ax.text(5, 0.8, "Builds higher-degree forms from 1-forms",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[7], "dg_v2_08_3_wedge")


def fig_08_4():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "Exterior derivative $d$", fontsize=14, color=C["dark"], ha="center",
            fontweight="bold")
    items = [
        ("$d(f\\,dx^I) = df \\wedge dx^I$", C["blue"]),
        ("$d^2 = 0$ (key identity)", C["red"]),
        ("$d(\\alpha \\wedge \\beta) = d\\alpha \\wedge \\beta + (-1)^p \\alpha \\wedge d\\beta$", C["purple"]),
    ]
    for i, (f, col) in enumerate(items):
        y = 4.1 - i * 1.1
        card(ax, 0.5, y - 0.3, 9, 0.7, col, f, fontsize=12)
    ax.text(5, 0.4, "Generalizes grad, curl, div uniformly",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[7], "dg_v2_08_4_exterior_d")


def fig_08_5():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    th = np.linspace(0, 2 * np.pi, 50)
    ax.plot(1.5 + 0.8 * np.cos(th), 2.5 + 0.6 * np.sin(th), color=C["dark"], lw=2)
    ax.fill(1.5 + 0.8 * np.cos(th), 2.5 + 0.6 * np.sin(th), color=C["red"], alpha=0.3)
    ax.text(1.5, 2.5, "$M$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.plot(7.5 + 1.0 * np.cos(th), 2.5 + 0.8 * np.sin(th), color=C["dark"], lw=2)
    ax.fill(7.5 + 1.0 * np.cos(th), 2.5 + 0.8 * np.sin(th), color=C["blue"], alpha=0.3)
    ax.text(7.5, 2.5, "$N$", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(6.4, 2.5), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle="->", color=C["dark"], lw=2.5))
    ax.text(4.5, 2.9, "$f$", fontsize=14, color=C["dark"], ha="center")
    ax.annotate("", xy=(2.5, 1.5), xytext=(6.4, 1.5),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=2.5))
    ax.text(4.5, 1.7, "$f^*$  (pullback of forms)", fontsize=12, color=C["purple"], ha="center",
            fontweight="bold")
    ax.text(5, 0.4, "$(f^*\\omega)(v_1, ..., v_p) = \\omega(df(v_1), ..., df(v_p))$",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 4.8, "Forms pull back, vectors push forward", fontsize=12, color=C["dark"],
            ha="center", fontweight="bold")
    return save(fig, SLUGS[7], "dg_v2_08_5_pullback")


def fig_08_6():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.0, "de Rham cohomology $H^k_{dR}(M) = \\ker d / \\operatorname{im} d$",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    # chain
    boxes = [("$\\Omega^0$", C["red"]), ("$\\Omega^1$", C["amber"]),
             ("$\\Omega^2$", C["green"]), ("$\\Omega^3$", C["blue"])]
    for i, (lbl, col) in enumerate(boxes):
        x = 0.7 + i * 2.5
        card(ax, x, 2.0, 1.5, 1.0, col, lbl, fontsize=14)
        if i < 3:
            ax.annotate("", xy=(x + 2.3, 2.5), xytext=(x + 1.6, 2.5),
                        arrowprops=dict(arrowstyle="->", color=C["dark"], lw=2))
            ax.text(x + 1.95, 2.85, "$d$", fontsize=11, color=C["dark"], ha="center")
    ax.text(5.5, 1.0, "closed: $d\\omega = 0$  /  exact: $\\omega = d\\eta$",
            fontsize=11, color=C["dark"], ha="center")
    return save(fig, SLUGS[7], "dg_v2_08_6_de_rham")


def fig_08_7():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.0, "Classical forms — vector calculus reborn",
            fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    rows = [
        ("0-form", "$f$", "scalar function", C["red"]),
        ("1-form", "$\\sum F_i\\,dx^i$", "covector / gradient", C["amber"]),
        ("2-form", "$F_{ij}\\,dx^i\\wedge dx^j$", "flux / curl", C["green"]),
        ("$n$-form", "$f\\,dx^1\\wedge...\\wedge dx^n$", "volume / density", C["purple"]),
    ]
    for i, (a, b, c, col) in enumerate(rows):
        y = 4.0 - i * 0.85
        card(ax, 0.3, y - 0.25, 1.5, 0.55, col, a, fontsize=10)
        ax.text(2.0, y, b, fontsize=12, color=C["dark"], va="center")
        ax.text(6.5, y, "→ " + c, fontsize=11, color=C["gray"], va="center")
    return save(fig, SLUGS[7], "dg_v2_08_7_examples")


# ============================================================
# 09 - Integration & Stokes
# ============================================================
def fig_09_1():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    # Two manifolds, one with consistent arrows, one not
    th = np.linspace(0, 2 * np.pi, 100)
    # left: orientable
    ax.plot(2.5 + 1.5 * np.cos(th), 3 + 1 * np.sin(th), color=C["dark"], lw=2)
    ax.fill(2.5 + 1.5 * np.cos(th), 3 + 1 * np.sin(th), color=C["green"], alpha=0.3)
    # add small CCW arrows
    for ti in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = 2.5 + 1.0 * np.cos(ti); y = 3 + 0.7 * np.sin(ti)
        dx = -np.sin(ti) * 0.2; dy = np.cos(ti) * 0.2
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x - dx, y - dy),
                    arrowprops=dict(arrowstyle="->", color=C["dark"], lw=1.2))
    ax.text(2.5, 1.5, "orientable: consistent choice", fontsize=11, color=C["dark"], ha="center")
    # right: Möbius hint - cross arrow
    ax.plot(7.5 + 1.5 * np.cos(th), 3 + 1 * np.sin(th), color=C["dark"], lw=2)
    ax.fill(7.5 + 1.5 * np.cos(th), 3 + 1 * np.sin(th), color=C["red"], alpha=0.3)
    ax.text(7.5, 3, "?", fontsize=24, ha="center", va="center", color=C["red"], fontweight="bold")
    ax.text(7.5, 1.5, "non-orientable: no global choice", fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 5.0, "Orientation of a manifold", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    return save(fig, SLUGS[8], "dg_v2_09_1_orientation")


def fig_09_2():
    fig, ax = setup_fig(figsize=(9, 6))
    # Disk with boundary, inward/outward arrows
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(th), np.sin(th), color=C["amber"], alpha=0.3)
    ax.plot(np.cos(th), np.sin(th), color=C["red"], lw=2.5)
    ax.text(0, 0, "$M$", fontsize=18, ha="center", va="center", fontweight="bold")
    ax.text(1.4, 0.1, "$\\partial M$", fontsize=14, color=C["red"], fontweight="bold")
    # arrow direction
    for ti in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x, y = np.cos(ti), np.sin(ti)
        tx, ty = -np.sin(ti) * 0.15, np.cos(ti) * 0.15
        ax.annotate("", xy=(x + tx, y + ty), xytext=(x - tx, y - ty),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5))
    ax.set_xlim(-1.7, 1.9); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    title_2d(ax, "Manifold with boundary $\\partial M$")
    return save(fig, SLUGS[8], "dg_v2_09_2_boundary")


def fig_09_3():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    # Hemisphere
    u = np.linspace(0, 2 * np.pi, 50); v = np.linspace(0, np.pi / 2, 25)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V); Y = np.sin(U) * np.sin(V); Z = np.cos(V)
    ax.plot_surface(X, Y, Z, color=C["blue"], alpha=0.4, edgecolor="none")
    # Boundary (equator) with orientation arrows
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color=C["red"], lw=3)
    for ti in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.quiver(np.cos(ti), np.sin(ti), 0,
                  -np.sin(ti) * 0.3, np.cos(ti) * 0.3, 0,
                  color=C["red"], lw=1.5)
    ax.text2D(0.05, 0.93,
              "$\\int_M d\\omega = \\int_{\\partial M} \\omega$",
              transform=ax.transAxes, fontsize=14, color=C["dark"], fontweight="bold")
    setup_3d(ax, elev=25, azim=-50, title="Stokes' theorem")
    return save(fig, SLUGS[8], "dg_v2_09_3_stokes")


def fig_09_4():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.1, "Classical theorems unified by Stokes",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    rows = [
        ("FTC ($n=1$)", "$\\int_a^b f'\\,dx = f(b) - f(a)$", C["red"]),
        ("Green ($n=2$)", "$\\oint P\\,dx + Q\\,dy = \\iint \\!(Q_x - P_y)\\,dA$", C["amber"]),
        ("Stokes ($n=2,3$)", "$\\oint \\mathbf{F}\\cdot d\\mathbf{r} = \\iint \\!(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{A}$", C["green"]),
        ("Divergence ($n=3$)", "$\\iiint \\nabla\\cdot\\mathbf{F}\\,dV = \\oiint \\mathbf{F}\\cdot d\\mathbf{A}$", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(rows):
        y = 4.1 - i * 0.85
        card(ax, 0.3, y - 0.25, 2.5, 0.55, col, a, fontsize=10)
        ax.text(3.0, y, b, fontsize=11, color=C["dark"], va="center")
    ax.text(5.5, 0.3, "All special cases of $\\int_M d\\omega = \\int_{\\partial M} \\omega$",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[8], "dg_v2_09_4_classical_unify")


def fig_09_5():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "de Rham cohomology measures topology",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    items = [
        ("$H^0(M)$", "= number of connected components", C["red"]),
        ("$H^1(M)$", "= 1-dim 'holes' (loops not bounding)", C["amber"]),
        ("$H^k(M)$", "= $k$-dim holes", C["green"]),
        ("dimensions = Betti numbers $b_k$", "$\\chi = \\sum (-1)^k b_k$", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.0 - i * 0.85
        card(ax, 0.5, y - 0.25, 3.5, 0.55, col, a, fontsize=10)
        ax.text(4.5, y, b, fontsize=11, color=C["dark"], va="center")
    return save(fig, SLUGS[8], "dg_v2_09_5_de_rham_coh")


def fig_09_6():
    fig, ax = setup_fig(figsize=(9, 6))
    # Show a 1-chain (curve) and a 2-chain (filled region)
    th = np.linspace(0.2, np.pi - 0.2, 100)
    x = th; y = np.sin(th) * 1.2
    ax.plot(x, y, color=C["red"], lw=3, label="1-chain $\\gamma$ (curve)")
    # 2-chain
    th2 = np.linspace(0, 2 * np.pi, 100)
    cx = 1.8 + 0.6 * np.cos(th2); cy = -0.8 + 0.4 * np.sin(th2)
    ax.fill(cx, cy, color=C["blue"], alpha=0.4, label="2-chain $\\Sigma$ (region)")
    ax.set_xlim(-0.3, 3.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=11)
    ax.text(0.0, -1.3, "Integrate $k$-forms over $k$-chains",
            fontsize=11, color=C["dark"])
    title_2d(ax, "Integration over chains")
    return save(fig, SLUGS[8], "dg_v2_09_6_chain_integration")


def fig_09_7():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Stokes — applications", fontsize=14, color=C["dark"], ha="center",
            fontweight="bold")
    items = [
        ("Maxwell's equations", "Faraday + Ampère via $d, \\wedge$", C["red"]),
        ("Hodge theory", "harmonic representatives in cohomology", C["amber"]),
        ("Index theorems", "Atiyah-Singer generalizes GB", C["green"]),
        ("Conservation laws", "Noether: symmetries → conserved currents", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.1 - i * 0.85
        card(ax, 0.3, y - 0.25, 3.5, 0.55, col, a, fontsize=10)
        ax.text(4.2, y, b, fontsize=11, color=C["dark"], va="center")
    return save(fig, SLUGS[8], "dg_v2_09_7_examples")


# ============================================================
# 10 - Riemannian Geometry
# ============================================================
def fig_10_1():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Riemannian metric $g$", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$g_p: T_p M \\times T_p M \\to \\mathbb{R}$ — symmetric, positive definite"
    card(ax, 0.5, 3.4, 9, 1.0, C["blue"], eq, fontsize=12)
    ax.text(5, 2.6, "$\\langle v, w\\rangle_p = g_{ij}(p)\\, v^i w^j$",
            fontsize=13, color=C["dark"], ha="center")
    ax.text(5, 1.8, "Gives length, angle, volume on each tangent space",
            fontsize=11, color=C["green"], ha="center")
    ax.text(5, 1.1, "Smoothly varying with $p$", fontsize=10, color=C["gray"], ha="center")
    return save(fig, SLUGS[9], "dg_v2_10_1_riem_metric")


def fig_10_2():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Levi-Civita connection $\\nabla$", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    items = [
        ("Torsion-free: $\\nabla_X Y - \\nabla_Y X = [X, Y]$", C["red"]),
        ("Compatible with metric: $X g(Y, Z) = g(\\nabla_X Y, Z) + g(Y, \\nabla_X Z)$", C["green"]),
        ("Unique such connection", C["purple"]),
    ]
    for i, (txt, col) in enumerate(items):
        y = 3.8 - i * 0.85
        card(ax, 0.3, y - 0.25, 9.4, 0.55, col, txt, fontsize=11)
    ax.text(5, 0.8, "Fundamental theorem of Riemannian geometry",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[9], "dg_v2_10_2_levi_civita")


def fig_10_3():
    fig, ax = setup_fig(figsize=(9, 7), n3d=True)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 25)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["blue"], alpha=0.35, edgecolor="none")
    # A geodesic loop on sphere (great circle triangle path)
    A = np.array([0, 0, 1])
    B = np.array([1, 0, 0])
    Cp = np.array([0, 1, 0])
    def arc(p, q, n=30):
        ts = np.linspace(0, 1, n); pts = []
        for t in ts:
            v = (1 - t) * p + t * q; pts.append(v / np.linalg.norm(v))
        return np.array(pts)
    full = []
    for p, q in [(A, B), (B, Cp), (Cp, A)]:
        a = arc(p, q); ax.plot(a[:, 0], a[:, 1], a[:, 2], color=C["red"], lw=2.5); full.append(a)
    # Parallel-transport a tangent vector along path - it rotates
    angles = [0, np.pi / 2, np.pi]
    for arc_data, ang in zip(full, angles):
        p = arc_data[len(arc_data) // 2]
        n = p
        # tangent perpendicular to n, rotated
        e1 = np.array([0, 0, 1]) - np.dot([0, 0, 1], n) * n
        if np.linalg.norm(e1) < 0.1:
            e1 = np.array([1, 0, 0]) - n[0] * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        v = np.cos(ang) * e1 + np.sin(ang) * e2
        ax.quiver(*p, *(v * 0.3), color=C["green"], lw=2.5)
    setup_3d(ax, elev=25, azim=-50, title="Parallel transport on $S^2$ — vector rotates by holonomy")
    return save(fig, SLUGS[9], "dg_v2_10_3_parallel_transport")


def fig_10_4():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Holonomy: rotation after parallel transport around a loop",
            fontsize=13, color=C["dark"], ha="center", fontweight="bold")
    eq = "$\\theta_{\\text{holonomy}} = \\int_\\Sigma K\\,dA$  (Gauss-Bonnet form)"
    card(ax, 1.0, 3.0, 8, 1.2, C["purple"], eq, fontsize=14)
    ax.text(5, 2.0, "On $S^2$: octant loop → 90° rotation",
            fontsize=11, color=C["red"], ha="center")
    ax.text(5, 1.4, "Holonomy group $\\subset$ O($n$) measures curvature globally",
            fontsize=11, color=C["dark"], ha="center")
    return save(fig, SLUGS[9], "dg_v2_10_4_holonomy")


def fig_10_5():
    fig, ax = setup_fig(figsize=(9, 6))
    # Plane vs Punctured plane: extends vs. doesn't
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.axis("off")
    # Geodesic in plane
    ax.plot([-2.5, 2.5], [1, 1], color=C["green"], lw=2.5)
    ax.text(0, 1.3, "Geodesic extends ∞ — complete", fontsize=11,
            color=C["green"], ha="center", fontweight="bold")
    # Punctured: hits puncture
    ax.plot([-2.5, 0], [-1, -1], color=C["red"], lw=2.5)
    ax.plot(0, -1, "o", color=C["dark"], markersize=12, mfc="white", mec=C["dark"], mew=2)
    ax.plot([0.1, 2.5], [-1, -1], "--", color=C["red"], lw=1, alpha=0.4)
    ax.text(0, -1.4, "incomplete — hits puncture", fontsize=11,
            color=C["red"], ha="center", fontweight="bold")
    title_2d(ax, "Geodesic completeness")
    return save(fig, SLUGS[9], "dg_v2_10_5_geodesic_complete")


def fig_10_6():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "Hopf-Rinow theorem", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    rows = [
        ("Geodesic completeness", C["red"]),
        ("Metric completeness", C["amber"]),
        ("Heine-Borel: closed bounded ⇒ compact", C["green"]),
        ("Any two points joined by minimal geodesic", C["purple"]),
    ]
    for i, (txt, col) in enumerate(rows):
        y = 4.3 - i * 0.85
        card(ax, 0.5, y - 0.25, 9, 0.55, col, txt, fontsize=11)
    ax.text(5, 0.4, "All four equivalent on connected Riemannian manifold",
            fontsize=11, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[9], "dg_v2_10_6_hopf_rinow")


def fig_10_7():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.0, "Classical Riemannian metrics", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    rows = [
        ("Euclidean $\\mathbb{R}^n$", "$g = \\sum dx_i^2$", C["red"]),
        ("Sphere $S^n$", "round metric, $K = 1$", C["amber"]),
        ("Hyperbolic $\\mathbb{H}^n$", "$ds^2 = (dx^2 + dy^2)/y^2$, $K = -1$", C["green"]),
        ("FLRW (cosmology)", "$-dt^2 + a(t)^2 g_{\\Sigma}$", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(rows):
        y = 4.0 - i * 0.85
        card(ax, 0.3, y - 0.25, 3.5, 0.55, col, a, fontsize=10)
        ax.text(4.2, y, b, fontsize=11, color=C["dark"], va="center")
    return save(fig, SLUGS[9], "dg_v2_10_7_examples")


# ============================================================
# 11 - Curvature on Manifolds
# ============================================================
def fig_11_1():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Riemann curvature tensor", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$R(X, Y)Z = \\nabla_X \\nabla_Y Z - \\nabla_Y \\nabla_X Z - \\nabla_{[X,Y]} Z$"
    card(ax, 0.3, 3.0, 9.4, 1.2, C["blue"], eq, fontsize=13)
    ax.text(5, 2.0, "Measures non-commutativity of covariant derivatives",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.3, "Components $R^i_{jkl}$ — full local curvature info",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[10], "dg_v2_11_1_riemann_tensor")


def fig_11_2():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Sectional curvature $K(\\sigma)$", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$K(X, Y) = \\dfrac{\\langle R(X, Y) Y, X\\rangle}{\\|X\\|^2 \\|Y\\|^2 - \\langle X, Y\\rangle^2}$"
    card(ax, 0.5, 3.0, 9, 1.2, C["red"], eq, fontsize=14)
    ax.text(5, 2.1, "Generalizes Gauss curvature: pick a 2-plane $\\sigma \\subset T_p M$",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.4, "Determines $R$ entirely on Riemannian manifold",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[10], "dg_v2_11_2_sectional")

def fig_11_3():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Ricci curvature $\\text{Ric}(X, Y)$", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$\\text{Ric}(X, Y) = \\text{tr}(Z \\mapsto R(Z, X) Y)$"
    card(ax, 1.0, 3.2, 8.0, 1.2, C["amber"], eq, fontsize=14)
    ax.text(5, 2.3, "Trace of Riemann tensor — averages sectional curvatures",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.6, "$\\text{Ric}_{ij} = R^k_{ikj}$",
            fontsize=12, color=C["green"], ha="center")
    ax.text(5, 1.0, "Drives Einstein equations and Ricci flow",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[10], "dg_v2_11_3_ricci")


def fig_11_4():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Scalar curvature $S$", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$S = \\text{tr}_g(\\text{Ric}) = g^{ij} R_{ij}$"
    card(ax, 1.5, 3.2, 7.0, 1.2, C["green"], eq, fontsize=15)
    ax.text(5, 2.3, "Trace of Ricci — single number per point",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.5, "On surface ($n=2$): $S = 2K$",
            fontsize=11, color=C["red"], ha="center", fontweight="bold")
    return save(fig, SLUGS[10], "dg_v2_11_4_scalar")


def fig_11_5():
    fig, ax = setup_fig(figsize=(11, 5.5))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.text(5.5, 5.0, "Curvature decomposition", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$R = W \\,\\oplus\\, \\widehat{\\text{Ric}}_0\\, \\oplus\\, S$"
    card(ax, 2.0, 3.4, 7.0, 1.0, C["purple"], eq, fontsize=15)
    items = [
        ("$W$ — Weyl tensor", "conformally invariant, traceless", C["red"]),
        ("$\\text{Ric}_0$ — traceless Ricci", "shape distortion", C["amber"]),
        ("$S$ — scalar", "volume distortion", C["green"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 2.6 - i * 0.7
        card(ax, 0.3, y - 0.2, 3.5, 0.45, col, a, fontsize=10)
        ax.text(4.2, y, b, fontsize=10, color=C["dark"], va="center")
    return save(fig, SLUGS[10], "dg_v2_11_5_decomp")


def fig_11_6():
    fig = plt.figure(figsize=(13, 4.8), facecolor=BG, dpi=DPI)
    # Sphere K>0
    ax = fig.add_subplot(131, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V),
                    color=C["red"], alpha=0.7, edgecolor="none")
    ax.set_title("Sphere $K = +1$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=20, azim=-50)
    # Plane K=0
    ax = fig.add_subplot(132, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(-1, 1, 20); v = np.linspace(-1, 1, 20)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(U, V, np.zeros_like(U), color=C["amber"], alpha=0.7, edgecolor="none")
    ax.set_title("Plane $K = 0$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=25, azim=-50)
    # Hyperbolic K<0 (pseudosphere)
    ax = fig.add_subplot(133, projection='3d'); ax.set_facecolor(BG)
    u = np.linspace(-1.2, 1.2, 30); v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    # Pseudosphere: x = sech(u) cos v, y = sech(u) sin v, z = u - tanh(u)
    sech = 1 / np.cosh(U)
    X = sech * np.cos(V); Y = sech * np.sin(V); Z = U - np.tanh(U)
    ax.plot_surface(X, Y, Z, color=C["purple"], alpha=0.7, edgecolor="none")
    ax.set_title("Pseudosphere $K = -1$", fontsize=11, color=C["dark"])
    setup_3d(ax, elev=10, azim=-60)
    fig.suptitle("Three constant-curvature model spaces", fontsize=13, color=C["dark"])
    return save(fig, SLUGS[10], "dg_v2_11_6_constant_K")


def fig_11_7():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Einstein manifolds: $\\text{Ric} = \\lambda g$",
            fontsize=14, color=C["dark"], ha="center", fontweight="bold")
    items = [
        ("Examples", "spheres, hyperbolic, complex projective", C["red"]),
        ("Vacuum GR", "$\\text{Ric} = 0$ → Einstein", C["amber"]),
        ("Cosmological constant", "$\\text{Ric} = \\Lambda g$", C["green"]),
        ("Ricci flow fixed points", "→ Einstein metrics", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.1 - i * 0.85
        card(ax, 0.3, y - 0.25, 3.0, 0.55, col, a, fontsize=10)
        ax.text(3.7, y, b, fontsize=11, color=C["dark"], va="center")
    return save(fig, SLUGS[10], "dg_v2_11_7_einstein")


# ============================================================
# 12 - Bundles & Physics
# ============================================================
def fig_12_1():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # Base manifold (line)
    ax.plot([1, 9], [1, 1], color=C["dark"], lw=2.5)
    ax.text(5, 0.5, "$M$ (base)", fontsize=12, color=C["dark"], ha="center")
    # Fibers
    for x in np.linspace(1.5, 8.5, 6):
        ax.plot([x, x], [1, 4.5], color=C["blue"], lw=1.5, alpha=0.7)
        ax.plot(x, 1, "o", color=C["red"], markersize=8)
    # Total space rectangle
    ax.add_patch(Rectangle((1, 1), 8, 3.5, facecolor=C["amber"], alpha=0.15,
                            edgecolor=C["dark"], lw=1.5, ls="--"))
    ax.text(5, 5.0, "$E$ (total space)", fontsize=12, color=C["dark"], ha="center", fontweight="bold")
    # arrows for projection
    ax.annotate("", xy=(2.5, 1.1), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=1.5))
    ax.text(2.9, 2.3, "$\\pi$", fontsize=14, color=C["purple"], fontweight="bold")
    ax.text(0.7, 3, "fiber\n$F = \\pi^{-1}(p)$", fontsize=11, color=C["blue"], ha="center")
    title_2d(ax, "Fiber bundle: $\\pi: E \\to M$, fibers $F$")
    return save(fig, SLUGS[11], "dg_v2_12_1_fiber_bundle")


def fig_12_2():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Principal $G$-bundle", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    items = [
        ("Fiber = Lie group $G$", C["red"]),
        ("Right action of $G$ free + transitive on fibers", C["amber"]),
        ("Examples: Frame bundle ($GL_n$), spin ($Spin(n)$), gauge ($U(1)$)", C["green"]),
        ("All fiber bundles arise via associated bundles", C["purple"]),
    ]
    for i, (txt, col) in enumerate(items):
        y = 4.1 - i * 0.85
        card(ax, 0.5, y - 0.25, 9, 0.55, col, txt, fontsize=11)
    return save(fig, SLUGS[11], "dg_v2_12_2_principal")


def fig_12_3():
    fig, ax = setup_fig(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # Base curve and horizontal lift
    t = np.linspace(0, 8, 100)
    base_y = 1 + 0 * t
    ax.plot(t + 1, base_y, color=C["dark"], lw=2.5)
    ax.text(5, 0.5, "path in base $\\gamma(t)$", fontsize=11, color=C["dark"], ha="center")
    # Fibers
    for x in np.linspace(1.5, 8.5, 8):
        ax.plot([x, x], [1, 4.5], color=C["blue"], lw=1, alpha=0.4)
    # Horizontal lift - rising path
    lift_y = 1.5 + 0.3 * t
    ax.plot(t + 1, lift_y, color=C["red"], lw=3, label="horizontal lift")
    for x, y in zip(np.linspace(1.5, 8.5, 5), 1.5 + 0.3 * np.linspace(0.5, 7.5, 5)):
        ax.plot(x, y, "o", color=C["red"], markersize=6)
    ax.legend(fontsize=11, loc="upper right")
    title_2d(ax, "Connection: horizontal lift of paths")
    return save(fig, SLUGS[11], "dg_v2_12_3_connection")


def fig_12_4():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Curvature 2-form", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$F = dA + A \\wedge A$"
    card(ax, 2.5, 3.3, 5.0, 1.2, C["red"], eq, fontsize=18)
    ax.text(5, 2.4, "$A$ — connection 1-form (gauge potential)",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.7, "$F$ — curvature 2-form (field strength)",
            fontsize=11, color=C["green"], ha="center")
    ax.text(5, 1.0, "Bianchi identity: $dF + A \\wedge F - F \\wedge A = 0$",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[11], "dg_v2_12_4_curvature_form")


def fig_12_5():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Chern classes", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    items = [
        ("$c_1 = \\dfrac{i}{2\\pi} \\text{tr}\\,F$", "magnetic monopole / first Chern", C["red"]),
        ("$c_2 = \\dfrac{1}{8\\pi^2} \\text{tr}(F \\wedge F)$", "instanton number", C["amber"]),
        ("Topological invariants of bundles", "integer-valued cohomology classes", C["green"]),
        ("Chern-Weil theory", "char classes from curvature", C["purple"]),
    ]
    for i, (a, b, col) in enumerate(items):
        y = 4.1 - i * 0.85
        card(ax, 0.3, y - 0.25, 4.0, 0.55, col, a, fontsize=10)
        ax.text(4.7, y, b, fontsize=10, color=C["dark"], va="center")
    return save(fig, SLUGS[11], "dg_v2_12_5_chern")


def fig_12_6():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Yang-Mills theory", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    eq = "$\\mathcal{S}_{YM} = \\dfrac{1}{2} \\int_M \\text{tr}(F \\wedge {*}F)$"
    card(ax, 2.0, 3.2, 6.0, 1.2, C["blue"], eq, fontsize=15)
    ax.text(5, 2.3, "Generalizes Maxwell ($U(1)$) to non-abelian groups",
            fontsize=11, color=C["dark"], ha="center")
    ax.text(5, 1.6, "$SU(3)$ — strong force; $SU(2)\\times U(1)$ — electroweak",
            fontsize=11, color=C["red"], ha="center")
    ax.text(5, 0.9, "Equations: $d_A {*} F = 0$",
            fontsize=10, color=C["gray"], ha="center", style="italic")
    return save(fig, SLUGS[11], "dg_v2_12_6_yang_mills")


def fig_12_7():
    fig, ax = setup_fig(figsize=(10, 5.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.text(5, 5.0, "Gauge theory dictionary", fontsize=14, color=C["dark"],
            ha="center", fontweight="bold")
    rows = [
        ("Geometry", "Physics", "—", C["dark"]),
        ("principal bundle", "gauge theory", C["red"], None),
        ("connection $A$", "gauge potential / vector boson", C["amber"], None),
        ("curvature $F$", "field strength", C["green"], None),
        ("gauge transformation", "change of frame", C["purple"], None),
        ("section of associated bundle", "matter field (e.g. Higgs)", C["blue"], None),
    ]
    for i, row in enumerate(rows):
        y = 4.3 - i * 0.7
        if i == 0:
            ax.text(2.5, y, row[0], fontsize=12, color=C["dark"], ha="center", fontweight="bold")
            ax.text(7.5, y, row[1], fontsize=12, color=C["dark"], ha="center", fontweight="bold")
            ax.plot([0.5, 9.5], [y - 0.2, y - 0.2], color=C["dark"], lw=0.8)
        else:
            a, b, col, _ = row
            card(ax, 0.5, y - 0.22, 4.2, 0.45, col, a, fontsize=10)
            ax.text(7.0, y, b, fontsize=11, color=C["dark"], va="center", ha="center")
    return save(fig, SLUGS[11], "dg_v2_12_7_gauge")


# ============================================================
# Main
# ============================================================
ALL_FIGS = [
    fig_01_1, fig_01_2, fig_01_3, fig_01_4, fig_01_5, fig_01_6, fig_01_7,
    fig_02_1, fig_02_2, fig_02_3, fig_02_4, fig_02_5, fig_02_6, fig_02_7,
    fig_03_1, fig_03_2, fig_03_3, fig_03_4, fig_03_5, fig_03_6, fig_03_7,
    fig_04_1, fig_04_2, fig_04_3, fig_04_4, fig_04_5, fig_04_6, fig_04_7,
    fig_05_1, fig_05_2, fig_05_3, fig_05_4, fig_05_5, fig_05_6, fig_05_7,
    fig_06_1, fig_06_2, fig_06_3, fig_06_4, fig_06_5, fig_06_6, fig_06_7,
    fig_07_1, fig_07_2, fig_07_3, fig_07_4, fig_07_5, fig_07_6, fig_07_7,
    fig_08_1, fig_08_2, fig_08_3, fig_08_4, fig_08_5, fig_08_6, fig_08_7,
    fig_09_1, fig_09_2, fig_09_3, fig_09_4, fig_09_5, fig_09_6, fig_09_7,
    fig_10_1, fig_10_2, fig_10_3, fig_10_4, fig_10_5, fig_10_6, fig_10_7,
    fig_11_1, fig_11_2, fig_11_3, fig_11_4, fig_11_5, fig_11_6, fig_11_7,
    fig_12_1, fig_12_2, fig_12_3, fig_12_4, fig_12_5, fig_12_6, fig_12_7,
]


if __name__ == "__main__":
    import sys
    print(f"Generating {len(ALL_FIGS)} figures...")
    ok = 0; fail = 0
    for i, f in enumerate(ALL_FIGS):
        try:
            p = f()
            ok += 1
            print(f"[{i+1:2d}/{len(ALL_FIGS)}] OK  {p}")
        except Exception as e:
            fail += 1
            print(f"[{i+1:2d}/{len(ALL_FIGS)}] FAIL {f.__name__}: {e}")
            import traceback; traceback.print_exc()
        sys.stdout.flush()
    print(f"\nDONE: {ok} ok, {fail} fail")
