#!/usr/bin/env python3
"""Generate all 6 figures for the Differential Geometry series on chenk.top."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import subprocess
import os

# Design tokens
BG_COLOR = "#fdfcf9"
COLORS = {
    "red": "#e85d4a",
    "amber": "#f5a834",
    "purple": "#8b5cf6",
    "blue": "#3b82f6",
    "green": "#10b981",
}
DPI = 160
SAVE_KWARGS = {"dpi": DPI, "bbox_inches": "tight", "facecolor": BG_COLOR, "pad_inches": 0.1}

OUT_DIR = "/tmp/dg_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def fig01_frenet_frame():
    """Frenet frame (T, N, B vectors) at a point on a 3D helix."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)

    # Helix parameters
    a, b = 1.0, 0.3
    t = np.linspace(0, 4 * np.pi, 300)
    x = a * np.cos(t)
    y = a * np.sin(t)
    z = b * t

    ax.plot(x, y, z, color=COLORS["blue"], linewidth=2, alpha=0.8)

    # Pick a point
    t0 = 2.5 * np.pi
    p = np.array([a * np.cos(t0), a * np.sin(t0), b * t0])

    # Frenet frame at t0
    c = np.sqrt(a**2 + b**2)
    T = np.array([-a * np.sin(t0), a * np.cos(t0), b]) / c
    N = np.array([-np.cos(t0), -np.sin(t0), 0])
    B = np.cross(T, N)

    scale = 0.6
    ax.quiver(*p, *T*scale, color=COLORS["red"], arrow_length_ratio=0.15, linewidth=2.5)
    ax.quiver(*p, *N*scale, color=COLORS["green"], arrow_length_ratio=0.15, linewidth=2.5)
    ax.quiver(*p, *B*scale, color=COLORS["purple"], arrow_length_ratio=0.15, linewidth=2.5)

    # Labels
    ax.text(*(p + T*scale*1.15), "T", color=COLORS["red"], fontsize=14, fontweight='bold')
    ax.text(*(p + N*scale*1.15), "N", color=COLORS["green"], fontsize=14, fontweight='bold')
    ax.text(*(p + B*scale*1.15), "B", color=COLORS["purple"], fontsize=14, fontweight='bold')

    ax.scatter(*p, color=COLORS["amber"], s=60, zorder=5)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title("Frenet Frame on a Helix", fontsize=13, pad=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    path = os.path.join(OUT_DIR, "fig01_frenet_frame.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig02_first_fundamental_form():
    """First fundamental form: surface patch with tangent vectors and grid."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)

    # Surface patch (paraboloid-like)
    u = np.linspace(-1.5, 1.5, 30)
    v = np.linspace(-1.5, 1.5, 30)
    U, V = np.meshgrid(u, v)
    X = U
    Y = V
    Z = 0.3 * (U**2 + V**2)

    ax.plot_surface(X, Y, Z, alpha=0.3, color=COLORS["blue"], edgecolor='none')

    # Grid lines on surface
    for ui in np.linspace(-1.5, 1.5, 7):
        vi = np.linspace(-1.5, 1.5, 50)
        ax.plot(np.full_like(vi, ui), vi, 0.3*(ui**2 + vi**2),
                color=COLORS["blue"], alpha=0.4, linewidth=0.8)
    for vi_val in np.linspace(-1.5, 1.5, 7):
        ui = np.linspace(-1.5, 1.5, 50)
        ax.plot(ui, np.full_like(ui, vi_val), 0.3*(ui**2 + vi_val**2),
                color=COLORS["blue"], alpha=0.4, linewidth=0.8)

    # Tangent vectors at a point
    u0, v0 = 0.5, 0.5
    p = np.array([u0, v0, 0.3*(u0**2 + v0**2)])
    xu = np.array([1, 0, 0.6*u0])
    xv = np.array([0, 1, 0.6*v0])
    xu = xu / np.linalg.norm(xu) * 0.8
    xv = xv / np.linalg.norm(xv) * 0.8

    ax.quiver(*p, *xu, color=COLORS["red"], arrow_length_ratio=0.12, linewidth=2.5)
    ax.quiver(*p, *xv, color=COLORS["green"], arrow_length_ratio=0.12, linewidth=2.5)
    ax.scatter(*p, color=COLORS["amber"], s=60, zorder=5)

    ax.text(*(p + xu*1.1), r"$\mathbf{x}_u$", color=COLORS["red"], fontsize=13)
    ax.text(*(p + xv*1.1), r"$\mathbf{x}_v$", color=COLORS["green"], fontsize=13)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title("First Fundamental Form: Tangent Vectors on a Surface", fontsize=13, pad=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    path = os.path.join(OUT_DIR, "fig02_first_form.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig03_principal_curvatures():
    """Principal curvatures on a saddle surface (max/min curvature directions)."""
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)

    # Saddle surface z = x^2 - y^2
    u = np.linspace(-1.5, 1.5, 40)
    v = np.linspace(-1.5, 1.5, 40)
    U, V = np.meshgrid(u, v)
    Z = U**2 - V**2

    ax.plot_surface(U, V, Z, alpha=0.35, cmap='coolwarm', edgecolor='none')

    # At origin: principal directions along x and y axes
    p = np.array([0, 0, 0])
    ax.scatter(*p, color=COLORS["amber"], s=80, zorder=5)

    # Direction of max curvature (kappa1 = 2, along x)
    t_line = np.linspace(-1.2, 1.2, 50)
    ax.plot(t_line, np.zeros_like(t_line), t_line**2,
            color=COLORS["red"], linewidth=2.5, label=r"$\kappa_1 = +2$ (max)")

    # Direction of min curvature (kappa2 = -2, along y)
    ax.plot(np.zeros_like(t_line), t_line, -t_line**2,
            color=COLORS["blue"], linewidth=2.5, label=r"$\kappa_2 = -2$ (min)")

    # Arrows for principal directions
    scale = 0.7
    ax.quiver(*p, scale, 0, 0, color=COLORS["red"], arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(*p, 0, scale, 0, color=COLORS["blue"], arrow_length_ratio=0.15, linewidth=2)

    ax.text(scale*1.1, 0, 0.1, r"$\mathbf{e}_1$", color=COLORS["red"], fontsize=13)
    ax.text(0, scale*1.1, -0.1, r"$\mathbf{e}_2$", color=COLORS["blue"], fontsize=13)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title("Principal Curvatures on a Saddle Surface", fontsize=13, pad=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    path = os.path.join(OUT_DIR, "fig03_principal_curvatures.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig04_geodesics():
    """Geodesic paths on a sphere (great circles) vs non-geodesic (latitude line)."""
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)

    # --- Sphere subplot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor(BG_COLOR)

    # Draw sphere wireframe
    phi = np.linspace(0, 2*np.pi, 40)
    theta = np.linspace(0, np.pi, 20)
    PHI, THETA = np.meshgrid(phi, theta)
    X = np.sin(THETA)*np.cos(PHI)
    Y = np.sin(THETA)*np.sin(PHI)
    Z = np.cos(THETA)
    ax1.plot_surface(X, Y, Z, alpha=0.1, color=COLORS["blue"], edgecolor='gray', linewidth=0.2)

    # Great circle (geodesic) - equator
    t = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(t), np.sin(t), np.zeros_like(t),
             color=COLORS["red"], linewidth=2.5, label="Geodesic (great circle)")

    # Great circle - meridian
    ax1.plot(np.zeros_like(t), np.sin(t), np.cos(t),
             color=COLORS["green"], linewidth=2.5)

    # Non-geodesic latitude
    lat = np.pi/4
    ax1.plot(np.sin(lat)*np.cos(t), np.sin(lat)*np.sin(t),
             np.full_like(t, np.cos(lat)),
             color=COLORS["amber"], linewidth=2, linestyle='--', label="Non-geodesic (latitude)")

    ax1.set_title("Geodesics on a Sphere", fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # --- Torus subplot ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor(BG_COLOR)

    R, r = 2.0, 0.7
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    XT = (R + r*np.cos(U))*np.cos(V)
    YT = (R + r*np.cos(U))*np.sin(V)
    ZT = r*np.sin(U)
    ax2.plot_surface(XT, YT, ZT, alpha=0.15, color=COLORS["purple"], edgecolor='gray', linewidth=0.2)

    # Geodesic on torus: outer equator (u=0)
    v_line = np.linspace(0, 2*np.pi, 100)
    ax2.plot((R+r)*np.cos(v_line), (R+r)*np.sin(v_line), np.zeros_like(v_line),
             color=COLORS["red"], linewidth=2.5, label="Outer equator (geodesic)")

    # Meridian geodesic (v=0)
    u_line = np.linspace(0, 2*np.pi, 100)
    ax2.plot((R+r*np.cos(u_line)), np.zeros_like(u_line), r*np.sin(u_line),
             color=COLORS["green"], linewidth=2.5, label="Meridian (geodesic)")

    ax2.set_title("Geodesics on a Torus", fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig04_geodesics.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig05_tangent_bundle():
    """Tangent spaces at multiple points on a curve (2D manifold illustration)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw a curve (the "manifold")
    t = np.linspace(0, 2*np.pi, 200)
    x = t
    y = 0.8*np.sin(t) + 0.3*np.sin(2*t)
    ax.plot(x, y, color=COLORS["blue"], linewidth=3, label="Manifold M")

    # Tangent spaces at several points
    points = [0.8, 2.0, 3.5, 5.0]
    colors_list = [COLORS["red"], COLORS["green"], COLORS["purple"], COLORS["amber"]]

    for i, t0 in enumerate(points):
        px = t0
        py = 0.8*np.sin(t0) + 0.3*np.sin(2*t0)
        # Tangent direction
        dx = 1.0
        dy = 0.8*np.cos(t0) + 0.6*np.cos(2*t0)
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/norm, dy/norm

        # Draw tangent line segment
        length = 0.6
        ax.plot([px - length*dx, px + length*dx],
                [py - length*dy, py + length*dy],
                color=colors_list[i], linewidth=2, alpha=0.8)

        # Tangent vector arrow
        ax.annotate("", xy=(px + length*dx, py + length*dy),
                   xytext=(px, py),
                   arrowprops=dict(arrowstyle="->", color=colors_list[i], lw=2))

        ax.scatter(px, py, color=colors_list[i], s=50, zorder=5)
        ax.text(px - 0.15, py + 0.25, f"$T_{{p_{i+1}}}M$",
                color=colors_list[i], fontsize=11)

    ax.set_xlim(-0.3, 6.8)
    ax.set_ylim(-1.5, 2.0)
    ax.set_aspect('equal')
    ax.set_title("Tangent Spaces at Points on a Manifold", fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(OUT_DIR, "fig05_tangent_bundle.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig06_euler_characteristic():
    """Euler characteristic examples: sphere chi=2, torus chi=0, double torus chi=-2."""
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)

    # --- Sphere ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor(BG_COLOR)
    phi = np.linspace(0, 2*np.pi, 40)
    theta = np.linspace(0, np.pi, 20)
    PHI, THETA = np.meshgrid(phi, theta)
    X = np.sin(THETA)*np.cos(PHI)
    Y = np.sin(THETA)*np.sin(PHI)
    Z = np.cos(THETA)
    ax1.plot_surface(X, Y, Z, alpha=0.5, color=COLORS["red"], edgecolor='none')
    ax1.set_title(r"Sphere: $\chi = 2$" + "\n" + r"$\int K\,dA = 4\pi$", fontsize=12)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    # --- Torus ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor(BG_COLOR)
    R, r = 1.5, 0.5
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(u, v)
    XT = (R + r*np.cos(U))*np.cos(V)
    YT = (R + r*np.cos(U))*np.sin(V)
    ZT = r*np.sin(U)
    ax2.plot_surface(XT, YT, ZT, alpha=0.5, color=COLORS["green"], edgecolor='none')
    ax2.set_title(r"Torus: $\chi = 0$" + "\n" + r"$\int K\,dA = 0$", fontsize=12)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # --- Double torus (two tori connected) ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor(BG_COLOR)
    # Approximate with two offset tori
    R2, r2 = 0.8, 0.35
    offset = 1.2
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, 2*np.pi, 30)
    U, V = np.meshgrid(u, v)

    # First torus
    XT1 = (R2 + r2*np.cos(U))*np.cos(V) - offset
    YT1 = (R2 + r2*np.cos(U))*np.sin(V)
    ZT1 = r2*np.sin(U)
    ax3.plot_surface(XT1, YT1, ZT1, alpha=0.5, color=COLORS["purple"], edgecolor='none')

    # Second torus
    XT2 = (R2 + r2*np.cos(U))*np.cos(V) + offset
    YT2 = (R2 + r2*np.cos(U))*np.sin(V)
    ZT2 = r2*np.sin(U)
    ax3.plot_surface(XT2, YT2, ZT2, alpha=0.5, color=COLORS["purple"], edgecolor='none')

    ax3.set_title(r"Double Torus: $\chi = -2$" + "\n" + r"$\int K\,dA = -4\pi$", fontsize=12)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig06_euler_characteristic.png")
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def upload_to_oss(local_path, oss_paths):
    """Upload figure to OSS for both EN and ZH."""
    for oss_path in oss_paths:
        cmd = f"ossutil cp -u {local_path} oss://blog-pic-ck/{oss_path}"
        print(f"  Uploading: {oss_path}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: upload failed: {result.stderr.strip()}")
        else:
            print(f"  OK")


if __name__ == "__main__":
    print("Generating Differential Geometry figures...")
    print()

    figures = {
        "01-curves-in-space": ("fig01_frenet_frame", fig01_frenet_frame),
        "02-surfaces-first-form": ("fig02_first_form", fig02_first_fundamental_form),
        "03-curvature-of-surfaces": ("fig03_principal_curvatures", fig03_principal_curvatures),
        "04-intrinsic-geometry": ("fig04_geodesics", fig04_geodesics),
        "05-manifolds-and-connections": ("fig05_tangent_bundle", fig05_tangent_bundle),
        "06-gauss-bonnet": ("fig06_euler_characteristic", fig06_euler_characteristic),
    }

    zh_slugs = {
        "01-curves-in-space": "01-空间曲线",
        "02-surfaces-first-form": "02-曲面与第一基本形式",
        "03-curvature-of-surfaces": "03-曲面的曲率",
        "04-intrinsic-geometry": "04-内蕴几何",
        "05-manifolds-and-connections": "05-流形与联络",
        "06-gauss-bonnet": "06-Gauss-Bonnet定理",
    }

    for en_slug, (fig_name, gen_func) in figures.items():
        print(f"--- {en_slug} ---")
        path = gen_func()
        zh_slug = zh_slugs[en_slug]
        oss_paths = [
            f"posts/en/differential-geometry/{en_slug}/fig_concept.png",
            f"posts/zh/differential-geometry/{zh_slug}/fig_concept.png",
        ]
        upload_to_oss(path, oss_paths)
        print()

    print("Done! All 6 figures generated and uploaded.")
