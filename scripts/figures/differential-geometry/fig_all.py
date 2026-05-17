#!/usr/bin/env python3
"""Generate 6 matplotlib figures for differential-geometry series."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withSimplePatchShadow
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, subprocess

BG = "#fdfcf9"
C = {"red":"#e85d4a","amber":"#f5a834","purple":"#8b5cf6","blue":"#3b82f6","green":"#10b981","gray":"#6b7280","dark":"#1f2937"}
OUT = "/tmp/math_figs"
os.makedirs(OUT, exist_ok=True)

# --- Fig 1: Helix with Frenet frame ---
def fig1():
    f = plt.figure(figsize=(10, 7))
    f.patch.set_facecolor(BG)
    ax = f.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    t = np.linspace(0, 4*np.pi, 300)
    a, b = 1.0, 0.3
    x, y, z = a*np.cos(t), a*np.sin(t), b*t
    ax.plot(x, y, z, color=C["blue"], lw=2.5, label="Helix $\\gamma(t)$")
    # Frenet frame at 3 points
    for t0 in [np.pi, 2*np.pi, 3*np.pi]:
        p = np.array([a*np.cos(t0), a*np.sin(t0), b*t0])
        T = np.array([-a*np.sin(t0), a*np.cos(t0), b])
        T = T/np.linalg.norm(T)
        N = np.array([-np.cos(t0), -np.sin(t0), 0])
        B = np.cross(T, N)
        s = 0.5
        ax.quiver(*p, *T*s, color=C["red"], arrow_length_ratio=0.15, lw=2)
        ax.quiver(*p, *N*s, color=C["green"], arrow_length_ratio=0.15, lw=2)
        ax.quiver(*p, *B*s, color=C["purple"], arrow_length_ratio=0.15, lw=2)
    ax.quiver([],[],[],[],[],[], color=C["red"], label="$\\mathbf{T}$ (tangent)")
    ax.quiver([],[],[],[],[],[], color=C["green"], label="$\\mathbf{N}$ (normal)")
    ax.quiver([],[],[],[],[],[], color=C["purple"], label="$\\mathbf{B}$ (binormal)")
    ax.set_title("Frenet Frame on a Helix", fontsize=14, fontweight="bold", color=C["dark"], pad=15)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    f.savefig(f"{OUT}/dg_fig1_frenet.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 2: Surface patch with tangent plane ---
def fig2():
    f = plt.figure(figsize=(10, 7))
    f.patch.set_facecolor(BG)
    ax = f.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    u = np.linspace(-2, 2, 40)
    v = np.linspace(-2, 2, 40)
    U, V = np.meshgrid(u, v)
    X = U; Y = V; Z = np.sin(U)*np.cos(V)
    ax.plot_surface(X, Y, Z, alpha=0.5, color=C["blue"], edgecolor="none")
    # Tangent plane at (0,0)
    u0, v0 = 0, 0
    p = np.array([u0, v0, np.sin(u0)*np.cos(v0)])
    du = np.array([1, 0, np.cos(u0)*np.cos(v0)])
    dv = np.array([0, 1, -np.sin(u0)*np.sin(v0)])
    n = np.cross(du, dv); n = n/np.linalg.norm(n)
    s = 1.5
    uu = np.linspace(-s, s, 5); vv = np.linspace(-s, s, 5)
    UU, VV = np.meshgrid(uu, vv)
    XX = p[0]+UU*du[0]+VV*dv[0]
    YY = p[1]+UU*du[1]+VV*dv[1]
    ZZ = p[2]+UU*du[2]+VV*dv[2]
    ax.plot_surface(XX, YY, ZZ, alpha=0.3, color=C["amber"], edgecolor="none")
    ax.quiver(*p, *n*0.8, color=C["red"], arrow_length_ratio=0.15, lw=2.5, label="Normal $\\mathbf{n}$")
    ax.quiver(*p, *du*0.6, color=C["green"], arrow_length_ratio=0.15, lw=2, label="$\\partial_u$")
    ax.quiver(*p, *dv*0.6, color=C["purple"], arrow_length_ratio=0.15, lw=2, label="$\\partial_v$")
    ax.set_title("Surface with Tangent Plane and Normal", fontsize=14, fontweight="bold", color=C["dark"], pad=15)
    ax.legend(loc="upper left", fontsize=9.5)
    f.savefig(f"{OUT}/dg_fig2_tangent_plane.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 3: Gaussian curvature comparison ---
def fig3():
    f, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={"projection":"3d"})
    f.patch.set_facecolor(BG)
    f.suptitle("Gaussian Curvature: $K = \\kappa_1 \\cdot \\kappa_2$", fontsize=14, fontweight="bold", color=C["dark"], y=0.97)
    # Sphere (K > 0)
    u = np.linspace(0, 2*np.pi, 40); v = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u, v)
    axes[0].plot_surface(np.cos(U)*np.sin(V), np.sin(U)*np.sin(V), np.cos(V), alpha=0.6, color=C["blue"], edgecolor="none")
    axes[0].set_title("Sphere: $K > 0$", fontsize=11, color=C["blue"], fontweight="bold")
    # Cylinder (K = 0)
    t = np.linspace(0, 2*np.pi, 40); z = np.linspace(-1, 1, 20)
    T, Z = np.meshgrid(t, z)
    axes[1].plot_surface(np.cos(T), np.sin(T), Z, alpha=0.6, color=C["green"], edgecolor="none")
    axes[1].set_title("Cylinder: $K = 0$", fontsize=11, color=C["green"], fontweight="bold")
    # Saddle (K < 0)
    x = np.linspace(-1, 1, 40); y = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(x, y)
    axes[2].plot_surface(X, Y, X**2-Y**2, alpha=0.6, color=C["red"], edgecolor="none")
    axes[2].set_title("Saddle: $K < 0$", fontsize=11, color=C["red"], fontweight="bold")
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout(rect=[0,0,1,0.92])
    f.savefig(f"{OUT}/dg_fig3_curvature.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 4: Geodesics on sphere ---
def fig4():
    f = plt.figure(figsize=(10, 7))
    f.patch.set_facecolor(BG)
    ax = f.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    u = np.linspace(0, 2*np.pi, 50); v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.cos(U)*np.sin(V), np.sin(U)*np.sin(V), np.cos(V), alpha=0.2, color=C["blue"], edgecolor="none")
    # Great circles (geodesics)
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), color=C["red"], lw=2.5, label="Equator")
    ax.plot(np.cos(t), np.zeros_like(t), np.sin(t), color=C["green"], lw=2.5, label="Meridian")
    ax.plot(np.cos(t)*np.cos(np.pi/4), np.sin(t), np.cos(t)*np.sin(np.pi/4), color=C["purple"], lw=2.5, label="Tilted great circle")
    ax.set_title("Geodesics on $S^2$ (Great Circles)", fontsize=14, fontweight="bold", color=C["dark"], pad=15)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    f.savefig(f"{OUT}/dg_fig4_geodesics.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 5: Chart transition (manifold concept diagram) ---
def fig5():
    shadow = withSimplePatchShadow(offset=(1.2,-1.2), shadow_rgbFace="#000000", alpha=0.18)
    def rbox(ax, x, y, w, h, color, text="", fs=10, tc="white"):
        b = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=color, ec="none")
        b.set_path_effects([shadow]); ax.add_patch(b)
        if text: ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")
    def arr(ax, x1, y1, x2, y2, color="#9ca3af"):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16))
    f, ax = plt.subplots(figsize=(13, 5.5))
    f.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 97, "Manifold: Charts and Transition Maps", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    # Manifold M in center
    theta = np.linspace(0, 2*np.pi, 100)
    mx, my = 50+18*np.cos(theta), 55+12*np.sin(theta)
    ax.fill(mx, my, alpha=0.15, color=C["blue"])
    ax.plot(mx, my, color=C["blue"], lw=2)
    ax.text(50, 55, "$M$", fontsize=16, ha="center", va="center", color=C["blue"], fontweight="bold")
    # Chart U_alpha (left overlap region)
    ax.add_patch(plt.Circle((40, 58), 12, fill=True, alpha=0.15, color=C["green"]))
    ax.text(33, 65, "$U_\\alpha$", fontsize=12, color=C["green"], fontweight="bold")
    # Chart U_beta (right overlap region)
    ax.add_patch(plt.Circle((60, 58), 12, fill=True, alpha=0.15, color=C["purple"]))
    ax.text(67, 65, "$U_\\beta$", fontsize=12, color=C["purple"], fontweight="bold")
    # R^n boxes at bottom
    rbox(ax, 5, 5, 25, 16, C["green"], "$\\varphi_\\alpha(U_\\alpha) \\subset \\mathbb{R}^n$", fs=10)
    rbox(ax, 70, 5, 25, 16, C["purple"], "$\\varphi_\\beta(U_\\beta) \\subset \\mathbb{R}^n$", fs=10)
    # Arrows
    arr(ax, 35, 48, 17, 22, C["green"])
    ax.text(22, 36, "$\\varphi_\\alpha$", fontsize=12, color=C["green"], fontweight="bold")
    arr(ax, 65, 48, 83, 22, C["purple"])
    ax.text(77, 36, "$\\varphi_\\beta$", fontsize=12, color=C["purple"], fontweight="bold")
    arr(ax, 31, 13, 69, 13, C["amber"])
    ax.text(50, 17, "$\\varphi_\\beta \\circ \\varphi_\\alpha^{-1}$", ha="center", fontsize=11, color=C["amber"], fontweight="bold")
    ax.text(50, 8, "(transition map — must be smooth)", ha="center", fontsize=9.5, color=C["gray"], style="italic")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/dg_fig5_manifold.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 6: Gauss-Bonnet illustration ---
def fig6():
    shadow = withSimplePatchShadow(offset=(1.2,-1.2), shadow_rgbFace="#000000", alpha=0.18)
    def rbox(ax, x, y, w, h, color, text="", fs=10, tc="white"):
        b = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=color, ec="none")
        b.set_path_effects([shadow]); ax.add_patch(b)
        if text: ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")
    f, ax = plt.subplots(figsize=(13, 5.5))
    f.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 97, "Gauss-Bonnet Theorem: $\\int_M K\\,dA = 2\\pi\\chi(M)$", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 90, "Total curvature is a topological invariant", ha="center", va="top", fontsize=11, color=C["gray"], style="italic")
    surfaces = [("Sphere", "$\\chi=2$\n$\\int K\\,dA = 4\\pi$", C["blue"], "genus 0"),
                ("Torus", "$\\chi=0$\n$\\int K\\,dA = 0$", C["green"], "genus 1"),
                ("Double Torus", "$\\chi=-2$\n$\\int K\\,dA = -4\\pi$", C["red"], "genus 2")]
    for i,(name,formula,col,genus) in enumerate(surfaces):
        x = 8 + i*32
        rbox(ax, x, 25, 26, 45, col, "", fs=10)
        ax.text(x+13, 64, name, ha="center", va="center", fontsize=13, color="white", fontweight="bold")
        ax.text(x+13, 48, formula, ha="center", va="center", fontsize=11, color="#ffffffe6")
        ax.text(x+13, 32, genus, ha="center", va="center", fontsize=10, color="#ffffffbf", style="italic")
    # Formula at bottom
    rbox(ax, 15, 5, 70, 12, C["amber"], "$\\chi(M) = 2 - 2g$ where $g$ = genus (number of holes)", fs=11)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/dg_fig6_gauss_bonnet.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

for i,fn in enumerate([fig1,fig2,fig3,fig4,fig5,fig6],1):
    print(f"Generating DG fig {i}...")
    fn()

BUCKET = "oss://blog-pic-ck"
slugs = ["01-curves-in-space","02-surfaces-first-form","03-curvature-of-surfaces",
         "04-intrinsic-geometry","05-manifolds-and-connections","06-gauss-bonnet"]
fnames = ["dg_fig1_frenet.png","dg_fig2_tangent_plane.png","dg_fig3_curvature.png",
          "dg_fig4_geodesics.png","dg_fig5_manifold.png","dg_fig6_gauss_bonnet.png"]
for slug,fname in zip(slugs,fnames):
    local = f"{OUT}/{fname}"
    for lang in ["en","zh"]:
        oss = f"{BUCKET}/posts/{lang}/differential-geometry/{slug}/{fname}"
        r = subprocess.run(["ossutil","cp","-u",local,oss,"--meta","Cache-Control:public,max-age=300,must-revalidate"], capture_output=True, text=True)
        print(f"  {lang}/{slug}: {'OK' if r.returncode==0 else r.stderr[:80]}")
print("Differential Geometry figures done!")
