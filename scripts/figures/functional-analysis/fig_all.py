#!/usr/bin/env python3
"""Generate 6 matplotlib figures for functional-analysis series."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np
import os, subprocess

BG = "#fdfcf9"
C = {"red":"#e85d4a","amber":"#f5a834","purple":"#8b5cf6","blue":"#3b82f6","green":"#10b981","gray":"#6b7280","dark":"#1f2937"}
OUT = "/tmp/math_figs"
os.makedirs(OUT, exist_ok=True)
shadow = withSimplePatchShadow(offset=(1.2,-1.2), shadow_rgbFace="#000000", alpha=0.18)

def make_fig(w, h):
    f, ax = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    return f, ax

def rbox(ax, x, y, w, h, color, text="", fs=10, tc="white"):
    b = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=color, ec="none")
    b.set_path_effects([shadow]); ax.add_patch(b)
    if text: ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")

def arr(ax, x1, y1, x2, y2, color="#9ca3af"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16))

# --- Fig 1: Unit balls in l^1, l^2, l^inf ---
def fig1():
    f, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    f.patch.set_facecolor(BG)
    f.suptitle("Unit Balls in Different Norms", fontsize=14, fontweight="bold", color="#1f2937", y=0.97)
    norms = [("$\\ell^1$ norm\n$\\|x\\|_1 = |x_1|+|x_2|$", "diamond"),
             ("$\\ell^2$ norm\n$\\|x\\|_2 = \\sqrt{x_1^2+x_2^2}$", "circle"),
             ("$\\ell^\\infty$ norm\n$\\|x\\|_\\infty = \\max(|x_1|,|x_2|)$", "square")]
    cols = [C["blue"], C["green"], C["red"]]
    for ax, (title, shape), col in zip(axes, norms, cols):
        ax.set_facecolor(BG)
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_title(title, fontsize=10.5, color="#1f2937", pad=10)
        t = np.linspace(0, 2*np.pi, 200)
        if shape == "diamond":
            pts = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]])
            ax.fill(pts[:,0], pts[:,1], alpha=0.25, color=col)
            ax.plot(pts[:,0], pts[:,1], color=col, lw=2)
        elif shape == "circle":
            ax.fill(np.cos(t), np.sin(t), alpha=0.25, color=col)
            ax.plot(np.cos(t), np.sin(t), color=col, lw=2)
        else:
            pts = np.array([[1,1],[-1,1],[-1,-1],[1,-1],[1,1]])
            ax.fill(pts[:,0], pts[:,1], alpha=0.25, color=col)
            ax.plot(pts[:,0], pts[:,1], color=col, lw=2)
        ax.set_xlabel("$x_1$", fontsize=10); ax.set_ylabel("$x_2$", fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.92])
    f.savefig(f"{OUT}/fa_fig1_unit_balls.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 2: Orthogonal projection in Hilbert space ---
def fig2():
    f, ax = make_fig(12, 5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 96, "Orthogonal Projection in Hilbert Space", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 89, "$\\hat{x} = P_M x$ minimizes $\\|x - m\\|$ over $m \\in M$", ha="center", va="top", fontsize=11, color=C["gray"], style="italic")
    # Draw subspace M as a shaded band
    ax.fill([5,95,95,5],[20,40,50,30], alpha=0.15, color=C["blue"])
    ax.plot([5,95],[25,45], color=C["blue"], lw=2, label="$M$ (closed subspace)")
    ax.text(85, 48, "$M$", fontsize=14, color=C["blue"], fontweight="bold")
    # Point x above the subspace
    ax.plot(50, 75, "o", color=C["red"], markersize=10, zorder=5)
    ax.text(53, 76, "$x$", fontsize=14, color=C["red"], fontweight="bold")
    # Projection point
    ax.plot(50, 35, "o", color=C["green"], markersize=10, zorder=5)
    ax.text(53, 33, "$\\hat{x} = P_M x$", fontsize=12, color=C["green"], fontweight="bold")
    # Vertical line (perpendicular)
    ax.plot([50,50],[35,75], "--", color=C["purple"], lw=2)
    ax.text(42, 55, "$x - \\hat{x} \\perp M$", fontsize=11, color=C["purple"], rotation=90, va="center")
    # Right angle mark
    ax.plot([50,53,53],[38,38,41], color=C["purple"], lw=1.5)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/fa_fig2_projection.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 3: Bounded vs unbounded operators ---
def fig3():
    f, ax = make_fig(13, 4.5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 96, "Bounded vs Unbounded Operators", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    # Bounded operator box
    rbox(ax, 3, 25, 42, 50, C["blue"], "", fs=10)
    ax.text(24, 70, "Bounded ($T \\in B(X,Y)$)", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.text(24, 60, "$\\|Tx\\| \\leq M\\|x\\|$", ha="center", fontsize=11, color="#ffffffe6")
    ax.text(24, 50, "continuous ↔ bounded\n(on normed spaces)", ha="center", fontsize=9.5, color="#ffffffcc")
    ax.text(24, 35, "Examples:\n• Matrix operators\n• Integral operators\n• Shift operators", ha="center", fontsize=9.5, color="#ffffffcc")
    # Unbounded operator box
    rbox(ax, 55, 25, 42, 50, C["red"], "", fs=10)
    ax.text(76, 70, "Unbounded", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.text(76, 60, "sup $\\|Tx\\|/\\|x\\| = \\infty$", ha="center", fontsize=11, color="#ffffffe6")
    ax.text(76, 50, "densely defined,\nclosed but not continuous", ha="center", fontsize=9.5, color="#ffffffcc")
    ax.text(76, 35, "Examples:\n• $d/dx$ on $L^2$\n• Multiplication by $x$\n• Laplacian $\\Delta$", ha="center", fontsize=9.5, color="#ffffffcc")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/fa_fig3_operators.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 4: The four big theorems ---
def fig4():
    f, ax = make_fig(13, 5.5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 97, "The Four Pillars of Functional Analysis", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 91, "All require completeness (Banach spaces)", ha="center", va="top", fontsize=10.5, color=C["gray"], style="italic")
    thms = [("Hahn-Banach", "Extend functionals\nfrom subspaces", C["blue"]),
            ("Open Mapping", "Surjective → open\n(between Banach)", C["green"]),
            ("Closed Graph", "Closed graph →\ncontinuous", C["purple"]),
            ("Uniform\nBoundedness", "Pointwise bounded →\nuniformly bounded", C["red"])]
    for i,(name,desc,col) in enumerate(thms):
        x = 3 + i*24.5
        rbox(ax, x, 35, 22, 38, col, "", fs=10)
        ax.text(x+11, 67, name, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(x+11, 50, desc, ha="center", va="center", fontsize=9.5, color="#ffffffd9")
    # Baire category at bottom
    rbox(ax, 20, 8, 60, 14, C["amber"], "Baire Category Theorem (complete metric spaces)", fs=11)
    for x in [14,38.5,63,87.5]:
        arr(ax, x, 22, x, 35, C["amber"])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/fa_fig4_big_theorems.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 5: Spectrum decomposition ---
def fig5():
    f, ax = make_fig(12, 5)
    ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
    ax.set_facecolor(BG); ax.set_aspect("equal")
    ax.set_title("Spectrum of an Operator $T$", fontsize=14, fontweight="bold", color=C["dark"], pad=15)
    # Complex plane
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Re", fontsize=10); ax.set_ylabel("Im", fontsize=10)
    # Point spectrum (eigenvalues) — discrete points
    eigs = [(-1.5, 0), (0.5, 1), (0.5, -1), (1.8, 0)]
    for ex,ey in eigs:
        ax.plot(ex, ey, "o", color=C["red"], markersize=8, zorder=5)
    ax.plot([], [], "o", color=C["red"], markersize=8, label="Point spectrum $\\sigma_p(T)$")
    # Continuous spectrum — arc
    t = np.linspace(-0.8, 0.8, 100)
    ax.plot(-0.3+0.5*np.cos(t*np.pi), 1.5*np.sin(t*np.pi), color=C["blue"], lw=3, label="Continuous spectrum $\\sigma_c(T)$")
    # Residual spectrum — shaded region
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(2+0.4*np.cos(theta), 1.5+0.4*np.sin(theta), alpha=0.3, color=C["purple"])
    ax.plot(2+0.4*np.cos(theta), 1.5+0.4*np.sin(theta), color=C["purple"], lw=2, label="Residual spectrum $\\sigma_r(T)$")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    f.savefig(f"{OUT}/fa_fig5_spectrum.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 6: Sobolev space inclusions ---
def fig6():
    f, ax = make_fig(13, 5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 96, "Sobolev Space Embedding Chain", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 89, "Higher regularity → more embeddings (Sobolev embedding theorem)", ha="center", va="top", fontsize=10.5, color=C["gray"], style="italic")
    spaces = ["$W^{k,p}$","$W^{k-1,p^*}$","$\\cdots$","$L^q$","$C^{0,\\alpha}$"]
    cols = [C["red"],C["purple"],C["gray"],C["blue"],C["green"]]
    bw = 16; n = len(spaces); gap = (100-n*bw)/(n+1)
    for i,(sp,col) in enumerate(zip(spaces,cols)):
        x = gap+i*(bw+gap)
        rbox(ax, x, 40, bw, 18, col, sp, fs=12)
        if i < n-1:
            arr(ax, x+bw+1, 49, x+bw+gap-1, 49, C["dark"])
    ax.text(50, 30, "$p^* = np/(n-p)$ when $kp < n$ (Sobolev conjugate exponent)", ha="center", fontsize=10, color=C["gray"])
    ax.text(50, 22, "$W^{k,p}(\\Omega) \\hookrightarrow C^{0,\\alpha}(\\bar\\Omega)$ when $kp > n$ (Morrey)", ha="center", fontsize=10, color=C["green"])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/fa_fig6_sobolev.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

for i,fn in enumerate([fig1,fig2,fig3,fig4,fig5,fig6],1):
    print(f"Generating FA fig {i}...")
    fn()

BUCKET = "oss://blog-pic-ck"
slugs = ["01-metric-and-normed-spaces","02-hilbert-spaces","03-bounded-operators",
         "04-big-theorems","05-spectral-theory","06-distributions-sobolev"]
fnames = ["fa_fig1_unit_balls.png","fa_fig2_projection.png","fa_fig3_operators.png",
          "fa_fig4_big_theorems.png","fa_fig5_spectrum.png","fa_fig6_sobolev.png"]
for slug,fname in zip(slugs,fnames):
    local = f"{OUT}/{fname}"
    for lang in ["en","zh"]:
        oss = f"{BUCKET}/posts/{lang}/functional-analysis/{slug}/{fname}"
        r = subprocess.run(["ossutil","cp","-u",local,oss,"--meta","Cache-Control:public,max-age=300,must-revalidate"], capture_output=True, text=True)
        print(f"  {lang}/{slug}: {'OK' if r.returncode==0 else r.stderr[:80]}")
print("Functional Analysis figures done!")
