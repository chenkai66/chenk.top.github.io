#!/usr/bin/env python3
"""Generate 6 matplotlib figures for abstract-algebra series."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

# --- Fig 1: Group axioms / Cayley table of Z/4Z ---
def fig1():
    f, ax = make_fig(12, 5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 95, "Cayley Table of $\\mathbb{Z}/4\\mathbb{Z}$", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 88, "Every row and column is a permutation of the group elements", ha="center", va="top", fontsize=10.5, color=C["gray"], style="italic")
    # Draw Cayley table
    elems = ["0","1","2","3"]
    add_mod4 = [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
    colors = [C["blue"],C["green"],C["amber"],C["red"]]
    tx, ty, cw, ch = 25, 15, 12, 12
    # Header
    ax.text(tx-2, ty+4*ch+ch/2, "+", ha="center", va="center", fontsize=12, fontweight="bold", color=C["dark"])
    for j,e in enumerate(elems):
        rbox(ax, tx+j*cw, ty+4*ch, cw-1, ch-1, "#e5e7eb", e, fs=12, tc=C["dark"])
    for i in range(4):
        rbox(ax, tx-cw, ty+(3-i)*ch, cw-1, ch-1, "#e5e7eb", elems[i], fs=12, tc=C["dark"])
        for j in range(4):
            v = add_mod4[i][j]
            rbox(ax, tx+j*cw, ty+(3-i)*ch, cw-1, ch-1, colors[v], str(v), fs=12, tc="white")
    # Side: properties
    props = [("Closure", "a + b mod 4 ∈ {0,1,2,3}"),
             ("Associative", "(a+b)+c = a+(b+c)"),
             ("Identity", "0 + a = a + 0 = a"),
             ("Inverse", "a + (4−a) ≡ 0")]
    for k,(title,desc) in enumerate(props):
        yy = 70 - k*16
        rbox(ax, 68, yy, 28, 12, C["purple"] if k%2==0 else C["blue"], "", fs=10)
        ax.text(82, yy+8, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(82, yy+3, desc, ha="center", va="center", fontsize=8.5, color="#ffffffd9")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig1_cayley_table.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 2: Homomorphism diagram ---
def fig2():
    f, ax = make_fig(12, 5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 95, "First Isomorphism Theorem", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 88, "$G/\\ker\\varphi \\cong \\mathrm{im}\\,\\varphi$", ha="center", va="top", fontsize=12, color=C["gray"])
    # G box
    rbox(ax, 5, 30, 22, 18, C["blue"], "$G$", fs=16)
    # H box
    rbox(ax, 72, 30, 22, 18, C["green"], "$H$", fs=16)
    # G/ker box
    rbox(ax, 35, 5, 28, 14, C["purple"], "$G/\\ker\\varphi$", fs=13)
    # im(phi) box
    rbox(ax, 72, 5, 22, 14, C["amber"], "$\\mathrm{im}\\,\\varphi$", fs=11)
    # Arrows
    arr(ax, 28, 39, 71, 39, C["dark"])
    ax.text(50, 43, "$\\varphi$", ha="center", fontsize=13, color=C["dark"], fontweight="bold")
    arr(ax, 16, 30, 42, 20, C["purple"])
    ax.text(24, 22, "$\\pi$", ha="center", fontsize=12, color=C["purple"], fontweight="bold")
    arr(ax, 63, 12, 72, 12, C["amber"])
    ax.text(67, 16, "$\\cong$", ha="center", fontsize=13, color=C["red"], fontweight="bold")
    arr(ax, 83, 30, 83, 20, C["green"])
    ax.text(87, 24, "incl", ha="left", fontsize=10, color=C["green"])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig2_isomorphism_thm.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 3: Ring hierarchy ---
def fig3():
    f, ax = make_fig(13, 4.5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 95, "Hierarchy of Ring Structures", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 88, "Each level adds constraints: commutativity → no zero divisors → unique factorization → all ideals principal", ha="center", va="top", fontsize=9.5, color=C["gray"], style="italic")
    names = ["Ring","Commutative\nRing","Integral\nDomain","UFD","PID","Field"]
    cols = [C["gray"],C["blue"],C["blue"],C["purple"],C["purple"],C["red"]]
    examples = ["$M_2(\\mathbb{R})$","$\\mathbb{Z}[x,y]$","$\\mathbb{Z}[x]$","$\\mathbb{Z}[x]$","$\\mathbb{Z}$","$\\mathbb{Q}$"]
    n = len(names); bw = 13; gap = (100 - n*bw)/(n+1)
    for i in range(n):
        x = gap + i*(bw+gap)
        rbox(ax, x, 35, bw, 25, cols[i], names[i], fs=9.5)
        ax.text(x+bw/2, 28, examples[i], ha="center", va="top", fontsize=9, color=C["dark"])
        if i < n-1:
            arr(ax, x+bw+0.5, 47, x+bw+gap-0.5, 47, cols[i+1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig3_ring_hierarchy.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 4: Field extension tower ---
def fig4():
    f, ax = make_fig(8, 7)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 97, "Tower of Field Extensions", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    fields = [("$\\mathbb{Q}$", C["blue"]),
              ("$\\mathbb{Q}(\\sqrt{2})$", C["green"]),
              ("$\\mathbb{Q}(\\sqrt{2},\\sqrt{3})$", C["purple"]),
              ("$\\mathbb{Q}(\\sqrt{2},\\sqrt{3},i)$", C["red"])]
    degrees = ["2","2","2"]
    bw, bh = 36, 12
    for i,(name,col) in enumerate(fields):
        y = 10 + i*20
        rbox(ax, 50-bw/2, y, bw, bh, col, name, fs=12)
        if i > 0:
            ax.plot([50,50],[y, y-8], color=C["dark"], lw=2)
            ax.text(54, y-4, f"[degree {degrees[i-1]}]", fontsize=9.5, color=C["gray"], va="center")
    ax.text(50, 5, "$[\\mathbb{Q}(\\sqrt{2},\\sqrt{3},i):\\mathbb{Q}]=8$", ha="center", fontsize=11, color=C["dark"], style="italic")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig4_field_tower.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 5: Galois correspondence ---
def fig5():
    f, ax = make_fig(13, 6)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 97, "Galois Correspondence", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    ax.text(50, 91, "Subgroups of Gal(E/F) ↔ Intermediate fields between F and E", ha="center", va="top", fontsize=10.5, color=C["gray"], style="italic")
    # Left: subgroup lattice
    ax.text(25, 82, "Subgroups of $\\mathrm{Gal}(E/F)$", ha="center", fontsize=11, fontweight="bold", color=C["purple"])
    sg = [("$\\{e\\}$",25,20),("$\\langle\\sigma\\rangle$",10,40),("$\\langle\\tau\\rangle$",25,40),("$\\langle\\sigma\\tau\\rangle$",40,40),("$G$",25,62)]
    for t,x,y in sg:
        rbox(ax, x-7, y, 14, 10, C["purple"], t, fs=10)
    for (x1,y1),(x2,y2) in [((25,30),(10,40)),((25,30),(25,40)),((25,30),(40,40)),((10,50),(25,62)),((25,50),(25,62)),((40,50),(25,62))]:
        ax.plot([x1,x2],[y1,y2], color=C["purple"], lw=1.5, alpha=0.6)
    # Right: field lattice
    ax.text(75, 82, "Intermediate Fields", ha="center", fontsize=11, fontweight="bold", color=C["green"])
    fl = [("$E$",75,62),("$F(\\alpha)$",60,40),("$F(\\beta)$",75,40),("$F(\\alpha\\beta)$",90,40),("$F$",75,20)]
    for t,x,y in fl:
        rbox(ax, x-8, y, 16, 10, C["green"], t, fs=10)
    for (x1,y1),(x2,y2) in [((75,62),(60,50)),((75,62),(75,50)),((75,62),(90,50)),((60,40),(75,30)),((75,40),(75,30)),((90,40),(75,30))]:
        ax.plot([x1,x2],[y1,y2], color=C["green"], lw=1.5, alpha=0.6)
    # Double arrows between
    for y in [25,45,67]:
        ax.annotate("", xy=(55,y), xytext=(45,y), arrowprops=dict(arrowstyle="<->", color=C["amber"], lw=2.5, mutation_scale=18))
    ax.text(50, 73, "order-reversing\nbijection", ha="center", fontsize=9.5, color=C["amber"], fontweight="bold")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig5_galois_correspondence.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

# --- Fig 6: Applications overview ---
def fig6():
    f, ax = make_fig(13, 4.5)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.text(50, 95, "Abstract Algebra in the Real World", ha="center", va="top", fontsize=14, fontweight="bold", color=C["dark"])
    apps = [("Cryptography\n(RSA, ECC)", "$\\mathbb{Z}/n\\mathbb{Z}$, elliptic\ncurves over $\\mathbb{F}_p$", C["red"]),
            ("Error Correction\n(Reed-Solomon)", "Polynomial rings\nover $\\mathbb{F}_q$", C["blue"]),
            ("Physics\n(Symmetry)", "Lie groups,\nrepresentation theory", C["purple"]),
            ("CS Theory\n(Complexity)", "Boolean algebras,\nfinite fields", C["green"])]
    bw, bh = 20, 30
    for i,(title,desc,col) in enumerate(apps):
        x = 5 + i*24
        rbox(ax, x, 30, bw, bh, col, "", fs=10)
        ax.text(x+bw/2, 55, title, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        ax.text(x+bw/2, 40, desc, ha="center", va="center", fontsize=8.5, color="#ffffffd9")
    # Central label
    rbox(ax, 35, 10, 30, 12, C["amber"], "Abstract Algebra", fs=12)
    for x in [15,39,63,87]:
        arr(ax, 50, 22, x, 30, C["amber"])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    f.savefig(f"{OUT}/aa_fig6_applications.png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.1)
    plt.close(f)

for i,fn in enumerate([fig1,fig2,fig3,fig4,fig5,fig6],1):
    print(f"Generating AA fig {i}...")
    fn()

# Upload to OSS
BUCKET = "oss://blog-pic-ck"
slugs = ["01-groups-and-subgroups","02-homomorphisms-and-quotients","03-rings-and-ideals",
         "04-fields-and-extensions","05-galois-theory","06-applications"]
fnames = ["aa_fig1_cayley_table.png","aa_fig2_isomorphism_thm.png","aa_fig3_ring_hierarchy.png",
          "aa_fig4_field_tower.png","aa_fig5_galois_correspondence.png","aa_fig6_applications.png"]
for slug,fname in zip(slugs,fnames):
    local = f"{OUT}/{fname}"
    for lang in ["en","zh"]:
        oss = f"{BUCKET}/posts/{lang}/abstract-algebra/{slug}/{fname}"
        r = subprocess.run(["ossutil","cp","-u",local,oss,"--meta","Cache-Control:public,max-age=300,must-revalidate"], capture_output=True, text=True)
        print(f"  {lang}/{slug}: {'OK' if r.returncode==0 else r.stderr[:80]}")
print("Abstract Algebra figures done!")
