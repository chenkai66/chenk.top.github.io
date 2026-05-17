#!/usr/bin/env python3
"""AA figures part 1 (articles 01-04). Uploads to OSS at end."""
import os, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrow, Wedge
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np

BG = "#fdfcf9"
C = {"red":"#e85d4a","amber":"#f5a834","purple":"#8b5cf6","blue":"#3b82f6",
     "green":"#10b981","gray":"#6b7280","dark":"#1f2937","light":"#e5e7eb"}
shadow = withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.15)
OUT = "/tmp/aa_figs_v2"

def mk(w=12,h=7):
    f,a = plt.subplots(figsize=(w,h))
    f.patch.set_facecolor(BG); a.set_facecolor(BG); a.axis('off')
    return f,a

def rb(ax,x,y,w,h,color,t='',fs=12,tc='white',alpha=1):
    b = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=0.6",
                       fc=color,ec='none',alpha=alpha,lw=0)
    b.set_path_effects([shadow]); ax.add_patch(b)
    if t: ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold')

def ar(ax,x1,y1,x2,y2,col=C['gray'],lw=2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,mutation_scale=18))

def title(ax,t,sub=''):
    ax.text(0.5, 0.97, t, transform=ax.transAxes, ha='center', va='top',
            fontsize=15, fontweight='bold', color=C['dark'])
    if sub:
        ax.text(0.5, 0.93, sub, transform=ax.transAxes, ha='center', va='top',
                fontsize=11, color=C['gray'], style='italic')

def save(fig, slug, name):
    d = f"{OUT}/{slug}"
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/{name}.png", dpi=180, bbox_inches='tight',
                facecolor=BG, pad_inches=0.15)
    plt.close(fig)

# ===== Article 01: Groups =====

def aa_01_1():  # Cayley table Z/4Z
    f,a = mk(11,8); a.set_xlim(0,100); a.set_ylim(0,100)
    title(a, "Cayley Table of $\\mathbb{Z}/4\\mathbb{Z}$", "Addition modulo 4")
    cs = 14; gx, gy = 18, 18; rs = 12
    # header
    rb(a, gx, 78, cs, rs, C['dark'], '+', 14)
    cols = [C['blue'], C['green'], C['amber'], C['red']]
    for i in range(4):
        rb(a, gx+(i+1)*cs, 78, cs, rs, C['dark'], str(i), 14)
        rb(a, gx, 78-(i+1)*rs, cs, rs, C['dark'], str(i), 14)
    for i in range(4):
        for j in range(4):
            v = (i+j) % 4
            rb(a, gx+(j+1)*cs, 78-(i+1)*rs, cs, rs, cols[v], str(v), 14)
    a.text(50, 8, "Each value appears once per row and column — a Latin square structure.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "01-groups-first-encounter", "aa_v2_01_1_cayley_z4")

def aa_01_2():  # Dihedral D4
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,75)
    title(a, "The Dihedral Group $D_4$", "8 symmetries of a square: 4 rotations and 4 reflections")
    sym = [("$e$",C['blue']),("$r$",C['blue']),("$r^2$",C['blue']),("$r^3$",C['blue']),
           ("$s$",C['red']),("$rs$",C['red']),("$r^2s$",C['red']),("$r^3s$",C['red'])]
    labs = ["identity","rot 90°","rot 180°","rot 270°","flip −","flip /","flip |","flip \\"]
    for i,((sm,col),lb) in enumerate(zip(sym,labs)):
        x = 6 + i*16; y = 30
        # square
        sq = Rectangle((x, y), 10, 10, fc=col, ec='none', alpha=0.85)
        sq.set_path_effects([shadow]); a.add_patch(sq)
        # rotation arrow indicator
        ang_map = {1:90, 2:180, 3:270}
        if i in [1,2,3]:
            ang = ang_map[i]
            ang_rad = np.radians(ang)
            dx, dy = 0, 5
            a.annotate('', xy=(x+5+dx*np.cos(ang_rad)*0.4, y+5+dy*np.sin(ang_rad)*0.4+0.5),
                       xytext=(x+5, y+5), arrowprops=dict(arrowstyle='->',color='white',lw=2))
        if i==0:
            a.text(x+5, y+5, '•', ha='center', va='center', fontsize=20, color='white')
        if i==4:
            a.plot([x+1, x+9], [y+5, y+5], 'w-', lw=2)
        if i==5:
            a.plot([x+1, x+9], [y+1, y+9], 'w-', lw=2)
        if i==6:
            a.plot([x+5, x+5], [y+1, y+9], 'w-', lw=2)
        if i==7:
            a.plot([x+1, x+9], [y+9, y+1], 'w-', lw=2)
        a.text(x+5, y+15, sm, ha='center', fontsize=14, color=col, fontweight='bold')
        a.text(x+5, y-5, lb, ha='center', fontsize=9, color=C['gray'])
    a.text(70, 60, "Rotations form a cyclic subgroup; reflections do not.",
           ha='center', fontsize=11, color=C['dark'])
    a.text(70, 8, "Order 8, non-abelian, generators $r$ (90° rot) and $s$ (any flip)",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "01-groups-first-encounter", "aa_v2_01_2_dihedral_d4")

def aa_01_3():  # Subgroup lattice D4 Hasse diagram
    f,a = mk(11,9); a.set_xlim(0,100); a.set_ylim(0,100)
    title(a, "Subgroup Lattice of $D_4$", "Hasse diagram showing inclusions")
    # Top: D4
    rb(a, 42, 82, 16, 8, C['dark'], '$D_4$', 14)
    # Middle level
    nodes_m = [(15, 55, '$\\langle r\\rangle$', C['blue']),
               (40, 55, '$V_1$', C['purple']),
               (65, 55, '$V_2$', C['purple'])]
    for x,y,t,col in nodes_m:
        rb(a, x, y, 16, 8, col, t, 13)
        ar(a, 50, 82, x+8, y+8)
    # Lower level
    nodes_l = [(8, 28, '$\\langle r^2\\rangle$', C['green']),
               (24, 28, '$\\langle s\\rangle$', C['amber']),
               (40, 28, '$\\langle r^2 s\\rangle$', C['amber']),
               (56, 28, '$\\langle rs\\rangle$', C['amber']),
               (72, 28, '$\\langle r^3 s\\rangle$', C['amber'])]
    for x,y,t,col in nodes_l:
        rb(a, x, y, 14, 7, col, t, 11)
    # Connections (simplified)
    for i, (x,y,t,col) in enumerate(nodes_l):
        if i == 0:  # <r^2> connects to <r>, V1, V2
            for mx,my,_,_ in nodes_m:
                ar(a, mx+8, my, x+7, y+7)
        elif i in [1,2]:  # <s>, <r^2 s> connect to V1
            ar(a, 48, 55, x+7, y+7)
        elif i in [3,4]:  # <rs>, <r^3 s> connect to V2
            ar(a, 73, 55, x+7, y+7)
    # Bottom
    rb(a, 42, 5, 16, 8, C['gray'], '$\\{e\\}$', 14)
    for x,y,_,_ in nodes_l:
        ar(a, x+7, y, 50, 13)
    a.text(50, 70, "$V_1 = \\{e, r^2, s, r^2 s\\}$, $V_2 = \\{e, r^2, rs, r^3 s\\}$",
           ha='center', fontsize=10, color=C['gray'], style='italic')
    save(f, "01-groups-first-encounter", "aa_v2_01_3_subgroup_lattice")

def aa_01_4():  # Lagrange partition
    f,a = mk(11,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Lagrange's Theorem: Cosets Partition the Group", 
          "$|G| = 12$, subgroup $H$ of order $4$ → 3 cosets")
    # Three coset boxes with 4 elements each
    cosets = [(10, "$H$", C['blue']), (40, "$gH$", C['green']), (70, "$g'H$", C['amber'])]
    for x, lb, col in cosets:
        rb(a, x, 25, 22, 25, col, '', alpha=0.25)
        a.text(x+11, 47, lb, ha='center', fontsize=15, color=col, fontweight='bold')
        for i in range(4):
            cx = x + 4 + (i%2)*12
            cy = 30 + (i//2)*8
            cir = Circle((cx,cy), 2, fc=col, ec='none')
            cir.set_path_effects([shadow]); a.add_patch(cir)
    a.text(50, 13, "$|G|/|H|$ = 3 cosets, each of size 4. Hence $|H|$ divides $|G|$.",
           ha='center', fontsize=12, color=C['dark'])
    a.text(50, 5, "This is why subgroup orders must divide the group's order.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "01-groups-first-encounter", "aa_v2_01_4_lagrange_partition")

def aa_01_5():  # Cyclic Z/6Z hexagon
    f,a = mk(10,8); a.set_xlim(-5,5); a.set_ylim(-4,5)
    title(a, "Cyclic Group $\\mathbb{Z}/6\\mathbb{Z}$ as Hexagon Rotations",
          "Generated by a single 60° rotation")
    R = 3
    for k in range(6):
        ang = np.pi/2 + k*np.pi/3
        x, y = R*np.cos(ang), R*np.sin(ang)
        col = [C['red'],C['amber'],C['green'],C['blue'],C['purple'],C['gray']][k]
        cir = Circle((x,y), 0.4, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
        a.text(x*1.32, y*1.32, str(k), ha='center', va='center', fontsize=14,
               color=col, fontweight='bold')
    # Connect with arc arrows
    for k in range(6):
        a1 = np.pi/2 + k*np.pi/3
        a2 = np.pi/2 + (k+1)*np.pi/3
        x1, y1 = R*np.cos(a1), R*np.sin(a1)
        x2, y2 = R*np.cos(a2), R*np.sin(a2)
        a.annotate('', xy=(x2*0.85,y2*0.85), xytext=(x1*0.85,y1*0.85),
                   arrowprops=dict(arrowstyle='-|>',color=C['gray'],lw=1.5,
                                   connectionstyle="arc3,rad=0.25", mutation_scale=14))
    a.text(0, 0, "$+1$", ha='center', va='center', fontsize=18,
           color=C['dark'], fontweight='bold')
    a.text(0, -3.7, "Adding 1 cyclically advances through the elements 0,1,2,3,4,5,0,...",
           ha='center', fontsize=10, color=C['gray'], style='italic')
    a.set_aspect('equal')
    save(f, "01-groups-first-encounter", "aa_v2_01_5_cyclic_z6")

def aa_01_6():  # Homomorphism Z -> Z/4Z
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "A Group Homomorphism $f: \\mathbb{Z} \\to \\mathbb{Z}/4\\mathbb{Z}$",
          "Reduction modulo 4 sends infinitely many integers to each residue class")
    # Z line
    a.plot([10, 130], [50, 50], color=C['blue'], lw=2.5)
    for v in range(-2, 11):
        x = 10 + (v+2)*9
        a.plot([x], [50], 'o', color=C['blue'], markersize=8)
        a.text(x, 56, str(v), ha='center', fontsize=10, color=C['blue'])
    a.text(135, 50, "...", ha='left', fontsize=14, color=C['blue'])
    a.text(70, 64, "$\\mathbb{Z}$", fontsize=18, color=C['blue'], fontweight='bold')
    # Z/4Z target
    targets = [(35, 18, '0', C['red']), (60, 18, '1', C['amber']),
               (85, 18, '2', C['green']), (110, 18, '3', C['purple'])]
    for x,y,t,col in targets:
        cir = Circle((x,y), 4, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
        a.text(x, y, t, ha='center', va='center', fontsize=14, color='white',
               fontweight='bold')
    a.text(72, 4, "$\\mathbb{Z}/4\\mathbb{Z}$", ha='center', fontsize=18,
           color=C['gray'], fontweight='bold')
    # Arrows
    for v in range(0, 9):
        x = 10 + (v+2)*9
        target_idx = v % 4
        tx, ty, _, col = targets[target_idx]
        ar(a, x, 47, tx, ty+4, col=col, lw=1)
    a.text(72, 33, "$f(n) = n \\,\\mathrm{mod}\\, 4$, kernel $= 4\\mathbb{Z}$",
           ha='center', fontsize=12, color=C['dark'], fontweight='bold')
    save(f, "01-groups-first-encounter", "aa_v2_01_6_homomorphism")

def aa_01_7():  # Group axioms flowchart
    f,a = mk(12,8); a.set_xlim(0,100); a.set_ylim(0,80)
    title(a, "The Four Group Axioms", "A checklist for verifying that $(G, \\cdot)$ is a group")
    boxes = [
        (10, 55, "Closure", "$\\forall a,b \\in G: a \\cdot b \\in G$", C['blue']),
        (10, 35, "Associativity", "$(a \\cdot b) \\cdot c = a \\cdot (b \\cdot c)$", C['green']),
        (55, 55, "Identity", "$\\exists e: e \\cdot a = a \\cdot e = a$", C['amber']),
        (55, 35, "Inverse", "$\\forall a, \\exists a^{-1}: a \\cdot a^{-1} = e$", C['red']),
    ]
    for x,y,t,form,col in boxes:
        rb(a, x, y, 35, 13, col, '', alpha=0.95)
        a.text(x+17.5, y+9, t, ha='center', va='center', fontsize=14,
               color='white', fontweight='bold')
        a.text(x+17.5, y+3.5, form, ha='center', va='center', fontsize=11,
               color='white')
    rb(a, 30, 8, 40, 13, C['dark'], 'Group $(G, \\cdot)$', 16)
    for x,y,_,_,col in boxes:
        ar(a, x+17.5, y, 50, 21, col=col)
    a.text(50, 73, "Forget any one of these and the structure isn't a group.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "01-groups-first-encounter", "aa_v2_01_7_axioms")

# ===== Article 02: Group Actions =====

def aa_02_1():  # Orbit-stabilizer
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "Orbit-Stabilizer Theorem", "$|G| = |\\mathrm{Orb}(x)| \\cdot |\\mathrm{Stab}(x)|$")
    # Group on left
    rb(a, 8, 25, 22, 25, C['blue'], '$G$', 18, alpha=0.9)
    a.text(19, 22, "$|G| = 12$", ha='center', fontsize=11, color=C['blue'])
    # X set on right with orbit highlighted
    rb(a, 75, 12, 55, 50, C['gray'], '', alpha=0.15)
    a.text(102, 64, "$X$", ha='center', fontsize=18, color=C['gray'], fontweight='bold')
    # Orbit points
    orb = [(85,30),(95,40),(105,30),(115,40),(105,50),(95,30)]
    for x,y in orb:
        cir = Circle((x,y), 2.5, fc=C['amber'], ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
    a.text(100, 18, "Orbit: 6 points", ha='center', fontsize=11,
           color=C['amber'], fontweight='bold')
    # Other points
    for x,y in [(82,55),(118,18),(125,55)]:
        cir = Circle((x,y), 2, fc=C['gray'], ec='none', alpha=0.6)
        a.add_patch(cir)
    # Arrow G -> X
    ar(a, 30, 38, 75, 38, col=C['blue'], lw=2.5)
    a.text(52, 42, "$g \\cdot x$", ha='center', fontsize=13, color=C['blue'],
           fontweight='bold')
    # Equation
    a.text(70, 6, "$|G| = |\\mathrm{Orb}(x)| \\cdot |\\mathrm{Stab}(x)|$ → $12 = 6 \\cdot 2$",
           ha='center', fontsize=14, color=C['dark'], fontweight='bold')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_1_orbit_stabilizer")

def aa_02_2():  # Cube rotations
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,75)
    title(a, "The 24 Rotational Symmetries of the Cube",
          "Classified by axis type and rotation angle")
    rows = [
        ("Identity", "1", C['gray'], "no rotation"),
        ("Face axes (×3)", "$3 \\times 3 = 9$", C['blue'], "90°, 180°, 270° about a face"),
        ("Edge axes (×6)", "$6 \\times 1 = 6$", C['green'], "180° about an edge midpoint"),
        ("Vertex axes (×4)", "$4 \\times 2 = 8$", C['amber'], "120°, 240° about a body diagonal"),
    ]
    for i, (lab, count, col, desc) in enumerate(rows):
        y = 55 - i*11
        rb(a, 10, y, 28, 8, col, lab, 12)
        rb(a, 42, y, 18, 8, col, count, 13, alpha=0.85)
        a.text(64, y+4, desc, ha='left', va='center', fontsize=11, color=C['dark'])
    rb(a, 10, 5, 50, 7, C['dark'], "Total: $1 + 9 + 6 + 8 = 24$", 13)
    a.text(95, 8.5, "$\\cong S_4$ (vertices permuted)",
           ha='center', fontsize=12, color=C['red'], fontweight='bold', style='italic')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_2_cube_rotations")

def aa_02_3():  # Burnside necklace
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Burnside Counting: 2-Color Necklaces with 4 Beads",
          "Cyclic group $C_4$ acts by rotation; fixed colorings give the count")
    # Show a sample necklace
    cx, cy = 25, 35
    for k in range(4):
        ang = np.pi/2 + k*np.pi/2
        x = cx + 8*np.cos(ang); y = cy + 8*np.sin(ang)
        col = [C['blue'], C['amber'], C['blue'], C['amber']][k]
        cir = Circle((x,y), 2.2, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
    a.plot([cx + 8*np.cos(np.pi/2 + k*np.pi/2) for k in range(5)],
           [cy + 8*np.sin(np.pi/2 + k*np.pi/2) for k in range(5)],
           '-', color=C['gray'], lw=1, alpha=0.5)
    a.text(cx, 22, "Sample BABA", ha='center', fontsize=11, color=C['gray'])
    # Table
    table = [("$g$", "Fixed by $g$", "Count"),
             ("$e$", "any coloring", "$2^4 = 16$"),
             ("$r$ (90°)", "all same", "$2$"),
             ("$r^2$ (180°)", "opposite pairs same", "$2^2 = 4$"),
             ("$r^3$ (270°)", "all same", "$2$")]
    for i, (a1, b1, c1) in enumerate(table):
        y = 55 - i*8
        col = C['dark'] if i==0 else C['gray']
        fs = 12 if i==0 else 11
        a.text(50, y, a1, ha='center', fontsize=fs, color=col, fontweight='bold' if i==0 else 'normal')
        a.text(72, y, b1, ha='center', fontsize=fs, color=col)
        a.text(92, y, c1, ha='center', fontsize=fs, color=col)
    a.text(50, 12, "Total fixed = $16+2+4+2 = 24$, so distinct necklaces = $24/4 = 6$",
           ha='center', fontsize=12, color=C['red'], fontweight='bold')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_3_burnside_necklace")

def aa_02_4():  # Action arrow diagram
    f,a = mk(12,6); a.set_xlim(0,100); a.set_ylim(0,55)
    title(a, "Group Action $G \\times X \\to X$",
          "An action assigns each $g \\in G$ a permutation of $X$")
    rb(a, 5, 25, 18, 14, C['blue'], '$G$', 22)
    rb(a, 30, 25, 18, 14, C['amber'], '$X$', 22)
    rb(a, 75, 25, 18, 14, C['amber'], '$X$', 22)
    a.text(14, 18, "group", ha='center', fontsize=10, color=C['blue'])
    a.text(39, 18, "set", ha='center', fontsize=10, color=C['amber'])
    a.text(84, 18, "set", ha='center', fontsize=10, color=C['amber'])
    ar(a, 48, 32, 75, 32, lw=2.5)
    a.text(61, 36, "$g \\cdot$", ha='center', fontsize=18, color=C['dark'], fontweight='bold')
    a.text(50, 8, "Identity acts trivially: $e \\cdot x = x$.    Composition: $(gh) \\cdot x = g \\cdot (h \\cdot x)$",
           ha='center', fontsize=11, color=C['gray'])
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_4_action_arrow")

def aa_02_5():  # Orbit partition
    f,a = mk(11,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Orbits Partition the Set $X$", "Disjoint orbits are an equivalence-class decomposition")
    # Big set rectangle
    rb(a, 8, 12, 84, 45, C['gray'], '', alpha=0.1)
    a.text(50, 52, "$X$", fontsize=18, color=C['gray'], ha='center', fontweight='bold')
    # 3 orbits with different colors
    orbits_pts = [
        ([(15,30),(20,40),(28,35),(22,25)], C['red']),
        ([(40,42),(48,38),(45,32),(38,28)], C['green']),
        ([(60,30),(68,40),(78,35),(72,25),(65,18),(80,18)], C['amber']),
    ]
    for pts, col in orbits_pts:
        for x,y in pts:
            cir = Circle((x,y), 2, fc=col, ec='none')
            cir.set_path_effects([shadow]); a.add_patch(cir)
        # connect
        for i in range(len(pts)):
            x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
            a.plot([x1,x2],[y1,y2], '--', color=col, alpha=0.4, lw=1)
    a.text(22, 18, "Orbit 1\nsize 4", ha='center', fontsize=10, color=C['red'])
    a.text(43, 18, "Orbit 2\nsize 4", ha='center', fontsize=10, color=C['green'])
    a.text(72, 8, "Orbit 3, size 6", ha='center', fontsize=10, color=C['amber'])
    a.text(50, 4, "Every $x \\in X$ lies in exactly one orbit. $|X| = \\sum |\\mathrm{Orb}_i|$.",
           ha='center', fontsize=11, color=C['dark'], style='italic')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_5_orbit_partition")

def aa_02_6():  # Class equation
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "The Class Equation for $S_4$", "$|G| = |Z(G)| + \\sum [G:C_G(x_i)]$")
    classes = [("$\\{e\\}$", 1, C['gray']), ("(12)-type", 6, C['blue']),
               ("(123)-type", 8, C['green']), ("(12)(34)-type", 3, C['amber']),
               ("(1234)-type", 6, C['red'])]
    x0 = 12; bar_w = 14; bar_max = 30
    for i,(lab, sz, col) in enumerate(classes):
        x = x0 + i*16
        h = (sz/8) * bar_max
        rb(a, x, 18, bar_w, h, col, str(sz), 16, alpha=0.9)
        a.text(x+bar_w/2, 13, lab, ha='center', fontsize=10, color=col, fontweight='bold')
    a.plot([8, 92], [18, 18], color=C['dark'], lw=2)
    a.text(50, 60, "Class sizes divide $|G| = 24$: " + " + ".join(str(c[1]) for c in classes) + " = 24",
           ha='center', fontsize=12, color=C['dark'], fontweight='bold')
    a.text(50, 5, "Each conjugacy class has size $|G|/|\\text{centralizer}|$.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_6_class_equation")

def aa_02_7():  # Polya square 4-color
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Pólya Enumeration: 4-Coloring a Square Modulo Rotation",
          "Cycle index counts distinct colorings")
    # Show formula
    a.text(50, 50, "$Z(C_4) = \\frac{1}{4}(a_1^4 + a_1^2 a_2 + 2a_4)$",
           ha='center', fontsize=15, color=C['dark'])
    a.text(50, 40, "Substitute $a_k = 4$ → $\\frac{1}{4}(256 + 64 + 32) = 88$ distinct colorings",
           ha='center', fontsize=13, color=C['blue'], fontweight='bold')
    # Sample squares
    cols_palette = [C['red'], C['amber'], C['green'], C['blue']]
    for k in range(5):
        cx = 12 + k*18; cy = 20
        cs = 5
        for i in range(2):
            for j in range(2):
                col = cols_palette[(k*7 + i*3 + j*5) % 4]
                rb(a, cx + j*cs, cy + i*cs, cs, cs, col, '', alpha=1)
    a.text(50, 8, "Compare to $4^4 = 256$ if rotations are distinguished — Pólya cuts to 88.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "02-group-actions-and-symmetry", "aa_v2_02_7_polya_square")

# ===== Article 03: Quotient Groups =====

def aa_03_1():  # Normal vs subgroup
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "Normal Subgroup vs Arbitrary Subgroup",
          "$N \\triangleleft G$ when left and right cosets coincide")
    # Left: arbitrary subgroup (cosets differ)
    rb(a, 10, 18, 50, 38, C['gray'], '', alpha=0.1)
    a.text(35, 60, "Subgroup $H$: $gH \\neq Hg$", ha='center', fontsize=12,
           color=C['red'], fontweight='bold')
    # Show different left and right cosets
    rb(a, 14, 28, 18, 8, C['blue'], '$gH$', 12, alpha=0.7)
    rb(a, 38, 28, 18, 8, C['red'], '$Hg$', 12, alpha=0.7)
    # Right: normal subgroup
    rb(a, 75, 18, 55, 38, C['gray'], '', alpha=0.1)
    a.text(102, 60, "Normal $N$: $gN = Ng$", ha='center', fontsize=12,
           color=C['green'], fontweight='bold')
    rb(a, 79, 28, 22, 8, C['green'], '$gN = Ng$', 12, alpha=0.85)
    rb(a, 105, 28, 22, 8, C['green'], '$g\'N = Ng\'$', 12, alpha=0.85)
    a.text(70, 8, "Normal subgroups let us form the quotient $G/N$ as a group; arbitrary subgroups do not.",
           ha='center', fontsize=11, color=C['dark'], style='italic')
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_1_normal_vs_subgroup")

def aa_03_2():  # Cosets in Z
    f,a = mk(13,6); a.set_xlim(-2,15); a.set_ylim(-2,5)
    title(a, "Cosets of $4\\mathbb{Z}$ in $\\mathbb{Z}$", "Four cosets, each an arithmetic progression")
    palette = [C['red'], C['amber'], C['green'], C['blue']]
    labels = ["$0+4\\mathbb{Z}$","$1+4\\mathbb{Z}$","$2+4\\mathbb{Z}$","$3+4\\mathbb{Z}$"]
    for v in range(-1, 13):
        col = palette[v % 4]
        cir = Circle((v, 1), 0.3, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
        a.text(v, 1.7, str(v), ha='center', fontsize=10, color=col, fontweight='bold')
    a.plot([-1.5, 13.5], [1, 1], '-', color=C['gray'], alpha=0.4, lw=1)
    for i, lb in enumerate(labels):
        a.text(2 + i*3, -1, lb, ha='center', fontsize=11, color=palette[i],
               fontweight='bold')
    a.text(6, -1.8, "$\\mathbb{Z}/4\\mathbb{Z} = \\{[0],[1],[2],[3]\\}$",
           ha='center', fontsize=12, color=C['dark'], fontweight='bold')
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_2_cosets_z4")

def aa_03_3():  # Quotient construction
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Building the Quotient $G/N$",
          "Collapse each coset to a single point")
    # G as bag of dots in 4 cosets
    rb(a, 8, 30, 35, 30, C['gray'], '', alpha=0.1)
    a.text(25, 63, "$G$", ha='center', fontsize=18, color=C['gray'], fontweight='bold')
    cosets_g = [(C['red'], 13, 50), (C['amber'], 25, 50), (C['green'], 37, 50),
                (C['red'], 13, 38), (C['amber'], 25, 38), (C['green'], 37, 38)]
    for col, x, y in cosets_g:
        cir = Circle((x,y), 1.8, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
    # G/N as 3 points
    rb(a, 60, 30, 32, 30, C['gray'], '', alpha=0.1)
    a.text(76, 63, "$G/N$", ha='center', fontsize=18, color=C['gray'], fontweight='bold')
    for col, x in [(C['red'],65),(C['amber'],76),(C['green'],87)]:
        cir = Circle((x, 45), 3, fc=col, ec='none')
        cir.set_path_effects([shadow]); a.add_patch(cir)
        a.text(x, 38, "[" + col[1:].upper()[:2] + "]", ha='center', fontsize=10,
               color=col, fontweight='bold')
    ar(a, 43, 45, 60, 45, lw=2.5)
    a.text(51, 49, "$\\pi$", fontsize=16, color=C['dark'], fontweight='bold')
    a.text(50, 18, "Each coset $gN$ becomes one element $[g]$. Multiplication: $[g][h] = [gh]$.",
           ha='center', fontsize=11, color=C['dark'])
    a.text(50, 10, "Well-defined precisely when $N$ is normal.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_3_quotient_construction")

def aa_03_4():  # First iso theorem
    f,a = mk(11,8); a.set_xlim(0,100); a.set_ylim(0,80)
    title(a, "First Isomorphism Theorem", "$G/\\ker(f) \\cong \\mathrm{im}(f)$")
    # Triangle layout
    rb(a, 10, 60, 18, 12, C['blue'], '$G$', 18)
    rb(a, 70, 60, 22, 12, C['amber'], '$\\mathrm{im}(f) \\subseteq H$', 14)
    rb(a, 38, 18, 24, 12, C['green'], '$G/\\ker(f)$', 16)
    ar(a, 28, 66, 70, 66, lw=2.5)
    a.text(48, 70, "$f$", fontsize=18, color=C['dark'], fontweight='bold')
    ar(a, 19, 60, 50, 30, lw=2.5, col=C['gray'])
    a.text(31, 47, "$\\pi$", fontsize=16, color=C['gray'])
    ar(a, 62, 30, 79, 60, lw=2.5, col=C['red'])
    a.text(73, 47, "$\\widetilde{f}$", fontsize=16, color=C['red'])
    a.text(50, 7, "$\\widetilde{f}([g]) = f(g)$ is the unique isomorphism making the diagram commute.",
           ha='center', fontsize=11, color=C['dark'])
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_4_first_iso_diagram")

def aa_03_5():  # Kernel and image
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "Kernel and Image of a Homomorphism",
          "$\\ker(f) = f^{-1}(e)$ in $G$,  $\\mathrm{im}(f) = f(G)$ in $H$")
    # G with kernel highlighted
    rb(a, 8, 18, 50, 38, C['gray'], '', alpha=0.1)
    a.text(33, 60, "$G$", fontsize=18, color=C['gray'], ha='center', fontweight='bold')
    rb(a, 14, 30, 16, 16, C['red'], '$\\ker(f)$', 14, alpha=0.7)
    rb(a, 35, 30, 16, 16, C['blue'], '$g$', 14, alpha=0.7)
    # H with image highlighted
    rb(a, 75, 18, 55, 38, C['gray'], '', alpha=0.1)
    a.text(102, 60, "$H$", fontsize=18, color=C['gray'], ha='center', fontweight='bold')
    rb(a, 80, 38, 16, 8, C['amber'], '$e_H$', 12, alpha=0.85)
    rb(a, 100, 30, 25, 16, C['green'], '$\\mathrm{im}(f)$', 14, alpha=0.7)
    ar(a, 22, 38, 88, 42, col=C['red'], lw=2)
    a.text(55, 47, "$\\ker \\to e_H$", ha='center', fontsize=11, color=C['red'])
    ar(a, 43, 38, 113, 38, col=C['blue'], lw=2)
    a.text(78, 33, "$g \\mapsto f(g)$", ha='center', fontsize=11, color=C['blue'])
    a.text(70, 8, "$f$ is injective $\\Leftrightarrow \\ker(f) = \\{e_G\\}$;   $f$ is surjective $\\Leftrightarrow \\mathrm{im}(f) = H$",
           ha='center', fontsize=11, color=C['dark'])
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_5_kernel_image")

def aa_03_6():  # Homomorphism types
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "Three Types of Homomorphisms",
          "Injective (mono), surjective (epi), bijective (iso)")
    types = [
        ("Injective", "1-to-1, $\\ker = \\{e\\}$", C['blue'], 5),
        ("Surjective", "Onto, $\\mathrm{im} = H$", C['green'], 50),
        ("Isomorphism", "Both, $G \\cong H$", C['red'], 95),
    ]
    for lab, desc, col, x0 in types:
        rb(a, x0, 25, 18, 22, col, lab, 13, alpha=0.9)
        rb(a, x0+22, 25, 18, 22, C['gray'], '', alpha=0.2)
        a.text(x0+9, 55, "$G$", ha='center', fontsize=14, color=col, fontweight='bold')
        a.text(x0+31, 55, "$H$", ha='center', fontsize=14, color=C['gray'], fontweight='bold')
        # Arrows showing type
        if lab == "Injective":
            for k in range(3):
                ar(a, x0+18, 32+k*5, x0+22, 32+k*5, lw=1.5)
        elif lab == "Surjective":
            for k in range(3):
                ar(a, x0+18, 30+k*4, x0+22, 33+k*3, lw=1.5)
            ar(a, x0+18, 42, x0+22, 33, lw=1.5)
        else:
            for k in range(4):
                ar(a, x0+18, 30+k*4, x0+22, 30+k*4, lw=1.5)
        a.text(x0+19, 17, desc, ha='center', fontsize=10, color=col, style='italic')
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_6_homo_types")

def aa_03_7():  # Correspondence theorem
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "The Correspondence Theorem",
          "Subgroups of $G$ containing $N$ ↔ Subgroups of $G/N$")
    # Left lattice
    rb(a, 5, 50, 14, 8, C['blue'], '$G$', 16)
    rb(a, 5, 32, 14, 8, C['blue'], '$H$', 14)
    rb(a, 5, 14, 14, 8, C['blue'], '$N$', 14)
    a.plot([12,12], [40,50], color=C['gray'], lw=2)
    a.plot([12,12], [22,32], color=C['gray'], lw=2)
    # Right lattice
    rb(a, 95, 50, 26, 8, C['green'], '$G/N$', 16)
    rb(a, 95, 32, 26, 8, C['green'], '$H/N$', 14)
    rb(a, 95, 14, 26, 8, C['green'], '$\\{e\\}$', 14)
    a.plot([108,108], [40,50], color=C['gray'], lw=2)
    a.plot([108,108], [22,32], color=C['gray'], lw=2)
    # Bijection arrow
    ar(a, 22, 36, 92, 36, lw=2.5, col=C['red'])
    ar(a, 92, 32, 22, 32, lw=2.5, col=C['red'])
    a.text(57, 42, "bijection", ha='center', fontsize=13, color=C['red'],
           fontweight='bold')
    a.text(57, 28, "$H \\leftrightarrow H/N$", ha='center', fontsize=13, color=C['red'])
    a.text(70, 8, "Normal subgroups correspond to normal subgroups; $H/N \\triangleleft G/N \\Leftrightarrow H \\triangleleft G$.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "03-quotient-groups-and-homomorphisms", "aa_v2_03_7_correspondence")

# ===== Article 04: Sylow =====

def aa_04_1():  # p-group chain
    f,a = mk(11,8); a.set_xlim(0,100); a.set_ylim(0,80)
    title(a, "Nested Subgroup Chain in a $p$-Group",
          "Every $p$-group has subgroups of every order $p^k$ dividing $|G|$")
    # Chain of nested boxes
    sizes = [(50, 5, 40, 65, "$\\{e\\}$", C['gray']),
             (40, 12, 50, 50, "$|H_1| = p$", C['blue']),
             (30, 20, 60, 35, "$|H_2| = p^2$", C['green']),
             (20, 28, 70, 22, "$|H_3| = p^3$", C['amber']),
             (10, 36, 80, 12, "$|G| = p^4$", C['red'])]
    for x, y, w, h, lb, col in sizes:
        rb(a, x, y, w, h, col, '', alpha=0.6)
        a.text(x+w/2, y+h-2, lb, ha='center', va='top', fontsize=12,
               color='white' if col != C['gray'] else C['dark'], fontweight='bold')
    a.text(50, 5, "Each step: $|H_{k+1}|/|H_k| = p$, with $H_k \\triangleleft H_{k+1}$",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_1_p_group_chain")

def aa_04_2():  # Sylow count
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,75)
    title(a, "Constraints on $n_p$ (Number of Sylow $p$-Subgroups)",
          "Theorem: $n_p \\equiv 1 (\\,\\mathrm{mod}\\,\\, p)$ and $n_p \\mid |G|/p^k$")
    rows = [("Group", "$|G|$", "$p$", "$n_p \\equiv 1$", "$n_p \\mid m$", "Possible $n_p$"),
            ("$\\mathbb{Z}/15\\mathbb{Z}$", "15", "3", "1, 4, 7,...", "1, 5", "1"),
            ("$S_3$", "6", "3", "1, 4,...", "1, 2", "1"),
            ("$A_4$", "12", "3", "1, 4,...", "1, 2, 4", "1, 4"),
            ("$S_4$", "24", "3", "1, 4,...", "1, 2, 4, 8", "1, 4")]
    for i, row in enumerate(rows):
        y = 60 - i*9
        for j, val in enumerate(row):
            x = 8 + j*15
            col = C['dark'] if i==0 else C['gray']
            fs = 11 if i==0 else 10
            fw = 'bold' if i==0 else 'normal'
            a.text(x+7, y, val, ha='center', fontsize=fs, color=col, fontweight=fw)
        if i > 0:
            a.plot([5, 95], [y+3, y+3], '-', color=C['light'], alpha=0.5, lw=0.5)
    a.text(50, 6, "$n_p = 1$ means the Sylow subgroup is normal — a strong structural constraint.",
           ha='center', fontsize=11, color=C['red'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_2_sylow_count")

def aa_04_3():  # S4 Sylow subgroups
    f,a = mk(13,7); a.set_xlim(0,140); a.set_ylim(0,70)
    title(a, "Sylow Subgroups of $S_4$ (order 24 = $2^3 \\cdot 3$)",
          "$n_2 = 3$ Sylow 2-subgroups (each $\\cong D_4$), $n_3 = 4$ Sylow 3-subgroups")
    # Sylow 2 (3 of them)
    a.text(35, 62, "Three Sylow 2-subgroups (order 8)", ha='center',
           fontsize=12, color=C['blue'], fontweight='bold')
    syl2 = [("$P_1$", 12), ("$P_2$", 35), ("$P_3$", 58)]
    for lb, x in syl2:
        rb(a, x, 30, 18, 22, C['blue'], lb, 16, alpha=0.85)
        a.text(x+9, 25, "$\\cong D_4$", ha='center', fontsize=10,
               color=C['blue'], style='italic')
    # Sylow 3 (4 of them)
    a.text(105, 62, "Four Sylow 3-subgroups (order 3)", ha='center',
           fontsize=12, color=C['green'], fontweight='bold')
    syl3 = [("$Q_1$", 80), ("$Q_2$", 95), ("$Q_3$", 110), ("$Q_4$", 125)]
    for lb, x in syl3:
        rb(a, x, 35, 11, 18, C['green'], lb, 14, alpha=0.85)
        a.text(x+5.5, 30, "$\\cong \\mathbb{Z}/3\\mathbb{Z}$", ha='center', fontsize=9,
               color=C['green'])
    a.text(70, 12, "All Sylow $p$-subgroups for fixed $p$ are conjugate; together they generate $G$.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_3_s4_sylow")

def aa_04_4():  # Conjugacy action on Sylow
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Conjugation Acts Transitively on Sylow $p$-Subgroups",
          "$\\mathrm{Syl}_p(G) = \\{P_1, P_2, ..., P_{n_p}\\}$ all conjugate")
    # Sylow subgroups in a circle
    n = 4
    R = 18
    cx, cy = 50, 38
    for k in range(n):
        ang = np.pi/2 + k*2*np.pi/n
        x = cx + R*np.cos(ang); y = cy + R*np.sin(ang)
        rb(a, x-7, y-4, 14, 8, C['blue'], f"$P_{k+1}$", 14, alpha=0.85)
    # Center: conjugation arrows
    for k in range(n):
        a1 = np.pi/2 + k*2*np.pi/n
        a2 = np.pi/2 + ((k+1)%n)*2*np.pi/n
        x1 = cx + (R-3)*np.cos(a1); y1 = cy + (R-3)*np.sin(a1)
        x2 = cx + (R-3)*np.cos(a2); y2 = cy + (R-3)*np.sin(a2)
        a.annotate('', xy=(x2,y2), xytext=(x1,y1),
                   arrowprops=dict(arrowstyle='-|>', color=C['amber'], lw=1.5,
                                   connectionstyle="arc3,rad=-0.4", mutation_scale=14))
    a.text(cx, cy, "$g P g^{-1}$", ha='center', va='center', fontsize=14,
           color=C['dark'], fontweight='bold')
    a.text(50, 8, "Number of Sylow $p$-subgroups = $|G : N_G(P)|$",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_4_conjugacy_action")

def aa_04_5():  # Classify order 15
    f,a = mk(12,8); a.set_xlim(0,100); a.set_ylim(0,80)
    title(a, "Classifying Groups of Order 15",
          "Sylow analysis forces $G \\cong \\mathbb{Z}/15\\mathbb{Z}$")
    steps = [
        ("$|G| = 15 = 3 \\cdot 5$", C['gray']),
        ("$n_3 \\equiv 1 (\\,\\mathrm{mod}\\,\\, 3)$ and $n_3 \\mid 5$  →  $n_3 = 1$", C['blue']),
        ("$n_5 \\equiv 1 (\\,\\mathrm{mod}\\,\\, 5)$ and $n_5 \\mid 3$  →  $n_5 = 1$", C['green']),
        ("Both Sylow subgroups are normal", C['amber']),
        ("$G \\cong \\mathbb{Z}/3 \\times \\mathbb{Z}/5 \\cong \\mathbb{Z}/15\\mathbb{Z}$", C['red']),
    ]
    for i, (txt, col) in enumerate(steps):
        y = 65 - i*11
        rb(a, 10, y, 80, 8, col, txt, 12)
        if i < len(steps)-1:
            ar(a, 50, y, 50, y-3, col=C['gray'])
    a.text(50, 5, "Same logic: every group of order $pq$ with $p < q$ and $p \\nmid q-1$ is cyclic.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_5_classify_15")

def aa_04_6():  # Normalizer
    f,a = mk(12,7); a.set_xlim(0,100); a.set_ylim(0,70)
    title(a, "Normalizer $N_G(P)$ Counts Sylow Subgroups",
          "$n_p = |G : N_G(P)|$")
    # Concentric structure: G > N_G(P) > P
    rb(a, 10, 15, 80, 45, C['gray'], '', alpha=0.15)
    a.text(50, 56, "$G$", fontsize=18, color=C['gray'], ha='center', fontweight='bold')
    rb(a, 22, 22, 56, 30, C['blue'], '', alpha=0.4)
    a.text(50, 47, "$N_G(P)$", fontsize=15, color=C['blue'], ha='center', fontweight='bold')
    rb(a, 35, 28, 30, 18, C['red'], 'Sylow $P$', 14)
    a.text(50, 8, "$|G| / |N_G(P)|$ counts conjugates of $P$, which is exactly $n_p$.",
           ha='center', fontsize=11, color=C['dark'])
    save(f, "04-sylow-theorems", "aa_v2_04_6_normalizer")

def aa_04_7():  # Simple group of order 60
    f,a = mk(12,8); a.set_xlim(0,100); a.set_ylim(0,80)
    title(a, "Simple Groups of Order 60: $A_5$ is the Unique Example",
          "Sylow analysis rules out a normal Sylow subgroup")
    rows = [("$|G| = 60 = 2^2 \\cdot 3 \\cdot 5$", C['gray']),
            ("$n_5 \\in \\{1, 6\\}$.  If $n_5 = 1$: normal $\\Rightarrow$ not simple", C['blue']),
            ("$n_5 = 6$: gives 24 elements of order 5", C['green']),
            ("$n_3 \\in \\{1, 4, 10\\}$. $n_3 \\geq 4$ adds $\\geq 8$ order-3 elements", C['amber']),
            ("Counting forces tight structure $\\Rightarrow G \\cong A_5$", C['red'])]
    for i, (txt, col) in enumerate(rows):
        y = 65 - i*11
        rb(a, 8, y, 84, 8, col, txt, 12)
    a.text(50, 5, "$A_5$ is the smallest non-abelian simple group — the start of the classification of finite simple groups.",
           ha='center', fontsize=11, color=C['gray'], style='italic')
    save(f, "04-sylow-theorems", "aa_v2_04_7_simple_60")

def main():
    # Article 01
    aa_01_1(); aa_01_2(); aa_01_3(); aa_01_4(); aa_01_5(); aa_01_6(); aa_01_7()
    # Article 02
    aa_02_1(); aa_02_2(); aa_02_3(); aa_02_4(); aa_02_5(); aa_02_6(); aa_02_7()
    # Article 03
    aa_03_1(); aa_03_2(); aa_03_3(); aa_03_4(); aa_03_5(); aa_03_6(); aa_03_7()
    # Article 04
    aa_04_1(); aa_04_2(); aa_04_3(); aa_04_4(); aa_04_5(); aa_04_6(); aa_04_7()
    print("Part 1: 28 figures generated")

if __name__ == "__main__":
    main()
