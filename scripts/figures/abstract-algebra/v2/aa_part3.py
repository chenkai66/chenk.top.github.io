#!/usr/bin/env python3
"""AA figures part 3: articles 09-12 (28 figures)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Wedge, Ellipse
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np

BG="#fdfcf9"
C={"red":"#e85d4a","amber":"#f5a834","purple":"#8b5cf6","blue":"#3b82f6",
   "green":"#10b981","gray":"#6b7280","dark":"#1f2937","light":"#e5e7eb"}
shadow=withSimplePatchShadow(offset=(1.2,-1.2),shadow_rgbFace="#000000",alpha=0.15)
OUT="/tmp/aa_figs_v2"

def mk(w=12,h=7):
    f,a=plt.subplots(figsize=(w,h));f.patch.set_facecolor(BG)
    a.set_facecolor(BG);a.axis('off');return f,a
def rb(ax,x,y,w,h,col,t='',fs=12,tc='white',alpha=1):
    b=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=0.6",
                     fc=col,ec='none',alpha=alpha,lw=0)
    b.set_path_effects([shadow]);ax.add_patch(b)
    if t:ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold')
def ar(ax,x1,y1,x2,y2,col=C['gray'],lw=2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,mutation_scale=18))
def title(ax,t,sub=''):
    ax.text(0.5,0.97,t,transform=ax.transAxes,ha='center',va='top',
            fontsize=15,fontweight='bold',color=C['dark'])
    if sub:ax.text(0.5,0.93,sub,transform=ax.transAxes,ha='center',va='top',
                   fontsize=11,color=C['gray'],style='italic')
def save(fig,slug,name):
    d=f"{OUT}/{slug}";os.makedirs(d,exist_ok=True)
    fig.savefig(f"{d}/{name}.png",dpi=180,bbox_inches='tight',facecolor=BG,pad_inches=0.15)
    plt.close(fig)

# ===== Article 09: Modules =====
def aa_09_1():
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"Examples of Modules","Replacing fields with rings expands the structures we can study")
    types=[("$\\mathbb{Z}$-modules","abelian groups",C['blue'],"$\\mathbb{Z}/n\\mathbb{Z},\\, \\mathbb{Z}^n$"),
           ("$F[x]$-modules","vector spaces with linear operator",C['green'],"$V$ with operator $T$"),
           ("$F$-modules","vector spaces over field $F$",C['amber'],"$F^n$"),
           ("Ideals","submodules of $R$ as $R$-module",C['red'],"$(p) \\subset \\mathbb{Z}$")]
    for i,(t,d,col,e) in enumerate(types):
        x=8+(i%2)*65;y=42-(i//2)*22
        rb(a,x,y,55,16,col,'',alpha=0.85)
        a.text(x+27,y+12,t,ha='center',fontsize=14,color='white',fontweight='bold')
        a.text(x+27,y+8,d,ha='center',fontsize=10,color='white')
        a.text(x+27,y+3,e,ha='center',fontsize=11,color='white',fontweight='bold')
    save(f,"09-modules","aa_v2_09_1_module_examples")

def aa_09_2():
    f,a=mk(11,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Free Module $R^n$ with Standard Basis","Every element is a unique $R$-combination")
    rb(a,15,30,70,25,C['blue'],'',alpha=0.3)
    a.text(50,55,"$R^3 = R e_1 \\oplus R e_2 \\oplus R e_3$",ha='center',fontsize=14,color=C['blue'],fontweight='bold')
    bases=[(25,"$e_1$",C['red']),(50,"$e_2$",C['green']),(75,"$e_3$",C['amber'])]
    for x,lab,col in bases:
        cir=Circle((x,40),3.5,fc=col,ec='none')
        cir.set_path_effects([shadow]);a.add_patch(cir)
        a.text(x,40,lab,ha='center',va='center',fontsize=14,color='white',fontweight='bold')
    a.text(50,15,"General element: $r_1 e_1 + r_2 e_2 + r_3 e_3$ with $r_i \\in R$",
           ha='center',fontsize=12,color=C['dark'])
    save(f,"09-modules","aa_v2_09_2_free_module")

def aa_09_3():
    f,a=mk(11,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Torsion: When Multiplication by $r$ Annihilates",
          "$M = \\mathbb{Z}/4\\mathbb{Z}$ as a $\\mathbb{Z}$-module: $4 \\cdot x = 0$")
    rb(a,8,40,84,18,C['blue'],'',alpha=0.4)
    a.text(50,52,"$M = \\mathbb{Z}/4\\mathbb{Z}$",ha='center',fontsize=15,color=C['blue'],fontweight='bold')
    elts=[(20,'$0$'),(40,'$1$'),(60,'$2$'),(80,'$3$')]
    for x,lab in elts:
        rb(a,x-5,42,10,8,C['blue'],lab,12)
    ar(a,50,38,50,28,col=C['red'])
    rb(a,30,15,40,12,C['red'],'$4 \\cdot x = 0$ for all $x$',13)
    a.text(50,5,"Torsion submodule = $\\{m : rm = 0$ for some $r \\neq 0\\}$",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"09-modules","aa_v2_09_3_torsion")

def aa_09_4():
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"Structure Theorem for Finitely Generated Modules over a PID",
          "Every such module decomposes uniquely")
    rb(a,10,50,28,16,C['blue'],'$M$',18)
    a.text(24,46,"f.g. over PID",ha='center',fontsize=10,color=C['blue'],style='italic')
    rb(a,55,55,30,8,C['green'],'$R^r$ (free)',13)
    rb(a,55,40,35,8,C['amber'],'$R/(p_1^{a_1})$',13)
    rb(a,95,40,35,8,C['amber'],'$\\cdots$',13)
    rb(a,55,25,35,8,C['amber'],'$R/(p_n^{a_n})$',13)
    a.text(50,60,"$\\oplus$",fontsize=20,color=C['gray'],ha='center')
    a.text(50,44,"$\\oplus$",fontsize=20,color=C['gray'],ha='center')
    a.text(50,29,"$\\oplus$",fontsize=20,color=C['gray'],ha='center')
    ar(a,38,58,55,58)
    a.text(70,15,"Free part rank $r$ + torsion: invariant factor / elementary divisor form",
           ha='center',fontsize=11,color=C['dark'])
    a.text(70,8,"Specializes to fund. theorem of f.g. abelian groups when $R = \\mathbb{Z}$.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"09-modules","aa_v2_09_4_structure_thm")

def aa_09_5():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Smith Normal Form","Reduce a matrix over a PID to invariant factor diagonal")
    a.text(15,50,"$A = (6,\\,4;\\,8,\\,6)$",
           ha='center',fontsize=15,color=C['blue'])
    ar(a,30,50,50,50,col=C['red'])
    a.text(40,55,"row/col ops",fontsize=11,color=C['red'])
    a.text(70,50,"$(2,\\,0;\\,0,\\,6)$",
           ha='center',fontsize=15,color=C['green'])
    a.text(50,28,"Invariant factors: $d_1 = 2,\\, d_2 = 6$ with $d_1 \\mid d_2$",
           ha='center',fontsize=12,color=C['dark'])
    a.text(50,18,"$\\mathbb{Z}^2 / A\\mathbb{Z}^2 \\cong \\mathbb{Z}/2 \\oplus \\mathbb{Z}/6$",
           ha='center',fontsize=13,color=C['amber'],fontweight='bold')
    a.text(50,7,"Smith normal form is the algorithmic engine behind the structure theorem.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"09-modules","aa_v2_09_5_smith_normal")

def aa_09_6():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Jordan Normal Form via $F[x]$-Module Structure",
          "View vector space $V$ as $F[x]$-module with $x$ acting as $T$")
    rb(a,8,40,28,18,C['blue'],'$(V, T)$',16)
    a.text(22,36,"f.d. over $F$",ha='center',fontsize=10,color=C['blue'],style='italic')
    rb(a,55,40,38,18,C['green'],'$F[x]$-module',14)
    a.text(74,36,"$x \\cdot v := T v$",ha='center',fontsize=11,color=C['green'])
    ar(a,36,49,55,49)
    rb(a,15,12,70,16,C['red'],'',alpha=0.85)
    a.text(50,22,"Jordan form $= F[x]/(x-\\lambda)^k$ summands",
           ha='center',fontsize=13,color='white',fontweight='bold')
    a.text(50,16,"Each block ↔ elementary divisor of $T$",
           ha='center',fontsize=11,color='white')
    save(f,"09-modules","aa_v2_09_6_jordan")

def aa_09_7():
    f,a=mk(13,6);a.set_xlim(0,140);a.set_ylim(0,55)
    title(a,"Short Exact Sequence and Splitting","$0 \\to A \\to B \\to C \\to 0$")
    rb(a,5,28,18,10,C['gray'],'$0$',14)
    rb(a,28,28,18,10,C['blue'],'$A$',16)
    rb(a,52,28,18,10,C['green'],'$B$',16)
    rb(a,76,28,18,10,C['amber'],'$C$',16)
    rb(a,99,28,18,10,C['gray'],'$0$',14)
    for i in range(4):
        ar(a,5+i*24+18,33,5+(i+1)*24,33)
    a.text(50,42,"$f$",fontsize=14,color=C['dark'],fontweight='bold',ha='center')
    a.text(74,42,"$g$",fontsize=14,color=C['dark'],fontweight='bold',ha='center')
    a.text(60,15,"$f$ injective, $g$ surjective, $\\ker(g) = \\mathrm{im}(f)$",
           ha='center',fontsize=11,color=C['gray'])
    a.text(60,7,"Splits ⟺ $B \\cong A \\oplus C$",
           ha='center',fontsize=12,color=C['red'],fontweight='bold')
    save(f,"09-modules","aa_v2_09_7_exact_seq")

# ===== Article 10: Representation Theory =====
def aa_10_1():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Representation $\\rho: G \\to \\mathrm{GL}(V)$",
          "Each group element becomes a linear transformation")
    rb(a,8,30,22,16,C['blue'],'$G$',18)
    rb(a,70,30,22,16,C['green'],'$\\mathrm{GL}(V)$',16)
    ar(a,30,38,70,38,lw=2.5)
    a.text(50,42,"$\\rho$",fontsize=20,color=C['dark'],fontweight='bold')
    a.text(50,30,"matrix",fontsize=11,color=C['gray'],style='italic')
    a.text(19,26,"abstract",fontsize=10,color=C['blue'])
    a.text(81,26,"concrete",fontsize=10,color=C['green'])
    a.text(50,17,"Properties: $\\rho(gh) = \\rho(g)\\rho(h)$, $\\rho(e) = I$",
           ha='center',fontsize=12,color=C['dark'])
    a.text(50,8,"Faithful if $\\rho$ injective; degree $= \\dim V$",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"10-representation-theory","aa_v2_10_1_rep_def")

def aa_10_2():
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"Three Irreducible Representations of $S_3$",
          "Trivial, sign, and 2-dim standard")
    reps=[("Trivial","$g \\mapsto 1$","dim 1",C['blue']),
          ("Sign","$g \\mapsto \\mathrm{sgn}(g)$","dim 1",C['green']),
          ("Standard","2D rotation/reflection","dim 2",C['amber'])]
    for i,(n,m,d,col) in enumerate(reps):
        x=10+i*43
        rb(a,x,30,38,22,col,'',alpha=0.85)
        a.text(x+19,46,n,ha='center',fontsize=14,color='white',fontweight='bold')
        a.text(x+19,40,m,ha='center',fontsize=11,color='white')
        a.text(x+19,34,d,ha='center',fontsize=12,color='white',fontweight='bold')
    a.text(70,15,"$|S_3| = 1^2 + 1^2 + 2^2 = 6$ ✓",
           ha='center',fontsize=14,color=C['red'],fontweight='bold')
    save(f,"10-representation-theory","aa_v2_10_2_s3_irreps")

def aa_10_3():
    f,a=mk(11,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Character Table of $S_3$","Each row is a character; columns are conjugacy classes")
    headers=["","$e$","$(12)$","$(123)$"]
    rows=[("$\\chi_1$ (triv)","1","1","1",C['blue']),
          ("$\\chi_2$ (sign)","1","-1","1",C['green']),
          ("$\\chi_3$ (std)","2","0","-1",C['amber'])]
    cw=18;rh=8
    for j,h in enumerate(headers):
        rb(a,8+j*cw,55,cw-1,rh,C['dark'],h,12,tc='white')
    a.text(35,50,"size:",fontsize=10,color=C['gray'])
    sizes=["","1","3","2"]
    for j,s in enumerate(sizes):
        if s:a.text(8+j*cw+cw/2,46,f"({s})",ha='center',fontsize=10,color=C['gray'],style='italic')
    for i,(name,*vals,col) in enumerate(rows):
        y=35-i*8
        rb(a,8,y,cw-1,rh,col,name,11,alpha=0.85)
        for j,v in enumerate(vals):
            rb(a,8+(j+1)*cw,y,cw-1,rh,col,v,12,alpha=0.6)
    save(f,"10-representation-theory","aa_v2_10_3_character_table")

def aa_10_4():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Character Orthogonality Relations","Two ways to slice the character table")
    rb(a,5,40,40,20,C['blue'],'',alpha=0.4)
    a.text(25,55,"First Orthogonality",ha='center',fontsize=12,color=C['blue'],fontweight='bold')
    a.text(25,46,"$\\frac{1}{|G|}\\sum_g \\overline{\\chi_i(g)} \\chi_j(g) = \\delta_{ij}$",
           ha='center',fontsize=11,color='white')
    rb(a,55,40,40,20,C['green'],'',alpha=0.4)
    a.text(75,55,"Second Orthogonality",ha='center',fontsize=12,color=C['green'],fontweight='bold')
    a.text(75,46,"$\\sum_i \\overline{\\chi_i(g)} \\chi_i(h) = \\delta_{[g][h]} \\frac{|G|}{|[g]|}$",
           ha='center',fontsize=11,color='white')
    a.text(50,25,"Rows are orthogonal under inner product weighted by $1/|G|$",
           ha='center',fontsize=11,color=C['dark'])
    a.text(50,15,"Columns are orthogonal weighted by class sizes",
           ha='center',fontsize=11,color=C['dark'])
    a.text(50,5,"$\\Rightarrow$ #irreducibles = #conjugacy classes",
           ha='center',fontsize=12,color=C['red'],fontweight='bold')
    save(f,"10-representation-theory","aa_v2_10_4_orthogonality")

def aa_10_5():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Decomposition of the Regular Representation",
          "Each irrep $V_i$ appears with multiplicity $\\dim V_i$")
    rb(a,8,40,18,18,C['dark'],'$\\mathbb{C}[G]$',16)
    a.text(17,36,"regular rep",ha='center',fontsize=10,color=C['dark'],style='italic')
    ar(a,26,49,40,49)
    rb(a,40,50,12,8,C['blue'],'$V_1$',12)
    rb(a,55,50,12,8,C['green'],'$V_2$',12)
    rb(a,70,50,12,8,C['amber'],'$V_3$',12)
    a.text(53,40,"with multiplicities",ha='center',fontsize=10,color=C['gray'])
    a.text(46,33,"$\\dim V_1$",ha='center',fontsize=11,color=C['blue'])
    a.text(61,33,"$\\dim V_2$",ha='center',fontsize=11,color=C['green'])
    a.text(76,33,"$\\dim V_3$",ha='center',fontsize=11,color=C['amber'])
    a.text(50,15,"$|G| = \\sum (\\dim V_i)^2$",
           ha='center',fontsize=14,color=C['red'],fontweight='bold')
    save(f,"10-representation-theory","aa_v2_10_5_regular_decomp")

def aa_10_6():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Schur's Lemma",
          "Morphisms between irreducible representations are scalar multiples of identity")
    rb(a,8,30,28,16,C['blue'],'$V$',18)
    a.text(22,26,"irreducible",ha='center',fontsize=10,color=C['blue'],style='italic')
    rb(a,64,30,28,16,C['green'],'$W$',18)
    a.text(78,26,"irreducible",ha='center',fontsize=10,color=C['green'],style='italic')
    ar(a,36,38,64,38,lw=2.5)
    a.text(50,42,"$\\varphi$ G-equivariant",fontsize=11,color=C['dark'])
    a.text(50,15,"$V \\cong W$: $\\varphi$ is multiplication by a scalar $\\lambda$",
           ha='center',fontsize=12,color=C['red'])
    a.text(50,8,"$V \\not\\cong W$: $\\varphi = 0$",
           ha='center',fontsize=12,color=C['red'],fontweight='bold')
    save(f,"10-representation-theory","aa_v2_10_6_schur")

def aa_10_7():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"$\\mathrm{SU}(2)$ Representations and Quantum Spin",
          "Irreducibles indexed by half-integers $j = 0, 1/2, 1, 3/2, \\ldots$")
    spins=[(0,"$j=0$","singlet",C['gray']),(1/2,"$j=1/2$","spin-1/2",C['blue']),
           (1,"$j=1$","spin-1",C['green']),(3/2,"$j=3/2$","spin-3/2",C['amber'])]
    for i,(j,lab,desc,col) in enumerate(spins):
        x=10+i*22;dim=int(2*j+1)
        rb(a,x,35,18,15,col,'',alpha=0.85)
        a.text(x+9,46,lab,ha='center',fontsize=13,color='white',fontweight='bold')
        a.text(x+9,40,f"dim {dim}",ha='center',fontsize=11,color='white')
        a.text(x+9,30,desc,ha='center',fontsize=10,color=col)
    a.text(50,15,"Photon ($j=1$), electron ($j=1/2$), graviton ($j=2$)",
           ha='center',fontsize=11,color=C['dark'])
    a.text(50,8,"Tensor products of irreps decompose via Clebsch-Gordan rule.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"10-representation-theory","aa_v2_10_7_su2_spin")

# ===== Article 11: Category Theory =====
def aa_11_1():
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"Three Categories: $\\mathbf{Set}$, $\\mathbf{Grp}$, $\\mathbf{Top}$",
          "Different objects, different morphisms")
    cats=[("$\\mathbf{Set}$","sets","functions",C['blue']),
          ("$\\mathbf{Grp}$","groups","group homomorphisms",C['green']),
          ("$\\mathbf{Top}$","topological spaces","continuous maps",C['amber'])]
    for i,(n,o,m,col) in enumerate(cats):
        x=10+i*43
        rb(a,x,30,38,30,col,'',alpha=0.85)
        a.text(x+19,53,n,ha='center',fontsize=20,color='white',fontweight='bold')
        a.text(x+19,44,"objects:",ha='center',fontsize=10,color='white')
        a.text(x+19,40,o,ha='center',fontsize=12,color='white',fontweight='bold')
        a.text(x+19,36,"morphisms:",ha='center',fontsize=10,color='white')
        a.text(x+19,32,m,ha='center',fontsize=11,color='white',fontweight='bold')
    a.text(70,15,"Each category has identity morphisms and associative composition.",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"11-category-theory","aa_v2_11_1_categories")

def aa_11_2():
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"Functor $F: \\mathcal{C} \\to \\mathcal{D}$",
          "Sends objects to objects, morphisms to morphisms, preserves composition")
    rb(a,8,30,40,30,C['blue'],'$\\mathcal{C}$',24,alpha=0.4)
    rb(a,90,30,40,30,C['green'],'$\\mathcal{D}$',24,alpha=0.4)
    a.plot([16,16,32],[42,52,52],'-',color='white',lw=2)
    a.plot([16,16,32],[38,28,28],'-',color='white',lw=2)
    a.plot([100,100,116],[42,52,52],'-',color='white',lw=2)
    a.plot([100,100,116],[38,28,28],'-',color='white',lw=2)
    cir=Circle((20,52),2,fc=C['blue'],ec='none');a.add_patch(cir)
    cir=Circle((20,28),2,fc=C['blue'],ec='none');a.add_patch(cir)
    cir=Circle((104,52),2,fc=C['green'],ec='none');a.add_patch(cir)
    cir=Circle((104,28),2,fc=C['green'],ec='none');a.add_patch(cir)
    a.text(20,57,"$X$",fontsize=12,color='white',fontweight='bold',ha='center')
    a.text(20,23,"$Y$",fontsize=12,color='white',fontweight='bold',ha='center')
    a.text(104,57,"$F(X)$",fontsize=11,color='white',fontweight='bold',ha='center')
    a.text(104,23,"$F(Y)$",fontsize=11,color='white',fontweight='bold',ha='center')
    ar(a,48,40,90,40,lw=2.5)
    a.text(70,44,"$F$",fontsize=18,color=C['dark'],fontweight='bold')
    a.text(70,8,"Examples: forgetful $\\mathbf{Grp} \\to \\mathbf{Set}$, free $\\mathbf{Set} \\to \\mathbf{Grp}$, $\\pi_1: \\mathbf{Top}_* \\to \\mathbf{Grp}$",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"11-category-theory","aa_v2_11_2_functor")

def aa_11_3():
    f,a=mk(11,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Natural Transformation $\\eta: F \\Rightarrow G$",
          "A square commutes for every morphism $f$")
    rb(a,15,42,18,10,C['blue'],'$F(X)$',13)
    rb(a,67,42,18,10,C['blue'],'$F(Y)$',13)
    rb(a,15,18,18,10,C['green'],'$G(X)$',13)
    rb(a,67,18,18,10,C['green'],'$G(Y)$',13)
    ar(a,33,47,67,47);a.text(50,51,"$F(f)$",ha='center',fontsize=11,color=C['dark'])
    ar(a,33,23,67,23);a.text(50,15,"$G(f)$",ha='center',fontsize=11,color=C['dark'])
    ar(a,24,42,24,28);a.text(28,35,"$\\eta_X$",fontsize=11,color=C['red'])
    ar(a,76,42,76,28);a.text(80,35,"$\\eta_Y$",fontsize=11,color=C['red'])
    a.text(50,8,"$\\eta_Y \\circ F(f) = G(f) \\circ \\eta_X$ for all morphisms $f$",
           ha='center',fontsize=12,color=C['gray'])
    save(f,"11-category-theory","aa_v2_11_3_natural_trans")

def aa_11_4():
    f,a=mk(12,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Universal Property of the Product",
          "$X \\times Y$ is the universal cone over $\\{X, Y\\}$")
    rb(a,40,60,20,10,C['red'],'$Z$',16)
    rb(a,15,30,20,10,C['blue'],'$X$',16)
    rb(a,65,30,20,10,C['blue'],'$Y$',16)
    rb(a,38,5,24,10,C['green'],'$X \\times Y$',14)
    ar(a,45,60,25,40);a.text(28,52,"$f$",fontsize=12,color=C['dark'],fontweight='bold')
    ar(a,55,60,75,40);a.text(72,52,"$g$",fontsize=12,color=C['dark'],fontweight='bold')
    ar(a,50,60,50,15,col=C['red']);a.text(53,38,"$\\exists ! \\, h$",fontsize=12,color=C['red'],fontweight='bold')
    ar(a,45,15,25,30,col=C['gray']);a.text(28,21,"$\\pi_X$",fontsize=11,color=C['gray'])
    ar(a,55,15,75,30,col=C['gray']);a.text(72,21,"$\\pi_Y$",fontsize=11,color=C['gray'])
    save(f,"11-category-theory","aa_v2_11_4_universal")

def aa_11_5():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Free-Forgetful Adjunction $F \\dashv U$",
          "$\\mathrm{Hom}_{\\mathbf{Grp}}(F(S), G) \\cong \\mathrm{Hom}_{\\mathbf{Set}}(S, U(G))$")
    rb(a,10,30,30,18,C['blue'],'$\\mathbf{Set}$',18,alpha=0.5)
    rb(a,60,30,30,18,C['green'],'$\\mathbf{Grp}$',18,alpha=0.5)
    a.annotate('',xy=(60,42),xytext=(40,42),
               arrowprops=dict(arrowstyle='-|>',color=C['red'],lw=2.5,mutation_scale=18))
    a.annotate('',xy=(40,36),xytext=(60,36),
               arrowprops=dict(arrowstyle='-|>',color=C['amber'],lw=2.5,mutation_scale=18))
    a.text(50,46,"$F$ (free)",fontsize=12,color=C['red'],fontweight='bold',ha='center')
    a.text(50,32,"$U$ (forget)",fontsize=12,color=C['amber'],fontweight='bold',ha='center')
    a.text(50,15,"$F$ is left adjoint to $U$ ($F \\dashv U$)",
           ha='center',fontsize=13,color=C['dark'],fontweight='bold')
    a.text(50,8,"Free group on $S$ is the most efficient group containing $S$.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"11-category-theory","aa_v2_11_5_adjoint")

def aa_11_6():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Yoneda Lemma","An object is determined by its hom-functor")
    rb(a,8,38,28,14,C['blue'],'$X \\in \\mathcal{C}$',16)
    ar(a,36,45,55,45,lw=2.5)
    rb(a,55,38,40,14,C['green'],'$\\mathrm{Hom}(-, X)$',14)
    a.text(75,55,"functor $\\mathcal{C}^{op} \\to \\mathbf{Set}$",ha='center',
           fontsize=10,color=C['gray'],style='italic')
    a.text(50,28,"$\\mathrm{Nat}(\\mathrm{Hom}(-, X), F) \\cong F(X)$",
           ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    a.text(50,18,"In particular: $\\mathrm{Hom}(-, X) \\cong \\mathrm{Hom}(-, Y) \\Rightarrow X \\cong Y$",
           ha='center',fontsize=12,color=C['red'])
    a.text(50,8,"You are determined by your relationships with everyone else.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"11-category-theory","aa_v2_11_6_yoneda")

def aa_11_7():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Limits and Colimits","Universal cones (limits) and cocones (colimits)")
    rb(a,5,30,40,25,C['blue'],'',alpha=0.4)
    a.text(25,50,"Limit",fontsize=14,color=C['blue'],fontweight='bold',ha='center')
    a.text(25,42,"$\\lim_{\\leftarrow}$",fontsize=18,color='white',fontweight='bold',ha='center')
    a.text(25,35,"products, equalizers,\npullbacks, terminals",
           ha='center',fontsize=10,color='white')
    rb(a,55,30,40,25,C['red'],'',alpha=0.4)
    a.text(75,50,"Colimit",fontsize=14,color=C['red'],fontweight='bold',ha='center')
    a.text(75,42,"$\\lim_{\\rightarrow}$",fontsize=18,color='white',fontweight='bold',ha='center')
    a.text(75,35,"coproducts, coequalizers,\npushouts, initials",
           ha='center',fontsize=10,color='white')
    a.text(50,15,"Limits universally absorb cones from above; colimits universally factor cocones from below.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"11-category-theory","aa_v2_11_7_limits")

# ===== Article 12: Applications =====
def aa_12_1():
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"RSA Encryption Flow",
          "Based on Euler theorem: $a^{\\phi(n)} \\equiv 1 $ (mod $n$)")
    rb(a,8,55,28,12,C['blue'],'Key Generation',13)
    a.text(22,50,"$n=pq,\\, e,\\, d = e^{-1} \\,\\mathrm{mod}\\, \\phi(n)$",ha='center',fontsize=10,color=C['blue'])
    rb(a,8,30,28,12,C['green'],'Encrypt',13)
    a.text(22,25,"$c = m^e \\,\\mathrm{mod}\\, n$",ha='center',fontsize=11,color=C['green'])
    rb(a,8,5,28,12,C['amber'],'Decrypt',13)
    a.text(22,1,"$m = c^d \\,\\mathrm{mod}\\, n$",ha='center',fontsize=11,color=C['amber'])
    a.text(75,55,"public: $(n, e)$",fontsize=12,color=C['dark'])
    a.text(75,40,"$m → (public key) c$",fontsize=13,color=C['dark'])
    a.text(75,25,"private: $d$",fontsize=12,color=C['dark'])
    a.text(75,10,"$c → (private key) m$",fontsize=13,color=C['dark'])
    a.text(120,40,"Security:\nfactoring $n$\nis hard",ha='center',fontsize=11,color=C['red'],fontweight='bold')
    save(f,"12-applications","aa_v2_12_1_rsa")

def aa_12_2():
    f,a=mk(11,8);a.set_xlim(-3,3);a.set_ylim(-3,3)
    title(a,"Geometric Group Law on an Elliptic Curve",
          "$y^2 = x^3 - x + 1$: line through $P, Q$ meets curve at third point")
    xs=np.linspace(-2,2.5,400)
    for sign in [1,-1]:
        ys_sq=xs**3-xs+1
        valid=ys_sq>=0
        ys=sign*np.sqrt(np.maximum(ys_sq,0))
        a.plot(xs[valid],ys[valid],color=C['blue'],lw=2)
    P=(-1,1);Q=(0,1);R=(1,-1)  # P+Q = reflection of third intersection (1,1)
    pts=[(P,'$P$',C['red']),(Q,'$Q$',C['amber']),((1,1),'(third)',C['gray']),(R,'$P+Q$',C['green'])]
    for (x,y),lab,col in pts:
        cir=Circle((x,y),0.08,fc=col,ec='none')
        cir.set_path_effects([shadow]);a.add_patch(cir)
        a.text(x,y+0.18,lab,ha='center',fontsize=12,color=col,fontweight='bold')
    # line through P,Q (extended)
    a.plot([-2,2.5],[1,1],'--',color=C['gray'],lw=1.5,alpha=0.7)
    a.set_aspect('equal');a.axhline(0,color=C['gray'],lw=0.3,alpha=0.3)
    a.axvline(0,color=C['gray'],lw=0.3,alpha=0.3)
    a.text(0,-2.7,"Reflect the third intersection across $x$-axis to get $P + Q$",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"12-applications","aa_v2_12_2_elliptic_curve")

def aa_12_3():
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"ECDSA Signature Flow","Elliptic curve scalar multiplication is the trapdoor")
    steps=[("Sign","$k$ random,\n$r = (kG).x$,\n$s = k^{-1}(z + rd)$",C['blue']),
           ("Public output","$(r, s)$",C['amber']),
           ("Verify","$u_1 = z s^{-1}$,\n$u_2 = r s^{-1}$,\n$P = u_1 G + u_2 Q$",C['green']),
           ("Accept","$P.x = r$?",C['red'])]
    for i,(t,d,col) in enumerate(steps):
        x=8+i*32
        rb(a,x,28,28,28,col,'',alpha=0.85)
        a.text(x+14,52,t,ha='center',fontsize=14,color='white',fontweight='bold')
        a.text(x+14,42,d,ha='center',fontsize=10,color='white')
    for i in range(3):
        ar(a,8+i*32+28,42,8+(i+1)*32,42)
    a.text(70,12,"Tied to discrete log on elliptic curves: smaller keys, same security as RSA.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"12-applications","aa_v2_12_3_ecdsa")

def aa_12_4():
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Reed-Solomon Codes from Polynomial Evaluation",
          "Encode message as polynomial coefficients, send polynomial values")
    rb(a,8,40,25,16,C['blue'],'Message',13)
    a.text(20,36,"$m_0, m_1, \\ldots, m_{k-1}$",ha='center',fontsize=11,color=C['blue'])
    rb(a,40,40,25,16,C['green'],'Polynomial',13)
    a.text(53,36,"$p(x) = \\sum m_i x^i$",ha='center',fontsize=11,color=C['green'])
    rb(a,72,40,22,16,C['amber'],'Codeword',13)
    a.text(83,36,"$p(\\alpha_1), \\ldots, p(\\alpha_n)$",ha='center',fontsize=11,color=C['amber'])
    ar(a,33,48,40,48);ar(a,65,48,72,48)
    a.text(50,18,"Decoding corrects up to $\\lfloor (n-k)/2 \\rfloor$ errors via polynomial interpolation",
           ha='center',fontsize=12,color=C['dark'])
    a.text(50,8,"Used in: CDs, DVDs, QR codes, deep-space communication.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"12-applications","aa_v2_12_4_reed_solomon")

def aa_12_5():
    f,a=mk(11,8);a.set_xlim(0,21);a.set_ylim(0,21)
    title(a,"QR Code Algebraic Structure",
          "Reed-Solomon over $\\mathrm{GF}(256)$ provides error correction")
    np.random.seed(42)
    # Draw a 21x21 QR-like grid
    grid=np.random.choice([0,1],size=(21,21),p=[0.55,0.45])
    # Three corner finder patterns
    for cx,cy in [(0,14),(14,14),(0,0)]:
        for i in range(7):
            for j in range(7):
                if i in [0,6] or j in [0,6] or (1<i<5 and 1<j<5):
                    grid[cy+i,cx+j]=1
                else:grid[cy+i,cx+j]=0
    for i in range(21):
        for j in range(21):
            if grid[i,j]:
                rb(a,j,20-i,1,1,C['dark'],'',alpha=1)
    a.text(10,-1.5,"Reed-Solomon over $\\mathrm{GF}(256)$ corrects up to 30% data loss",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    a.set_aspect('equal')
    save(f,"12-applications","aa_v2_12_5_qr_code")

def aa_12_6():
    f,a=mk(11,8);a.set_xlim(-2.5,2.5);a.set_ylim(-2,2)
    title(a,"$\\mathrm{SU}(3)$ Flavor Symmetry: Quark Octet",
          "Eightfold way of Gell-Mann")
    quarks=[(0,1.5,"$n$","udd",C['blue']),
            (1,1.5,"$p$","uud",C['blue']),
            (-1.2,0,"$\\Sigma^-$","dds",C['green']),
            (1.2,0,"$\\Sigma^+$","uus",C['green']),
            (0,0,"$\\Sigma^0$, $\\Lambda$","uds, uds",C['amber']),
            (-0.6,-1.5,"$\\Xi^-$","dss",C['red']),
            (0.6,-1.5,"$\\Xi^0$","uss",C['red'])]
    for x,y,lab,c,col in quarks:
        cir=Circle((x,y),0.18,fc=col,ec='none')
        cir.set_path_effects([shadow]);a.add_patch(cir)
        a.text(x,y+0.32,lab,ha='center',fontsize=11,color=col,fontweight='bold')
        a.text(x,y-0.32,c,ha='center',fontsize=8,color=C['gray'],style='italic')
    # Hexagonal layout indicator
    a.plot([-1.2,1.2,1.2,-1.2,-1.2],[0,0,0,0,0],color=C['gray'],lw=0.5,alpha=0.3)
    a.set_aspect('equal');a.axis('on')
    a.set_xlabel("$T_3$ (isospin)",fontsize=10);a.set_ylabel("$Y$ (hypercharge)",fontsize=10)
    save(f,"12-applications","aa_v2_12_6_quark_su3")

def aa_12_7():
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"Wallpaper Groups: 17 Symmetry Types in 2D",
          "Combinations of translations, rotations, reflections, glides")
    # Show 4 sample patterns
    samples=[("p1","translation only",C['blue']),
             ("p4m","square + reflections",C['green']),
             ("p6m","hexagonal full",C['amber']),
             ("pgg","glide reflections",C['red'])]
    for i,(n,d,col) in enumerate(samples):
        x=8+i*32
        rb(a,x,30,28,22,col,'',alpha=0.85)
        a.text(x+14,46,n,ha='center',fontsize=18,color='white',fontweight='bold')
        a.text(x+14,38,d,ha='center',fontsize=10,color='white')
    a.text(70,15,"All 17 are accounted for by the crystallographic restriction theorem.",
           ha='center',fontsize=12,color=C['dark'])
    a.text(70,8,"Used in: Islamic art, Escher tilings, crystal classification.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"12-applications","aa_v2_12_7_wallpaper")

def main():
    aa_09_1();aa_09_2();aa_09_3();aa_09_4();aa_09_5();aa_09_6();aa_09_7()
    aa_10_1();aa_10_2();aa_10_3();aa_10_4();aa_10_5();aa_10_6();aa_10_7()
    aa_11_1();aa_11_2();aa_11_3();aa_11_4();aa_11_5();aa_11_6();aa_11_7()
    aa_12_1();aa_12_2();aa_12_3();aa_12_4();aa_12_5();aa_12_6();aa_12_7()
    print("Part 3: 28 figures generated")

if __name__=="__main__":main()
