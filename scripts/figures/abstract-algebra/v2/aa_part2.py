#!/usr/bin/env python3
"""AA figures part 2: articles 05-08 (28 figures)."""
import os, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Wedge
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

# ===== Article 05: Rings =====
def aa_05_1():  # Ring hierarchy
    f,a=mk(11,9);a.set_xlim(0,100);a.set_ylim(0,90)
    title(a,"Hierarchy of Ring Structures","From general rings down to fields")
    levels=[(80,"Rings",C['gray']),(68,"Commutative rings",C['blue']),
            (56,"Integral domains",C['green']),(44,"UFDs",C['amber']),
            (32,"PIDs",C['purple']),(20,"Euclidean domains",C['red']),(8,"Fields",C['dark'])]
    for y,lab,col in levels:
        rb(a,15,y,70,8,col,lab,13)
    for i in range(len(levels)-1):
        ar(a,50,levels[i][0],50,levels[i+1][0]+8,col=C['gray'])
    a.text(92,40,"$\\subseteq$",fontsize=20,color=C['gray'],ha='center',rotation=90)
    save(f,"05-rings-and-ideals","aa_v2_05_1_ring_hierarchy")

def aa_05_2():  # Z ideals
    f,a=mk(12,6);a.set_xlim(-1,13);a.set_ylim(-1,4)
    title(a,"Ideals of $\\mathbb{Z}$ as Multiples of $n$","$(n) = n\\mathbb{Z}$ is principal")
    ideals=[(1,C['blue']),(2,C['green']),(3,C['amber']),(4,C['red'])]
    for i,(n,col) in enumerate(ideals):
        y=2.5-i*0.7
        for k in range(-1,12):
            v=k*n
            if -1<=v<=12:
                cir=Circle((v,y),0.18,fc=col,ec='none')
                cir.set_path_effects([shadow]);a.add_patch(cir)
        a.text(-1.5,y,f"$({n})$",ha='right',va='center',fontsize=12,color=col,fontweight='bold')
    a.plot([-0.5,12.5],[2.5,2.5],'-',color=C['gray'],alpha=0.2,lw=0.5)
    a.text(6,-0.5,"$(1) \\supset (2) \\supset (4)$ but $(2)$ and $(3)$ are incomparable",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"05-rings-and-ideals","aa_v2_05_2_z_ideals")

def aa_05_3():  # Principal ideal in Z[x]
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Principal Ideal $(x^2+1) \\subset \\mathbb{Z}[x]$",
          "Multiples of $x^2+1$ form an ideal")
    rb(a,8,15,84,40,C['gray'],'',alpha=0.1)
    a.text(50,58,"$\\mathbb{Z}[x]$",ha='center',fontsize=18,color=C['gray'],fontweight='bold')
    rb(a,15,22,70,28,C['blue'],'',alpha=0.4)
    a.text(50,46,"$(x^2+1) = \\{f(x)(x^2+1) : f \\in \\mathbb{Z}[x]\\}$",
           ha='center',fontsize=13,color='white',fontweight='bold')
    examples=["$x^2+1$","$2x^2+2$","$x^3+x$","$(x-1)(x^2+1)$"]
    for i,ex in enumerate(examples):
        rb(a,18+i*17,28,15,8,C['blue'],ex,11)
    a.text(50,8,"Quotient $\\mathbb{Z}[x]/(x^2+1) \\cong \\mathbb{Z}[i]$ — the Gaussian integers!",
           ha='center',fontsize=12,color=C['red'],fontweight='bold')
    save(f,"05-rings-and-ideals","aa_v2_05_3_principal_ideal")

def aa_05_4():  # Z[x]/(x^2+1) ~ Z[i]
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"$\\mathbb{Z}[x]/(x^2+1) \\cong \\mathbb{Z}[i]$",
          "Setting $x = i$ gives the Gaussian integers")
    rb(a,8,30,38,24,C['blue'],'$\\mathbb{Z}[x]/(x^2+1)$',16)
    rb(a,90,30,42,24,C['green'],'$\\mathbb{Z}[i]$',18)
    ar(a,46,42,90,42,lw=2.5,col=C['red'])
    a.text(68,46,"$x \\mapsto i$",fontsize=16,color=C['red'],fontweight='bold')
    a.text(27,27,"$a + bx + (x^2+1)$",ha='center',fontsize=11,color=C['blue'])
    a.text(111,27,"$a + bi$",ha='center',fontsize=14,color=C['green'])
    examples="$(2+x)(3+x) = 6 + 5x + x^2 = 5 + 5x \\,\\mathrm{mod}\\, (x^2+1)$"
    a.text(70,12,examples,ha='center',fontsize=12,color=C['dark'])
    a.text(70,5,"Same as $(2+i)(3+i) = 6 + 5i + i^2 = 5 + 5i$",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"05-rings-and-ideals","aa_v2_05_4_quotient_ring")

def aa_05_5():  # Prime vs maximal
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Prime Ideals ⊇ Maximal Ideals",
          "All maximal ideals are prime; not all primes are maximal")
    # Two overlapping circles
    c1=Circle((35,35),22,fc=C['blue'],alpha=0.3,ec='none')
    c2=Circle((60,35),18,fc=C['amber'],alpha=0.5,ec='none')
    c1.set_path_effects([shadow]);c2.set_path_effects([shadow])
    a.add_patch(c1);a.add_patch(c2)
    a.text(20,55,"Prime ideals",fontsize=14,color=C['blue'],fontweight='bold')
    a.text(60,55,"Maximal\nideals",fontsize=14,color=C['amber'],ha='center',fontweight='bold')
    a.text(25,30,"$(0) \\subset \\mathbb{Z}[x]$\nis prime, not maximal",
           ha='center',fontsize=10,color=C['dark'])
    a.text(60,30,"$(p) \\subset \\mathbb{Z}$\nis both",
           ha='center',fontsize=10,color=C['dark'])
    a.text(50,8,"In a PID, nonzero prime ideals are maximal.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"05-rings-and-ideals","aa_v2_05_5_prime_maximal")

def aa_05_6():  # Ring homo
    f,a=mk(12,6);a.set_xlim(0,100);a.set_ylim(0,55)
    title(a,"Ring Homomorphism $\\varphi: R \\to S$",
          "Preserves both addition and multiplication")
    rb(a,10,18,20,16,C['blue'],'$R$',20)
    rb(a,70,18,20,16,C['green'],'$S$',20)
    ar(a,30,26,70,26,lw=2.5)
    a.text(50,30,"$\\varphi$",fontsize=18,color=C['dark'],fontweight='bold')
    rules=["$\\varphi(a+b) = \\varphi(a) + \\varphi(b)$",
           "$\\varphi(ab) = \\varphi(a)\\varphi(b)$",
           "$\\varphi(1_R) = 1_S$"]
    for i,r in enumerate(rules):
        a.text(50,12-i*4,r,ha='center',fontsize=11,color=C['dark'])
    save(f,"05-rings-and-ideals","aa_v2_05_6_ring_homo")

def aa_05_7():  # Ring examples
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"Catalog of Classical Rings",
          "Different examples illuminate different properties")
    rings=[("$\\mathbb{Z}$","integers","ED",C['blue']),
           ("$\\mathbb{Q}, \\mathbb{R}, \\mathbb{C}$","fields","Field",C['red']),
           ("$\\mathbb{Z}[i]$","Gaussian","ED",C['blue']),
           ("$\\mathbb{Z}[\\sqrt{-5}]$","not UFD","domain",C['amber']),
           ("$\\mathbb{Z}[x]$","poly. ring","UFD",C['green']),
           ("$\\mathbb{F}_p[x]$","poly. over field","ED",C['blue']),
           ("$M_n(\\mathbb{R})$","matrices","non-comm.",C['purple']),
           ("$\\mathbb{Z}/6\\mathbb{Z}$","not domain","ring",C['gray'])]
    for i,(r,d,t,col) in enumerate(rings):
        x=10+(i%4)*32;y=55-(i//4)*22
        rb(a,x,y,28,16,col,'',alpha=0.85)
        a.text(x+14,y+11,r,ha='center',fontsize=14,color='white',fontweight='bold')
        a.text(x+14,y+7,d,ha='center',fontsize=10,color='white')
        a.text(x+14,y+3,t,ha='center',fontsize=9,color='white',style='italic')
    a.text(70,5,"Each example sits at a different level of the ring hierarchy.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"05-rings-and-ideals","aa_v2_05_7_ring_examples")

# ===== Article 06: Polynomial Rings =====
def aa_06_1():  # Long division (simplified)
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Polynomial Long Division",
          "$x^3 - 1 = (x-1)(x^2 + x + 1)$")
    rb(a,15,42,30,12,C['blue'],'$x^3 - 1$',16)
    rb(a,55,42,30,12,C['green'],'$\div\, (x - 1)$',16)
    a.text(50,32,"$=$",ha='center',fontsize=20,color=C['dark'],fontweight='bold')
    rb(a,30,15,40,12,C['red'],'$x^2 + x + 1$',18)
    a.text(50,5,"Remainder is 0, so $x - 1$ divides $x^3 - 1$.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"06-polynomial-rings","aa_v2_06_1_division")

def aa_06_2():  # Factor tree x^4-1
    f,a=mk(12,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Factorization Tree of $x^4 - 1$ over $\\mathbb{Q}$",
          "Each step splits into irreducible factors")
    rb(a,40,68,20,8,C['dark'],'$x^4 - 1$',14)
    rb(a,18,50,20,8,C['blue'],'$x^2 - 1$',13)
    rb(a,62,50,20,8,C['blue'],'$x^2 + 1$',13)
    ar(a,46,68,28,58);ar(a,54,68,72,58)
    rb(a,5,30,18,8,C['green'],'$x - 1$',12)
    rb(a,28,30,18,8,C['green'],'$x + 1$',12)
    rb(a,62,30,20,8,C['amber'],'$x^2 + 1$',12)
    a.text(72,22,"(irred. over $\\mathbb{Q}$)",ha='center',fontsize=10,color=C['gray'],style='italic')
    ar(a,24,50,14,38);ar(a,32,50,37,38)
    ar(a,72,50,72,38)
    rb(a,15,8,70,8,C['red'],'$x^4 - 1 = (x-1)(x+1)(x^2+1)$',13)
    save(f,"06-polynomial-rings","aa_v2_06_2_factor_tree")

def aa_06_3():  # Irreducibility test
    f,a=mk(12,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Irreducibility Tests over $\\mathbb{Q}$",
          "Decision tree for testing irreducibility")
    rb(a,30,68,40,8,C['dark'],'Polynomial $f(x)$',14)
    rb(a,5,50,28,10,C['blue'],'Degree 1?',12)
    rb(a,40,50,20,10,C['green'],'Find roots?',12)
    rb(a,70,50,25,10,C['amber'],'Eisenstein?',12)
    ar(a,40,68,19,60);ar(a,50,68,50,60);ar(a,60,68,82,60)
    rb(a,5,30,28,8,C['blue'],'Always irred.',11)
    rb(a,40,30,20,8,C['green'],'Has root → red.',11)
    rb(a,70,30,25,8,C['amber'],'Eisenstein → irred.',11)
    ar(a,19,50,19,38);ar(a,50,50,50,38);ar(a,82,50,82,38)
    rb(a,15,8,70,8,C['red'],'Else: try mod $p$ reduction or rational root',12)
    save(f,"06-polynomial-rings","aa_v2_06_3_irreducible_test")

def aa_06_4():  # Eisenstein
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Eisenstein's Criterion",
          "If a prime $p$ satisfies these three conditions, $f$ is irreducible")
    poly="$f(x) = x^4 + 6x^3 + 12x^2 + 18x + 6$"
    a.text(50,58,poly,ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    a.text(50,52,"Try $p = 3$:",ha='center',fontsize=12,color=C['gray'])
    rb(a,12,30,25,12,C['green'],'$3 \\nmid 1$\n(leading)',11)
    rb(a,40,30,25,12,C['green'],'$3 \\mid 6, 12, 18, 6$',11)
    rb(a,68,30,25,12,C['green'],'$9 \\nmid 6$\n(constant)',11)
    rb(a,25,10,55,10,C['red'],'$\\Rightarrow$ irreducible over $\\mathbb{Q}$',13)
    save(f,"06-polynomial-rings","aa_v2_06_4_eisenstein")

def aa_06_5():  # UFD chain
    f,a=mk(11,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Unique Factorization in a UFD",
          "Every nonzero, non-unit element factors into irreducibles, uniquely up to associates")
    rb(a,5,40,20,12,C['dark'],'$60$',16)
    ar(a,15,40,15,30)
    rb(a,5,18,20,8,C['blue'],'$2 \\cdot 30$',12)
    ar(a,25,22,38,22)
    rb(a,40,18,22,8,C['blue'],'$2 \\cdot 2 \\cdot 15$',12)
    ar(a,62,22,75,22)
    rb(a,77,18,20,8,C['green'],'$2 \\cdot 2 \\cdot 3 \\cdot 5$',12)
    a.text(50,7,"Different splitting orders give the same multiset of primes — uniqueness.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"06-polynomial-rings","aa_v2_06_5_ufd_chain")

def aa_06_6():  # Polynomial grid Z[x]
    f,a=mk(11,8);a.set_xlim(-0.5,5.5);a.set_ylim(-0.5,5.5)
    title(a,"Monomials in $\\mathbb{Z}[x]$","Polynomials as $\\mathbb{Z}$-linear combinations of $x^n$")
    for d in range(6):
        for c in range(6):
            sz=0.3*np.exp(-0.2*(d+c))
            col=[C['red'],C['amber'],C['green'],C['blue'],C['purple'],C['gray']][d % 6]
            rb(a,d-sz,c-sz,2*sz,2*sz,col,'',alpha=0.7)
    for d in range(6):
        if d == 0:
            a.text(d,-0.3,"$1$",ha='center',fontsize=12,color=C['dark'])
        elif d == 1:
            a.text(d,-0.3,"$x$",ha='center',fontsize=12,color=C['dark'])
        else:
            a.text(d,-0.3,f"$x^{d}$",ha='center',fontsize=12,color=C['dark'])
    for c in range(1,6):
        a.text(-0.3,c,f"${c}$"+"$\\cdot$",ha='right',va='center',fontsize=11,color=C['dark'])
    a.text(2.5,5,"$f(x) = \\sum_{n=0}^{d} a_n x^n$, $a_n \\in \\mathbb{Z}$",
           ha='center',fontsize=12,color=C['gray'],style='italic')
    a.set_aspect('equal')
    save(f,"06-polynomial-rings","aa_v2_06_6_poly_grid")

def aa_06_7():  # x^3-2 roots in C
    f,a=mk(10,8);a.set_xlim(-2,2);a.set_ylim(-2,2)
    title(a,"Roots of $x^3 - 2 = 0$ in the Complex Plane",
          "Three roots forming an equilateral triangle")
    a.axhline(0,color=C['gray'],lw=0.5,alpha=0.5)
    a.axvline(0,color=C['gray'],lw=0.5,alpha=0.5)
    cir=Circle((0,0),2**(1/3),fill=False,ec=C['gray'],alpha=0.5,lw=1,linestyle='--')
    a.add_patch(cir)
    cube_root=2**(1/3)
    roots=[(cube_root,0),(cube_root*np.cos(2*np.pi/3),cube_root*np.sin(2*np.pi/3)),
           (cube_root*np.cos(4*np.pi/3),cube_root*np.sin(4*np.pi/3))]
    cols=[C['red'],C['blue'],C['green']]
    for (rx,ry),col in zip(roots,cols):
        cir2=Circle((rx,ry),0.1,fc=col,ec='none')
        cir2.set_path_effects([shadow]);a.add_patch(cir2)
    a.text(cube_root+0.15,0,"$\\sqrt[3]{2}$",fontsize=12,color=cols[0],fontweight='bold')
    a.text(roots[1][0]-0.4,roots[1][1]+0.1,"$\\sqrt[3]{2}\\,\\omega$",fontsize=12,
           color=cols[1],fontweight='bold')
    a.text(roots[2][0]-0.4,roots[2][1]-0.15,"$\\sqrt[3]{2}\\,\\omega^2$",fontsize=12,
           color=cols[2],fontweight='bold')
    a.text(0,-1.8,"$\\omega = e^{2\\pi i/3}$",ha='center',fontsize=11,color=C['gray'])
    a.set_aspect('equal')
    a.set_xlabel('Re',fontsize=10);a.set_ylabel('Im',fontsize=10)
    a.axis('on')
    save(f,"06-polynomial-rings","aa_v2_06_7_root_complex")

# ===== Article 07: Field Extensions =====
def aa_07_1():  # Extension tower
    f,a=mk(11,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Tower of Field Extensions",
          "$\\mathbb{Q} \\subset \\mathbb{Q}(\\sqrt{2}) \\subset \\mathbb{Q}(\\sqrt{2}, \\sqrt{3})$")
    rb(a,30,55,40,10,C['red'],'$\\mathbb{Q}(\\sqrt{2}, \\sqrt{3})$',16)
    rb(a,30,32,40,10,C['blue'],'$\\mathbb{Q}(\\sqrt{2})$',16)
    rb(a,40,8,20,10,C['green'],'$\\mathbb{Q}$',18)
    a.plot([50,50],[18,32],color=C['gray'],lw=2)
    a.plot([50,50],[42,55],color=C['gray'],lw=2)
    a.text(56,25,"degree 2",fontsize=11,color=C['gray'],style='italic')
    a.text(56,49,"degree 2",fontsize=11,color=C['gray'],style='italic')
    a.text(72,49,"basis: $\\{1, \\sqrt{3}\\}$",fontsize=10,color=C['blue'])
    a.text(72,25,"basis: $\\{1, \\sqrt{2}\\}$",fontsize=10,color=C['blue'])
    a.text(50,72,"$[\\mathbb{Q}(\\sqrt{2},\\sqrt{3}) : \\mathbb{Q}] = 4$",
           ha='center',fontsize=14,color=C['red'],fontweight='bold')
    save(f,"07-field-extensions","aa_v2_07_1_extension_tower")

def aa_07_2():  # Minimal poly
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Minimal Polynomial Determines Extension Degree",
          "If $\\alpha$ has minimal polynomial of degree $n$, then $[\\mathbb{Q}(\\alpha):\\mathbb{Q}] = n$")
    examples=[("$\\sqrt{2}$","$x^2 - 2$","2",C['blue']),
              ("$\\sqrt[3]{2}$","$x^3 - 2$","3",C['green']),
              ("$\\zeta_5$","$x^4 + x^3 + x^2 + x + 1$","4",C['amber']),
              ("$i + \\sqrt{2}$","$x^4 - 2x^2 + 9$","4",C['red'])]
    a.text(15,55,"$\\alpha$",ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    a.text(50,55,"Minimal poly",ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    a.text(85,55,"degree",ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    for i,(al,mp,d,col) in enumerate(examples):
        y=42-i*8
        rb(a,8,y,15,6,col,al,12,alpha=0.85)
        rb(a,28,y,45,6,col,mp,11,alpha=0.85)
        rb(a,78,y,15,6,col,d,12,alpha=0.85)
    a.text(50,5,"Smaller minimal polynomial = simpler arithmetic in the extension.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"07-field-extensions","aa_v2_07_2_minimal_poly")

def aa_07_3():  # Algebraic vs transcendental
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Algebraic vs Transcendental Numbers",
          "$\\alpha$ algebraic ⟺ root of some polynomial over $\\mathbb{Q}$")
    c1=Circle((35,35),20,fc=C['blue'],alpha=0.4,ec='none')
    c1.set_path_effects([shadow]);a.add_patch(c1)
    a.text(28,52,"Algebraic",fontsize=14,color=C['blue'],fontweight='bold')
    examples_alg=["$\\sqrt{2}$","$\\sqrt[3]{5}$","$\\zeta_n$","$1+\\sqrt{2}$","$i$","$\\frac{1}{3}$"]
    for i,e in enumerate(examples_alg):
        ang=np.pi/6+i*np.pi/3
        x=35+12*np.cos(ang);y=35+12*np.sin(ang)
        a.text(x,y,e,ha='center',va='center',fontsize=11,color='white',fontweight='bold')
    c2=Circle((78,35),17,fc=C['red'],alpha=0.4,ec='none')
    c2.set_path_effects([shadow]);a.add_patch(c2)
    a.text(78,52,"Transcendental",fontsize=14,color=C['red'],fontweight='bold',ha='center')
    a.text(78,40,"$\\pi$",fontsize=18,color='white',fontweight='bold',ha='center')
    a.text(78,32,"$e$",fontsize=18,color='white',fontweight='bold',ha='center')
    a.text(78,24,"$\\sum 10^{-n!}$",fontsize=11,color='white',ha='center')
    a.text(50,5,"Algebraic numbers form a countable subfield of $\\mathbb{C}$.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"07-field-extensions","aa_v2_07_3_alg_vs_trans")

def aa_07_4():  # Splitting field
    f,a=mk(12,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Splitting Field of $x^3 - 2$",
          "Smallest extension of $\\mathbb{Q}$ containing all roots")
    rb(a,35,65,30,10,C['red'],'$\\mathbb{Q}(\\sqrt[3]{2}, \\omega)$',14)
    a.text(50,57,"degree 6",ha='center',fontsize=11,color=C['gray'])
    rb(a,10,40,25,8,C['blue'],'$\\mathbb{Q}(\\sqrt[3]{2})$',13)
    rb(a,40,40,20,8,C['amber'],'$\\mathbb{Q}(\\omega)$',13)
    rb(a,65,40,25,8,C['blue'],'$\\mathbb{Q}(\\sqrt[3]{2}\\omega)$',12)
    a.plot([22,50],[48,65],color=C['gray'],lw=1.5)
    a.plot([50,50],[48,65],color=C['gray'],lw=1.5)
    a.plot([78,50],[48,65],color=C['gray'],lw=1.5)
    rb(a,40,15,20,8,C['green'],'$\\mathbb{Q}$',16)
    a.plot([50,22],[23,40],color=C['gray'],lw=1.5)
    a.plot([50,50],[23,40],color=C['gray'],lw=1.5)
    a.plot([50,78],[23,40],color=C['gray'],lw=1.5)
    a.text(50,7,"Roots: $\\sqrt[3]{2},\\, \\omega\\sqrt[3]{2},\\, \\omega^2\\sqrt[3]{2}$ where $\\omega = e^{2\\pi i/3}$",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"07-field-extensions","aa_v2_07_4_splitting_field")

def aa_07_5():  # Finite field
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Constructing $\\mathrm{GF}(p^n)$",
          "Quotient of polynomial ring by an irreducible polynomial")
    rb(a,8,40,30,12,C['blue'],'$\\mathbb{F}_p[x]$',14)
    rb(a,55,40,38,12,C['green'],'$\\mathbb{F}_p[x]/(f(x))$',14)
    a.text(74,55,"$f$ irreducible of degree $n$",ha='center',fontsize=11,
           color=C['gray'],style='italic')
    ar(a,38,46,55,46,lw=2.5)
    a.text(46,49,"mod $f$",fontsize=11,color=C['dark'])
    a.text(50,30,"Result: a field of size $p^n$",ha='center',fontsize=14,
           color=C['red'],fontweight='bold')
    a.text(50,18,"Example: $\\mathbb{F}_4 = \\mathbb{F}_2[x]/(x^2+x+1) = \\{0, 1, \\alpha, \\alpha+1\\}$",
           ha='center',fontsize=12,color=C['dark'])
    a.text(50,8,"Up to isomorphism, only one finite field of each prime power order.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"07-field-extensions","aa_v2_07_5_finite_field")

def aa_07_6():  # Constructible numbers
    f,a=mk(11,8);a.set_xlim(0,100);a.set_ylim(0,80)
    title(a,"Constructible Numbers Form a 2-Tower",
          "Each step is a degree-2 extension")
    levels=[(70,"$\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}, \\sqrt{2+\\sqrt{3}})$",C['red']),
            (52,"$\\mathbb{Q}(\\sqrt{2}, \\sqrt{3})$",C['amber']),
            (34,"$\\mathbb{Q}(\\sqrt{2})$",C['blue']),
            (16,"$\\mathbb{Q}$",C['green'])]
    for y,lab,col in levels:
        rb(a,15,y,70,8,col,lab,13)
    for i in range(len(levels)-1):
        a.plot([50,50],[levels[i+1][0]+8,levels[i][0]],color=C['gray'],lw=1.5)
        a.text(53,(levels[i][0]+levels[i+1][0]+8)/2,"deg 2",fontsize=10,color=C['gray'])
    a.text(50,6,"Cube duplication, angle trisection, circle squaring impossible — degrees not powers of 2.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"07-field-extensions","aa_v2_07_6_constructible")

def aa_07_7():  # Extension examples
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"Classical Field Extensions","Each illustrates a different phenomenon")
    rows=[("$\\mathbb{Q}(\\sqrt{2})$","quadratic","Galois of order 2",C['blue']),
          ("$\\mathbb{Q}(\\sqrt[3]{2})$","cubic","not Galois (incomplete roots)",C['green']),
          ("$\\mathbb{Q}(i)$","cyclotomic","Gaussian rationals",C['amber']),
          ("$\\mathbb{Q}(\\zeta_n)$","cyclotomic","Galois group $(\\mathbb{Z}/n)^*$",C['purple']),
          ("$\\mathbb{F}_q(t)$","function field","transcendental",C['red'])]
    for i,(ex,typ,desc,col) in enumerate(rows):
        y=63-i*11
        rb(a,8,y,28,8,col,ex,13)
        rb(a,40,y,24,8,col,typ,12,alpha=0.85)
        rb(a,68,y,68,8,col,desc,11,alpha=0.7)
    save(f,"07-field-extensions","aa_v2_07_7_extension_examples")

# ===== Article 08: Galois =====
def aa_08_1():  # Galois group Q(sqrt2)
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Galois Group $\\mathrm{Gal}(\\mathbb{Q}(\\sqrt{2})/\\mathbb{Q})$",
          "Two automorphisms: identity and $\\sqrt{2} \\mapsto -\\sqrt{2}$")
    rb(a,8,30,38,18,C['blue'],'',alpha=0.4)
    a.text(27,50,"$\\mathbb{Q}(\\sqrt{2})$",ha='center',fontsize=15,color=C['blue'],fontweight='bold')
    a.text(27,42,"$\\{a + b\\sqrt{2} : a, b \\in \\mathbb{Q}\\}$",ha='center',fontsize=11,color='white')
    rb(a,55,30,38,18,C['green'],'',alpha=0.4)
    a.text(74,50,"$\\mathbb{Q}(\\sqrt{2})$",ha='center',fontsize=15,color=C['green'],fontweight='bold')
    a.text(74,42,"$\\{a - b\\sqrt{2} : a, b \\in \\mathbb{Q}\\}$",ha='center',fontsize=11,color='white')
    ar(a,46,40,55,40,col=C['red'],lw=2.5)
    a.text(50,44,"$\\sigma$",fontsize=18,color=C['red'],fontweight='bold')
    a.text(50,11,"$\\mathrm{Gal} = \\{\\mathrm{id}, \\sigma\\} \\cong \\mathbb{Z}/2\\mathbb{Z}$",
           ha='center',fontsize=14,color=C['dark'],fontweight='bold')
    save(f,"08-galois-theory","aa_v2_08_1_galois_group")

def aa_08_2():  # Galois correspondence
    f,a=mk(13,8);a.set_xlim(0,140);a.set_ylim(0,80)
    title(a,"Galois Correspondence",
          "Subgroups of $\\mathrm{Gal}(L/K)$ ↔ Intermediate fields between $K$ and $L$")
    # Left: subgroups
    a.text(20,72,"Subgroups",fontsize=14,color=C['blue'],fontweight='bold',ha='center')
    rb(a,5,55,30,8,C['blue'],'$\\{e\\}$',13)
    rb(a,5,38,30,8,C['blue'],'$H$',13)
    rb(a,5,21,30,8,C['blue'],'$\\mathrm{Gal}$',13)
    a.plot([20,20],[29,38],color=C['gray'],lw=1.5)
    a.plot([20,20],[46,55],color=C['gray'],lw=1.5)
    # Right: fields
    a.text(120,72,"Intermediate fields",fontsize=14,color=C['green'],fontweight='bold',ha='center')
    rb(a,105,55,30,8,C['green'],'$L$',13)
    rb(a,105,38,30,8,C['green'],'$F$',13)
    rb(a,105,21,30,8,C['green'],'$K$',13)
    a.plot([120,120],[29,38],color=C['gray'],lw=1.5)
    a.plot([120,120],[46,55],color=C['gray'],lw=1.5)
    # Bijection arrows
    ar(a,35,42,105,42,col=C['red'],lw=2.5)
    ar(a,105,38,35,38,col=C['red'],lw=2.5)
    a.text(70,46,"order-reversing bijection",ha='center',fontsize=12,color=C['red'],fontweight='bold')
    a.text(70,8,"$H \\leftrightarrow L^H$ (fixed field of $H$);   $|H| = [L : L^H]$",
           ha='center',fontsize=12,color=C['dark'])
    save(f,"08-galois-theory","aa_v2_08_2_correspondence")

def aa_08_3():  # x^4 - 2 correspondence
    f,a=mk(13,9);a.set_xlim(0,140);a.set_ylim(0,90)
    title(a,"Galois Correspondence for $x^4 - 2$",
          "Splitting field $\\mathbb{Q}(\\sqrt[4]{2}, i)$, Galois group $D_4$ (order 8)")
    # Subgroup lattice (left)
    rb(a,8,75,18,7,C['blue'],'$\\{e\\}$',12)
    rb(a,2,55,12,7,C['blue'],'$\\langle r^2\\rangle$',11)
    rb(a,16,55,12,7,C['blue'],'$\\langle s\\rangle$',11)
    rb(a,30,55,12,7,C['blue'],'$\\langle rs\\rangle$',11)
    rb(a,2,37,12,7,C['blue'],'$\\langle r\\rangle$',11)
    rb(a,16,37,12,7,C['blue'],'$V_1$',11)
    rb(a,30,37,12,7,C['blue'],'$V_2$',11)
    rb(a,8,15,18,7,C['blue'],'$D_4$',13)
    # Field lattice (right)
    rb(a,98,75,30,7,C['green'],'$\\mathbb{Q}$',13)
    rb(a,90,55,18,7,C['green'],'$\\mathbb{Q}(\\sqrt{2})$',11)
    rb(a,108,55,18,7,C['green'],'$\\mathbb{Q}(i)$',11)
    rb(a,108,37,18,7,C['green'],'$\\mathbb{Q}(\\sqrt[4]{2})$',11)
    rb(a,90,37,18,7,C['green'],'$\\mathbb{Q}(i\\sqrt[4]{2})$',11)
    rb(a,98,15,30,7,C['green'],'$\\mathbb{Q}(\\sqrt[4]{2}, i)$',12)
    # Bijection
    ar(a,30,40,98,40,col=C['red'],lw=2)
    a.text(64,44,"order-reversing",ha='center',fontsize=11,color=C['red'])
    a.text(70,8,"$|D_4| = 8 = [\\mathbb{Q}(\\sqrt[4]{2}, i) : \\mathbb{Q}]$",
           ha='center',fontsize=12,color=C['dark'])
    save(f,"08-galois-theory","aa_v2_08_3_x4_minus_2")

def aa_08_4():  # Solvable chain
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Solvable Group: Chain with Abelian Quotients",
          "$G = G_0 \\triangleright G_1 \\triangleright \\cdots \\triangleright G_n = \\{e\\}$")
    levels=[(50,"$G$",C['red']),(38,"$G_1$",C['amber']),
            (26,"$G_2$",C['green']),(14,"$\\{e\\}$",C['blue'])]
    for y,lab,col in levels:
        rb(a,30,y,40,8,col,lab,13)
    quotient_labels=["$G/G_1$ abelian","$G_1/G_2$ abelian","$G_2$ abelian"]
    for i,ql in enumerate(quotient_labels):
        a.plot([50,50],[levels[i+1][0]+8,levels[i][0]],color=C['gray'],lw=1.5)
        a.text(72,(levels[i][0]+levels[i+1][0]+8)/2,ql,fontsize=11,color=C['gray'],style='italic')
    a.text(50,5,"Solvability transfers: $G$ solvable ⟺ root extension of associated polynomial.",
           ha='center',fontsize=11,color=C['dark'])
    save(f,"08-galois-theory","aa_v2_08_4_solvable_chain")

def aa_08_5():  # Quintic unsolvable
    f,a=mk(13,7);a.set_xlim(0,140);a.set_ylim(0,70)
    title(a,"Why the Quintic Is Unsolvable in Radicals",
          "$\\mathrm{Gal}$ of generic quintic = $S_5$, which contains the simple non-abelian $A_5$")
    rb(a,8,42,30,12,C['red'],'Quintic $f(x)$',13)
    rb(a,50,42,28,12,C['amber'],'$\\mathrm{Gal}(f) = S_5$',14)
    rb(a,90,42,40,12,C['blue'],'$A_5 \\triangleleft S_5$, simple',13)
    ar(a,38,48,50,48);ar(a,78,48,90,48)
    rb(a,40,16,55,12,C['dark'],'$A_5$ not solvable',16,alpha=0.95)
    ar(a,110,42,75,28,col=C['gray'])
    a.text(70,8,"Cubic and quartic have solvable Galois groups; quintic and beyond can fail.",
           ha='center',fontsize=11,color=C['gray'],style='italic')
    save(f,"08-galois-theory","aa_v2_08_5_quintic")

def aa_08_6():  # Cyclotomic
    f,a=mk(11,8);a.set_xlim(-2,2);a.set_ylim(-2,2)
    title(a,"Cyclotomic Extension $\\mathbb{Q}(\\zeta_8)$",
          "Roots of $x^8 - 1$ on the unit circle")
    cir=Circle((0,0),1,fill=False,ec=C['gray'],alpha=0.5,lw=1,linestyle='--')
    a.add_patch(cir)
    cols=[C['red'],C['amber'],C['green'],C['blue'],C['purple'],C['red'],C['amber'],C['green']]
    for k in range(8):
        ang=k*np.pi/4
        x=np.cos(ang);y=np.sin(ang)
        cir2=Circle((x,y),0.07,fc=cols[k],ec='none')
        cir2.set_path_effects([shadow]);a.add_patch(cir2)
        a.text(x*1.18,y*1.18,f"$\\zeta_8^{k}$",ha='center',va='center',fontsize=10,
               color=cols[k],fontweight='bold')
    a.text(0,-1.7,"$\\mathrm{Gal}(\\mathbb{Q}(\\zeta_8)/\\mathbb{Q}) \\cong (\\mathbb{Z}/8\\mathbb{Z})^* \\cong \\mathbb{Z}/2 \\times \\mathbb{Z}/2$",
           ha='center',fontsize=11,color=C['dark'])
    a.set_aspect('equal');a.axhline(0,color=C['gray'],lw=0.3,alpha=0.3)
    a.axvline(0,color=C['gray'],lw=0.3,alpha=0.3)
    save(f,"08-galois-theory","aa_v2_08_6_cyclotomic")

def aa_08_7():  # Normal extension
    f,a=mk(12,7);a.set_xlim(0,100);a.set_ylim(0,70)
    title(a,"Normal Subgroups Correspond to Normal Extensions",
          "$H \\triangleleft G$ ⟺ $L^H/K$ is normal")
    rb(a,8,30,32,18,C['blue'],'',alpha=0.5)
    a.text(24,46,"$H \\triangleleft \\mathrm{Gal}(L/K)$",ha='center',fontsize=13,
           color='white',fontweight='bold')
    a.text(24,38,"normal subgroup",ha='center',fontsize=11,color='white',style='italic')
    rb(a,60,30,32,18,C['green'],'',alpha=0.5)
    a.text(76,46,"$L^H/K$ normal",ha='center',fontsize=13,color='white',fontweight='bold')
    a.text(76,38,"contains all conjugates",ha='center',fontsize=11,color='white',style='italic')
    ar(a,40,40,60,40,col=C['red'],lw=2.5)
    ar(a,60,38,40,38,col=C['red'],lw=2.5)
    a.text(50,15,"$\\mathrm{Gal}(L^H/K) = \\mathrm{Gal}(L/K)/H$",
           ha='center',fontsize=13,color=C['dark'],fontweight='bold')
    save(f,"08-galois-theory","aa_v2_08_7_normal_field")

def main():
    aa_05_1();aa_05_2();aa_05_3();aa_05_4();aa_05_5();aa_05_6();aa_05_7()
    aa_06_1();aa_06_2();aa_06_3();aa_06_4();aa_06_5();aa_06_6();aa_06_7()
    aa_07_1();aa_07_2();aa_07_3();aa_07_4();aa_07_5();aa_07_6();aa_07_7()
    aa_08_1();aa_08_2();aa_08_3();aa_08_4();aa_08_5();aa_08_6();aa_08_7()
    print("Part 2: 28 figures generated")

if __name__=="__main__":main()
