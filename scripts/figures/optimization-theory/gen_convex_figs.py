"""Generate 5 pedagogical matplotlib figures for the convex analysis article."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection

# Color palette
BLUE = "#2E5BFF"
COPPER = "#D97706"
GREEN = "#059669"
PURPLE = "#7C3AED"
GRAY = "#475569"
LIGHT_BG = "#F8FAFC"

def style_ax(ax, keep_axes=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not keep_axes:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_facecolor('white')

# ---------------------------------------------------------------
# Figure 1: Convex vs non-convex set
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

# Left: convex set (an ellipse)
ax = axes[0]
theta = np.linspace(0, 2*np.pi, 200)
ex, ey = 1.6*np.cos(theta), 1.05*np.sin(theta)
ax.fill(ex, ey, color=BLUE, alpha=0.18, edgecolor=BLUE, linewidth=2)
# Two interior points and segment
p1 = np.array([-1.0, 0.4])
p2 = np.array([1.1, -0.5])
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COPPER, linewidth=2.2)
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=70, color=COPPER, zorder=5, edgecolor='white', linewidth=1.5)
ax.text(p1[0]-0.05, p1[1]+0.15, r'$x$', fontsize=15, color=COPPER, ha='right')
ax.text(p2[0]+0.05, p2[1]-0.05, r'$y$', fontsize=15, color=COPPER, ha='left', va='top')
ax.set_title('Convex set\n(every segment stays inside)', fontsize=13, color=GRAY, pad=12)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
style_ax(ax)

# Right: non-convex (crescent / annulus segment)
ax = axes[1]
# Outer circle minus inner circle — annulus is non-convex
t = np.linspace(0, 2*np.pi, 200)
outer = np.column_stack([1.5*np.cos(t), 1.2*np.sin(t)])
# Make a kidney/crescent: outer ellipse minus a smaller offset disk
ax.fill(outer[:,0], outer[:,1], color=BLUE, alpha=0.18, edgecolor=BLUE, linewidth=2)
# Subtract inner circle (white over)
inner_t = np.linspace(0, 2*np.pi, 100)
ix = 0.6 + 0.6*np.cos(inner_t)
iy = 0.6*np.sin(inner_t)
ax.fill(ix, iy, color='white', edgecolor=BLUE, linewidth=2)
# Two points such that segment passes through hole
p1 = np.array([-1.2, 0.2])
p2 = np.array([1.3, 0.3])
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COPPER, linewidth=2.2, linestyle='--')
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=70, color=COPPER, zorder=5, edgecolor='white', linewidth=1.5)
# Mark the offending midpoint
mid = (p1 + p2)/2
ax.scatter(mid[0], mid[1], s=80, color='red', marker='x', zorder=6, linewidth=2.5)
ax.text(mid[0], mid[1]-0.28, 'midpoint\noutside', fontsize=10.5, color='red', ha='center')
ax.text(p1[0]-0.05, p1[1]+0.18, r'$x$', fontsize=15, color=COPPER, ha='right')
ax.text(p2[0]+0.05, p2[1]+0.18, r'$y$', fontsize=15, color=COPPER, ha='left')
ax.set_title('Non-convex set\n(segment can leave the set)', fontsize=13, color=GRAY, pad=12)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
style_ax(ax)

plt.tight_layout()
plt.savefig('/tmp/fig1_convex_set.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("fig1 done")

# ---------------------------------------------------------------
# Figure 2: Projection onto a convex set
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor('white')

# Convex set: rounded polygon (e.g. ellipse-ish)
theta = np.linspace(0, 2*np.pi, 200)
cx, cy = 1.4*np.cos(theta) - 0.3, 1.1*np.sin(theta) + 0.1
ax.fill(cx, cy, color=BLUE, alpha=0.18, edgecolor=BLUE, linewidth=2.2, label='Convex set $C$')

# Point y outside
y_pt = np.array([2.5, 1.8])
# Find projection numerically: the closest point on ellipse to y_pt
def ellipse_pt(t):
    return np.array([1.4*np.cos(t) - 0.3, 1.1*np.sin(t) + 0.1])
ts = np.linspace(0, 2*np.pi, 5000)
dists = np.array([np.linalg.norm(ellipse_pt(t) - y_pt) for t in ts])
t_star = ts[np.argmin(dists)]
z = ellipse_pt(t_star)

# Draw y, z
ax.scatter(*y_pt, s=120, color=COPPER, zorder=5, edgecolor='white', linewidth=2)
ax.scatter(*z, s=120, color=GREEN, zorder=5, edgecolor='white', linewidth=2)
ax.plot([y_pt[0], z[0]], [y_pt[1], z[1]], color=COPPER, linewidth=2.4)
ax.text(y_pt[0]+0.1, y_pt[1]+0.05, r'$y$', fontsize=18, color=COPPER)
ax.text(z[0]-0.18, z[1]+0.18, r'$\pi_C(y)=z$', fontsize=15, color=GREEN, ha='right')

# Tangent line at z: perpendicular to (y - z)
diff = y_pt - z
diff_unit = diff / np.linalg.norm(diff)
perp = np.array([-diff_unit[1], diff_unit[0]])
tan_a = z - 1.4*perp
tan_b = z + 1.4*perp
ax.plot([tan_a[0], tan_b[0]], [tan_a[1], tan_b[1]],
        color=GRAY, linewidth=1.6, linestyle='--', alpha=0.85)
ax.text(tan_b[0]+0.05, tan_b[1]-0.05, 'supporting hyperplane', fontsize=10.5, color=GRAY)

# Right-angle marker at z
sz = 0.12
sq = np.array([z, z + sz*diff_unit, z + sz*diff_unit + sz*perp, z + sz*perp])
ax.plot(sq[:,0].tolist()+[sq[0,0]], sq[:,1].tolist()+[sq[0,1]],
        color=GRAY, linewidth=1.2)

# Another point x in C with the obtuse-angle inequality illustrated
x_inC = np.array([-1.0, -0.4])
ax.scatter(*x_inC, s=80, color=PURPLE, zorder=5, edgecolor='white', linewidth=1.5)
ax.text(x_inC[0]-0.12, x_inC[1]+0.12, r'$x \in C$', fontsize=13, color=PURPLE, ha='right', va='bottom')
# arrow x - z
ax.annotate('', xy=x_inC, xytext=z,
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.6))
# arrow y - z
ax.annotate('', xy=y_pt, xytext=z,
            arrowprops=dict(arrowstyle='->', color=COPPER, lw=1.6))

ax.text(0.5, -1.85,
        r'$\langle y - z,\; x - z \rangle \leq 0$ for every $x \in C$',
        fontsize=13, color=GRAY, ha='center', style='italic')

ax.set_xlim(-2.5, 3.4)
ax.set_ylim(-2.3, 2.7)
ax.set_aspect('equal')
ax.set_title('Projection onto a closed convex set', fontsize=14, color=GRAY, pad=14)
style_ax(ax)
plt.tight_layout()
plt.savefig('/tmp/fig2_projection.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("fig2 done")

# ---------------------------------------------------------------
# Figure 3: Convex function – tangent lower bound + epigraph
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

xs = np.linspace(-2.0, 2.0, 400)
f = lambda x: 0.5*x**2 + 0.2

# Left: first-order condition (tangent lower bound)
ax = axes[0]
ax.plot(xs, f(xs), color=BLUE, linewidth=2.5, label=r'$f(x)$')
x0 = -0.6
fx0 = f(x0)
slope = x0  # f'(x) = x
tangent = fx0 + slope*(xs - x0)
ax.plot(xs, tangent, color=COPPER, linewidth=2, linestyle='--',
        label=r'$f(x_0)+\langle\nabla f(x_0),x-x_0\rangle$')
ax.scatter(x0, fx0, s=90, color=COPPER, zorder=5, edgecolor='white', linewidth=1.5)
ax.text(x0-0.05, fx0-0.25, r'$x_0$', fontsize=14, color=COPPER, ha='center')

# Highlight inequality at some y
y0 = 1.0
fy = f(y0)
ty = fx0 + slope*(y0 - x0)
ax.plot([y0, y0], [ty, fy], color=GREEN, linewidth=2)
ax.scatter([y0, y0], [fy, ty], s=70, color=GREEN, zorder=5, edgecolor='white', linewidth=1.2)
ax.annotate('', xy=(y0+0.06, fy), xytext=(y0+0.06, ty),
            arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.4))
ax.text(y0+0.18, (fy+ty)/2, 'gap = $f(y) -$ tangent\n$\\geq 0$', fontsize=10.5, color=GREEN, va='center')

ax.set_xlim(-2.1, 2.4)
ax.set_ylim(-1.0, 2.8)
ax.set_title('First-order condition:\ntangent line is a global lower bound', fontsize=12.5, color=GRAY, pad=10)
ax.legend(loc='upper center', fontsize=10, frameon=False)
style_ax(ax, keep_axes=True)
ax.set_xticks([])
ax.set_yticks([])
ax.axhline(0, color=GRAY, linewidth=0.6)
ax.axvline(0, color=GRAY, linewidth=0.6)

# Right: epigraph
ax = axes[1]
ax.plot(xs, f(xs), color=BLUE, linewidth=2.5, label=r'$f(x)$')
# Shade epigraph
xfill = np.concatenate([xs, xs[::-1]])
yfill = np.concatenate([f(xs), 3.0*np.ones_like(xs)])
ax.fill(xfill, yfill, color=BLUE, alpha=0.16, label=r'$\mathrm{epi}(f)$')
# Two points in epigraph and segment
A = np.array([-1.3, 1.6])
B = np.array([1.5, 2.4])
ax.scatter(*A, s=80, color=COPPER, zorder=5, edgecolor='white', linewidth=1.5)
ax.scatter(*B, s=80, color=COPPER, zorder=5, edgecolor='white', linewidth=1.5)
ax.plot([A[0], B[0]], [A[1], B[1]], color=COPPER, linewidth=2)
ax.text(A[0]-0.05, A[1]+0.1, r'$(x,s)$', fontsize=12, color=COPPER, ha='right')
ax.text(B[0]+0.1, B[1]+0.05, r'$(y,t)$', fontsize=12, color=COPPER)

ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-0.6, 3.2)
ax.set_title('Epigraph characterization:\n$f$ convex $\\Leftrightarrow \\mathrm{epi}(f)$ is convex', fontsize=12.5, color=GRAY, pad=10)
ax.legend(loc='upper center', fontsize=10, frameon=False)
style_ax(ax, keep_axes=True)
ax.set_xticks([])
ax.set_yticks([])
ax.axhline(0, color=GRAY, linewidth=0.6)
ax.axvline(0, color=GRAY, linewidth=0.6)

plt.tight_layout()
plt.savefig('/tmp/fig3_convex_function.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("fig3 done")

# ---------------------------------------------------------------
# Figure 4: Fenchel conjugate geometric picture
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('white')

xs = np.linspace(-1.6, 2.0, 400)
f = lambda x: 0.5*x**2 + 0.4
ax.plot(xs, f(xs), color=BLUE, linewidth=2.6, label=r'$f(x)$')

# Choose slope y = 1.2; the supporting affine of slope y is y x - f*(y)
y_slope = 1.2
# x* maximizes y x - f(x): for f = 0.5 x^2 + 0.4, derivative gives x* = y
x_star = y_slope
# f*(y) = y x* - f(x*) = y^2 - (0.5 y^2 + 0.4) = 0.5 y^2 - 0.4
fstar_y = 0.5*y_slope**2 - 0.4
# Affine line y x - f*(y)
line = y_slope*xs - fstar_y
ax.plot(xs, line, color=COPPER, linewidth=2, linestyle='--',
        label=fr'$x \mapsto y\,x - f^*(y)$,  slope $y={y_slope}$')

# Mark the touching point (x*, f(x*))
ax.scatter(x_star, f(x_star), s=110, color=GREEN, zorder=6, edgecolor='white', linewidth=1.6)
ax.text(x_star+0.1, f(x_star)-0.05, r'tangency', fontsize=11, color=GREEN, va='top')

# Vertical drop showing -f*(y) at x = 0
ax.plot([0, 0], [0, -fstar_y], color=PURPLE, linewidth=2.2)
ax.scatter([0, 0], [0, -fstar_y], s=60, color=PURPLE, zorder=5, edgecolor='white', linewidth=1.2)
ax.annotate('', xy=(-0.18, 0), xytext=(-0.18, -fstar_y),
            arrowprops=dict(arrowstyle='<->', color=PURPLE, lw=1.5))
ax.text(-0.30, -fstar_y/2, r'$-f^*(y)$', fontsize=14, color=PURPLE, ha='right', va='center')

# Origin
ax.scatter(0, 0, s=40, color='black', zorder=5)

# Axes
ax.axhline(0, color=GRAY, linewidth=0.7)
ax.axvline(0, color=GRAY, linewidth=0.7)

ax.set_xlim(-1.7, 2.1)
ax.set_ylim(-1.5, 2.6)
ax.set_title(r'Conjugate $f^*(y)$: vertical gap between origin and the highest affine minorant of slope $y$',
             fontsize=12.5, color=GRAY, pad=12)
ax.legend(loc='upper left', fontsize=11, frameon=False)
style_ax(ax, keep_axes=True)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig('/tmp/fig4_conjugate.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("fig4 done")

# ---------------------------------------------------------------
# Figure 5: Subgradients at |x|=0
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

xs = np.linspace(-2.0, 2.0, 400)
ax.plot(xs, np.abs(xs), color=BLUE, linewidth=2.6, label=r'$f(x) = |x|$', zorder=4)

# Multiple supporting lines through origin with slopes in [-1, 1]
slopes = [-1.0, -0.5, 0.0, 0.5, 1.0]
colors = [COPPER, COPPER, GREEN, COPPER, COPPER]
# clip the lines to the visible range only (we'll only draw for x in [-1.4, 1.4] so they don't escape)
xs_line = np.linspace(-1.4, 1.4, 200)
for s, c in zip(slopes, colors):
    ax.plot(xs_line, s*xs_line, color=c, linewidth=1.6, linestyle='--', alpha=0.85, zorder=2)
# place slope labels inside the visible area, on a tilted line
for s, c in zip(slopes, colors):
    if s > 0:
        lx = 1.5
        ly = s*1.4
        ax.text(lx, ly, f'slope ${s:+.1f}$', fontsize=10.5, color=c, va='center')
    elif s < 0:
        lx = 1.5
        ly = s*1.4
        ax.text(lx, ly, f'slope ${s:+.1f}$', fontsize=10.5, color=c, va='center')
    else:
        ax.text(1.5, 0.0, r'slope $0$', fontsize=10.5, color=c, va='center')

# Highlight origin
ax.scatter(0, 0, s=120, color=PURPLE, zorder=6, edgecolor='white', linewidth=1.8)
ax.text(0.05, -0.25, 'kink at $x=0$', fontsize=12, color=PURPLE)

# Annotate subdifferential — placed at bottom-left where there's empty space
ax.text(-2.0, -1.55,
        r'$\partial f(0) = [-1,\, 1]$' + '\nevery slope in this interval gives a tangent lower bound',
        fontsize=11.5, color=GRAY, ha='left', va='bottom', style='italic')

ax.axhline(0, color=GRAY, linewidth=0.6)
ax.axvline(0, color=GRAY, linewidth=0.6)
ax.set_xlim(-2.1, 2.6)
ax.set_ylim(-1.8, 2.3)
ax.set_title(r'Subgradients of $|x|$ at the kink $x = 0$', fontsize=13.5, color=GRAY, pad=12)
ax.legend(loc='upper right', fontsize=11, frameon=False)
style_ax(ax, keep_axes=True)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig('/tmp/fig5_subgradient.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("fig5 done")

print("ALL DONE")
