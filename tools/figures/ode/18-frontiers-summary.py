"""
Chapter 18 (FINALE): Frontiers and Series Summary.

Figures:
  fig1_concept_map.png        -- 18-chapter concept map (networkx graph) connecting topics
  fig2_neural_odes.png        -- ResNet -> Neural ODE: continuous-depth + adjoint trajectory
  fig3_method_selection.png   -- Method-selection flowchart for solving any ODE
  fig4_ode_ml_connection.png  -- Three faces of ODE+ML: Neural ODE, PINN, score-based diffusion
  fig5_series_journey.png     -- 18-chapter learning journey timeline (the send-off)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import networkx as nx
from scipy.integrate import solve_ivp

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

BLUE   = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN  = COLORS["success"]
RED    = COLORS["danger"]

OUT_DIRS = [
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/ode/18-advanced-topics-summary',
    '/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/ode/18-前沿专题与总结',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  saved {name}')


# ---------------------------------------------------------------------------
# fig1: 18-chapter concept map
# ---------------------------------------------------------------------------
def fig1_concept_map():
    print('Building fig1_concept_map...')
    fig, ax = plt.subplots(figsize=(15, 10))

    G = nx.DiGraph()

    # Group: foundations
    nodes = {
        # name : (x, y, group_color, label)
        'Ch1\nOrigins':        (0.04, 0.85, BLUE,   '1. Origins\n& Intuition'),
        'Ch2\nFirst-order':    (0.18, 0.92, BLUE,   '2. First-order\nMethods'),
        'Ch3\nLinear theory':  (0.34, 0.92, BLUE,   '3. Higher-order\nLinear'),
        'Ch4\nLaplace':        (0.50, 0.85, BLUE,   '4. Laplace\nTransform'),
        'Ch5\nSeries':         (0.66, 0.92, BLUE,   '5. Series &\nSpecial fns'),
        # systems / nonlinear core
        'Ch6\nSystems':        (0.18, 0.62, PURPLE, '6. Systems &\nMatrix exp'),
        'Ch7\nPhase plane':    (0.34, 0.62, PURPLE, '7. Phase\nPlane'),
        'Ch8\nStability':      (0.50, 0.62, PURPLE, '8. Nonlinear\nStability'),
        'Ch9\nChaos':          (0.66, 0.62, PURPLE, '9. Chaos &\nLorenz'),
        'Ch10\nBifurc.':       (0.82, 0.78, PURPLE, '10. Bifurcation\nTheory'),
        # numerical / BVP / PDE
        'Ch11\nNumerical':     (0.04, 0.40, GREEN,  '11. Numerical\nMethods'),
        'Ch12\nBVP':           (0.20, 0.40, GREEN,  '12. Boundary\nValue Problems'),
        'Ch13\nPDE intro':     (0.36, 0.40, GREEN,  '13. PDE\nIntroduction'),
        # applications
        'Ch14\nEpid.':         (0.55, 0.40, RED,    '14. Epidemiology\n(SIR)'),
        'Ch15\nEcology':       (0.72, 0.40, RED,    '15. Population\nDynamics'),
        'Ch16\nControl':       (0.88, 0.40, RED,    '16. Control\nTheory'),
        'Ch17\nPhysics':       (0.30, 0.18, RED,    '17. Physics &\nEngineering'),
        # finale
        'Ch18\nFrontiers':     (0.65, 0.18, COLORS["warning"], '18. Frontiers &\nSummary'),
    }
    for n, (x, y, c, lbl) in nodes.items():
        G.add_node(n, pos=(x, y), color=c, label=lbl)

    # Edges show topical dependencies
    edges = [
        # foundations chain
        ('Ch1\nOrigins', 'Ch2\nFirst-order'),
        ('Ch2\nFirst-order', 'Ch3\nLinear theory'),
        ('Ch3\nLinear theory', 'Ch4\nLaplace'),
        ('Ch3\nLinear theory', 'Ch5\nSeries'),
        # foundations -> systems
        ('Ch3\nLinear theory', 'Ch6\nSystems'),
        ('Ch4\nLaplace', 'Ch16\nControl'),
        # systems chain
        ('Ch6\nSystems', 'Ch7\nPhase plane'),
        ('Ch7\nPhase plane', 'Ch8\nStability'),
        ('Ch8\nStability', 'Ch9\nChaos'),
        ('Ch8\nStability', 'Ch10\nBifurc.'),
        # systems -> applications
        ('Ch7\nPhase plane', 'Ch15\nEcology'),
        ('Ch7\nPhase plane', 'Ch14\nEpid.'),
        ('Ch6\nSystems', 'Ch16\nControl'),
        # numerical / BVP
        ('Ch2\nFirst-order', 'Ch11\nNumerical'),
        ('Ch11\nNumerical', 'Ch12\nBVP'),
        ('Ch12\nBVP', 'Ch13\nPDE intro'),
        # applications -> physics & engineering
        ('Ch6\nSystems', 'Ch17\nPhysics'),
        ('Ch16\nControl', 'Ch17\nPhysics'),
        ('Ch11\nNumerical', 'Ch17\nPhysics'),
        # everything funnels into Chapter 18
        ('Ch9\nChaos', 'Ch18\nFrontiers'),
        ('Ch10\nBifurc.', 'Ch18\nFrontiers'),
        ('Ch13\nPDE intro', 'Ch18\nFrontiers'),
        ('Ch14\nEpid.', 'Ch18\nFrontiers'),
        ('Ch15\nEcology', 'Ch18\nFrontiers'),
        ('Ch17\nPhysics', 'Ch18\nFrontiers'),
        ('Ch16\nControl', 'Ch18\nFrontiers'),
    ]
    G.add_edges_from(edges)

    pos    = nx.get_node_attributes(G, 'pos')
    colors = [nodes[n][2] for n in G.nodes()]
    labels = {n: nodes[n][3] for n in G.nodes()}

    # Edges first
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12,
                           edge_color='gray', width=0.9, alpha=0.55,
                           connectionstyle='arc3,rad=0.06')
    # Nodes (rounded boxes via scatter then redraw labels in text)
    for n, (x, y) in pos.items():
        c = nodes[n][2]; lbl = nodes[n][3]
        is_finale = n == 'Ch18\nFrontiers'
        box = FancyBboxPatch(
            (x - 0.058, y - 0.038), 0.116, 0.076,
            boxstyle='round,pad=0.012',
            linewidth=2.5 if is_finale else 1.3,
            edgecolor=c, facecolor=c, alpha=0.78 if is_finale else 0.18,
        )
        ax.add_patch(box)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=8.5, fontweight='bold',
                color='white' if is_finale else 'black')

    # Group labels (legend-like)
    legend_items = [
        ('Foundations  (Ch 1-5)',        BLUE),
        ('Systems & nonlinear  (Ch 6-10)', PURPLE),
        ('Numerical & PDE  (Ch 11-13)',  GREEN),
        ('Applications  (Ch 14-17)',     RED),
        ('Finale (Ch 18)',               COLORS["warning"]),
    ]
    for i, (txt, c) in enumerate(legend_items):
        ax.add_patch(Rectangle((0.005, 0.005 + i * 0.04), 0.016, 0.025,
                               facecolor=c, alpha=0.85, edgecolor='black', lw=0.8))
        ax.text(0.028, 0.018 + i * 0.04, txt, fontsize=10,
                fontweight='bold', va='center')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('Concept Map of the 18-Chapter ODE Series  --  '
                 'how the ideas connect',
                 fontsize=15, fontweight='bold')
    save(fig, 'fig1_concept_map.png')


# ---------------------------------------------------------------------------
# fig2: ResNet -> Neural ODE
# ---------------------------------------------------------------------------
def fig2_neural_odes():
    print('Building fig2_neural_odes...')
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0],
                          hspace=0.45, wspace=0.30)

    # (a) ResNet vs Neural ODE depth diagram
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    # ResNet stack (discrete blocks)
    ax.text(2.2, 5.6, 'ResNet  $h_{t+1} = h_t + f(h_t,\\theta_t)$',
            ha='center', fontsize=12, fontweight='bold', color=BLUE)
    for i in range(5):
        y0 = 0.3 + i * 0.95
        ax.add_patch(Rectangle((1.3, y0), 1.8, 0.6,
                               facecolor=BLUE, alpha=0.25, edgecolor=BLUE, lw=1.5))
        ax.text(2.2, y0 + 0.3, f'layer {i+1}', ha='center', fontsize=9)
        if i < 4:
            ax.add_patch(FancyArrowPatch((2.2, y0 + 0.6), (2.2, y0 + 0.95),
                                         arrowstyle='->', mutation_scale=12,
                                         color='black'))

    # Neural ODE continuous "depth"
    ax.text(7.0, 5.6, r'Neural ODE  $\dfrac{dh}{dt} = f(h,t,\theta)$',
            ha='center', fontsize=12, fontweight='bold', color=RED)
    # Continuous gradient bar
    n_seg = 80
    ys = np.linspace(0.3, 5.0, n_seg)
    for i in range(n_seg - 1):
        col = plt.cm.plasma(i / n_seg)
        ax.add_patch(Rectangle((6.0, ys[i]), 1.8, ys[i+1] - ys[i],
                               facecolor=col, alpha=0.85, edgecolor='none'))
    ax.add_patch(Rectangle((6.0, 0.3), 1.8, 4.7, facecolor='none',
                           edgecolor=RED, lw=2.0))
    ax.text(6.9, 0.05, r'$t = 0$', ha='center', fontsize=10)
    ax.text(6.9, 5.15, r'$t = T$', ha='center', fontsize=10)
    ax.add_patch(FancyArrowPatch((4.0, 2.7), (5.7, 2.7),
                                 arrowstyle='->', mutation_scale=22, color='black', lw=1.6))
    ax.text(4.85, 3.0, 'depth $\\to \\infty$',
            ha='center', fontsize=10, style='italic')

    # (b) Continuous trajectory in 2D state-space (toy spiral)
    ax = fig.add_subplot(gs[0, 1])
    # A learnable vector field; here just a damped spiral with several initial conditions
    def f(t, h):
        x, y = h
        return [-0.3*x - 1.0*y, 1.0*x - 0.3*y]
    cmap = plt.cm.viridis
    for k, (x0, y0) in enumerate([(2, 0), (-1, 1.6), (0, -2), (-1.8, -1.0)]):
        sol = solve_ivp(f, (0, 8), [x0, y0],
                        t_eval=np.linspace(0, 8, 800), rtol=1e-9)
        # color by time
        for i in range(len(sol.t) - 1):
            ax.plot(sol.y[0, i:i+2], sol.y[1, i:i+2],
                    color=cmap(i / len(sol.t)), lw=1.6, alpha=0.85)
        ax.scatter(sol.y[0, 0],  sol.y[1, 0],  color=GREEN, s=60,
                   edgecolor='white', zorder=5)
        ax.scatter(sol.y[0, -1], sol.y[1, -1], color=RED, s=60,
                   edgecolor='white', zorder=5)
    # vector field arrows
    X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 18), np.linspace(-2.5, 2.5, 18))
    U = -0.3*X - 1.0*Y; V = 1.0*X - 0.3*Y
    ax.quiver(X, Y, U, V, color='gray', alpha=0.35, scale=40, width=0.003)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel(r'$h_1$'); ax.set_ylabel(r'$h_2$')
    ax.set_title('Hidden-state trajectory under a learned vector field',
                 fontsize=11, fontweight='bold')

    # (c) Adjoint backward-pass schematic
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    # Forward
    ax.text(2.5, 3.7, 'Forward integration', ha='center',
            fontsize=11, fontweight='bold', color=BLUE)
    ax.add_patch(FancyArrowPatch((1.0, 3.0), (4.5, 3.0),
                                 arrowstyle='->', mutation_scale=22,
                                 color=BLUE, lw=2.4))
    ax.text(0.85, 2.6, '$h(0)$', ha='center', fontsize=10)
    ax.text(4.65, 2.6, '$h(T)$', ha='center', fontsize=10)
    ax.text(2.75, 2.55, r'$\dot h = f(h, t, \theta)$', ha='center',
            fontsize=10, style='italic')

    # Backward (adjoint)
    ax.text(7.5, 3.7, 'Adjoint backward', ha='center',
            fontsize=11, fontweight='bold', color=RED)
    ax.add_patch(FancyArrowPatch((9.0, 1.4), (5.5, 1.4),
                                 arrowstyle='->', mutation_scale=22,
                                 color=RED, lw=2.4))
    ax.text(9.15, 1.0, '$a(T)=\\partial L/\\partial h$', fontsize=10)
    ax.text(5.35, 1.0, '$a(0)$', ha='center', fontsize=10)
    ax.text(7.25, 0.95, r'$\dot a = -a^{\!T}(\partial f/\partial h)$',
            ha='center', fontsize=10, style='italic')

    ax.text(5.0, 0.1, 'Memory cost of backprop:  $O(1)$ '
                     '-- no activation storage',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7',
                      edgecolor=RED))

    # (d) Toy training example: fit a damped sine with Euler steps vs ODE solve
    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 10, 400)
    target = np.exp(-0.2*t) * np.cos(2*t)
    # "ResNet" with N=10 layers approximates ODE with 10 Euler steps
    for N, color, label in [(5, BLUE, 'ResNet  N=5'),
                            (15, PURPLE, 'ResNet  N=15'),
                            (60, GREEN,  'ResNet  N=60'),
                            (None, RED,   'Neural ODE  (adaptive)')]:
        if N is None:
            ax.plot(t, target, color=color, lw=2.3, label=label)
        else:
            tt = np.linspace(0, 10, N + 1)
            yy = np.exp(-0.2*tt) * np.cos(2*tt)
            # interpolate piecewise linearly
            yy_full = np.interp(t, tt, yy)
            ax.plot(t, yy_full, color=color, lw=1.4, label=label, alpha=0.9)
    ax.set_xlabel('time / depth'); ax.set_ylabel('hidden state')
    ax.set_title('More layers in ResNet $\\approx$ finer Euler steps; ODE solver picks step size',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Neural ODEs:  deep networks as continuous dynamical systems',
                 fontsize=14, fontweight='bold', y=1.00)
    save(fig, 'fig2_neural_odes.png')


# ---------------------------------------------------------------------------
# fig3: method selection flowchart
# ---------------------------------------------------------------------------
def fig3_method_selection():
    print('Building fig3_method_selection...')
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis('off')

    def node(x, y, w, h, text, color, kind='box', fontsize=10, fontweight='bold'):
        if kind == 'diamond':
            # rotated square
            from matplotlib.patches import Polygon
            poly = Polygon([[x, y - h/2], [x + w/2, y], [x, y + h/2], [x - w/2, y]],
                           closed=True, facecolor=color, alpha=0.22,
                           edgecolor=color, lw=1.8)
            ax.add_patch(poly)
        else:
            box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle='round,pad=0.05',
                                 facecolor=color, alpha=0.22,
                                 edgecolor=color, lw=1.8)
            ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight)

    def arr(x1, y1, x2, y2, label='', color='black'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='->', mutation_scale=18,
                                     color=color, lw=1.4))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2, label, fontsize=9,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.85))

    # Top-level: encounter ODE
    node(8, 10.2, 4.2, 0.7, 'Encounter an ODE', PURPLE, fontsize=13)

    # Q1: linear?
    node(8, 9.0, 4.2, 0.8, 'Is it linear?', BLUE, kind='diamond',
         fontsize=11)
    arr(8, 9.85, 8, 9.4)

    # Linear path
    node(3.5, 7.5, 3.6, 0.7, 'Constant coefficients?', BLUE, kind='diamond',
         fontsize=10)
    arr(7.0, 8.7, 4.5, 7.7, 'YES (Ch 3-4)', color=BLUE)

    node(2.0, 6.2, 3.2, 0.7, 'Characteristic eqn (Ch 3)\nor Laplace (Ch 4)',
         BLUE, fontsize=10)
    arr(2.6, 7.15, 2.0, 6.55, 'YES', color=BLUE)

    node(5.0, 6.2, 3.2, 0.7, 'Series solution (Ch 5)\nFrobenius / Bessel',
         BLUE, fontsize=10)
    arr(4.4, 7.15, 5.0, 6.55, 'NO (singular)', color=BLUE)

    # Nonlinear path
    node(12.5, 7.5, 4.0, 0.8, 'Need a closed form?', PURPLE, kind='diamond',
         fontsize=11)
    arr(9.0, 8.7, 11.5, 7.7, 'NO (nonlinear)', color=PURPLE)

    node(11.0, 6.0, 3.4, 0.8, 'First-order tricks (Ch 2)\nseparable / exact /\nintegrating factor',
         PURPLE, fontsize=9)
    arr(11.5, 7.15, 11.0, 6.4, 'YES, simple', color=PURPLE)

    # Big numerical / qualitative branch (right of nonlinear)
    node(14.5, 5.4, 2.8, 0.9, 'Stiff?', RED, kind='diamond', fontsize=10)
    arr(13.5, 7.15, 14.5, 5.85, 'NO -> numeric', color=PURPLE)

    node(14.5, 4.2, 2.8, 0.7, 'BDF / Radau\n(Ch 11)', RED, fontsize=10)
    arr(14.5, 4.95, 14.5, 4.55, 'YES', color=RED)

    node(11.5, 4.2, 2.8, 0.7, 'RK45 / DOP853\n(Ch 11)', GREEN, fontsize=10)
    arr(13.6, 5.4, 12.4, 4.55, 'NO', color=GREEN)

    # Qualitative analysis
    node(8, 5.0, 3.2, 0.8, 'Qualitative behaviour?',
         GREEN, kind='diamond', fontsize=10)
    arr(8, 8.6, 8, 5.4, 'phase / stability', color=GREEN)

    node(4.5, 3.2, 3.4, 0.7, 'Phase plane (Ch 7)\nLyapunov (Ch 8)',
         GREEN, fontsize=10)
    arr(7.0, 4.7, 5.0, 3.55, 'planar', color=GREEN)

    node(8.5, 3.2, 3.4, 0.7, 'Bifurcation (Ch 10)\nChaos / Lorenz (Ch 9)',
         GREEN, fontsize=10)
    arr(8.0, 4.6, 8.5, 3.55, 'parameter / 3D', color=GREEN)

    # Bottom tier: specialised problem types
    node(2.0, 1.8, 3.4, 0.7, 'BVP -> shooting /\nfinite difference (Ch 12)',
         RED, fontsize=10)
    node(6.5, 1.8, 3.4, 0.7, 'PDE -> separation /\nspectral (Ch 13)',
         RED, fontsize=10)
    node(11.0, 1.8, 3.4, 0.7, 'Hamiltonian -> symplectic\nintegrator (Ch 11/17)',
         RED, fontsize=10)
    node(14.5, 1.8, 2.6, 0.7, 'Modern: Neural ODE,\nSDE, fractional (Ch 18)',
         COLORS["warning"], fontsize=10)

    arr(3.5, 6.55, 2.5, 2.15)
    arr(8.0, 4.6, 6.5, 2.15)
    arr(12.5, 4.6, 11.0, 2.15)
    arr(14.0, 4.6, 14.5, 2.15)

    ax.text(8, 0.5,
            'Real problems usually need TWO methods:  qualitative (phase/stability) + numerical (RK45)',
            ha='center', fontsize=11, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f3f4f6',
                      edgecolor='gray'))

    ax.set_title('Method-Selection Flowchart  --  matching the right tool to the right ODE',
                 fontsize=14, fontweight='bold', pad=10)
    save(fig, 'fig3_method_selection.png')


# ---------------------------------------------------------------------------
# fig4: ODE + ML connection
# ---------------------------------------------------------------------------
def fig4_ode_ml_connection():
    print('Building fig4_ode_ml_connection...')
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.40, wspace=0.30)

    # (a) Neural ODE: solve ODE with learnable f
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 4, 200)
    # learnt vs true vector fields (toy)
    true = np.sin(2*t) * np.exp(-0.2*t)
    learnt = true + 0.04 * np.sin(20*t) * np.exp(-0.1*t)
    ax.plot(t, true,   color=BLUE, lw=2.2, label='true trajectory')
    ax.plot(t, learnt, color=RED, lw=1.4, ls='--', label='Neural ODE fit')
    ax.set_xlabel('time'); ax.set_ylabel('h(t)')
    ax.set_title('Neural ODE\n$\\dot h = f_\\theta(h, t)$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # (b) PINN: residual + data
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 1, 80)
    # Suppose y'' = -pi^2 sin(pi x), y(0)=y(1)=0  => y = sin(pi x)
    true = np.sin(np.pi * x)
    rng = np.random.default_rng(3)
    noise = 0.08 * rng.standard_normal(8)
    xs = np.linspace(0.05, 0.95, 8)
    ys = np.sin(np.pi * xs) + noise
    pinn = true + 0.02 * np.sin(7 * np.pi * x)  # pretend fit
    ax.plot(x, true, color=BLUE, lw=2.0, label='exact PDE solution')
    ax.scatter(xs, ys, color=PURPLE, s=40, zorder=5, label='sparse data')
    ax.plot(x, pinn, color=RED, lw=1.4, ls='--', label='PINN  (data + residual)')
    ax.set_xlabel('x'); ax.set_ylabel('y(x)')
    ax.set_title('PINN\n$L = L_\\mathrm{data} + \\lambda L_\\mathrm{phys}$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', fontsize=8)

    # (c) Diffusion / score-based generation: dX = f(X)dt + g(X)dW reversed
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(7)
    n = 600
    # samples of "data" distribution (mixture of two gaussians)
    samples = np.concatenate([rng.normal(-1.5, 0.5, n//2),
                              rng.normal(1.0, 0.4, n//2)])
    # forward (data -> noise)
    bins = np.linspace(-5, 5, 60)
    ax.hist(samples, bins=bins, alpha=0.5, color=BLUE, density=True,
            label='data $p_0$')
    ax.hist(rng.normal(0, 1.6, n), bins=bins, alpha=0.45, color=RED,
            density=True, label='noise $p_T$')
    # arrows showing forward / reverse SDE
    ax.annotate('', xy=(2.0, 0.2), xytext=(-2.0, 0.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0, 0.23, 'forward SDE', ha='center', fontsize=9, color='gray')
    ax.annotate('', xy=(-2.0, 0.05), xytext=(2.0, 0.05),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
    ax.text(0, 0.08, 'reverse SDE  (learn score)', ha='center',
            fontsize=9, color=GREEN, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_xlabel('x'); ax.set_ylabel('density')
    ax.set_title('Score-based diffusion\n$dX = f\\,dt + g\\,dW$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # (d) Comparison table
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    rows = [
        ['Topic',                'Equation',                                          'Use of ODE/SDE',                              'Where it appears'],
        ['Neural ODE',          r'$\dot h = f_\theta(h,t)$',                          'Continuous-depth network; adjoint backprop',   'Time-series, normalising flows'],
        ['PINN',                r'$L = L_\mathrm{data} + \lambda L_\mathrm{phys}$',   'Penalty enforces ODE/PDE on neural net',       'Inverse problems, physics ML'],
        ['Score-based diffusion', r'$dX = f\,dt + g\,dW$',                            'Forward noise + learned reverse drift',        'Image / audio / molecule generation'],
        ['Optimal transport',    r'$\partial_t \rho + \nabla\cdot(\rho v) = 0$',      'Flow matching; continuous transport map',      'Generative modelling, alignment'],
        ['Hamiltonian net',     r'$\dot q = \partial_p H, \ \dot p = -\partial_q H$', 'Architectural prior on conserved energy',      'Mechanical systems, MD'],
    ]
    tbl = ax.table(cellText=rows, loc='center', cellLoc='left',
                   colWidths=[0.18, 0.26, 0.30, 0.26])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for j in range(4):
        tbl[(0, j)].set_facecolor('#e0e7ff')
        tbl[(0, j)].set_text_props(weight='bold')
    # color rows
    palette = [BLUE, PURPLE, GREEN, RED, COLORS["info"]]
    for i in range(1, 6):
        for j in range(4):
            tbl[(i, j)].set_facecolor(palette[i-1] if False else 'white')
            tbl[(i, 0)].set_text_props(color=palette[i-1], weight='bold')

    fig.suptitle('ODEs $\\leftrightarrow$ Modern Machine Learning  --  the same calculus, new machinery',
                 fontsize=14, fontweight='bold', y=1.00)
    save(fig, 'fig4_ode_ml_connection.png')


# ---------------------------------------------------------------------------
# fig5: 18-chapter journey diagram (the send-off)
# ---------------------------------------------------------------------------
def fig5_series_journey():
    print('Building fig5_series_journey...')
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 18.5); ax.set_ylim(-3, 5); ax.axis('off')

    # Timeline backbone (a smooth curve representing the journey)
    xs = np.linspace(0.5, 18, 400)
    ys = 0.6 * np.sin(xs * 0.55) * np.exp(-0.05 * xs)
    ax.plot(xs, ys, color=COLORS["text2"], lw=3, alpha=0.6)
    ax.plot(xs, ys, color=PURPLE, lw=1.2, alpha=0.9)

    # Chapter milestones along the curve
    chapters = [
        (1,  'Origins',      'A falling apple,\na heated bar', BLUE),
        (2,  'First-order',  'Separable, exact,\nlinear',       BLUE),
        (3,  'Linear theory','Superposition,\nresonance',       BLUE),
        (4,  'Laplace',      'Algebra in the\n$s$-domain',      BLUE),
        (5,  'Series',       'Bessel, Hermite,\nLegendre',      BLUE),
        (6,  'Systems',      'Matrix\nexponential',             PURPLE),
        (7,  'Phase plane',  'Geometry beats\nformulas',        PURPLE),
        (8,  'Stability',    'Lyapunov\nfunctions',             PURPLE),
        (9,  'Chaos',        'Lorenz butterfly,\nstrange attractor', PURPLE),
        (10, 'Bifurcation',  'Hopf, pitchfork,\ncodimension-2',  PURPLE),
        (11, 'Numerics',     'RK4, BDF,\nsymplectic',           GREEN),
        (12, 'BVP',          'Shooting, FD,\ncollocation',      GREEN),
        (13, 'PDE intro',    'Heat, wave,\nLaplace',            GREEN),
        (14, 'Epidemiology', 'SIR + $R_0$',                     RED),
        (15, 'Ecology',      'Lotka-Volterra,\ncompetition',    RED),
        (16, 'Control',      'PID, LQR,\nfeedback',             RED),
        (17, 'Physics',      'Pendulum, RLC,\norbits',          RED),
        (18, 'Frontiers',    'Neural ODE, SDE,\nfractional',    COLORS["warning"]),
    ]

    for i, (n, name, sub, color) in enumerate(chapters):
        x = 0.5 + (i + 0.5) * (17 / 18)
        y = 0.6 * np.sin(x * 0.55) * np.exp(-0.05 * x)
        # stem to label
        y_label = 2.4 if i % 2 == 0 else -2.0
        ax.plot([x, x], [y, y_label], color=color, lw=1.0, alpha=0.6)
        # node circle
        size = 280 if n == 18 else 150
        ax.scatter([x], [y], color=color, s=size, edgecolor='white',
                   linewidth=1.8, zorder=6)
        ax.text(x, y, str(n), ha='center', va='center', fontsize=9,
                fontweight='bold', color='white' if n != 18 else 'black')
        # label box
        box_y = y_label
        ax.text(x, box_y, f'{name}\n{sub}', ha='center',
                va='center' if i % 2 == 0 else 'center',
                fontsize=8.2, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.30',
                          facecolor='white', edgecolor=color, lw=1.4))

    # Start / End markers
    ax.text(0.3, 1.0, 'START', fontsize=11, fontweight='bold',
            color=BLUE, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dbeafe',
                      edgecolor=BLUE))
    ax.text(18.2, 1.0, 'FINALE', fontsize=11, fontweight='bold',
            color='#b45309', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                      edgecolor=COLORS["warning"]))

    # Horizontal scale: epochs of the journey
    epochs = [
        (1, 5,   'FOUNDATIONS\nch 1-5',              BLUE,   3.7),
        (6, 10,  'DYNAMICS & CHAOS\nch 6-10',        PURPLE, 3.7),
        (11, 13, 'COMPUTATION\nch 11-13',            GREEN,  3.7),
        (14, 17, 'APPLICATIONS\nch 14-17',           RED,    3.7),
        (18, 18, 'BEYOND\nch 18',                    COLORS["warning"], 3.7),
    ]
    for s, e, lbl, col, ypos in epochs:
        x1 = 0.5 + (s - 1 + 0.0) * (17 / 18)
        x2 = 0.5 + (e + 0.0) * (17 / 18)
        ax.plot([x1, x2], [ypos, ypos], color=col, lw=4, alpha=0.55)
        ax.text((x1 + x2) / 2, ypos + 0.5, lbl, ha='center',
                fontsize=10, fontweight='bold', color=col)

    # Closing message
    ax.text(9, -2.85,
            'Differential equations describe the laws of change.  '
            'Thank you for taking the journey.',
            ha='center', fontsize=12.5, style='italic', color='#1f2937',
            fontweight='bold')

    ax.set_title('The 18-Chapter ODE Journey  --  from Newton\'s apple to Neural ODEs',
                 fontsize=15, fontweight='bold', pad=16)
    save(fig, 'fig5_series_journey.png')


if __name__ == '__main__':
    fig1_concept_map()
    fig2_neural_odes()
    fig3_method_selection()
    fig4_ode_ml_connection()
    fig5_series_journey()
    print('\nChapter 18 figures complete.  Series finale ready.')
