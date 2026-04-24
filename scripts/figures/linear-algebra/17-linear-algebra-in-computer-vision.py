"""
Figure generation script for Linear Algebra Chapter 17:
Linear Algebra in Computer Vision.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_image_as_matrix         A grayscale image is a matrix; an RGB image
                                 is an H x W x 3 tensor. Show the gray array
                                 with pixel values overlaid plus the three
                                 stacked R/G/B channels.
    fig2_convolution_kernels     Three classic convolution kernels (edge,
                                 blur, sharpen) with their 3 x 3 stencils
                                 and the resulting filtered image.
    fig3_affine_transforms       Rotation, scaling, and translation applied
                                 to a synthetic image, with the 3 x 3
                                 homogeneous matrices written out.
    fig4_homography              A photographed plane (skewed quad) rectified
                                 to a frontal rectangle by a homography H,
                                 with point correspondences highlighted.
    fig5_camera_projection       Pinhole camera model: a 3D wireframe cube
                                 in world coordinates projected onto the
                                 2D image plane via P = K [R | t].
    fig6_svd_compression         SVD low-rank reconstruction of an image at
                                 several ranks plus the singular value
                                 spectrum and cumulative energy curve.
    fig7_optical_flow            Optical flow as a displacement vector field:
                                 two consecutive frames of a moving disk and
                                 the per-pixel arrows estimated by Lucas-
                                 Kanade-style local least squares.

Usage:
    python3 scripts/figures/linear-algebra/17-linear-algebra-in-computer-vision.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150
RNG = np.random.default_rng(20240429)

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "17-linear-algebra-in-computer-vision"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "17-计算机视觉中的线性代数"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------
def synthetic_scene(size: int = 128) -> np.ndarray:
    """A small synthetic 'scene': sky gradient, ground, sun, two buildings.

    Returns a float array in [0, 1] suitable for grayscale display.
    """
    h = w = size
    img = np.zeros((h, w), dtype=np.float64)

    # Sky vertical gradient (top dark, bottom lighter)
    yy = np.linspace(0.25, 0.75, h)[:, None] * np.ones((1, w))
    img[: int(0.6 * h)] = yy[: int(0.6 * h)]

    # Ground
    img[int(0.6 * h):] = 0.35

    # Sun
    cy, cx = int(0.25 * h), int(0.78 * w)
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[dist < 0.07 * size] = 0.95

    # Two rectangular buildings
    img[int(0.45 * h): int(0.85 * h), int(0.10 * w): int(0.30 * w)] = 0.15
    img[int(0.35 * h): int(0.85 * h), int(0.40 * w): int(0.55 * w)] = 0.55

    # Windows on the bigger building
    for ry in (0.42, 0.55, 0.68):
        for rx in (0.43, 0.48):
            r0 = int(ry * h)
            c0 = int(rx * w)
            img[r0: r0 + 4, c0: c0 + 3] = 0.95

    # A thin horizon strip (roof line)
    img[int(0.85 * h): int(0.86 * h)] = 0.05
    return np.clip(img, 0.0, 1.0)


def synthetic_rgb(size: int = 64) -> np.ndarray:
    """Tiny RGB image with three obviously-different channels."""
    h = w = size
    img = np.zeros((h, w, 3), dtype=np.float64)
    yy, xx = np.indices((h, w))
    # R: warm sun in upper right
    img[..., 0] = np.exp(-((xx - 0.75 * w) ** 2 + (yy - 0.25 * h) ** 2) /
                         (0.18 * size) ** 2)
    # G: ground gradient
    img[..., 1] = np.clip((yy / h - 0.4) * 1.6, 0, 1) * 0.85
    # B: sky tint
    img[..., 2] = np.clip(1.0 - yy / h, 0, 1) * 0.85
    # Add a building (low everywhere)
    img[int(0.45 * h):, int(0.15 * w): int(0.35 * w)] = (
        np.array([0.25, 0.25, 0.30])
    )
    return np.clip(img, 0.0, 1.0)


def convolve2d_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Naive 2D correlation with zero padding, returning same-size output."""
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * pad[i: i + img.shape[0], j: j + img.shape[1]]
    return out


# ---------------------------------------------------------------------------
# Figure 1: Image as a matrix and as an RGB tensor
# ---------------------------------------------------------------------------
def fig1_image_as_matrix() -> None:
    # Tiny 8 x 8 grayscale patch so we can label every pixel value.
    small = np.array([
        [0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.65, 0.70],
        [0.10, 0.14, 0.20, 0.30, 0.45, 0.65, 0.78, 0.82],
        [0.12, 0.18, 0.30, 0.50, 0.70, 0.85, 0.92, 0.94],
        [0.15, 0.25, 0.45, 0.70, 0.88, 0.95, 0.96, 0.95],
        [0.20, 0.35, 0.60, 0.82, 0.92, 0.90, 0.78, 0.65],
        [0.25, 0.40, 0.55, 0.62, 0.55, 0.42, 0.30, 0.22],
        [0.30, 0.32, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18],
        [0.35, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22],
    ])
    rgb = synthetic_rgb(80)

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22)

    # Left: grayscale matrix with overlaid numbers
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(small, cmap="gray", vmin=0, vmax=1)
    for i in range(small.shape[0]):
        for j in range(small.shape[1]):
            v = small[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.55 else C_DARK, fontsize=8)
    ax1.set_xticks(range(small.shape[1]))
    ax1.set_yticks(range(small.shape[0]))
    ax1.set_xticklabels([f"{j}" for j in range(small.shape[1])], fontsize=8)
    ax1.set_yticklabels([f"{i}" for i in range(small.shape[0])], fontsize=8)
    ax1.set_xlabel("column $j$", fontsize=10)
    ax1.set_ylabel("row $i$", fontsize=10)
    ax1.set_title(r"Grayscale image $\mathbf{I} \in \mathbb{R}^{H\times W}$, "
                  r"$I_{ij}\in[0,1]$",
                  color=C_DARK, fontsize=12)
    ax1.grid(False)

    # Right: RGB tensor as three stacked channels + composite
    gs_r = gs[0, 1].subgridspec(2, 3, hspace=0.30, wspace=0.18,
                                height_ratios=[1.0, 0.9])
    ax_r = fig.add_subplot(gs_r[0, 0])
    ax_g = fig.add_subplot(gs_r[0, 1])
    ax_b = fig.add_subplot(gs_r[0, 2])
    ax_full = fig.add_subplot(gs_r[1, :])

    ax_r.imshow(rgb[..., 0], cmap="Reds", vmin=0, vmax=1)
    ax_r.set_title("R channel", color="#dc2626", fontsize=10)
    ax_g.imshow(rgb[..., 1], cmap="Greens", vmin=0, vmax=1)
    ax_g.set_title("G channel", color="#16a34a", fontsize=10)
    ax_b.imshow(rgb[..., 2], cmap="Blues", vmin=0, vmax=1)
    ax_b.set_title("B channel", color="#2563eb", fontsize=10)
    for ax in (ax_r, ax_g, ax_b):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax_full.imshow(rgb)
    ax_full.set_xticks([]); ax_full.set_yticks([]); ax_full.grid(False)
    ax_full.set_title(r"Composite RGB tensor $\mathcal{I}\in"
                      r"\mathbb{R}^{H\times W\times 3}$",
                      color=C_DARK, fontsize=11)

    fig.suptitle("An image is just numbers: matrix (gray) or 3-channel tensor (color)",
                 color=C_DARK, fontsize=13, y=1.00)
    save(fig, "fig1_image_as_matrix")


# ---------------------------------------------------------------------------
# Figure 2: Convolution kernels - edge / blur / sharpen
# ---------------------------------------------------------------------------
def fig2_convolution_kernels() -> None:
    img = synthetic_scene(160)

    # Three kernels
    k_edge = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], dtype=np.float64)        # Laplacian
    k_blur = np.ones((5, 5), dtype=np.float64) / 25.0            # 5x5 box
    k_sharp = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]], dtype=np.float64)        # sharpen

    # Outputs (use a 3x3 box for "blur" display kernel matrix, but apply 5x5)
    out_edge = convolve2d_same(img, k_edge)
    out_blur = convolve2d_same(img, k_blur)
    out_sharp = convolve2d_same(img, k_sharp)

    # Normalise edges for display
    e = out_edge - out_edge.min()
    e = e / (e.max() + 1e-9)

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.6),
                             gridspec_kw={"width_ratios": [0.9, 1.0, 1.0],
                                          "wspace": 0.28, "hspace": 0.32})

    titles = ["Edge detection (Laplacian)",
              "Blur (box / mean filter)",
              "Sharpen"]
    kernels_display = [k_edge,
                       np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0,
                       k_sharp]
    outputs = [e, out_blur, np.clip(out_sharp, 0, 1)]
    accent = [C_AMBER, C_GREEN, C_PURPLE]

    for row, (title, k, out, col) in enumerate(
            zip(titles, kernels_display, outputs, accent)):
        # Column 0: kernel as a small grid with numbers
        ax_k = axes[row, 0]
        ax_k.imshow(np.zeros_like(k), cmap="gray", vmin=0, vmax=1, alpha=0.0)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                ax_k.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=col, alpha=0.10,
                                         edgecolor=col, linewidth=1.2))
                v = k[i, j]
                txt = f"{v:.2f}" if abs(v - round(v)) > 1e-3 else f"{int(round(v))}"
                ax_k.text(j, i, txt, ha="center", va="center",
                          color=C_DARK, fontsize=11)
        ax_k.set_xlim(-0.6, k.shape[1] - 0.4)
        ax_k.set_ylim(k.shape[0] - 0.4, -0.6)
        ax_k.set_aspect("equal")
        ax_k.set_xticks([]); ax_k.set_yticks([])
        ax_k.set_title(f"Kernel $K$\n{title}", color=col, fontsize=11)
        ax_k.grid(False)

        # Column 1: original image
        ax_in = axes[row, 1]
        ax_in.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax_in.set_title("Input image $I$", fontsize=10, color=C_DARK)
        ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.grid(False)

        # Column 2: output
        ax_out = axes[row, 2]
        ax_out.imshow(out, cmap="gray", vmin=0, vmax=1)
        ax_out.set_title(r"Output $G = I * K$", fontsize=10, color=col)
        ax_out.set_xticks([]); ax_out.set_yticks([]); ax_out.grid(False)

    fig.suptitle("Convolution kernels: a 3 x 3 matrix decides what the filter sees",
                 color=C_DARK, fontsize=13, y=0.995)
    save(fig, "fig2_convolution_kernels")


# ---------------------------------------------------------------------------
# Figure 3: Affine transforms (rotation / scale / translation) on an image
# ---------------------------------------------------------------------------
def warp_affine(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous affine M to an image using nearest-neighbour
    inverse warping. Output canvas matches input size, centered."""
    h, w = img.shape
    out = np.full_like(img, fill_value=1.0)  # white background
    Minv = np.linalg.inv(M)
    yy, xx = np.indices((h, w))
    ones = np.ones_like(xx)
    pts = np.stack([xx - w / 2, yy - h / 2, ones], axis=-1).astype(np.float64)
    src = pts @ Minv.T
    sx = src[..., 0] + w / 2
    sy = src[..., 1] + h / 2
    sxi = np.round(sx).astype(int)
    syi = np.round(sy).astype(int)
    valid = (sxi >= 0) & (sxi < w) & (syi >= 0) & (syi < h)
    out[valid] = img[syi[valid], sxi[valid]]
    return out


def fig3_affine_transforms() -> None:
    img = synthetic_scene(160)
    h, w = img.shape

    theta = np.deg2rad(25)
    c, s = np.cos(theta), np.sin(theta)
    M_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    M_scl = np.array([[1.4, 0, 0], [0, 0.7, 0], [0, 0, 1]])
    M_trn = np.array([[1, 0, 28], [0, 1, -22], [0, 0, 1]])

    rot = warp_affine(img, M_rot)
    scl = warp_affine(img, M_scl)
    trn = warp_affine(img, M_trn)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0),
                             gridspec_kw={"wspace": 0.18})

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original", color=C_DARK, fontsize=12)

    panels = [
        (rot, "Rotation $R(25^\\circ)$", C_BLUE,
         "[ cosθ  -sinθ   0 ]\n"
         "[ sinθ   cosθ   0 ]\n"
         "[   0       0     1 ]"),
        (scl, "Scale $(s_x, s_y) = (1.4, 0.7)$", C_PURPLE,
         "[ sx   0    0 ]\n"
         "[  0   sy   0 ]\n"
         "[  0    0    1 ]"),
        (trn, "Translation $(t_x, t_y) = (28, -22)$", C_AMBER,
         "[ 1   0   tx ]\n"
         "[ 0   1   ty ]\n"
         "[ 0   0    1 ]"),
    ]

    for ax, (im, name, col, mat) in zip(axes[1:], panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, color=col, fontsize=11)
        ax.text(0.5, -0.05, mat, transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color=col,
                family="monospace")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig.suptitle("Affine transforms in homogeneous coordinates: linear part + translation",
                 color=C_DARK, fontsize=13, y=1.02)
    save(fig, "fig3_affine_transforms")


# ---------------------------------------------------------------------------
# Figure 4: Homography for perspective correction
# ---------------------------------------------------------------------------
def warp_perspective(img: np.ndarray, H: np.ndarray,
                     out_size: tuple[int, int]) -> np.ndarray:
    """Inverse-warp an image by a 3x3 homography H to a destination canvas."""
    H_inv = np.linalg.inv(H)
    out_h, out_w = out_size
    out = np.full((out_h, out_w), 1.0, dtype=np.float64)
    yy, xx = np.indices((out_h, out_w))
    ones = np.ones_like(xx)
    pts = np.stack([xx, yy, ones], axis=-1).astype(np.float64)
    src = pts @ H_inv.T
    sx = src[..., 0] / src[..., 2]
    sy = src[..., 1] / src[..., 2]
    sxi = np.round(sx).astype(int)
    syi = np.round(sy).astype(int)
    valid = ((sxi >= 0) & (sxi < img.shape[1]) &
             (syi >= 0) & (syi < img.shape[0]))
    out[valid] = img[syi[valid], sxi[valid]]
    return out


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT homography from 4+ point pairs (Nx2 each)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def fig4_homography() -> None:
    # Build a "frontal" page: white rectangle with text-like horizontal stripes.
    src_w, src_h = 200, 280
    page = np.full((src_h, src_w), 1.0)
    # Horizontal text rows
    for y in range(30, src_h - 20, 22):
        for x in range(20, src_w - 20, 14):
            length = int(8 + 4 * np.sin(0.13 * x + 0.07 * y))
            page[y: y + 4, x: x + length] = 0.18
    # A title bar and a footer
    page[10: 22, 30: src_w - 30] = 0.10
    # A border
    page[:6] = page[-6:] = 0.45
    page[:, :6] = page[:, -6:] = 0.45

    # Camera-view quad (perspective skew)
    out_w, out_h = 320, 280
    src_pts = np.array([[0, 0], [src_w - 1, 0],
                        [src_w - 1, src_h - 1], [0, src_h - 1]], float)
    dst_pts = np.array([[55, 35], [275, 70],
                        [255, 245], [40, 220]], float)
    H_forward = estimate_homography(src_pts, dst_pts)
    captured = warp_perspective(page, H_forward, (out_h, out_w))

    # Now rectify back: H_inverse from captured quad to a clean rectangle.
    rect_w, rect_h = 220, 300
    dst_rect = np.array([[0, 0], [rect_w - 1, 0],
                         [rect_w - 1, rect_h - 1], [0, rect_h - 1]], float)
    H_back = estimate_homography(dst_pts, dst_rect)
    rectified = warp_perspective(captured, H_back, (rect_h, rect_w))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0),
                             gridspec_kw={"wspace": 0.18,
                                          "width_ratios": [1, 1.1, 0.85]})

    # Panel 1: captured image with marked corners
    ax0 = axes[0]
    ax0.imshow(captured, cmap="gray", vmin=0, vmax=1)
    quad = Polygon(dst_pts, closed=True, fill=False,
                   edgecolor=C_AMBER, linewidth=2.0)
    ax0.add_patch(quad)
    for i, (u, v) in enumerate(dst_pts):
        ax0.scatter(u, v, s=70, color=C_AMBER, zorder=5,
                    edgecolor="white", linewidth=1.4)
        ax0.text(u + 6, v - 6, f"$x_{i + 1}$", color=C_AMBER, fontsize=11)
    ax0.set_title("Captured image (skewed)",
                  color=C_DARK, fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.grid(False)

    # Panel 2: arrow / equation
    ax1 = axes[1]
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.add_patch(FancyArrowPatch((0.05, 0.5), (0.95, 0.5),
                                   arrowstyle="-|>", mutation_scale=22,
                                   color=C_BLUE, linewidth=2.5))
    ax1.text(0.5, 0.62, r"Apply homography $H^{-1}$",
             ha="center", va="bottom", fontsize=13, color=C_BLUE)
    ax1.text(0.5, 0.38,
             "(u', v', w')ᵀ = H · (u, v, 1)ᵀ\n"
             "x = u'/w',   y = v'/w'",
             ha="center", va="top", fontsize=11, color=C_DARK,
             family="monospace")
    ax1.text(0.5, 0.18,
             "4 point correspondences fix the 8 DoF\n"
             "of $H$ via DLT + SVD.",
             ha="center", va="top", fontsize=10, color=C_GRAY)

    # Panel 3: rectified frontal view
    ax2 = axes[2]
    ax2.imshow(rectified, cmap="gray", vmin=0, vmax=1)
    rect = Polygon(dst_rect, closed=True, fill=False,
                   edgecolor=C_GREEN, linewidth=2.0)
    ax2.add_patch(rect)
    ax2.set_title("Rectified front view",
                  color=C_GREEN, fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)

    fig.suptitle("Homography in action: perspective correction of a planar surface",
                 color=C_DARK, fontsize=13, y=1.00)
    save(fig, "fig4_homography")


# ---------------------------------------------------------------------------
# Figure 5: Camera projection matrix - 3D world to 2D image
# ---------------------------------------------------------------------------
def fig5_camera_projection() -> None:
    # Build a unit cube wireframe centered at origin
    s = 1.0
    verts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s,  s], [s, -s,  s], [s, s,  s], [-s, s,  s],
    ], dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    # Move it into the camera's field of view
    verts_world = verts + np.array([0.0, 0.0, 6.0])

    # Camera intrinsics
    f = 320.0
    cx, cy = 240.0, 180.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # Camera extrinsics: a small rotation so the cube looks 3D
    rx = np.deg2rad(-12)
    ry = np.deg2rad(18)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    R = Ry @ Rx
    # Translate so projected centroid lands near principal point
    centroid_world = verts_world.mean(axis=0)
    centroid_cam = R @ centroid_world
    # We want cam-centroid x,y to be 0 so it projects to (cx, cy)
    t = -np.array([centroid_cam[0], centroid_cam[1], 0.0])

    Xc = (R @ verts_world.T).T + t
    proj = (K @ Xc.T).T
    pix = proj[:, :2] / proj[:, [2]]

    fig = plt.figure(figsize=(13.0, 5.4))

    # Left: 3D cube
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for i, j in edges:
        xs = [verts_world[i, 0], verts_world[j, 0]]
        ys = [verts_world[i, 1], verts_world[j, 1]]
        zs = [verts_world[i, 2], verts_world[j, 2]]
        ax3d.plot(xs, ys, zs, color=C_BLUE, linewidth=1.7)
    ax3d.scatter(verts_world[:, 0], verts_world[:, 1], verts_world[:, 2],
                 color=C_BLUE, s=28)
    # Camera center and view rays
    ax3d.scatter([0], [0], [0], color=C_AMBER, s=80,
                 label="Camera center")
    for v in verts_world:
        ax3d.plot([0, v[0]], [0, v[1]], [0, v[2]],
                  color=C_AMBER, alpha=0.18, linewidth=0.9)
    # Image plane at z=2 (scaled symbolic)
    zp = 2.0
    plane_x = np.array([-1.5, 1.5, 1.5, -1.5, -1.5])
    plane_y = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    plane_z = np.full_like(plane_x, zp)
    ax3d.plot(plane_x, plane_y, plane_z, color=C_PURPLE, linewidth=1.5)
    ax3d.text(1.6, 0, zp, "image plane", color=C_PURPLE, fontsize=9)

    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title(r"3D world $\mathbf{X}_w = (X, Y, Z)$",
                   color=C_DARK, fontsize=11)
    ax3d.view_init(elev=15, azim=-65)
    ax3d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Right: 2D image
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, 480); ax2.set_ylim(360, 0)
    ax2.set_aspect("equal")
    for i, j in edges:
        ax2.plot([pix[i, 0], pix[j, 0]], [pix[i, 1], pix[j, 1]],
                 color=C_BLUE, linewidth=1.8)
    ax2.scatter(pix[:, 0], pix[:, 1], color=C_BLUE, s=32, zorder=5,
                edgecolor="white", linewidth=1.0)
    ax2.scatter([cx], [cy], color=C_AMBER, marker="+", s=120,
                linewidth=2.5, label="principal point $(c_x, c_y)$")
    ax2.set_title(r"2D image $\;\;\lambda\,(u,v,1)^\top = K\,[R\,|\,t]\,\mathbf{X}_w$",
                  color=C_DARK, fontsize=11)
    ax2.set_xlabel("$u$ (pixels)"); ax2.set_ylabel("$v$ (pixels)")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle("Pinhole camera projection: a 3 x 4 matrix sends 3D points to pixels",
                 color=C_DARK, fontsize=13, y=1.0)
    save(fig, "fig5_camera_projection")


# ---------------------------------------------------------------------------
# Figure 6: SVD low-rank image compression
# ---------------------------------------------------------------------------
def fig6_svd_compression() -> None:
    # Build a richer synthetic image: gradient + circles + stripes
    n = 180
    yy, xx = np.indices((n, n))
    img = 0.5 + 0.45 * np.sin(0.06 * xx) * np.cos(0.04 * yy)
    cy, cx = 60, 130
    img += 0.35 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)) / 200.0)
    img += 0.25 * np.exp(-(((xx - 50) ** 2 + (yy - 130) ** 2)) / 600.0)
    img += 0.10 * (((xx + yy) // 12) % 2)
    img = np.clip(img, 0, 1)

    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    ranks = [2, 8, 32]
    reconstructions = []
    for r in ranks:
        recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        reconstructions.append(np.clip(recon, 0, 1))

    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9],
                          hspace=0.38, wspace=0.25)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_orig.set_title(f"Original\n(rank = {len(S)})",
                      color=C_DARK, fontsize=11)
    ax_orig.set_xticks([]); ax_orig.set_yticks([]); ax_orig.grid(False)

    full_storage = n * n
    for i, (r, recon) in enumerate(zip(ranks, reconstructions)):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(recon, cmap="gray", vmin=0, vmax=1)
        compressed = r * (2 * n + 1)
        ratio = compressed / full_storage * 100
        ax.set_title(f"Rank-{r} reconstruction\nstorage: {ratio:.1f}% of original",
                     color=C_BLUE if r >= 8 else C_AMBER, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    # Bottom: singular value spectrum + cumulative energy
    ax_s = fig.add_subplot(gs[1, :2])
    idx = np.arange(1, len(S) + 1)
    ax_s.semilogy(idx, S, color=C_BLUE, linewidth=2.0,
                  marker="o", markersize=3)
    for r, col in zip(ranks, [C_AMBER, C_BLUE, C_GREEN]):
        ax_s.axvline(r, color=col, linestyle="--", linewidth=1.2, alpha=0.7)
        ax_s.text(r, S[0] * 1.3, f"r={r}", color=col,
                  ha="center", fontsize=9)
    ax_s.set_xlabel("Singular value index $i$")
    ax_s.set_ylabel(r"$\sigma_i$ (log scale)")
    ax_s.set_title("Singular value spectrum decays fast",
                   color=C_DARK, fontsize=11)

    ax_c = fig.add_subplot(gs[1, 2:])
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    ax_c.plot(idx, energy, color=C_PURPLE, linewidth=2.2)
    for r, col in zip(ranks, [C_AMBER, C_BLUE, C_GREEN]):
        e = energy[r - 1]
        ax_c.axvline(r, color=col, linestyle="--", linewidth=1.2, alpha=0.7)
        ax_c.scatter([r], [e], color=col, s=55, zorder=5)
        ax_c.text(r + 4, e - 0.04, f"{e * 100:.1f}% energy",
                  color=col, fontsize=9)
    ax_c.set_xlabel("Rank $k$")
    ax_c.set_ylabel("Cumulative energy fraction")
    ax_c.set_ylim(0, 1.04)
    ax_c.set_title("Most variance lives in the first few singular values",
                   color=C_DARK, fontsize=11)

    fig.suptitle(r"SVD image compression: $I \approx \sum_{i=1}^{k}\sigma_i u_i v_i^\top$",
                 color=C_DARK, fontsize=13, y=1.0)
    save(fig, "fig6_svd_compression")


# ---------------------------------------------------------------------------
# Figure 7: Optical flow as a displacement vector field
# ---------------------------------------------------------------------------
def make_disk_image(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.indices((size, size))
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = 0.85 - 0.65 * np.exp(-(d / r) ** 2 * 3.0)
    # Texture pattern so derivatives are non-trivial
    img += 0.10 * np.sin(0.35 * xx + 0.18 * yy)
    return np.clip(img, 0, 1)


def lucas_kanade(img1: np.ndarray, img2: np.ndarray,
                 win: int = 9, step: int = 12) -> tuple:
    """Compute sparse Lucas-Kanade flow on a regular grid.

    Returns (xs, ys, us, vs) for arrow plotting.
    """
    Iy, Ix = np.gradient(img1)
    It = img2 - img1
    h, w = img1.shape
    half = win // 2
    xs, ys, us, vs = [], [], [], []
    for y in range(half, h - half, step):
        for x in range(half, w - half, step):
            ix = Ix[y - half: y + half + 1, x - half: x + half + 1].ravel()
            iy = Iy[y - half: y + half + 1, x - half: x + half + 1].ravel()
            it = It[y - half: y + half + 1, x - half: x + half + 1].ravel()
            A = np.stack([ix, iy], axis=1)
            ATA = A.T @ A
            # Skip ill-conditioned (textureless) windows
            if np.linalg.cond(ATA) > 1e6:
                continue
            try:
                flow = np.linalg.solve(ATA, -A.T @ it)
            except np.linalg.LinAlgError:
                continue
            if np.linalg.norm(flow) < 0.05:
                continue
            xs.append(x); ys.append(y)
            us.append(flow[0]); vs.append(flow[1])
    return (np.array(xs), np.array(ys),
            np.array(us), np.array(vs))


def fig7_optical_flow() -> None:
    size = 140
    cx0, cy0, r = 50, 70, 18
    dx, dy = 14, 6  # ground-truth disk motion
    img1 = make_disk_image(size, cx0, cy0, r)
    img2 = make_disk_image(size, cx0 + dx, cy0 + dy, r)

    xs, ys, us, vs = lucas_kanade(img1, img2, win=9, step=10)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6),
                             gridspec_kw={"wspace": 0.22})

    axes[0].imshow(img1, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Frame $t$", color=C_DARK, fontsize=11)
    axes[1].imshow(img2, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Frame $t+1$ (object moved)",
                      color=C_DARK, fontsize=11)

    axes[2].imshow(img1, cmap="gray", vmin=0, vmax=1, alpha=0.55)
    if len(xs) > 0:
        axes[2].quiver(xs, ys, us, vs, color=C_BLUE, scale=1.0,
                       scale_units="xy", angles="xy", width=0.005,
                       headwidth=4.5, headlength=5.0)
    # Ground-truth arrow on the disk for reference
    axes[2].annotate("", xy=(cx0 + dx, cy0 + dy), xytext=(cx0, cy0),
                     arrowprops=dict(arrowstyle="->", color=C_AMBER,
                                     linewidth=2.2))
    legend = [
        Line2D([0], [0], color=C_BLUE, marker=">",
               markersize=8, linestyle="-", label="LK estimated flow"),
        Line2D([0], [0], color=C_AMBER, marker=">",
               markersize=8, linestyle="-", label="True object motion"),
    ]
    axes[2].legend(handles=legend, loc="upper right",
                   fontsize=9, framealpha=0.95)
    axes[2].set_title("Optical flow field $(u, v)$",
                      color=C_BLUE, fontsize=11)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig.suptitle("Optical flow: per-pixel displacement vectors from $I_x u + I_y v + I_t = 0$",
                 color=C_DARK, fontsize=13, y=1.02)
    save(fig, "fig7_optical_flow")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_image_as_matrix()
    fig2_convolution_kernels()
    fig3_affine_transforms()
    fig4_homography()
    fig5_camera_projection()
    fig6_svd_compression()
    fig7_optical_flow()
    print("All 7 figures written to:")
    print(f"  {EN_DIR}")
    print(f"  {ZH_DIR}")


if __name__ == "__main__":
    main()
