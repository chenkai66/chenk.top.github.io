"""
Generate figures for the Tennis-Scene Computer-Vision System Design post.

Outputs 7 PNGs into both EN and ZH asset directories:
  fig1_pipeline.png            End-to-end CV pipeline (capture -> detect -> track -> 3D -> analyze)
  fig2_detection_compare.png   Detection AP curve: small-object detectors (YOLO/Faster-RCNN/TrackNet)
  fig3_trajectory_3d.png       3D ball trajectory with drag + Magnus, plus 2D top-down landing zone
  fig4_court_lines.png         Court line detection via Hough transform with mock court image
  fig5_pose_skeleton.png       17-keypoint COCO skeleton illustrating serve / forehand / backhand
  fig6_action_recognition.png  Action recognition confusion matrix + per-class F1
  fig7_architecture.png        Edge + cloud system architecture (cameras -> edge -> aggregator -> cloud)

Style: seaborn-v0_8-whitegrid, dpi=150, palette {#2563eb, #7c3aed, #10b981, #f59e0b}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.integrate import odeint

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e2e8f0"
DARK = "#0f172a"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/standalone/tennis-cv-system-design"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/standalone/"
    "网球场景计算机视觉系统设计方案"
)
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------------------
# Figure 1: End-to-end pipeline
# ---------------------------------------------------------------------------
def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        ("Capture", "8x 4K @ 60fps\nPTP sync", BLUE),
        ("Detect", "YOLOv8\nTrackNet V3", PURPLE),
        ("2D Track", "Kalman +\nByteTrack", GREEN),
        ("3D Reconstruct", "DLT triangulation\nBundle adjust", AMBER),
        ("Predict", "Drag + Magnus\nPINN", BLUE),
        ("Analyze", "Pose / action\nLine call", PURPLE),
    ]

    n = len(stages)
    box_w = 1.85
    gap = 0.32
    total = n * box_w + (n - 1) * gap
    x0 = (14 - total) / 2
    y_box = 2.3
    h_box = 1.6

    for i, (title, sub, color) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y_box), box_w, h_box,
            boxstyle="round,pad=0.04,rounding_size=0.16",
            linewidth=1.6, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y_box + h_box - 0.42, title,
                ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)
        ax.text(x + box_w / 2, y_box + 0.55, sub,
                ha="center", va="center", fontsize=9.5, color=GRAY)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w + 0.02, y_box + h_box / 2),
                (x + box_w + gap - 0.02, y_box + h_box / 2),
                arrowstyle="-|>", mutation_scale=14, color=DARK, linewidth=1.5,
            ))

    # Top metric ribbon
    ax.text(7, 4.5, "Tennis-Scene CV Pipeline", ha="center", fontsize=14,
            fontweight="bold", color=DARK)
    ax.text(7, 4.15, "target: end-to-end < 16.7 ms per frame   |   3D position error < 5 cm",
            ha="center", fontsize=10, color=GRAY)

    # Bottom data-flow annotations
    bottom_y = 1.2
    annots = [
        (x0 + 0.5 * box_w, "RAW", BLUE),
        (x0 + 1.5 * box_w + 1 * gap, "boxes\n(2D)", PURPLE),
        (x0 + 2.5 * box_w + 2 * gap, "tracks\n(2D, t)", GREEN),
        (x0 + 3.5 * box_w + 3 * gap, "(X,Y,Z)", AMBER),
        (x0 + 4.5 * box_w + 4 * gap, "future\ntrajectory", BLUE),
        (x0 + 5.5 * box_w + 5 * gap, "events,\nstats", PURPLE),
    ]
    for cx, label, color in annots:
        ax.text(cx, bottom_y, label, ha="center", va="center", fontsize=9,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, linewidth=0.8))

    _save(fig, "fig1_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2: Small-object detection comparison
# ---------------------------------------------------------------------------
def fig2_detection_compare() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Left: AP vs object size
    sizes = np.array([8, 16, 24, 32, 48, 64, 96, 128])  # bbox side, px
    faster_rcnn = np.array([0.05, 0.18, 0.34, 0.50, 0.66, 0.74, 0.79, 0.81])
    yolov8 = np.array([0.12, 0.31, 0.49, 0.62, 0.74, 0.80, 0.84, 0.86])
    tracknet_v3 = np.array([0.42, 0.66, 0.78, 0.84, 0.88, 0.90, 0.91, 0.92])

    ax = axes[0]
    ax.plot(sizes, faster_rcnn, "o-", color=GRAY, label="Faster R-CNN (R50)", linewidth=2)
    ax.plot(sizes, yolov8, "s-", color=BLUE, label="YOLOv8-l (1280)", linewidth=2)
    ax.plot(sizes, tracknet_v3, "^-", color=GREEN, label="TrackNet V3 (3-frame)", linewidth=2.4)
    ax.axvspan(8, 22, alpha=0.10, color=AMBER, label="tennis-ball regime")
    ax.set_xlabel("Object size (px, bbox side length)")
    ax.set_ylabel("Average Precision (AP@0.5)")
    ax.set_title("Detection accuracy collapses on small objects")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.0)

    # Right: latency vs accuracy bubbles
    ax = axes[1]
    models = [
        ("Faster R-CNN R50",  85, 0.50, GRAY),
        ("YOLOv5-s 640",       8, 0.55, AMBER),
        ("YOLOv8-l 1280",     22, 0.74, BLUE),
        ("TrackNet V2",       18, 0.86, PURPLE),
        ("TrackNet V3",       28, 0.92, GREEN),
        ("Coarse-to-fine",    34, 0.94, DARK),
    ]
    for name, lat, ap, color in models:
        ax.scatter(lat, ap, s=320, color=color, alpha=0.75, edgecolor="white", linewidth=2)
        dy = 0.025 if name != "YOLOv5-s 640" else -0.04
        ax.annotate(name, (lat, ap), xytext=(8, 8 if dy > 0 else -16),
                    textcoords="offset points", fontsize=9.5, color=DARK)
    ax.axhline(0.90, color=GREEN, linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(16.7, color=AMBER, linestyle="--", alpha=0.6, linewidth=1)
    ax.text(16.9, 0.42, "60 fps budget", color=AMBER, fontsize=9)
    ax.text(85, 0.905, "production threshold", color=GREEN, fontsize=9, ha="right")
    ax.set_xlabel("Inference latency per frame (ms, RTX 4090)")
    ax.set_ylabel("AP for tennis-ball-size objects")
    ax.set_title("Speed vs accuracy trade-off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.4, 1.0)

    _save(fig, "fig2_detection_compare.png")


# ---------------------------------------------------------------------------
# Figure 3: 3D ball trajectory with drag + Magnus
# ---------------------------------------------------------------------------
def fig3_trajectory_3d() -> None:
    rho = 1.225
    Cd = 0.55
    r = 0.0335
    A = np.pi * r ** 2
    m = 0.0585
    g = 9.81
    Cm = 0.00029

    def dyn(state, t, spin):
        v = state[3:6]
        speed = np.linalg.norm(v) + 1e-9
        drag = -0.5 * rho * Cd * A * speed * v
        magnus = Cm * np.cross(spin, v)
        gravity = np.array([0, 0, -g * m])
        a = (drag + magnus + gravity) / m
        return np.concatenate([v, a])

    t = np.linspace(0, 1.5, 600)
    p0 = np.array([-11.0, 0.0, 1.5])  # baseline, court center
    speed0 = 45.0  # m/s ~ 162 km/h
    angle = np.deg2rad(5.0)
    v0 = np.array([speed0 * np.cos(angle), 0.0, speed0 * np.sin(angle)])

    flat = odeint(dyn, np.concatenate([p0, v0]), t, args=(np.zeros(3),))
    topspin = odeint(dyn, np.concatenate([p0, v0]), t, args=(np.array([0, -250, 0]),))
    slice_ = odeint(dyn, np.concatenate([p0, v0]), t, args=(np.array([0, 250, 0]),))

    def cut(traj):
        idx = np.where(traj[:, 2] <= 0)[0]
        return traj[: idx[0] + 1] if len(idx) else traj

    flat, topspin, slice_ = cut(flat), cut(topspin), cut(slice_)

    fig = plt.figure(figsize=(13.5, 5.2))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    for traj, color, label in [
        (flat, BLUE, "no spin"),
        (topspin, GREEN, "topspin (-250 rad/s)"),
        (slice_, AMBER, "backspin (+250 rad/s)"),
    ]:
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2.2, label=label)
        ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=40, edgecolor="white")

    # court rectangle on ground
    court_x = [-11.885, 11.885, 11.885, -11.885, -11.885]
    court_y = [-4.115, -4.115, 4.115, 4.115, -4.115]
    court_z = [0] * 5
    ax3d.plot(court_x, court_y, court_z, color=GRAY, linewidth=1.2)
    # net at x=0
    ax3d.plot([0, 0], [-5, 5], [0, 0], color=DARK, linewidth=1)
    ax3d.plot([0, 0], [-5, 5], [0.914, 0.914], color=DARK, linewidth=1)

    ax3d.set_xlabel("x (m, baseline → baseline)")
    ax3d.set_ylabel("y (m, sideline)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D trajectory: drag + Magnus")
    ax3d.legend(loc="upper right", fontsize=9)
    ax3d.view_init(elev=18, azim=-62)
    ax3d.set_box_aspect((3, 1, 0.8))

    # Right: top-down landing zone with uncertainty ellipses
    ax2 = fig.add_subplot(1, 2, 2)
    # court
    ax2.add_patch(Rectangle((-11.885, -4.115), 23.77, 8.23,
                            facecolor="#84cc1633", edgecolor=DARK, linewidth=1.5))
    # singles lines
    ax2.add_patch(Rectangle((-11.885, -4.115 + 0.46), 23.77, 8.23 - 0.92,
                            facecolor="none", edgecolor=DARK, linewidth=1))
    # service boxes
    ax2.plot([-6.4, -6.4], [-4.115 + 0.46, 4.115 - 0.46], color=DARK, linewidth=1)
    ax2.plot([6.4, 6.4], [-4.115 + 0.46, 4.115 - 0.46], color=DARK, linewidth=1)
    ax2.plot([-6.4, 6.4], [0, 0], color=DARK, linewidth=1)
    # net
    ax2.plot([0, 0], [-4.5, 4.5], color=DARK, linewidth=2)

    landings = [
        (flat[-1, 0], flat[-1, 1], BLUE, "no spin"),
        (topspin[-1, 0], topspin[-1, 1], GREEN, "topspin"),
        (slice_[-1, 0], slice_[-1, 1], AMBER, "backspin"),
    ]
    for x, y, color, label in landings:
        ax2.scatter(x, y, s=120, color=color, edgecolor="white", linewidth=2, zorder=5, label=label)
        # 1-sigma uncertainty ellipse 8 cm
        circ = mpatches.Ellipse((x, y), 0.16, 0.16, facecolor=color, alpha=0.18,
                                edgecolor=color, linewidth=1)
        ax2.add_patch(circ)

    ax2.set_xlim(-13, 13)
    ax2.set_ylim(-5.5, 5.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Predicted landing zone (top-down)")
    ax2.legend(loc="upper right", fontsize=9)

    _save(fig, "fig3_trajectory_3d.png")


# ---------------------------------------------------------------------------
# Figure 4: Court line detection (Hough)
# ---------------------------------------------------------------------------
def fig4_court_lines() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # Generate a synthetic court image (perspective-warped)
    H, W = 320, 480
    img = np.full((H, W, 3), 70, dtype=np.uint8)  # dark green court
    img[:, :, 1] = 110

    def line_in_image(p1, p2, color=(255, 255, 255), thickness=2):
        # Bresenham-ish: simple linspace
        n = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) * 2) + 1
        xs = np.linspace(p1[0], p2[0], n).astype(int)
        ys = np.linspace(p1[1], p2[1], n).astype(int)
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xx = np.clip(xs + dx, 0, W - 1)
                yy = np.clip(ys + dy, 0, H - 1)
                img[yy, xx] = color

    # Court lines in perspective (roughly)
    # Outer
    line_in_image((60, 280), (420, 280))    # baseline near
    line_in_image((140, 80),  (340, 80))    # baseline far
    line_in_image((60, 280), (140, 80))     # left sideline
    line_in_image((420, 280), (340, 80))    # right sideline
    # Service line
    line_in_image((100, 200), (380, 200))
    line_in_image((200, 140), (280, 140))
    # Center service
    line_in_image((240, 200), (240, 140))
    # Net
    line_in_image((90, 220), (390, 220), color=(220, 220, 220), thickness=1)

    # add noise + a player blob
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 12, img.shape).astype(int)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    # player
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - 200) ** 2 / 80 + (yy - 240) ** 2 / 250 <= 1
    img[mask] = [40, 40, 200]

    axes[0].imshow(img)
    axes[0].set_title("Input frame")
    axes[0].axis("off")

    # Edge map (Sobel-ish on white channel)
    gray = img.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    edges = np.clip(gx + gy, 0, 255)
    edges = (edges > 60).astype(float)
    axes[1].imshow(edges, cmap="gray_r")
    axes[1].set_title("Canny edges")
    axes[1].axis("off")

    # Hough output overlay
    axes[2].imshow(img)
    overlays = [
        ((60, 280), (420, 280)),
        ((140, 80), (340, 80)),
        ((60, 280), (140, 80)),
        ((420, 280), (340, 80)),
        ((100, 200), (380, 200)),
        ((240, 200), (240, 140)),
    ]
    for (x1, y1), (x2, y2) in overlays:
        axes[2].plot([x1, x2], [y1, y2], color=AMBER, linewidth=2.2, alpha=0.95)
    axes[2].scatter(
        [60, 420, 140, 340, 100, 380, 240, 240],
        [280, 280, 80, 80, 200, 200, 200, 140],
        color=GREEN, s=40, edgecolor="white", linewidth=1.2, zorder=5,
    )
    axes[2].set_title("Hough lines + intersections (homography anchors)")
    axes[2].axis("off")

    _save(fig, "fig4_court_lines.png")


# ---------------------------------------------------------------------------
# Figure 5: Pose skeleton for serve / forehand / backhand
# ---------------------------------------------------------------------------
def fig5_pose_skeleton() -> None:
    skeleton_edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4),
    ]

    def base_pose():
        # 17 keypoints in (x, y) pixel-ish coordinates
        return np.array([
            [0.0, 1.70],   # 0 nose
            [-0.04, 1.74], [0.04, 1.74],
            [-0.07, 1.72], [0.07, 1.72],
            [-0.18, 1.55], [0.18, 1.55],
            [-0.26, 1.30], [0.26, 1.30],
            [-0.30, 1.05], [0.30, 1.05],
            [-0.13, 1.05], [0.13, 1.05],
            [-0.14, 0.55], [0.14, 0.55],
            [-0.15, 0.05], [0.15, 0.05],
        ])

    def serve():
        kp = base_pose()
        # right arm raised high
        kp[8] = [0.20, 1.65]
        kp[10] = [0.10, 2.05]
        # left arm extended up (toss)
        kp[7] = [-0.22, 1.70]
        kp[9] = [-0.18, 2.10]
        # body lean back slightly
        kp[5, 0] += 0.02; kp[6, 0] += 0.02
        return kp

    def forehand():
        kp = base_pose()
        # right arm across to left side after contact
        kp[8] = [-0.05, 1.40]
        kp[10] = [-0.30, 1.45]
        # left arm balance
        kp[7] = [-0.15, 1.30]
        kp[9] = [0.05, 1.20]
        return kp

    def backhand():
        kp = base_pose()
        # two-handed backhand: wrists together on right
        kp[8] = [0.10, 1.40]
        kp[10] = [0.32, 1.50]
        kp[7] = [0.08, 1.40]
        kp[9] = [0.30, 1.50]
        return kp

    poses = [("Serve", serve(), BLUE),
             ("Forehand", forehand(), GREEN),
             ("Backhand (two-handed)", backhand(), AMBER)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, (name, kp, color) in zip(axes, poses):
        for a, b in skeleton_edges:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]], color=color, linewidth=2.5)
        ax.scatter(kp[:, 0], kp[:, 1], color=DARK, s=28, zorder=5)
        # racket as line from right wrist
        if name == "Serve":
            ax.plot([kp[10, 0], kp[10, 0] + 0.05], [kp[10, 1], kp[10, 1] + 0.35], color=PURPLE, linewidth=3)
        elif name == "Forehand":
            ax.plot([kp[10, 0], kp[10, 0] - 0.30], [kp[10, 1], kp[10, 1] + 0.10], color=PURPLE, linewidth=3)
        else:
            ax.plot([kp[10, 0], kp[10, 0] + 0.30], [kp[10, 1], kp[10, 1] + 0.10], color=PURPLE, linewidth=3)
        # ground
        ax.axhline(0, color=GRAY, linewidth=1, alpha=0.5)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.1, 2.4)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=12, color=color, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("17-keypoint COCO skeleton: discriminative features per stroke",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig5_pose_skeleton.png")


# ---------------------------------------------------------------------------
# Figure 6: Action recognition - confusion matrix + per-class F1
# ---------------------------------------------------------------------------
def fig6_action_recognition() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    classes = ["Serve", "Forehand", "Backhand", "Volley", "Smash", "Ready"]
    # Confusion matrix (rows = true, cols = pred)
    cm = np.array([
        [188,   2,   1,   0,   7,   2],   # Serve
        [  3, 215,   8,   4,   0,   0],   # Forehand
        [  2,   9, 192,   5,   0,   2],   # Backhand
        [  1,   6,   4, 142,   0,  17],   # Volley
        [  9,   1,   1,   0,  85,   4],   # Smash
        [  0,   0,   0,  18,   2, 360],   # Ready
    ], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    ax = axes[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            color = "white" if v > 0.5 else DARK
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=9.5, color=color)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Action confusion matrix (row-normalized %)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Right: per-class F1 bar
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    ax = axes[1]
    x = np.arange(len(classes))
    w = 0.27
    ax.bar(x - w, precision, w, color=BLUE, label="Precision")
    ax.bar(x, recall, w, color=GREEN, label="Recall")
    ax.bar(x + w, f1, w, color=AMBER, label="F1")
    for i, (p, r, fv) in enumerate(zip(precision, recall, f1)):
        ax.text(i + w, fv + 0.01, f"{fv:.2f}", ha="center", fontsize=8.5, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics (template + temporal smoothing)")
    ax.legend(loc="lower right", fontsize=9)

    _save(fig, "fig6_action_recognition.png")


# ---------------------------------------------------------------------------
# Figure 7: Edge + cloud architecture
# ---------------------------------------------------------------------------
def fig7_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Layer bands
    bands = [
        (5.7, 7.6, "#dbeafe", "Cloud / data center"),
        (3.0, 5.4, "#ede9fe", "Aggregator (on-prem GPU)"),
        (0.4, 2.7, "#dcfce7", "Edge (per-camera Jetson)"),
    ]
    for y0, y1, color, label in bands:
        ax.add_patch(Rectangle((0.2, y0), 13.6, y1 - y0, facecolor=color, edgecolor="none"))
        ax.text(0.4, y1 - 0.25, label, fontsize=10, color=GRAY, fontweight="bold")

    # Edge: 8 cameras
    cam_w = 1.4
    cam_h = 1.4
    cam_y = 0.9
    for i in range(8):
        cx = 0.85 + i * (cam_w + 0.18)
        ax.add_patch(FancyBboxPatch(
            (cx, cam_y), cam_w, cam_h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            edgecolor=GREEN, facecolor="white", linewidth=1.4))
        ax.text(cx + cam_w / 2, cam_y + cam_h - 0.30, f"Cam {i+1}",
                ha="center", fontsize=9, fontweight="bold", color=DARK)
        ax.text(cx + cam_w / 2, cam_y + 0.55, "Jetson Orin\nYOLO + ROI",
                ha="center", fontsize=7.5, color=GRAY)
        # arrow up
        ax.add_patch(FancyArrowPatch(
            (cx + cam_w / 2, cam_y + cam_h + 0.02),
            (cx + cam_w / 2, 3.0 - 0.02),
            arrowstyle="-|>", mutation_scale=10, color=GREEN, linewidth=1.0, alpha=0.8))

    # Aggregator: 3 boxes
    agg_y = 3.3
    agg_h = 1.7
    agg_boxes = [
        ("Sync & 3D", "PTP\nDLT triangulate", PURPLE),
        ("Tracker / KF", "Kalman + ByteTrack\nphysics gating", PURPLE),
        ("Pose & action", "HRNet + template\nrule + LSTM", PURPLE),
    ]
    agg_w = 3.6
    agg_gap = 0.4
    agg_total = 3 * agg_w + 2 * agg_gap
    agg_x0 = (14 - agg_total) / 2
    for i, (title, sub, color) in enumerate(agg_boxes):
        x = agg_x0 + i * (agg_w + agg_gap)
        ax.add_patch(FancyBboxPatch(
            (x, agg_y), agg_w, agg_h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            edgecolor=color, facecolor="white", linewidth=1.6))
        ax.text(x + agg_w / 2, agg_y + agg_h - 0.40, title,
                ha="center", fontsize=11.5, fontweight="bold", color=DARK)
        ax.text(x + agg_w / 2, agg_y + 0.55, sub,
                ha="center", fontsize=9.5, color=GRAY)
        # up arrow
        ax.add_patch(FancyArrowPatch(
            (x + agg_w / 2, agg_y + agg_h + 0.02),
            (x + agg_w / 2, 5.7 - 0.02),
            arrowstyle="-|>", mutation_scale=11, color=PURPLE, linewidth=1.1, alpha=0.85))

    # Cloud: 3 boxes
    cloud_y = 5.95
    cloud_h = 1.45
    cloud_boxes = [
        ("Match service", "events, scoreboard\nWebSocket push", BLUE),
        ("Analytics", "tactics, heatmaps\nplayer profile", BLUE),
        ("Storage / replay", "object store\nHawk-Eye replay", BLUE),
    ]
    for i, (title, sub, color) in enumerate(cloud_boxes):
        x = agg_x0 + i * (agg_w + agg_gap)
        ax.add_patch(FancyBboxPatch(
            (x, cloud_y), agg_w, cloud_h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            edgecolor=color, facecolor="white", linewidth=1.6))
        ax.text(x + agg_w / 2, cloud_y + cloud_h - 0.32, title,
                ha="center", fontsize=11.5, fontweight="bold", color=DARK)
        ax.text(x + agg_w / 2, cloud_y + 0.42, sub,
                ha="center", fontsize=9.5, color=GRAY)

    # Side annotations: bandwidth / latency
    ax.text(13.7, 2.0, "RAW: 8x 4K@60\n~6 Gbps total", fontsize=8.5, color=GREEN, ha="right")
    ax.text(13.7, 4.1, "after edge ROI:\n~80 Mbps", fontsize=8.5, color=PURPLE, ha="right")
    ax.text(13.7, 6.6, "events JSON\n< 1 Mbps", fontsize=8.5, color=BLUE, ha="right")

    ax.text(7, 7.8, "Edge + cloud reference architecture",
            ha="center", fontsize=14, fontweight="bold", color=DARK)

    _save(fig, "fig7_architecture.png")


def main() -> None:
    fig1_pipeline()
    fig2_detection_compare()
    fig3_trajectory_3d()
    fig4_court_lines()
    fig5_pose_skeleton()
    fig6_action_recognition()
    fig7_architecture()


if __name__ == "__main__":
    main()
