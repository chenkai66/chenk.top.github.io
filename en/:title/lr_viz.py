"""
Learning-rate visualization mini-lab.

This script generates:
- lr_schedules.png          : Constant vs Warmup+Cosine vs WSD
- gd_stability_1d.png       : 1D quadratic stable/borderline/unstable
- gd_path_2d.gif            : 2D quadratic descent path animation

Run:
  python lr_viz.py
"""

from __future__ import annotations

import math
import os
from typing import List

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _out_dir() -> str:
    # save next to this script (Hexo post asset folder)
    return os.path.dirname(os.path.abspath(__file__))


def plot_lr_schedules(
    total_steps: int = 400,
    warmup_steps: int = 40,
    cooldown_steps: int = 80,
    lr_max: float = 1.0,
    lr_min: float = 0.05,
) -> None:
    steps = np.arange(total_steps)

    def lr_cosine(t: int) -> float:
        if t < warmup_steps:
            return lr_max * (t + 1) / max(1, warmup_steps)
        tt = t - warmup_steps
        TT = max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * tt / TT))
        return lr_min + (lr_max - lr_min) * cos

    def lr_wsd(t: int) -> float:
        if t < warmup_steps:
            return lr_max * (t + 1) / max(1, warmup_steps)
        stable_end = total_steps - cooldown_steps
        if t < stable_end:
            return lr_max
        tt = t - stable_end
        frac = min(1.0, (tt + 1) / max(1, cooldown_steps))
        return lr_max + (lr_min - lr_max) * frac

    def lr_constant(_: int) -> float:
        return lr_max

    cos = np.array([lr_cosine(int(t)) for t in steps])
    wsd = np.array([lr_wsd(int(t)) for t in steps])
    const = np.array([lr_constant(int(t)) for t in steps])

    plt.figure(figsize=(9, 4.5), dpi=160)
    plt.plot(steps, const, label="Constant (no schedule)", linewidth=2)
    plt.plot(steps, cos, label="Warmup + Cosine", linewidth=2)
    plt.plot(steps, wsd, label="Warmup + Stable + Cooldown (WSD)", linewidth=2)
    plt.axvline(warmup_steps, color="k", alpha=0.2, linestyle="--")
    plt.axvline(total_steps - cooldown_steps, color="k", alpha=0.2, linestyle="--")
    plt.title("Learning-rate schedules (illustration)")
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(_out_dir(), "lr_schedules.png"))
    plt.close()


def plot_gd_stability_1d(a: float = 4.0, steps: int = 60, theta0: float = 2.0) -> None:
    # L(theta)=0.5*a*theta^2, stable if 0<eta<2/a
    eta_stable = 0.3  # < 2/a
    eta_border = 2.0 / a  # = 2/a (borderline)
    eta_unstable = 0.7  # > 2/a

    def run(eta: float) -> tuple[np.ndarray, np.ndarray]:
        th = theta0
        thetas: List[float] = []
        losses: List[float] = []
        for _ in range(steps):
            g = a * th
            th = th - eta * g
            thetas.append(th)
            losses.append(0.5 * a * th * th)
        return np.array(thetas), np.array(losses)

    th1, l1 = run(eta_stable)
    th2, l2 = run(eta_border)
    th3, l3 = run(eta_unstable)

    plt.figure(figsize=(9, 4.5), dpi=160)
    plt.subplot(1, 2, 1)
    plt.plot(th1, label=f"stable $\\eta={eta_stable}$")
    plt.plot(th2, label=f"borderline $\\eta={eta_border:.2f}$")
    plt.plot(th3, label=f"unstable $\\eta={eta_unstable}$")
    plt.title("Parameter trajectory $\\theta_t$")
    plt.xlabel("step")
    plt.ylabel("$\\theta$")
    plt.legend(frameon=False, fontsize=8)

    plt.subplot(1, 2, 2)
    plt.semilogy(l1 + 1e-18, label="stable")
    plt.semilogy(l2 + 1e-18, label="borderline")
    plt.semilogy(l3 + 1e-18, label="unstable")
    plt.title("Loss trajectory (log scale)")
    plt.xlabel("step")
    plt.ylabel("$L(\\theta)$")
    plt.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(_out_dir(), "gd_stability_1d.png"))
    plt.close()


def make_gd_path_2d_gif(
    ax: float = 1.0,
    ay: float = 12.0,
    eta: float = 0.12,
    x0: tuple[float, float] = (2.0, 1.5),
    steps: int = 45,
    duration_s: float = 0.08,
) -> None:
    # f(x,y)=0.5*(ax*x^2 + ay*y^2)
    x = np.array([x0[0], x0[1]], dtype=float)

    xs = np.linspace(-2.5, 2.5, 220)
    ys = np.linspace(-2.5, 2.5, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (ax * X**2 + ay * Y**2)

    path = [x.copy()]
    for _ in range(steps):
        g = np.array([ax * x[0], ay * x[1]])
        x = x - eta * g
        path.append(x.copy())

    frames = []
    for k in range(len(path)):
        fig = plt.figure(figsize=(5, 5), dpi=120)
        plt.contour(X, Y, Z, levels=25, linewidths=0.7, alpha=0.7)
        p = np.array(path[: k + 1])
        plt.plot(p[:, 0], p[:, 1], "-o", markersize=3, linewidth=1.5)
        plt.scatter([0], [0], s=40)
        plt.title(f"2D quadratic descent (step {k}/{len(path)-1}, $\\eta={eta}$)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.tight_layout()

        fig.canvas.draw()
        # Matplotlib >=3.8 recommends buffer_rgba; keep RGB for GIF
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = np.array(buf)[..., :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(os.path.join(_out_dir(), "gd_path_2d.gif"), frames, duration=duration_s)


def main() -> None:
    plot_lr_schedules()
    plot_gd_stability_1d()
    make_gd_path_2d_gif()
    print("Done. Assets written to:", _out_dir())


if __name__ == "__main__":
    main()

