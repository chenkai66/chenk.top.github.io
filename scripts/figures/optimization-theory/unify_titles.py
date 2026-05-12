#!/usr/bin/env python3
"""Shorten and unify all 12 optim article titles. Use ':' instead of '--' / '——'.
Match the Linux series style of brief, clean titles.
"""
import re
import os

# (filename, en_title, zh_title)
TITLES = [
    ('01-convex-analysis-foundations.md',
     'Optimization (1): Convex Analysis Foundations',
     '优化理论（一）：凸分析基础'),
    ('02-smoothness-strong-convexity-nesterov.md',
     'Optimization (2): Smoothness, Strong Convexity, and Nesterov Acceleration',
     '优化理论（二）：光滑性、强凸性与 Nesterov 加速'),
    ('03-gradient-descent-family.md',
     'Optimization (3): The Gradient Descent Family from SGD to AdamW',
     '优化理论（三）：梯度下降族——从 SGD 到 AdamW'),
    ('04-learning-rate-schedules.md',
     'Optimization (4): Learning Rate and Schedules',
     '优化理论（四）：学习率与调度策略'),
    ('05-acceleration-beyond-nesterov.md',
     'Optimization (5): Acceleration Beyond Nesterov',
     '优化理论（五）：Nesterov 之外的加速'),
    ('06-composite-proximal-methods.md',
     'Optimization (6): Composite Optimization and Proximal Methods',
     '优化理论（六）：复合优化与近端方法'),
    ('07-second-order-methods.md',
     'Optimization (7): Second-Order Methods',
     '优化理论（七）：二阶方法'),
    ('08-lagrangian-duality-kkt.md',
     'Optimization (8): Lagrangian Duality and KKT Conditions',
     '优化理论（八）：Lagrangian 对偶与 KKT 条件'),
    ('09-interior-point-barrier.md',
     'Optimization (9): Interior-Point Methods and Self-Concordant Barriers',
     '优化理论（九）：内点法与自和谐障碍函数'),
    ('10-stochastic-variance-reduction.md',
     'Optimization (10): Stochastic Optimization and Variance Reduction',
     '优化理论（十）：随机优化与方差缩减'),
    ('11-nonconvex-saddle-escape.md',
     'Optimization (11): Non-Convex Optimization and Saddle Escape',
     '优化理论（十一）：非凸优化与鞍点逃逸'),
    ('12-discrete-global-optimization.md',
     'Optimization (12): Discrete and Global Optimization',
     '优化理论（十二）：离散与全局优化'),
]

BASE = '/root/chenk-hugo/content'

def update_title(path, new_title):
    if not os.path.exists(path):
        print(f"  SKIP (missing): {path}")
        return
    with open(path) as f:
        content = f.read()
    # Use single-quoted title (most flexible, handles backslashes too)
    new_line = f"title: '{new_title.replace(chr(39), chr(39)+chr(39))}'"
    new_content = re.sub(r'^title:\s*.+$', new_line, content, count=1, flags=re.MULTILINE)
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"  ✓ {os.path.basename(path)}: {new_title}")
    else:
        print(f"  (no change): {path}")


for fname, en_title, zh_title in TITLES:
    update_title(os.path.join(BASE, 'en', 'optimization-theory', fname), en_title)
    update_title(os.path.join(BASE, 'zh', 'optimization-theory', fname), zh_title)

print("\nDone.")
