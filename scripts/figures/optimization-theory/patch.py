from pathlib import Path

files = {
    "content/en/optimization-theory/04-learning-rate-schedules.md": {
        "anchor1": "Modern large-batch results (LAMB, LARS) extend this idea, but the basic message is unchanged: **LR and $B$ are tied**.",
        "insert1": "\n\n![Linear scaling rule and gradient noise vs batch size](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/04-learning-rate-schedules/fig8.png)\n\nLeft: empirically, the linear rule $\\eta \\propto B$ holds to within a few percent up to a critical batch size, then plateaus — past that point, more data per step buys you no extra LR headroom. Right: the gradient standard error shrinks as $1/\\sqrt B$, which is exactly *why* a larger batch can absorb a larger step.",
        "anchor2": "Either way: **always warm up Adam**. 1–5% of total steps for vision/CNN, 5–10% for LLMs and very large batches.",
        "insert2": "\n\n![Effect of warmup on early training: smoother loss and bounded gradient norm](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/04-learning-rate-schedules/fig9.png)\n\nThe failure mode is dramatic. Without warmup the gradient norm spikes far above the clip threshold in the first ~30 steps, the loss takes a sharp upward bump, and the run never quite catches up to the warmed-up curve. A few hundred warmup steps are often the difference between a training run that converges and one that diverges or stalls.",
    },
    "content/zh/optimization-theory/04-learning-rate-schedules.md": {
        "anchor1": "后来 LAMB、LARS 等大 batch 算法把这一思路又推了一步，但本质没变：**LR 和 $B$ 是绑在一起的**。",
        "insert1": "\n\n![Linear scaling rule 与梯度噪声随 batch size 的变化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/04-learning-rate-schedules/fig8.png)\n\n左图：经验上，线性规则 $\\eta \\propto B$ 在达到一个临界 batch size 之前都能在几个百分点以内成立，超过该点后开始走平——再加 batch 不会多出 LR 余量。右图：梯度标准误差以 $1/\\sqrt B$ 衰减，这正是“大 batch 能承受大步长”背后的原因。",
        "anchor2": "无论哪种解释，结论都一样：**Adam 永远要 warmup**。视觉/CNN 用 1–5% 的总步数；LLM 和超大 batch 推荐 5–10%。",
        "insert2": "\n\n![Warmup 对早期训练的影响：损失更平滑，梯度范数不超限](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/04-learning-rate-schedules/fig9.png)\n\n不加 warmup 的崩坏样子很可观：前 30 步左右梯度范数冲出 clip 阈值很高，损失出现明显隐凸，之后的训练也几乎追不上加了 warmup 的曲线。几百步的 warmup 往往就是“能收敛”与“发散或卡住”的分界。",
    },
}

for path, ops in files.items():
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for ai, ii in (("anchor1", "insert1"), ("anchor2", "insert2")):
        a = ops[ai]
        if a not in text:
            print("MISSING:", path, ai)
            continue
        new = a + ops[ii]
        text = text.replace(a, new, 1)
    p.write_text(text, encoding="utf-8")
    print("patched", path)
