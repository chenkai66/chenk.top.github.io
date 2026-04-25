---
title: "阿里云百炼实战（四）：万相视频生成端到端"
date: 2026-02-28 09:00:00
tags:
  - 阿里云百炼
  - DashScope
  - 万相
  - 视频生成
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 4
description: "wan2.5-t2v-plus 与 wan2.5-i2v-plus 的异步任务模式，一段带退避的轮询代码，以及那条让所有团队至少踩过一次的 24 小时结果过期规则。"
disableNunjucks: true
---

营销团队的需求：每天 30 条 5 秒产品短片，每条由一张主图加一行 prompt 生成，1 小时内必须出在 OSS 上。试过 Sora（没有国内发票通道）、可灵（限流 + 价高）、Pika（图生视频质感不行），最后选定万相（`wan2.5-i2v-plus`）。万相赢的不是视频本身明显更好，而是 API 异步优先、配额给得宽、一个工程师一下午能搭完。这一篇就是那一下午——加上后来踩过的所有坑。

## 两个主力模型，外加老一代 v2.1

| model_id | 模式 | 最大时长 | 备注 |
|---|---|---|---|
| `wan2.5-t2v-plus` | 文生视频 | 5s | 纯 prompt 生成的默认选择 |
| `wan2.5-i2v-plus` | 图生视频 | 5s | 用首帧锚定动作 |
| `wanx2.1-t2v-turbo` | T2V，更快、画质略低 | 5s | 适合预览/草图 |
| `wanx2.1-i2v-turbo` | I2V，更快、画质略低 | 5s | 同上 |

我上线用的就是 2.5 plus。2.1 turbo 在你确实需要"快 30%、便宜 50%、画质打折扣"时才用；按我经验用户更愿意多等两分钟换更好画质。两个 2.5 模型都封顶 **5 秒**。需要 10 秒就生成两段拼接——同一个模型还可以把首段最后一帧作为 i2v 输入来做"续杯"。

## 异步任务三步曲

万相每个调用都走同样的三步：

1. **创建任务。** POST 时带请求头 `X-DashScope-Async: enable`，立刻拿到 `task_id`。
2. **轮询。** GET `/api/v1/tasks/{task_id}`，直到 `task_status` 变成 `SUCCEEDED` 或 `FAILED`。
3. **下载。** 成功响应里有 `output.video_url`。**24 小时内下载完**——过期 URL 直接 404，视频从此消失。

24 小时过期是这条流水线最大的运维炸弹。我亲自踩过、也见过别的团队踩过：轮询拿到 URL → 记日志 → 下载阶段因为不相关的 bug 失败 → 第二天才发现。把这个 URL 当成"一次性下载链接"对待：拿到立刻下，存到自家 OSS，永远不要假设它明天还在。

## 接口

```
POST https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis
GET  https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}
```

鉴权 `Authorization: Bearer $DASHSCOPE_API_KEY`。POST 上的 async 请求头是**强制**的，不带就直接 400。

## 创建文生视频任务

先看裸 HTTP，过程更清楚，再看 SDK。

```python
import os
import requests

def create_t2v_task(prompt: str, size: str = "1280*720") -> str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "wan2.5-t2v-plus",
        "input": {
            "prompt": prompt,
        },
        "parameters": {
            "size": size,           # "1280*720"、"720*1280"、"960*960" 等
            "duration": 5,           # 2.5-plus 最大 5
            "prompt_extend": True,   # 让模型自动扩写 prompt 以提升效果
            "seed": None,            # 想可复现就显式传
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    return body["output"]["task_id"]
```

几个关键参数：

- **`size`**：用 `宽*高`（注意是 `*` 不是 `x`）。1280x720 是质感和成本的甜蜜点。竖屏小红书/抖音用 960x960 或 720x1280。Plus 模型最高支持 1920x1080。
- **`prompt_extend`**：默认 `True`。服务端会把你的一句 prompt 扩写得更丰富。常规场景留着开；想严格控艺术风格就关掉。
- **`seed`**：想复现就传。同 prompt + 同 seed = 同一段视频（不计极少量服务端非确定性）。
- **`duration`**：1-5。成本线性。我默认 5，差价不大。

## 图生视频任务

结构一样，模型换一下，多一个 `img_url`：

```python
def create_i2v_task(image_url: str, prompt: str) -> str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "wan2.5-i2v-plus",
        "input": {
            "img_url": image_url,
            "prompt": prompt,        # 描述运动/镜头，不要描述主体
        },
        "parameters": {
            "duration": 5,
            "prompt_extend": True,
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["output"]["task_id"]
```

i2v 两个不那么显然的点：

1. **prompt 要描述动作，不要描述主体。** 写"一位女生举着这瓶水，微笑，自然柔光"是错的——女生已经在你的图里。应该写"女生缓缓抬起瓶子，转向镜头，发丝轻轻飘动"。描述主体会触发模型重画，丢失原图保真度。
2. **输出宽高比跟随原图。** i2v 不传 `size`。原图 1080x1920，视频就是 1080x1920。

## 轮询：好好写

最朴素的写法 `while True: sleep(5); check()`，到了 50 个并发任务时一分钟会请求 600 次 `/tasks/{id}`，被限流。

我自己一直在用的轮询，先紧后松的退避：

```python
import time
import requests

def poll_task(task_id: str, timeout: int = 600) -> dict:
    """轮询直到 SUCCEEDED/FAILED 或超时，返回任务体。"""
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"}

    start = time.time()
    # 退避策略：开始紧（部分任务 30 秒就完成），之后逐步放宽
    delays = [5, 5, 5, 8, 10, 15, 20, 30, 30, 30]
    i = 0

    while time.time() - start < timeout:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        body = r.json()
        status = body["output"]["task_status"]

        if status == "SUCCEEDED":
            return body
        if status == "FAILED":
            raise RuntimeError(f"任务 {task_id} 失败: {body['output'].get('message', '无信息')}")
        if status == "UNKNOWN":
            raise RuntimeError(f"任务 {task_id} 状态未知，多半已过期")

        # PENDING 或 RUNNING——等
        time.sleep(delays[min(i, len(delays) - 1)])
        i += 1

    raise TimeoutError(f"任务 {task_id} 在 {timeout}s 内未完成")
```

`wan2.5-t2v-plus` 上 5 秒 720p 的典型时长 60-180 秒。外层超时 600s，从来没合理触发过。

## 串成一条：生成、轮询、下载、入库

```python
import os
import time
import requests

API = "https://dashscope.aliyuncs.com/api/v1"
HEADERS_AUTH = {"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"}

def generate_video(prompt: str, out_path: str) -> dict:
    # 1. 创建
    create_url = f"{API}/services/aigc/video-generation/video-synthesis"
    create_headers = {**HEADERS_AUTH, "X-DashScope-Async": "enable", "Content-Type": "application/json"}
    create_payload = {
        "model": "wan2.5-t2v-plus",
        "input": {"prompt": prompt},
        "parameters": {"size": "1280*720", "duration": 5, "prompt_extend": True},
    }
    r = requests.post(create_url, headers=create_headers, json=create_payload, timeout=30)
    r.raise_for_status()
    task_id = r.json()["output"]["task_id"]
    print(f"已创建任务 {task_id}")

    # 2. 轮询
    body = poll_task(task_id)
    video_url = body["output"]["video_url"]
    print(f"取到视频 URL（24h 后过期）：{video_url[:80]}...")

    # 3. 立刻下载
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    return {
        "task_id": task_id,
        "video_url": video_url,
        "local_path": out_path,
        "usage": body.get("usage"),
    }

if __name__ == "__main__":
    info = generate_video(
        prompt="咖啡馆桌上一杯冰柠檬茶，杯壁慢慢起雾，午后暖光",
        out_path="./out.mp4",
    )
    print(info)
```

这就是生产代码的骨架，少了 OSS 上传和死信队列。整个形状是"创建-轮询-下载"在同一个函数里完成，**绝不**把下载留给后台异步。每一个想偷懒的地方，我都因此丢过视频。

## 真实数据

`wan2.5-t2v-plus`，30 条 5 秒 720p 视频顺序跑：

- p50 生成时间：~95s
- p95 生成时间：~180s
- 失败率：~1%（多为瞬时，重试一次）
- 单条成本：~¥1.5

并发起来后瓶颈是单 workspace 视频任务并发上限，默认大约 5-10。生产上找售后开高就行。

## 真正能用的 prompt 模式

跑过几百段后总结：

- **加镜头运动是性价比最高的画质提升。** "缓推 dolly in"、"静止广角"、"手持过肩跟拍"——加一句胜过再加一个形容词。
- **写明光线。** "金色时刻"、"柔光箱影棚光"、"冷色顶部办公灯"能锚住整体观感。不写就给你一个平均值。
- **避开文字。** 万相和所有视频模型一样不会渲染稳定的文字。需要文字就生成完了在后期叠。
- **prompt 控制在 80 中文字 / 50 英文词以内。** 太长会扰乱扩写器。短而精的 prompt 更可控。

> **实战提示：** 一旦你需要在一批视频里保持一致风格，立刻把 `prompt_extend` 关掉，自己写好扩写后的 prompt。自动扩写器是个黑盒，会随模型升级而变化；自己掌控才能保证模型升级不会悄悄改掉你的品牌质感。

## 错误处理

| 错误 | 多半原因 | 处理 |
|---|---|---|
| `task_status: FAILED, code: InvalidParameter.Prompt` | prompt 触发内容审核 | 改写，常见是品牌名或人名 |
| `task_status: FAILED, code: InternalError` | 服务端 | 重新创建任务 |
| 创建时 429 | 并发上限 | 排队，退避重试 |
| 下载时 404 | URL 已过 24h | 视频没了，重新生成 |
| 轮询长时间停在 RUNNING（>5min） | 卡死任务，极少见 | 取消任务并重建 |

内容审核是最烦的。万相对名人姓名、政治、特定商标词比较保守。商品类内容里我会把品牌在 prompt 里改成通用名，靠 i2v 模式用真实产品图来保证品牌一致。

## 顺便讲讲 SDK

`pip install dashscope` 提供 `dashscope.VideoSynthesis.async_call(...)`，把创建+轮询包成一次调用：

```python
from dashscope import VideoSynthesis

rsp = VideoSynthesis.call(
    model="wan2.5-t2v-plus",
    prompt="...",
    size="1280*720",
    duration=5,
)
print(rsp.output.video_url)
```

原型可以用。生产我更喜欢裸 HTTP，因为我要显式控制轮询节奏、超时、重试，以及"URL 拿到后第几毫秒就开始下载"这一刻。SDK 内置的轮询合理但不可调。

## 闭环

万相生视频，第三篇的 Qwen-Omni 理解视频，第五篇的 Qwen-TTS 给视频配音。整条流水线串起来，短片不到 60 秒能跑完一个端到端，这就是我手上大部分生成式营销自动化背后的发动机。下一篇就是 TTS 那一半——意外地，又是 DashScope 原生独有。
