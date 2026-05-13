---
title: "阿里云百炼（三）：Qwen-Omni 多模态理解"
date: 2026-02-27 09:00:00
tags:
  - Aliyun Bailian
  - Qwen-Omni
  - Multimodal
  - Video Understanding
  - Streaming
categories: 阿里云百炼
lang: zh
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 3
description: "Qwen-Omni 生产实践：四种输入、文档没强调的流式必填，加上一个真实可跑的视频理解示例和合理的像素预算。"
disableNunjucks: true
translationKey: "aliyun-bailian-3"
---
在所有百炼模型中，Qwen-Omni 帮我规避了最多的产品路线图问题：过去处理“帮我看看这段 2 分钟宣传视频讲了什么”这类需求，往往需要三周时间——先提取视频帧，再为每一帧生成描述，最后拼接成连贯文本；如今只需一个 HTTP 请求即可搞定。但文档对某些关键细节语焉不详，尤其是“必须启用流式传输”这一硬性要求，已让不止一个团队白白耗费半天排查问题。下面帮你避开这个坑。

![阿里云百链（3）：Qwen-Omni 用于视频、音频和图像理解 — 视觉](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_1.png)

## Qwen-Omni 能接收什么

根据 Qwen 多模态模型的 API 文档，单个用户消息的 `content` 数组可以自由混合文本、图像、音频和视频内容。这才是真正的核心能力——不是“支持图像”，而是“支持任意模态以任意组合输入”：

![Qwen-Omni 输入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig1_omni_inputs.png)

每种类型的结构如下：

| Type | Field | Notes |
|---|---|---|
| `text` | `text: "..."` | 普通字符串。 |
| `image_url` | `image_url: {url}` | URL 或 base64 data URI。`min_pixels` / `max_pixels` 控制缩放。 |
| `input_audio` | `data, format` | 支持 `mp3`、`wav` 等格式，可为 URL 或本地 base64。 |
| `video_url` | `video_url: {url}` | URL 或 data URI；也可使用由帧图像组成的 `video` 数组。 |

真实调用示例：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

stream = client.chat.completions.create(
    model="qwen3-omni-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what's in this video in two sentences."},
            {"type": "video_url",
             "video_url": {"url": "https://your-bucket.oss-cn-shanghai.aliyuncs.com/clips/promo.mp4"}},
        ],
    }],
    stream=True,           # <- mandatory
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## 流式传输不是可选项——这是个陷阱

文档将流式（streaming）描述为一项功能特性，却隐去了一个关键事实：**对 Qwen-Omni 而言，流式是强制要求**。若设置 `stream=False`，你会直接收到 400 错误，提示该模型必须使用流式。

![流式处理需求](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig2_omni_streaming.png)

原因其实很合理：模型需处理大型视频文件并生成长文本响应，其底层协议默认采用增量传输。若等待完整响应再返回，客户端会卡住数十秒，期间毫无进度反馈。

如果你的下游代码期望一个完整的字符串，只需自行缓冲流式返回的增量片段：

```python
def call_omni_buffered(messages):
    stream = client.chat.completions.create(
        model="qwen3-omni-flash", messages=messages, stream=True,
    )
    return "".join(c.choices[0].delta.content or "" for c in stream)
```

这只是一个额外函数，写一次即可复用。

## 像素与帧预算——钱到底花在哪

文档未充分说明的成本控制点在于：图像的 `min_pixels` 和 `max_pixels`，以及视频对应的帧率（fps）和缩放参数。默认情况下，Qwen-Omni 会以原始分辨率和默认帧率处理视频。对于一段 2 分钟的 1080p 视频，这会产生*海量*的视觉 token，账单也随之飙升。

我在生产环境中的实践：

- **用于理解任务的图像**：设 `max_pixels: 1280*720`。对于“图中有什么”这类任务，几乎无质量损失，却能大幅节省成本；同时设 `min_pixels: 640*480`，避免模型将微小裁剪区域强行放大。
- **用于描述任务的视频**：上传前预缩放到 720p；静态内容（如人物讲话）帧率降至 4 fps，动态内容（如体育赛事、快速剪辑）降至 8 fps。超过 8 fps 通常只会引入冗余帧，徒增视觉 token 消耗。
- **长视频**：必须分片处理。模型有上下文长度限制。对于超过约 3 分钟的内容，切成 90 秒一段，分别生成摘要，再用 `qwen-plus` 对摘要进行二次汇总——这与长文档 RAG 的“分块-摘要-聚合”模式一致。

## 发送本地视频文件

你有两种选择，文档均有覆盖。

![将本地视频发送到 Qwen-Omni](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig3_video_pipeline.png)

**路径一（推荐）：上传至 OSS，发送签名 URL。**

```python
import oss2, os, time
auth = oss2.Auth(os.environ["OSS_AK"], os.environ["OSS_SK"])
bucket = oss2.Bucket(auth, "https://oss-cn-shanghai.aliyuncs.com", "your-bucket")
bucket.put_object_from_file("clips/promo.mp4", "/tmp/promo.mp4")
signed = bucket.sign_url("GET", "clips/promo.mp4", 600)  # 10 min expiry
```

随后将 `signed` 作为 `url` 字段传入。对于超过 30 秒的视频，此方案更优，因为 base64 编码会使 payload 膨胀约 33%。

**路径二：内联 base64。** 适用于短片段，可省去与 OSS 交互的往返开销。

```python
import base64
with open("/tmp/short.mp4", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
content = [{
    "type": "video_url",
    "video_url": {"url": f"data:video/mp4;base64,{b64}"},
}]
```

> **实战建议**：调试 Qwen-Omni 返回 400 错误时，请确认该 URL 能从公网直接访问。模型服务无法访问你的 VPC。签名 URL 可用；未签名的私有 OSS 对象则不可用。

## 音频理解

结构基本相同，只需将类型设为 `type: "input_audio"`：

```python
content = [
    {"type": "text", "text": "Transcribe this and identify the speaker's mood."},
    {"type": "input_audio", "input_audio": {"data": signed_audio_url, "format": "mp3"}},
]
```

若仅需纯转录，百炼还提供专用的 **Paraformer** ASR 模型，成本更低。建议：纯转录用 Paraformer，需要语义理解（如情感分析、摘要、“通话中是否提及价格”）时才使用 Qwen-Omni。

## 一个真实产品用例

我在 AI 营销场景中反复落地的模式是：创意团队上传一段 60 秒产品视频，我们需要结构化输出（包括 `scene description`、`key product features visible`、`target audience guess`、`recommended music style`）。只需一次 Qwen-Omni 调用，开启 JSON 模式（没错，多模态也支持），在 720p 输入下端到端耗时不到 4 秒。

```python
sys = ("Analyze this product video and return JSON with keys: "
       "scene_description, product_features (list), target_audience, music_style.")
stream = client.chat.completions.create(
    model="qwen3.5-omni-plus",
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": signed_url}},
        ]},
    ],
    response_format={"type": "json_object"},
    stream=True,
)
text = "".join(c.choices[0].delta.content or "" for c in stream)
import json; result = json.loads(text)
```

## 音频输入编码：采样率、格式与最大时长

Qwen-Omni 的音频输入看似简单——只需在 `input_audio` 中提供 `data` 和 `format`——但其编码要求极为严格，且错误提示不友好。一旦配置不当，API 会直接返回 400 错误（信息为 “unsupported audio format”），却不会说明具体原因或修复方向。

经生产验证的有效参数如下：

- **采样率**：16 kHz 是最佳平衡点。模型虽支持 8 / 16 / 22.05 / 24 / 44.1 / 48 kHz，但除 16 kHz 外均会在服务端重采样，增加延迟。对于 VoIP 录音（原生 8 kHz），我会在客户端用 `librosa.resample` 上采样至 16 kHz——音质不变，延迟更可控。
- **格式**：支持 `wav`（PCM 16-bit）、`mp3`、`m4a`、`flac`、`ogg`。短语音（< 30 秒）我用 WAV（体积无碍），长音频则用 MP3。MP4 容器中的 AAC *有时* 可用，但依赖编码 profile——与其调试不如先转为 MP3。
- **声道**：优先使用单声道（mono）。立体声会被下混；若双声道对应不同说话人（如访谈），下混会丢失说话人分离信息。建议自行转为 mono（保留目标声道），或拆成两个请求。
- **位深**：16-bit PCM 是安全默认值。24-bit 和 32-bit float 虽有时可用，但偶有被拒——我曾遇到同一 WAV 文件本周 16-bit 通过、32-bit 被拒的情况。
- **最大时长**：`qwen3-omni-flash` 单次请求限制为 3 分钟。超限时必须分片，错误信息为 `Audio duration exceeds limit`，并附带具体上限值。
- **最大文件大小**：10 MB。以 16 kHz / 16-bit mono PCM 计，约可容纳 5 分钟 WAV；但 64 kbps 的 MP3 在同等体积下可容纳 20 多分钟。因此，长音频必须使用 MP3。

每次调用前，我会运行一个预检转码器：

```python
import subprocess, os

def normalize_audio(src: str, dst: str) -> None:
    """Transcode to the codec Qwen-Omni handles most reliably."""
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-c:a", "libmp3lame",  # MP3 codec
        "-b:a", "64k",         # 64 kbps — plenty for speech
        dst,
    ], check=True, capture_output=True)
```

在每次上传后、调用 API 前执行此步骤。200ms 的转码开销远小于 API 延迟，可忽略不计，却能彻底规避“不支持的音频格式”类 400 错误。

## 视频帧采样：口播内容 1 fps 足矣，动作内容需 8 fps

![阿里云百链（3）：Qwen-Omni 用于视频、音频和图像理解 —— 视觉](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_2.png)

Qwen-Omni 内部处理视频时，默认按其内置策略采样帧，并将每帧编码为视觉 token 块。视觉 token 消耗量与帧数线性相关，因此帧率（fps）是控制视频成本最关键的调节旋钮。

官方文档仅提及视频 URL 中可设 `fps` 参数，却未说明不同内容类型应如何选择。基于生产环境中约 2000 次视频调用的经验，我的规则如下：

- **口播内容（采访、产品演示、讲座）**：1 fps 已足够。关键信息来自音频和稀疏视觉线索（如幻灯片切换、手势）。更高帧率只会浪费 token 在几乎相同的帧上。我在同一段 2 分钟采访上测试了 1 / 4 / 8 / 16 fps，发现描述质量在 1 fps 时已趋于稳定；部分场景甚至可降至 0.5 fps 仍能产出合理摘要。
- **动作内容（体育集锦、快剪广告、运动镜头）**：8 fps。低于此值会遗漏动作定义的关键事件；高于此值则处理大量冗余中间帧。8 fps 是我与三家广告技术客户共同验证的最佳平衡点。
- **混合内容（UGC、用户上传的未知类型）**：默认 4 fps。虽非最优，但不会在任一极端场景下彻底失效。
- **幻灯片/静态叠加内容**：0.5 fps。模型需读取每页幻灯片，而非处理过渡动画。

显式设置帧率的方法：

```python
content = [{
    "type": "video_url",
    "video_url": {
        "url": signed_url,
        "fps": 4,   # default ≈ 2 — set this when you know your content
    },
}]
```

若需最大控制权，可自行*预提取*帧，并以 `video` 数组形式传入图像：

```python
import cv2
def extract_frames(path: str, fps: float) -> list[str]:
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(src_fps / fps)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % step == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}")
        i += 1
    return frames

video_part = {"type": "video", "video": extract_frames("/tmp/clip.mp4", fps=4)}
```

预提取能让你精确指定模型看到哪些帧。当自动采样器跳过关键瞬间（如表情突变、PPT 切换）时，此方法尤为可靠——我已多次遇到自动采样遗漏关键帧的情况。

## 视频 Token 成本核算

Qwen-Omni 的成本痛点并非单个视觉 token 的单价，而是视频输入所触发的 token 总量。我们从第一性原理计算：

一张 1280×720 的视频帧，经 Qwen-Omni 内部视觉编码器处理后，约消耗 **256 个视觉 token**。（具体数值因模型变体而异；`qwen3.5-omni-plus` 更密集，`qwen3-omni-flash` 更轻量，但 256 是实用的估算基准。）

因此，一段 30 秒、4 fps 的视频共产生 `30 × 4 × 256 = 30,720` 视觉 token。按 `qwen3-omni-flash` 的视觉 token 费率（12 元 / 百万 token）计算，仅输入部分成本即为 `30720 / 1e6 × 12 = 0.37 元`，尚未包含生成描述的输出 token。

扩展来看：

| 内容 | 时长 | 帧率 | 视觉标记 | 输入成本（人民币） |
|---|---|---|---|---|
| Talking head | 60s | 1 | 15,360 | 0.18 |
| Talking head | 60s | 4 | 61,440 | 0.74 |
| Ad spot | 15s | 8 | 30,720 | 0.37 |
| Long demo | 180s | 4 | 184,320 | 2.21 |
| Lecture, slides only | 600s | 0.5 | 76,800 | 0.92 |

两点结论：

- 帧率选择可使成本相差 4–8 倍，务必谨慎决策。
- 一段 10 分钟讲座以 0.5 fps 处理，成本反而低于 1 分钟口播以 4 fps 处理。只要采样足够稀疏，长内容*其实很便宜*。

作为对比，同样内容的纯音频调用约消耗每秒 50 个音频 token（内部转写为文本 token 等效量）。10 分钟音频即 `600 × 50 = 30,000` 音频 token——比 1 分钟高密度视频还便宜。**对于以音频承载核心信息的口播内容，仅发送音频即可**，无需视频——理解质量相当，成本却低约 5 倍。

## 延迟分析：视频理解的 TTFT 与批量处理模式

Qwen-Omni 的端到端延迟可分为三个阶段，分开测量很有价值：

1. **上传 + URL 签名**：200ms–2s，取决于文件大小和 OSS 接入点。若需多次迭代 prompt，记得复用签名 URL。
2. **视觉编码（服务端）**：30 秒 720p 视频约需 1–4s。这是用户感知“无响应”的阶段，直到 token 开始流式输出；耗时大致与帧数成正比。
3. **流式生成**：首 token 时间（TTFT）通常在视觉编码完成后 1.5–3s，后续生成速度约 30–60 token/秒。

一段 30 秒视频生成 60 词描述，在生产环境中端到端耗时约 **6–8 秒**。虽快于人工处理，但显著慢于纯文本 LLM。若你的用户体验要求亚秒级响应，则不能实时调用 Omni——应在视频摄入阶段完成预处理并缓存描述结果。

对于批量任务（如 overnight 处理 1000 个视频），Omni 并无类似 OpenAI 的原生 `batch` 接口。我采用的模式是：

- 使用异步队列（Celery / SQS / 任选）搭配 10 个工作进程；
- 每个工作进程持有一个 `OpenAI` 客户端，串行调用 Omni，单任务软超时设为 10s（长尾可达 30s）；
- 每个 API key 并发上限为 10（匹配默认工作区配额）；
- 遇 429 错误时指数退避重试，最长等待 3 分钟；
- 生产吞吐量：每个工作区约 600 视频/小时，上限由 `qwen3-omni-flash` 的 60 RPM 配额决定。

若需更高吞吐，可申请配额提升（第一章已介绍流程），并将任务分散至多个工作区。Bailian 未提供原生并行批处理原语，需自行构建。

## 接下来

第 4 篇将转向 *生产侧* —— **万相 text-to-video**。该 API 仅支持异步调用和原生协议，失败模式截然不同（如队列深度、输出 URL 过期）。这也是我投入最多时间优化提示词的 API。
