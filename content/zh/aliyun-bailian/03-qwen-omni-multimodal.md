---
title: "百炼实战（三）：Qwen-Omni 多模态理解"
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
在所有百炼模型里，Qwen-Omni 算是帮我填坑最多的一个。以前那种“帮我看看这 2 分钟 promo 视频里讲了啥”的需求，得搞帧提取、逐帧 caption 再拼接，起码折腾 3 周。现在 Qwen-Omni 一个 HTTP 请求就搞定。但文档里有些坑没写清楚，尤其是“必须流式传输”这一点，已经让好几个团队半天的时间打水漂了。别让你也成为其中一个。

![Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_1.png)

## Qwen-Omni 能收什么

查一下 Qwen 多模态模型的 API 参考，单个 user message 的 `content` 数组里能混排 text、image、audio 和 video。这才是重点——不是“支持图片”，而是“支持任意组合”：

![Qwen-Omni inputs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig1_omni_inputs.png)

每种类型的结构，直接摘自 API 参考：

| Type | Field | Notes |
|---|---|---|
| `text` | `text: "..."` | 普通字符串。 |
| `image_url` | `image_url: {url}` | URL 或 base64 data URI。`min_pixels` / `max_pixels` 控制 resize。 |
| `input_audio` | `data, format` | `mp3`, `wav` 等。URL 或本地 base64。 |
| `video_url` | `video_url: {url}` | URL 或 data URI。也可以用帧图片组成的 `video` 数组。 |

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

## 流式传输不是可选的——这是个坑

文档里把 streaming 写成一个功能特性，但藏了一个事实：**对 Qwen-Omni 来说这是必填项**。设成 `stream=False` 直接返回 400，报错说模型要求流式。

![Streaming requirement](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig2_omni_streaming.png)

想一下就明白了：模型要处理几 MB 的视频，生成很长的回复。传输协议本身就是按增量交付设计的。全攒在一起发会阻塞客户端几十秒，还没个进度提示。

如果你的下游代码期望一个完整字符串，自己把 delta 攒起来就行：

```python
def call_omni_buffered(messages):
    stream = client.chat.completions.create(
        model="qwen3-omni-flash", messages=messages, stream=True,
    )
    return "".join(c.choices[0].delta.content or "" for c in stream)
```

多写一个函数而已。写一次就够了。

## 像素和帧预算——钱花在哪

文档里没怎么提的成本控制旋钮：图片的 `min_pixels` 和 `max_pixels`，以及视频对应的 fps 和 resize 参数。默认情况下 Qwen-Omni 会按原生分辨率和默认 fps 处理视频。对于 2 分钟的 1080p 片段，这消耗的 token 量*非常大*，账单也跟着涨。

我在生产环境的做法：

- **理解任务的图片** — `max_pixels: 1280*720`。对于“图里有什么”这类任务几乎没质量损失，但能省不少钱。设 `min_pixels: 640*480` 防止模型把小图强行放大。
- **描述任务的视频** — 上传前预 resize 到 720p，静态内容（比如人物说话）fps 降到 4，动作内容（体育、快切）降到 8。超过 8 fps 通常是在为冗余帧买单。
- **长视频** — 切片。模型有 context 限制。超过 ~3 分钟的内容，切成 90 秒片段，分别总结，再用 `qwen-plus` 对摘要再做摘要。跟长文档 RAG 一个套路。

## 发送本地视频文件

你有两个选择。文档都覆盖了。

![Sending a local video to Qwen-Omni](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig3_video_pipeline.png)

**路径 1（推荐）：上传到 OSS，发 signed URL。**

```python
import oss2, os, time
auth = oss2.Auth(os.environ["OSS_AK"], os.environ["OSS_SK"])
bucket = oss2.Bucket(auth, "https://oss-cn-shanghai.aliyuncs.com", "your-bucket")
bucket.put_object_from_file("clips/promo.mp4", "/tmp/promo.mp4")
signed = bucket.sign_url("GET", "clips/promo.mp4", 600)  # 10 min expiry
```

然后把 `signed` 传给 `url` 字段。超过 30 秒的视频都用这个方案，因为 base64 会让 payload 膨胀 33%。

**路径 2：base64 内联。** 适合短片段，省去了往返 OSS 的过程。

```python
import base64
with open("/tmp/short.mp4", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
content = [{
    "type": "video_url",
    "video_url": {"url": f"data:video/mp4;base64,{b64}"},
}]
```

> **实战建议：** 调试 Qwen-Omni 400 错误时，检查 URL 能否从公网访问。模型服务进不了你的 VPC。Signed URL 没问题；没签名的私有 OSS 不行。

## 音频理解

结构差不多，只是 `type: "input_audio"`：

```python
content = [
    {"type": "text", "text": "Transcribe this and identify the speaker's mood."},
    {"type": "input_audio", "input_audio": {"data": signed_audio_url, "format": "mp3"}},
]
```

纯转录任务的话，百炼还有一个更便宜的专用 **Paraformer** ASR 模型。只要转录用 Paraformer，需要理解（情感、总结、“电话里提没提价格”）再用 Qwen-Omni。

## 一个真实的产品用例

我在 AI 营销场景里经常落地的模式：创意团队上传 60 秒产品视频；我们需要结构化的 caption（`scene description`, `key product features visible`, `target audience guess`, `recommended music style`）。一次 Qwen-Omni 调用，开启 JSON 模式（没错，多模态也支持），720p 输入下实际耗时不到 4 秒。

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

## 音频输入编码：采样率、格式、最大时长

Qwen-Omni 的音频输入看起来简单——传个 `input_audio` 部分带上 `data` 和 `format`——但编码要求很严，而且报错不友好。搞错了就是 400 "unsupported audio format"，没别的指引。

生产环境真正能用的参数：

- **采样率**：16 kHz 是最佳平衡点。模型接受 8 / 16 / 22.05 / 24 / 44.1 / 48 kHz，但除了 16 kHz 以外都会在服务端重采样，增加延迟。对于 VoIP 录音（原生 8 kHz），我用 `librosa.resample` 在客户端上采样到 16 kHz——质量一样，延迟更可控。
- **格式**：`wav` (PCM 16-bit), `mp3`, `m4a`, `flac`, `ogg`。短语音（< 30s）不在乎大小我用 WAV，长的用 MP3。MP4 里的 AAC *有时* 能行，取决于编码 profile——与其调试不如先转成 MP3。
- **声道**：首选 mono。立体声会被下混；如果两个声道是不同说话人（访谈类内容），下混会丢失分离信息。自己转成 mono 选关心的那个声道，或者拆成两个请求。
- **位深**：16-bit PCM 是安全默认值。24-bit 和 32-bit float 能用但有时被拒——我见过同一个 WAV 文件这周 16-bit 通过，32-bit 被拒。
- **最大时长**：`qwen3-omni-flash` 每次请求限制 3 分钟。超过必须切片。报错是 `Audio duration exceeds limit`，消息里会带具体上限。
- **最大文件大小**：10 MB。16 kHz / 16-bit mono PCM 大概是 5 分钟 WAV——但 64 kbps 的 MP3 同样大小能塞进 20 多分钟。长内容必须用 MP3。

每次音频调用前我都会跑一个预检转码：

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

每次上传后、调用 API 前都跑一遍。200ms 的转码成本比起 API 延迟完全可以忽略，而且能彻底消除那一类“格式不支持”的 400 错误。
## 视频帧采样：口播 1 fps 就够了，动作类要 8 fps

![Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_2.png)

Qwen-Omni 内部处理视频时，默认会按模型内部的速率采样帧，然后把每一帧编码成 vision token block。Token 消耗跟帧数成正比，所以帧率（fps）是你控制视频成本最大的那个旋钮。

官方文档只说了视频 URL 部分有个 `fps` 参数，但没告诉你不同类型的内容该设多少。在生产环境跑了大概 2000 次视频调用后，我总结了几条规则：

- **口播内容（采访、产品演示、讲座）**：1 fps 足够。关键信号在音频加上稀疏的视觉线索（PPT 切换、手势）。更高 fps 会把 Token 浪费在看起来一模一样的帧上。我在同一段 2 分钟采访片段上测过 1 / 4 / 8 / 16 fps，描述质量在 1 fps 就趋于平稳了。有时甚至降到 0.5 fps 也能拿到合理的总结。
- **动作类内容（体育集锦、带剪辑的广告、运动）**：8 fps。低于这个值会漏掉由动作定义的事件。高于这个值模型就在处理冗余的中间帧。8 fps 是我跟三个广告技术客户验证过的最佳点。
- **混合内容（UGC、用户上传未知类型）**：默认 4 fps。两头都不完美，但也不会彻底翻车。
- **幻灯片/静态加覆盖层内容**：0.5 fps。模型需要读每一页 slide，而不是处理溶解过渡效果。

显式设置 fps 的方法：

```python
content = [{
    "type": "video_url",
    "video_url": {
        "url": signed_url,
        "fps": 4,   # default ≈ 2 — set this when you know your content
    },
}]
```

想要最大控制权，可以自己*预提取*帧，然后作为 `video` 数组的图片部分发送：

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

预提取能让你精确控制模型看到哪些帧。当自动采样器跳过关键时刻时（这种情况在我这儿发生过不止一次），这招特别管用。

## 视频 Token 成本算账

Omni 让人意外的成本点不在于单 Token 费率，而在于每秒视频产生的 Token 数量。咱们从第一性原理算一下：

1280×720 的视频帧，经过 Qwen-Omni 内部 vision encoder 后，大概消耗 **每帧 256 个 vision tokens**。（具体数字取决于模型 variant；`qwen3.5-omni-plus` 密度更高，`qwen3-omni-flash` 更轻量，但 256 是个好用的规划数字。）

所以一个 30 秒的片段，4 fps 下就是 `30 * 4 * 256 = 30,720 vision tokens`。按 `qwen3-omni-flash` vision 费率 12 RMB / million tokens 算，光是输入部分就是 `30720 / 1e6 * 12 = 0.37 RMB`，这还没算生成描述的输出 Token。

放大来看：

| Content | Duration | fps | Vision tokens | Input cost (RMB) |
|---|---|---|---|---|
| Talking head | 60s | 1 | 15,360 | 0.18 |
| Talking head | 60s | 4 | 61,440 | 0.74 |
| Ad spot | 15s | 8 | 30,720 | 0.37 |
| Long demo | 180s | 4 | 184,320 | 2.21 |
| Lecture, slides only | 600s | 0.5 | 76,800 | 0.92 |

两点结论：

- fps 选择能让成本变化 4-8 倍。得刻意选。
- 10 分钟讲座用 0.5 fps 比 1 分钟口播用 4 fps 还便宜。长内容只要采样稀疏，*很便宜*。

对比一下，同样内容的纯音频调用，大概是每秒音频 50 tokens（内部转写成 text-token-equivalents）。10 分钟音频是 `600 * 50 = 30,000 audio tokens` —— 比 1 分钟密集视频还便宜。**对于音频承载意义的口播内容，只发音频**，别发视频 —— 理解质量相当，成本低大概 5 倍。

## 延迟 profile：视频理解的 TTFT 和批量模式

Qwen-Omni 的端到端延迟可以拆成三个阶段，分开测很有用：

1. **Upload + URL signing**：200ms-2s 取决于文件大小和 OSS endpoint。如果在迭代 prompt，记得复用 signed URL。
2. **Vision encoding (server-side)**：30 秒 720p 片段大概 1-4s。这是用户感觉“什么都没发生”直到 Token 开始流式输出的阶段。大概跟帧数成正比。
3. **Streaming generation**：TTFT 通常在 vision encoding 结束后 1.5-3s，然后 30-60 tokens/sec。

30 秒视频片段 → 60 词描述，端到端在生产环境大概 **6-8 秒 wall clock**。比人快，比纯文本 LLM 慢。如果你的 UX 需要亚秒级，别用 Omni —— 你得在摄入时预处理视频并缓存描述。

对于批量任务（比如 overnight 处理 1000 个视频），Omni 没有像 OpenAI 那样的原生 `batch` endpoint。我用的模式是：

- Async queue (Celery / SQS / 你喜欢的) 配 10 个 workers。
- 每个 worker 持有一个 `OpenAI` 客户端，串行调用 Omni，每个任务设 10s soft timeout（长尾到 30s）。
- 每 key 并发上限 10（匹配默认 workspace quota）。
- 遇到 429 重试，指数退避最多 3 分钟。
- 生产环境吞吐量：每个 workspace 大概 600 视频/小时，上限卡在 60 RPM `qwen3-omni-flash` quota。

想要更高吞吐量，申请 quota 提升（第 1 章讲了流程），然后分散到多个 workspace。Bailian 没有原生 parallel-batch 原语，得自己造。

## 接下来

第 4 篇跳到 *生产* 侧 —— **万相 text-to-video**。那是纯异步、纯原生协议，失败模式完全不同（队列深度、输出 URL 过期）。这也是我花最多时间调 prompt 的 API。