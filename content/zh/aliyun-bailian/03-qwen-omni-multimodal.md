---
title: "阿里云百炼实战（三）：Qwen-Omni 多模态——视频、音频、图像理解"
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
在所有百炼模型中，Qwen-Omni 是那个让我从最多产品路线图“坑”里爬出来的救星。以前如果有人问我：“这条 2 分钟的广告片到底在讲什么？”这通常是一个需要三周才能完成的项目——包括抽帧、逐帧生成字幕，再把内容拼接起来。而现在，用 Qwen-Omni，只需要发一个 HTTP 请求就能搞定。不过，文档对一些容易踩坑的地方描述得很少，尤其是那个“必须使用流式传输”的要求，已经让不止一个团队白白浪费了大半天时间。咱们可别重蹈覆辙。

![阿里云百炼实战（三）：Qwen-Omni 多模态——视频、音频、图像理解 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_1.jpg)
## Qwen-Omni 支持的输入类型

根据 Qwen API 文档中关于多模态模型的说明，单条用户消息可以通过 `content` 数组混合多种类型的内容，包括文本、图片、音频和视频。这种能力的核心亮点并不在于“支持图片”，而是能够灵活处理各种类型的组合输入：

![Qwen-Omni 输入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig1_omni_inputs.png)

以下是每种输入类型的结构说明（参考自 API 文档）：

| 类型         | 字段                     | 备注                                   |
|--------------|--------------------------|----------------------------------------|
| `text`       | `text: "..."`           | 普通字符串，用于传递纯文本内容。       |
| `image_url`  | `image_url: {url}`      | 支持 URL 或 base64 编码的数据 URI。可通过 `min_pixels` 和 `max_pixels` 参数控制图片缩放。 |
| `input_audio`| `data, format`          | 支持 `mp3`、`wav` 等格式。可以是文件的 URL 或本地 base64 编码数据。 |
| `video_url`  | `video_url: {url}`      | 支持 URL 或 base64 编码的数据 URI。也可以通过 `video` 数组传递一组帧图像。 |

以下是一个实际调用的示例代码：

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
            {"type": "text", "text": "请用两句话描述这条视频的内容。"},
            {"type": "video_url",
             "video_url": {"url": "https://your-bucket.oss-cn-shanghai.aliyuncs.com/clips/promo.mp4"}},
        ],
    }],
    stream=True,           # <- 必须设置为 True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```
## 流式处理不是可选项——这是一个隐藏的“坑”

官方文档将流式处理描述为一项功能，但却刻意淡化了这样一个事实：**对于 Qwen-Omni 模型来说，流式处理是强制要求的**。如果你尝试设置 `stream=False`，结果会直接返回一个 400 错误，提示该模型必须使用流式传输。

![流式传输是必需的](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig2_omni_streaming.png)

仔细想想，这个设计其实合情合理：Qwen-Omni 模型需要处理数 MB 的视频数据，并生成较长的输出内容。协议本身基于增量交付的设计理念，如果非要等到所有内容生成完毕再一次性返回，客户端可能会被阻塞几十秒甚至更久，期间没有任何进度反馈，体验会非常糟糕。

如果你的下游逻辑需要完整的字符串输出，可以通过手动拼接流式数据来实现：

```python
def call_omni_buffered(messages):
    stream = client.chat.completions.create(
        model="qwen3-omni-flash", messages=messages, stream=True,
    )
    return "".join(c.choices[0].delta.content or "" for c in stream)
```

虽然需要额外写一个函数来完成拼接，但这段代码只需写一次，之后就可以复用到任何地方。
## 像素和帧率预算——哪些地方在烧钱

文档里没细说的一个关键成本控制点：图像的 `min_pixels` 和 `max_pixels`，以及视频对应的 fps 和分辨率调整参数。Qwen-Omni 默认会以原始分辨率和默认帧率处理视频。如果你有一段 2 分钟的 1080p 视频，那相当于消耗了 *巨量* 的 token，账单自然也会水涨船高。

我在实际生产中的做法是这样的：

- **用于理解任务的图像** — 设置 `max_pixels: 1280*720`。对于“这张图里有什么”这类任务，基本不会损失质量，但能显著降低成本。同时设置 `min_pixels: 640*480`，避免模型对小尺寸裁剪图进行无谓的放大操作。
- **用于描述任务的视频** — 在上传前先把分辨率调整到 720p。如果是静态内容（比如人物对话），将帧率降到 4 fps；如果是动态内容（比如体育赛事或快速切换的场景），则降到 8 fps。超过 8 fps 后，通常是在为重复帧买单。
- **长视频处理** — 切分处理。模型有上下文长度限制，因此对于超过 3 分钟的视频，我会将其分割成每段 90 秒的小块，分别生成摘要，然后再用 `qwen-plus` 对这些摘要进行二次总结。这个流程和处理长文档的 RAG 方法类似。
## 上传本地视频文件

有两种方法可以选择，文档里都详细介绍了。

![将本地视频发送至 Qwen-Omni](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/fig3_video_pipeline.png)

**方法一（推荐）：上传到 OSS，使用带签名的 URL。**

```python
import oss2, os
auth = oss2.Auth(os.environ["OSS_AK"], os.environ["OSS_SK"])
bucket = oss2.Bucket(auth, "https://oss-cn-shanghai.aliyuncs.com", "your-bucket")
bucket.put_object_from_file("clips/promo.mp4", "/tmp/promo.mp4")
signed = bucket.sign_url("GET", "clips/promo.mp4", 600)  # 签名有效期 10 分钟
```

将 `signed` 作为 `url` 字段传递即可。对于超过 30 秒的视频，这是最佳选择，因为 base64 编码会让数据量增加 33%。

**方法二：直接内嵌 base64 编码。** 这种方式适合短小精悍的视频片段，同时还能省去与 OSS 的交互。

```python
import base64
with open("/tmp/short.mp4", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
content = [{
    "type": "video_url",
    "video_url": {"url": f"data:video/mp4;base64,{b64}"},
}]
```

> **实战建议：** 如果 Qwen-Omni 返回 400 错误，记得检查 URL 是否可以通过公网访问。模型服务无法访问你的 VPC 内部资源。带签名的 OSS URL 是可行的，但未签名的私有 OSS 则不行。
## 音频理解

整体结构类似，但类型需要改为 `input_audio`：

```python
content = [
    {"type": "text", "text": "请将这段音频转写成文字，并分析说话人的情绪。"},
    {"type": "input_audio", "input_audio": {"data": signed_audio_url, "format": "mp3"}},
]
```

如果只是单纯的语音转文字任务，百炼提供了一个专门优化的 **Paraformer** ASR 模型，成本更低。对于只需要转写的场景，推荐使用 Paraformer；而如果需要更深层次的理解（如情绪分析、内容总结、或者判断通话中是否提到价格等），则可以选择 Qwen-Omni。
## 一个实际的产品应用场景

在 AI 营销中，我经常会用到这样一个模式：创意团队上传一段 60 秒的产品宣传视频，我们需要从中提取结构化的描述信息，包括“场景描述”、“可见的核心产品特性”、“目标受众推测”以及“推荐的背景音乐风格”。通过调用 Qwen-Omni 的多模态能力，并启用 JSON 模式，处理一段 720p 的视频输入仅需不到 4 秒。

```python
sys = ("分析这段产品视频，返回包含以下字段的 JSON 数据："
       "scene_description（场景描述）、product_features（产品特性列表）、"
       "target_audience（目标受众）、music_style（推荐音乐风格）。")
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

Qwen-Omni 的音频处理路径看似简单——只需传递一个包含 `data` 和 `format` 的 `input_audio` 参数即可。然而，实际的编码要求非常严格，且错误提示并不友好。一旦格式不符合要求，系统只会返回“unsupported audio format”（不支持的音频格式），没有更多说明。

在生产环境中，经过验证的最佳实践如下：

- **采样率**：16 kHz 是最优选择。虽然模型支持 8 kHz、16 kHz、22.05 kHz、24 kHz、44.1 kHz 和 48 kHz 等多种采样率，但非 16 kHz 的音频会在服务端被重新采样，这会增加额外的延迟。对于语音通话录音（通常是 8 kHz），我建议在客户端使用 `librosa.resample` 将其上采样到 16 kHz——这样既能保证质量，也能更好地控制延迟。
  
- **格式**：支持的格式包括 `wav`（PCM 16-bit）、`mp3`、`m4a`、`flac` 和 `ogg`。对于短语音（小于 30 秒），可以直接使用 WAV 格式，无需担心文件大小；而对于较长的内容，推荐使用 MP3。需要注意的是，MP4 容器中的 AAC 编码有时可以正常工作，但这取决于具体的编解码器配置。与其调试兼容性问题，不如直接转码为 MP3。

- **声道**：单声道是首选。立体声音频会被自动混音为单声道，如果双声道分别记录了不同的说话人（如采访场景），混音会导致说话人信息丢失。为了避免这种情况，可以在客户端手动将目标声道提取为单声道，或者将双声道拆分为两个独立请求。

- **位深**：16-bit PCM 是最安全的选择。虽然 24-bit 和 32-bit 浮点数音频在某些情况下也能被接受，但它们偶尔会被拒绝。例如，同一个 WAV 文件在一周内可能以 16-bit 被接受，而以 32-bit 被拒绝。

- **最大时长**：对于 `qwen3-omni-flash` 模型，单次请求的音频时长不能超过 3 分钟。如果音频长度超出限制，必须进行分段处理。此时，系统会返回错误信息 `Audio duration exceeds limit`，并在消息中明确标注具体的时间上限。

- **最大文件大小**：单个音频文件的大小不得超过 10 MB。以 16 kHz、16-bit 单声道 PCM 编码为例，这样的限制大约相当于 5 分钟的 WAV 文件。但如果使用 64 kbps 的 MP3 编码，则可以在相同的文件大小限制下容纳超过 20 分钟的音频内容。因此，对于长时间的音频内容，使用 MP3 是必选项。

为了确保音频格式始终符合要求，我在每次调用 API 前都会运行一个预处理转码脚本：

```python
import subprocess, os

def normalize_audio(src: str, dst: str) -> None:
    """将音频转码为 Qwen-Omni 最稳定的编码格式。"""
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ac", "1",            # 单声道
        "-ar", "16000",        # 16 kHz
        "-c:a", "libmp3lame",  # 使用 MP3 编码器
        "-b:a", "64k",         # 64 kbps，适合语音内容
        dst,
    ], check=True, capture_output=True)
```

在每次上传音频之前运行这个脚本。虽然转码过程会增加约 200 毫秒的开销，但相比 API 的整体延迟，这点时间完全可以忽略不计。更重要的是，它彻底避免了因格式问题导致的“400 不支持的音频格式”错误。
## 视频帧采样：1 fps 适合对话场景，8 fps 适合动作场景

![阿里云百炼实战（三）：Qwen-Omni 多模态——视频、音频、图像理解 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/03-qwen-omni-multimodal/illustration_2.jpg)

在 Qwen-Omni 的内部处理逻辑中，视频会被按一定的帧率抽帧，每一帧会被编码为一个视觉 token 块。由于 token 的消耗与帧数成正比，因此帧率是控制视频处理成本的最关键参数。

官方文档提到，可以通过视频 URL 中的 `fps` 参数来设置帧率，但并没有说明不同类型的内容应该选择什么样的值。经过生产环境中大约 2000 次视频调用的实践，我总结了一些经验规则：

- **对话类内容（如访谈、产品演示、讲座等）**：1 fps 就足够了。这类内容的核心信息主要集中在音频和稀疏的视觉线索上，比如幻灯片切换或手势动作。更高的帧率只会浪费 token 在那些几乎完全相同的画面上。我曾在一段 2 分钟的访谈视频上测试了 1 / 4 / 8 / 16 fps，结果发现描述质量在 1 fps 时就已经达到瓶颈。某些情况下，甚至可以降到 0.5 fps，依然能生成合理的摘要。
  
- **动作类内容（如体育赛事、带有快速剪辑的广告、动态画面等）**：建议设置为 8 fps。低于这个值可能会错过一些由运动定义的关键事件，而高于这个值则会导致模型处理大量冗余的中间帧。经过三家广告技术客户的验证，8 fps 是一个非常合适的“黄金值”。

- **混合类内容（如用户生成内容 UGC、类型未知的用户上传视频等）**：默认设置为 4 fps。虽然这个值对两种极端情况都不是最优解，但至少不会导致灾难性的失败。

- **幻灯片或静态叠加内容**：建议设置为 0.5 fps。模型需要关注的是每张幻灯片的内容，而不是过渡效果或溶解动画。

如果需要显式设置帧率，可以参考以下代码：

```python
content = [{
    "type": "video_url",
    "video_url": {
        "url": signed_url,
        "fps": 4,   # 默认值约为 2，如果明确知道内容类型，建议显式设置
    },
}]
```

如果你希望对帧的选择有更精确的控制，可以预先提取帧，并将它们作为 `video` 数组的图像部分发送给模型：

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

通过预提取帧，你可以完全掌控模型看到的画面内容。自动采样有时会跳过关键帧，这种情况我已经遇到过不止一次，而预提取则是避免这种问题的有效手段。
## 视频 Token 成本分析

使用 Omni 时，成本的意外之处并不在于每 Token 的单价，而在于每秒视频所需的 Token 数量。我们从基本原理出发来计算一下：

一帧分辨率为 1280×720 的视频，经过 Qwen-Omni 内部视觉编码器处理后，大约需要 **256 个视觉 Token**。（具体数量取决于模型版本；例如 `qwen3.5-omni-plus` 更密集，而 `qwen3-omni-flash` 则更轻量化，但 256 是一个适合规划的参考值。）

假设一段 30 秒的视频以 4 帧每秒（fps）的速度处理，那么总的视觉 Token 数就是 `30 * 4 * 256 = 30,720`。按照 `qwen3-omni-flash` 的视觉 Token 定价 12 元/百万 Token 来算，仅输入部分的成本就达到了 `30720 / 1e6 * 12 = 0.37` 元，这还不包括生成描述所需的输出 Token。

以下是不同场景下的成本估算：

| 内容类型       | 时长   | 帧率 (fps) | 视觉 Token 数 | 输入成本（元） |
|----------------|--------|------------|---------------|----------------|
| 人物讲话       | 60 秒  | 1          | 15,360        | 0.18           |
| 人物讲话       | 60 秒  | 4          | 61,440        | 0.74           |
| 广告片段       | 15 秒  | 8          | 30,720        | 0.37           |
| 长篇演示       | 180 秒 | 4          | 184,320       | 2.21           |
| 讲座（仅幻灯片）| 600 秒 | 0.5        | 76,800        | 0.92           |

从中可以得出两点重要结论：

1. **帧率的选择对成本影响巨大**：不同的帧率可能导致成本相差 4 到 8 倍，因此需要根据实际需求谨慎选择。
2. **长视频稀疏采样成本更低**：例如，10 分钟的讲座如果以 0.5 fps 处理，其成本甚至低于 1 分钟的人物讲话视频（4 fps）。对于长时间的内容，采用较低帧率采样能够显著降低成本。

作为对比，纯音频处理的成本要低得多。音频通常以每秒约 50 Token 的速度进行处理（内部会将音频转录为等效的文本 Token）。10 分钟的音频相当于 `600 * 50 = 30,000` 个音频 Token，比 1 分钟高帧率视频的成本还低。**对于以语音为主的内容（如人物讲话），建议仅发送音频而非视频**，这样可以在保证理解质量的同时，将成本降低约 5 倍。
## 延迟分析：视频理解的 TTFT 与批处理模式

在 Qwen-Omni 的端到端延迟中，可以将其拆解为三个关键阶段，分别进行测量和优化：

1. **文件上传与 URL 签名**：耗时通常在 200 毫秒到 2 秒之间，具体取决于文件大小以及 OSS 的接入点。如果需要反复调整提示词（prompt），可以复用同一个已签名的 URL，避免重复生成。
2. **视觉编码（服务端处理）**：对于一段 30 秒的 720p 视频，处理时间大约在 1 到 4 秒之间。这是用户感知“系统似乎没有响应”的时间段，因为此时还没有开始生成 token。处理时间大致与视频帧数呈线性关系。
3. **流式生成**：视觉编码完成后，TTFT（首次生成 token 的时间）通常在 1.5 到 3 秒之间，随后的生成速度为每秒 30 到 60 个 token。

以一段 30 秒的视频生成 60 字描述为例，在生产环境中端到端的总耗时约为 **6 到 8 秒（墙钟时间）**。这一速度比人类快，但比纯文本 LLM 慢。如果用户体验要求亚秒级响应，那么 Qwen-Omni 并不适合——需要在视频入库时预先处理并缓存描述内容。

对于批量任务（例如一晚上处理 1000 条视频），Qwen-Omni 并不像 OpenAI 那样提供原生的 `batch` 接口。以下是我在实际场景中采用的方案：

- 使用异步队列（如 Celery、SQS 或其他你喜欢的工具），配置 10 个 worker。
- 每个 worker 持有一个 `OpenAI` 客户端，串行调用 Qwen-Omni，单任务设置 10 秒软超时（长尾任务可能延长至 30 秒）。
- 每个 API Key 的并发上限设置为 10（与默认工作空间配额一致）。
- 遇到 429 错误时，采用指数退避策略重试，最大重试时间为 3 分钟。
- 生产环境中的吞吐量：每个工作空间每小时可处理约 600 条视频，瓶颈在于 `qwen3-omni-flash` 的 60 RPM 配额限制。

如果需要更高的吞吐量，可以申请提升配额（第一章详细介绍了流程），并通过多个工作空间分散任务。百炼平台目前没有原生的并行批处理功能，因此需要自行实现相关逻辑。
## 下一篇

第四篇跳到*生产*那一侧——**万相文生视频**。完全异步、只能走原生协议，失败模式也完全不同（队列深度、输出 URL 过期）。这也是我花最多时间调 prompt 的一个 API。
