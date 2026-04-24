---
title: "阿里云百炼实战（三）：Qwen-Omni 多模态——视频、音频、图像理解"
date: 2026-04-22 09:00:00
tags:
  - 阿里云百炼
  - DashScope
  - 多模态
  - Qwen-Omni
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 3
description: "调用 qwen3-omni-flash 和 qwen3.5-omni-plus 处理真实视频，文档里被淹没的强制流式 + include_usage 规则，以及一份从两分钟产品演示视频里抽结构化字段的完整代码。"
disableNunjucks: true
---

商家上传一个 90 秒的产品视频，平台需要自动产出一份中文商品描述、一句英文广告 headline、三个抖音/TikTok 钩子。早期流水线是先用一个 API 转文字、再用一个 API 抽帧、再让大模型融合，三次往返、每周两次对齐 bug，而且模型完全不知道说话人手里举的 SKU 跟外包装上印的不是一个。Qwen-Omni 一个调用全包。它也是这个系列里最脆弱的 API——只要你按调普通文本模型的方式调它就报错。这一篇就把规则讲清楚。

## 两个模型，按延迟预算二选一

| model_id | 强项 | 1 分钟视频大约延迟 | 备注 |
|---|---|---|---|
| `qwen3-omni-flash` | 快、便宜、多数抽取够用 | 8-15s | 我的默认 |
| `qwen3.5-omni-plus` | 细粒度理解更好、推理更长 | 15-30s | 抽取质量优先于吞吐时用 |

两者都接受视频、音频、图像、文本混在同一个 `messages` 数组里。两者踩的坑也一样。下面所有示例都用 `qwen3-omni-flash`，换模型就是改一个字符串。

## 强制流式规则

这一段读两遍：

> **Qwen-Omni 强制 `stream=True` 且强制 `stream_options={"include_usage": True}`。非流式直接报错；不传 `include_usage` 也报错。**

第一次撞上时我以为第二条是文档笔误，并不是。模型会把感知 token（音频/帧 embedding）和文本 token 交织输出，服务端的计费会计依赖 usage 钩子来知道何时关流。不传 `include_usage`，你会拿到一个 400，错误信息含糊得像 "stream_options required"。

也就是说**根本不存在"同步 Qwen-Omni"这种调用方式**。每个调用都是流式调用，worker 池要按这个前提设计。

## 接口和请求结构

Qwen-Omni 走的多模态生成接口：

```
POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

也可以走 OpenAI 兼容接口，我自己就是这么干的——请求结构就是 OpenAI 的 content-block 形式，多了 `video_url`、`audio_url`、`image_url` 几种类型。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DASHSCOPE_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

stream = client.chat.completions.create(
    model="qwen3-omni-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.oss-cn-beijing.aliyuncs.com/demo.mp4"}},
                {"type": "text", "text": "描述视频里说话人手中拿的物品。"},
            ],
        }
    ],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if not chunk.choices:
        # 携带 usage 的最终 chunk choices 为空——直接跳过
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

两个不能忘的点：

1. **`content` 是一个类型块列表**，不是字符串。`text`、`image_url`、`video_url`、`audio_url` 可以混搭。顺序有意义——希望问题指向媒体，就把媒体放在前面。
2. **最后一个 chunk 的 `choices` 是空的**，因为它只携带 `usage`。`if not chunk.choices` 这一跳必须写，否则 `IndexError`。

## 媒体怎么传

Qwen-Omni 接受三种来源，按优先级排：

| 来源 | 大小限制 | 备注 |
|---|---|---|
| HTTPS URL（如 OSS） | 实际无限制 | 最佳。OSS 短期签名 URL 即可。 |
| OSS 内网 URL | 实际无限制 | 同区域内调用延迟最低。 |
| base64 data URL | 请求体 ~10MB | 视频用 base64 很痛苦，仅小图建议。 |

90 秒 720p 的视频常见 30-60MB，只能走 URL。我把所有媒体都放在 `cn-beijing` 的 OSS 里，签 15 分钟有效期，把签名 URL 直接塞进 `video_url`。如果你的存储在别处，模型会**服务端拉取**，所以只要 URL 在公网可访问就行。

一个常见错误：OSS URL 带了 `Content-Disposition: attachment`，模型的拉取器会尊重它而不解码媒体。把 `Content-Type` 设成 `video/mp4`（或对应的格式），别加 Content-Disposition。

## 完整真实示例：从产品演示视频里抽结构化字段

下面这条流水线就是干掉早期那套三 API 大杂烩的代码。一次调用、结构化输出、完整可跑：

```python
import os
import json
from openai import OpenAI

PROMPT = """你正在观看一段产品演示视频，请抽取：
1. product_name（字符串）
2. key_features（3 条，使用说话人的语言）
3. target_audience（字符串）
4. tone（取值之一：energetic、calm、technical、lifestyle）
5. cta_suggestions（3 条简短的英文 CTA 文案）

只回复一个 JSON 对象，不要任何说明文字。"""

def extract_product_info(video_url: str) -> dict:
    client = OpenAI(
        api_key=os.environ['DASHSCOPE_API_KEY'],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=90.0,
    )

    stream = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)

    raw = "".join(parts).strip()
    # 模型偶尔会用 ```json 包起来，即使你说了 "不要说明文字"
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()

    data = json.loads(raw)
    data["_usage"] = usage.model_dump() if usage else None
    return data

if __name__ == "__main__":
    info = extract_product_info("https://your-oss.oss-cn-beijing.aliyuncs.com/demo.mp4?Expires=...&Signature=...")
    print(json.dumps(info, ensure_ascii=False, indent=2))
```

这段代码踩过坑后才长出来的几个细节：

- 把整段流缓冲完再 parse JSON。`json.loads` 不能渐进解析半截文档。
- 即便 prompt 里要求"不要说明文字"，模型偶尔仍会包 ```json fence，要剥掉。
- 从最后一个 chunk 拿 `usage`，方便按视频归集成本。
- 客户端超时 90 秒。`qwen3.5-omni-plus` 上偶尔超过默认 60 秒。

90 秒 720p 视频在 `qwen3-omni-flash` 上 8-12 秒返回；同一视频在 `qwen3.5-omni-plus` 上 18-25 秒。

## 音频和图像是同一套写法

`content` 列表，换块类型即可：

```python
content = [
    {"type": "audio_url", "audio_url": {"url": "https://.../voicemail.mp3"}},
    {"type": "text", "text": "总结这段语音留言并列出待办事项。"},
]
```

```python
content = [
    {"type": "image_url", "image_url": {"url": "https://.../receipt.jpg"}},
    {"type": "text", "text": "以 JSON 形式抽取商家、总额、日期。"},
]
```

一个 message 里也可以塞多块媒体——比如三张商品图加一个问题。成本大致按媒体时长/数量线性增加。

## 延迟、成本、吞吐

我们一个生产 worker 池处理 30-90 秒短视频的真实数据，模型 `qwen3-omni-flash`：

- p50 延迟：11s
- p95 延迟：22s
- p99 延迟：38s
- 单 worker 有效吞吐：约 5 个视频/分钟（输出不阻塞的前提下）
- 单视频成本：随时长大约 ¥0.08-0.20

两条架构提示：

1. **不要在多个阻塞 worker 间共享同一个 HTTP client。** 流式会长时间占着连接；16 个 worker 共用一个 client、连接池默认 10，会死锁。要么每个 worker 独立 client，要么把池大小拉大。
2. **OSS URL 在投递任务时就预签好，而不是任务开始时再签。** 队列堆积时，投递时签的 URL 10 分钟后还能用；任务开始时签的 URL 在立即执行时也有效。看你队列模式取舍。

> **实战提示：** 想把 5 分钟视频切成 5 个 1 分钟分别调用再合并？除非合并极其简单，否则别这么做。模型会丢失跨段上下文。要么整段送进去（Qwen-Omni 能吃到约 10 分钟），要么真的把 prompt 改成段内问题（"只描述这一分钟"）。

## Omni 特有错误

| 错误 | 原因 | 修复 |
|---|---|---|
| `stream_options is required` | 漏了 `stream_options={"include_usage": True}` | 补上 |
| `stream is required` | 用了非流式 | 改成 `stream=True` |
| `media fetch failed` | URL 公网不可达、Content-Type 不对、签名 URL 过期 | 用 curl 从 VPC 外测一下 |
| `media duration exceeds limit` | 单段媒体超过模型上限 | 切分或下采样 |
| `media decode failed` | 模型不识别的编码（罕见，通常是奇怪的 webm） | 重编为 H.264 mp4 |

最日常的还是 "media fetch failed"。我的标准排查动作就是在家里 Mac 上 `curl -I "$url"`——返回 200 且 content-type 正常，模型才能读到；返回别的什么，就别去问百炼了，先把 URL 修好。

## 顺势能搭出什么

一次调用就同时拿到了字幕、视觉上下文和推理，下一步显而易见的应用：实时会议总结（按音频分片）、视频内容问答（在 Omni 输出之上做 RAG）、视频解说无障碍化（Omni 描述视频，第五篇的 TTS 把描述读出来）。说到这——下一篇先讲万相视频流水线，正好是 Omni 描述的视频从哪来。
