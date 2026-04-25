---
title: "阿里云百炼实战（五）：Qwen-TTS 多语言语音合成"
date: 2026-03-01 09:00:00
tags:
  - 阿里云百炼
  - DashScope
  - 语音合成
  - Qwen-TTS
categories: 阿里云百炼
lang: zh-CN
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 5
description: "qwen3-tts-flash 的 40+ 音色、市面上罕见的方言覆盖，以及那个让所有先试过 OpenAI TTS 的团队踩坑的\"只能走 DashScope 原生接口\"细节。"
disableNunjucks: true
---

广州的商家想给视频广告配粤语解说。Azure TTS 的粤语像海关广播，ElevenLabs 没上线粤语商用音色，OpenAI TTS 只有六个音色而且非英语全是英音味儿。Qwen-TTS-Flash 的粤语音色，听起来真的像香港当地人在念稿。那一刻是我彻底不再"先去试试别家"的转折点。这一篇讲怎么用、怎么选音色，以及那个最常见的错误（把它当成 OpenAI 兼容接口）。

## 唯一值得用的模型：`qwen3-tts-flash`

百炼上还挂着旧的 `sambert-*` 和 `cosyvoice-*` 系列。它们能跑，价格也许还更便宜，但新代码就用 `qwen3-tts-flash`：更快、音色更多、方言覆盖更好，并且阿里在持续迭代它。

| 属性 | 值 |
|---|---|
| model_id | `qwen3-tts-flash` |
| 接口 | `https://dashscope.aliyuncs.com/api/v1/services/audio/tts` |
| 输出格式 | mp3、wav、pcm |
| 采样率 | 8k / 16k / 22050 / 24k / 44100 |
| 语言 | 普通话、英语、粤语、川渝话、上海话、东北话、日语、韩语（按音色） |
| 音色 | 40+ |
| 计费 | 按音频秒数 |

## 只能走 DashScope 原生接口

这是大家都会踩的那一脚：

> **Qwen-TTS 不能通过 OpenAI 兼容接口调用，只能用 DashScope 原生。**

不能把 `openai` SDK 的 `audio.speech.create` 指到兼容 URL 上——TTS 没有兼容层。用 `dashscope` SDK 或裸 HTTP。

我见过三家公司的三个工程师犯同一个错——理由都是"既然 Qwen LLM 能走 OpenAI 兼容，TTS 当然也能"。报错是 404 或 "model not found"，看着完全没头绪。直接告诉你：装 `dashscope`。

## 最简单的调用

```python
import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Cherry")
audio_bytes = synth.call("欢迎收看本期产品演示，今天我们将展示三个新功能。")

with open("out.mp3", "wb") as f:
    f.write(audio_bytes)
```

就这。`audio_bytes` 默认是 mp3。10 秒一句话，延迟 1-2 秒。`synth` 对象可复用，批量场景直接复用。

## 音色选择：别浏览目录，照下面这张表挑

控制台有音色试听，确实有用。但下面这张是我一年生产里反复抽出来的短名单：

| 音色 | 语言 | 适用 |
|---|---|---|
| `Cherry` | 普通话、英语 | 亲和女声，绝大部分消费向旁白的默认 |
| `Ethan` | 普通话、英语 | 温暖男声，适合解说视频 |
| `Serena` | 英语 | 专业女声，B2B 内容 |
| `Chelsie` | 普通话 | 偏年轻女声，小红书/生活方式调性 |
| `Sunny` | 川渝话 | 川渝口音，性格强烈，少量使用 |
| `Lily` | 粤语 | 港式女声，南方市场的杀手级 |
| `Marcus` | 普通话、英语 | 权威男声，正式旁白 |
| `Jennifer` | 普通话（东北） | 东北口音，角色化使用 |

音色 ID 区分大小写。`cherry` 失败、`Cherry` 成功。完整列表以 `qwen3-tts-flash` 文档为准，但上面 8 个能覆盖我九成场景。

一个音色支不支持多语言取决于训练数据。`Cherry` 和 `Ethan` 是双语的主力，输入里中英文混排时会自然切换。单语言音色读另一种语言会变形，按需选。

## 流式合成降低首字节延迟

一句话旁白用非流式就够。长稿子或者"一边生成一边播放"的场景就上流式：

```python
import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

class FileCallback(ResultCallback):
    def __init__(self, path):
        self.f = open(path, "wb")
    def on_data(self, data: bytes):
        self.f.write(data)
    def on_complete(self):
        self.f.close()
    def on_error(self, message):
        print("出错:", message)
        self.f.close()

callback = FileCallback("out_stream.mp3")
synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Cherry", callback=callback)

# 可多次调 streaming_call 来分段喂入文本，最后 streaming_complete 收尾。
synth.streaming_call("第一句旁白。")
synth.streaming_call("第二句，可以来自上游的 LLM 流。")
synth.streaming_complete()
```

流式模式在"LLM 输出直接灌 TTS"场景里非常香——`qwen-plus` 流式生成文本，每段 delta 推给 TTS，音频在 LLM 还没写完时就开始播。用户感知延迟从"等 LLM + 等 TTS"压缩为"等 LLM 开口"。

## 格式与采样率怎么选

```python
synth = SpeechSynthesizer(
    model="qwen3-tts-flash",
    voice="Cherry",
    format="wav",        # mp3 | wav | pcm
    sample_rate=24000,   # 8000 | 16000 | 22050 | 24000 | 44100
)
```

我自己的选择：

- **mp3 / 24kHz**：上传到 Web 的内容，这是默认。文件最小，旁白场景听感和 wav 没区别。
- **wav / 24kHz**：要在 `ffmpeg` 里跟其他音频混音，省一道编码往返。
- **pcm / 16kHz**：实时播放库吃裸帧时用。

不要超过 24kHz。模型实际有效带宽到不了那么高，多出来的字节就是浪费。

## 真实业务：多语言产品旁白

这段函数支撑了一个"用 5 种音色配同一段产品文案"的功能。一段产品描述进来、5 个 mp3 文件出去。

```python
import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

NARRATION_VOICES = [
    ("zh-mandarin", "Cherry"),
    ("zh-mandarin-male", "Ethan"),
    ("zh-cantonese", "Lily"),
    ("zh-sichuan", "Sunny"),
    ("en-pro", "Serena"),
]

def narrate_all(text_by_locale: dict, out_dir: str) -> dict:
    """text_by_locale: {locale_id: text}，例如 {'zh-mandarin': '...', 'en-pro': '...'}
    返回 {locale_id: file_path}。"""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for locale, voice in NARRATION_VOICES:
        text = text_by_locale.get(locale)
        if not text:
            continue
        synth = SpeechSynthesizer(model="qwen3-tts-flash", voice=voice, format="mp3", sample_rate=24000)
        audio = synth.call(text)
        path = os.path.join(out_dir, f"{locale}.mp3")
        with open(path, "wb") as f:
            f.write(audio)
        results[locale] = path
    return results

if __name__ == "__main__":
    inputs = {
        "zh-mandarin": "这款保温杯采用医用级不锈钢，24小时保温保冷。",
        "zh-mandarin-male": "这款保温杯采用医用级不锈钢，24小时保温保冷。",
        "zh-cantonese": "呢個保溫杯用嘅係醫用級不鏽鋼，24個鐘保溫保冷。",
        "zh-sichuan": "这个保温杯用的是医用级不锈钢，二十四个小时保温保冷哈。",
        "en-pro": "This insulated bottle uses medical-grade stainless steel, keeping drinks hot or cold for 24 hours.",
    }
    print(narrate_all(inputs, "./narrations"))
```

两条生产经验：

1. **先翻译，再合成。** 不要把同一句普通话原文丢给粤语音色让它"用粤语读"——它会用粤语口音读普通话书面语，根本不是粤语。先用 `qwen-plus` 翻成真正的粤语口语，再合成。
2. **按 (text + voice + format) 哈希做缓存。** 旁白是不可变内容，同一段产品描述跑两次就该只付一次钱。哈希作为 OSS key，合成前先查。

> **实战提示：** SSML（`<break time='500ms'/>`、emphasis 等）支持是部分且不一致的。常规旁白用标点就够：逗号给短停顿、句号给长停顿、破折号（——）给那种"feature 高潮前"的戏剧性停顿。SSML 留给标点真的不够用的极少数场景。

## 成本与延迟

8 秒典型产品旁白：

- 延迟：非流式 1-2 秒；流式首字节 300-500ms
- 成本：每次几分钱，详细按秒计价见百炼定价页
- 吞吐：一个 SDK client 单进程能稳吃 10-20 次合成/秒

经济性很宽容。我从来没遇到过 TTS 账单是值得担忧的科目——LLM 和万相的成本永远比它高一个量级。

## 可能踩的坑

| 错误 | 原因 | 处理 |
|---|---|---|
| Model not found / 404 | 走了 OpenAI 兼容接口 | 换 `dashscope` SDK |
| `InvalidParameter.Voice` | 音色 ID 错（多半是大小写） | 严格用文档中的 ID（`Cherry` 不是 `cherry`） |
| 名字读音机械 / 不正确 | 词表外的 token | 改成发音友好的拼写（如把 "Aidge" 写成 "艾迪琪"），或者用 SSML phoneme 标 |
| 异语种听感不对 | 音色未训练该语言 | 换一个支持该语言的音色 |
| 音频被截断 | 输入超出单次最长字符（约 2000 字） | 按句切分、合成、拼接 |

"按发音友好的方式拼写"这个小技巧很值。品牌名是所有 TTS 的硬伤——我们 Aidge 这个品牌名，写成中文 "艾迪琪" 时模型读得相当准，写成 "Aidge" 时就完全乱来。每个品牌名测一次，固定一种写法即可。

## 系列收尾

五篇文章，一个平台，全是我自己每天用的部分。整个系列的逻辑是同一条：选对模型，搞清楚它在哪条接口（兼容/原生/异步）上，按规矩走流式，把它当任何普通供应商对待——重试、缓存、预算告警一样不少。百炼有它的怪癖，每个云上 AI 平台都有；它的真正价值在于：中文/方言场景几乎没有对手，剩下的能力栈也是 OpenAI/Anthropic 一份足够可信、价格明显更低的备选方案。

如果你基于这个系列搭出了什么有意思的东西，欢迎告诉我。等抽出一个完整周末，我会把这五篇里的演示代码整理成一个小仓库放上 GitHub。
