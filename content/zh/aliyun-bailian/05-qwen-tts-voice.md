---
title: "阿里云百炼（五）：Qwen-TTS 语音合成"
date: 2026-03-01 09:00:00
tags:
  - Aliyun Bailian
  - Qwen-TTS
  - TTS
  - Voice
  - Audio
categories: 阿里云百炼
lang: zh
mathjax: false
series: aliyun-bailian
series_title: "阿里云百炼实战手册"
series_order: 5
description: "Qwen-TTS-Flash 实战：只走原生 API、40+ 音色目录（含粤语 / 川语）、流式合成，以及文档一笔带过的 SSML 行为细节。"
disableNunjucks: true
translationKey: "aliyun-bailian-5"
---
我经手的所有中文产品最终都选用了 Qwen-TTS-Flash。这并非因为它是最便宜的 TTS API（市面上确实有更低价的选择），而是因为它是目前唯一能在同一 SDK 中同时支持**中国大陆方言**（如粤语、四川话、吴语）和英语的方案，且合成语音自然流畅，完全不像 2019 年那种机械感十足的海关广播音。在将它用于营销视频配音流水线半年后，我总结出一些早期容易踩的坑——这些正是我希望自己第一天就被告知的内容。

![阿里云百链 (5)：多语言语音 — Qwen-TTS 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/illustration_1.png)

## 声音目录

根据模型卡片，Qwen-TTS-Flash 提供了 40 多种声音。我最常用的是以下几种：

![Qwen-TTS 语音目录](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig1_tts_voice_map.png)

中文产品旁白方面，营销内容默认使用 `Cherry`（温暖、30 多岁女性），教程或解说类内容则用 `Ethan`（沉稳、40 多岁男性）。粤语广告通常选择 `Sunny`，效果稳妥可靠。虽然声音名称（voice ID）本身是稳定的，但阿里云会定期新增声音——因此，在生产代码中硬编码某个声音前，请务必从模型卡片获取最新权威列表进行核对。

## 只用原生 API

Qwen-TTS 不走 OpenAI 兼容层，必须通过 DashScope 原生 SDK 调用。

![Qwen-TTS 原生调用结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig2_tts_request_flow.png)

最小化请求示例如下：

```python
import os, dashscope, requests
from dashscope.audio.qwen_tts import SpeechSynthesizer

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

resp = SpeechSynthesizer.call(
    model="qwen3-tts-flash",
    text="欢迎来到杭州，全城最安静的咖啡馆就在西湖边上。",
    voice="Cherry",
    format="mp3",
)

audio_url = resp.output.audio["url"]
with open("/tmp/out.mp3", "wb") as f:
    f.write(requests.get(audio_url, timeout=30).content)
```

这里有两个关键点需要注意：

- 默认输出是一个 *URL*，而非原始字节流。和 Wanxiang 一样，**生成的音频链接有效期仅为 24 小时**（我通常立即下载并重新上传到自己的 OSS Bucket）。
- `format` 默认为 `mp3`，但也支持 WAV 格式；如果后续需要拼接多个片段，我更倾向使用 WAV，因为每个片段没有 MP3 的头部开销。

## 流式 TTS —— 延迟敏感场景

在语音机器人等实时对话界面中，流式传输必不可少。返回的 delta 是原始音频字节，可直接写入播放器或文件。

![流式 TTS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig3_tts_streaming.png)

```python
from dashscope.audio.qwen_tts import SpeechSynthesizer

with open("/tmp/streamed.mp3", "wb") as f:
    for resp in SpeechSynthesizer.call(
        model="qwen3-tts-flash",
        text="这段语音是从模型那边一段一段流式返回的。",
        voice="Cherry",
        stream=True,
    ):
        if resp.output and resp.output.audio:
            f.write(resp.output.audio["data"])
```

在上海区域，流式模式的首字节时间（Time-to-first-byte）通常低于 400 毫秒，用户几乎感知不到延迟。相比之下，非流式模式合成一段 30 秒的语音大约需要 4–6 秒——这对批量配音足够高效，但在实时聊天场景中就显得迟滞了。

## 多语言和方言细节

Qwen-TTS 能自动检测文本语言，但如果混合多种文字系统，建议显式指定 `language` 参数。我在生产环境中的规则如下：

- 纯普通话文本 → `language="zh"`（默认值）。
- 纯英文文本 → `language="en"`，此时 `Eric` 等声音表现尤为出色。
- 使用繁体字书写的粤语文本 → `language="zh-yue"`，推荐声音为 `Sunny` 或 `Lily`。
- 中英混杂（技术类旁白常见情况）→ 不设置 `language`，模型对语码转换（code-switching）的处理出人意料地好。

> **提示：** 涉及方言时，**上线前务必与母语者进行 A/B 测试**。Qwen-TTS 的粤语合成整体质量不错，但在声调上仍有瑕疵——而粤语恰恰是声调语言，一个音节的声调错误就可能彻底改变语义。

## SSML —— 哪些管用，哪些不管用

官方文档虽提到支持 SSML，却未明确说明哪些标签实际生效。根据我的实测经验：

- `<break time="500ms"/>` —— 有效，适合在营销文案的句子间插入停顿。
- `<emphasis level="strong">` —— 有效。
- `<prosody rate="slow">` —— 有效，支持 `slow`、`medium`、`fast` 或百分比数值。
- `<prosody pitch="...">` —— 支持相对调整（如 `+10%`）。
- `<say-as interpret-as="digits">` —— 对电话号码、验证码、日期等结构化数据发音准确。
- `<phoneme>` —— 部分有效；中文场景下，带声调的拼音比 IPA 更可靠。
- `<voice>` —— **无效**。无法在单次合成中切换声音，必须分多次调用再拼接。

## 拼接片段做旁白

营销脚本往往较长。以下是合成一段 60 秒旁白的典型做法：

```python
def synthesize_long(script: str, voice: str = "Cherry") -> str:
    sentences = split_sentences(script)  # your splitter; basic regex is fine
    parts = []
    for s in sentences:
        resp = SpeechSynthesizer.call(model="qwen3-tts-flash", text=s,
                                       voice=voice, format="wav")
        parts.append(download(resp.output.audio["url"]))
    # ffmpeg concat is the simplest reliable concat for WAV
    return concat_wavs(parts, output="/tmp/full.wav")
```

为什么要拆分？主要有两个原因：（1）短文本单次调用延迟更低，并行合成能显著提升整体速度；（2）若某一句效果不佳，只需重试该句，无需重新生成整段。我们在广告制作中就采用这种方式——用 `<break>` 控制句间停顿，并行合成让 60 秒的音频在约 4 秒内即可完成。

## 成本

计费按输出音频的秒数计算，流式与非流式价格相同。一段 60 秒的广告配音仅需几元人民币——远低于真人配音演员的小时费率，且生成速度极快，营销团队一个下午就能迭代几十个版本。

## 声音克隆 vs 预设声音 vs SSML 控制：选对工具

百炼提供了三层声音控制机制，选择哪一层比大多数人想象的更重要。成本、复杂度与控制力通常与功能的“炫酷”程度成反比。

**预设声音**  
即目录中的 40 多种声音。除非有特殊需求，否则优先使用它们。这些声音覆盖了普通话各年龄段、性别和语域，还包括粤语、四川话、吴语、英语（美式/英式），以及若干专为叙事设计的“角色音”。每秒费用为标准费率，延迟最低，音质达到录音棚级别。我约 90% 的生产流量都来自预设声音。

**基于预设声音的 SSML 控制**  
保持声音不变，仅调整韵律、停顿、强调和发音。费用与预设声音相同。适用于以下场景：
- 长篇旁白需要精确控制节奏（如技术教程、有声书）；
- 某些短语需要额外强调，而模型默认处理不足；
- 存在专有名词或技术术语被误读（尤其在中英混杂场景：“DSL” 常被读作 “D-S-L”，而非正确的字母发音 “deh-es-el”）。

我营销流水线中的真实案例：

```python
text = """
<speak>
  欢迎使用 <say-as interpret-as="characters">DSL</say-as> 翻译系统。
  <break time="300ms"/>
  本次更新支持<emphasis level="strong">多语种</emphasis>批量处理。
  <prosody rate="95%">详细文档请参见官网。</prosody>
</speak>
"""
resp = SpeechSynthesizer.call(model="qwen3-tts-flash", text=text, voice="Cherry", format="wav")
```

其中 `<say-as>` 确保 “DSL” 按字母逐字发音（符合缩略语规范）；`<break>` 在关键信息后插入刻意停顿，便于听众理解；`<prosody rate="95%">` 稍微放慢 URL 推荐部分，方便听众记录。这些效果都无法通过纯文本输入实现。

**声音克隆（音色迁移）**  
百炼提供独立的克隆 API：上传 10–30 秒参考音频，即可获得一个 voice ID 用于后续调用。该功能真实可用，但需注意两点：

- **质量不稳定**：清晰、单人、录音棚级别的参考音频克隆效果极佳；而电话录音、嘈杂环境或多说话人的音频克隆后可能显得诡异。务必人工审核后再投入生产。
- **法律与伦理风险**：未经许可克隆他人声音在许多地区已属违法，且明确违反百炼服务条款。仅可克隆你拥有版权或已获书面授权的声音。平台有权撤销疑似模仿公众人物的 voice ID。

在我的营销流程中，我们曾（在创始人同意下）克隆其声音，用于一个需要大量 30 秒口播变体的产品线。我们一次性完成克隆，并在 20 个样本上验证音质，之后该产品线所有内容均使用该 voice ID。克隆创建后，后续调用成本与预设声音一致。

决策树如下：
- 需要目录中不存在的声音？→ 克隆（需授权）。
- 预设声音可用，但发音或节奏不理想？→ 使用 SSML。
- 预设声音效果满意？→ 直接使用，无需过度设计。

## 延迟预算：流式分块与口型同步窗口

![阿里云百链（5）：Qwen-TTS 多语言语音 — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/illustration_2.png)

对于交互式语音产品（如语音机器人、实时虚拟人），延迟比边际音质更重要。流式 TTS 的延迟构成如下：

1. **请求建立**（TLS 握手、首包发送）：国内约 30–80 毫秒，跨区域约 200–400 毫秒。
2. **首字节时间（TTFT）**：在上海区域，新请求通常在 200–400 毫秒内返回首个音频块。
3. **分块节奏**：音频块约每 100–200 毫秒到达一次，每块包含 200–400 毫秒的音频。
4. **尾部延迟**：最后一个块在文本朗读完毕后约 100 毫秒到达。

在“用户输入文本 → 音频播放”的体验中，端到端的首声延迟通常在 250–500 毫秒之间。这是“即时响应”的心理阈值——低于 500 毫秒被视为实时，超过 1 秒则明显卡顿。

对于 **虚拟人口型同步**，你需要音频分块及每个音素（phoneme）的时间戳。但 Qwen-TTS 基础 API 并不返回音素级时间戳。可行的替代方案有两种：

- **对输出音频运行强制对齐器（forced-aligner）**：如 `whisper-timestamped` 或 `mfa`（Montreal Forced Aligner），输入合成音频和原文，即可输出词级或音素级时间戳。这会增加 200–500 毫秒的处理延迟，我将其用于非实时虚拟人项目。
- **基于文本长度估算时间戳**：在无法承受对齐开销的实时场景中，可粗略估算——默认语速下，每个汉字约对应 150 毫秒音频，每个英文单词约 250 毫秒。虽然误差会累积，但前 5–10 秒通常足够准确，肉眼观看时口型基本自然。

配合帧感知虚拟人的流式模式如下：

```python
import time
def stream_with_phoneme_sync(text: str, on_chunk, on_phoneme_estimate):
    start = time.time()
    elapsed_audio_ms = 0
    for resp in SpeechSynthesizer.call(model="qwen3-tts-flash", text=text,
                                        voice="Cherry", stream=True):
        if not resp.output or not resp.output.audio: continue
        chunk = resp.output.audio["data"]
        on_chunk(chunk)
        # rough estimate: 16kHz mono PCM = 32 KB per second
        chunk_duration_ms = len(chunk) / 32 * 1000
        elapsed_audio_ms += chunk_duration_ms
        on_phoneme_estimate(elapsed_audio_ms)
```

这对营销级虚拟人已足够。但若追求高保真配音（如影视级 dubbing），仍需先渲染成文件，再运行强制对齐。

## 音频后处理：响度标准化与静音修剪

原始 TTS 输出几乎无法直接用于生产。任何面向听众的流水线都必须包含以下两步后处理：

**响度标准化（Loudness normalization）**  
不同声音、不同文本，甚至同一声音多次调用，生成的音频感知响度都可能不同。若不做标准化，听众可能前一句被震得耳朵疼，后一句却听不清。行业标准为 EBU R128：播客或长内容目标为 -16 LUFS，流媒体平台通常为 -14 LUFS。

```bash
ffmpeg -y -i in.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 out.wav
```

这一行 ffmpeg 命令可将所有 TTS 输出统一至相同感知响度。**每次输出都必须执行，无一例外**。CPU 开销仅为微秒级。

**静音修剪（Silence trimming）**  
Qwen-TTS 常在音频开头和结尾生成 100–300 毫秒的静音。若用于拼接旁白，问题会被放大——一段 20 句的脚本可能多出 4–6 秒空白。修剪命令如下：

```bash
ffmpeg -y -i in.wav -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse" out.wav
```

应在拼接前先修剪静音，再通过显式的静音填充添加可控的句间停顿。这样处理后的旁白听起来自然连贯，而非生硬拼接。

**将响度标准化、静音修剪与拼接整合为流水线：**

```python
def post_process(parts: list[str], output: str):
    """Normalize, trim, concat."""
    cleaned = []
    for i, p in enumerate(parts):
        out = f"/tmp/clean_{i}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", p,
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,"
                               "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,"
                               "areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse",
                        out], check=True, capture_output=True)
        cleaned.append(out)
    # Concat with controlled 250ms gaps between sentences
    with open("/tmp/concat.txt", "w") as f:
        for i, c in enumerate(cleaned):
            f.write(f"file '{c}'\n")
            if i < len(cleaned) - 1:
                f.write("file '/tmp/silence_250ms.wav'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", "/tmp/concat.txt", "-c", "copy", output], check=True)
```

句间 250 毫秒的静音填充，正是让连续 TTS 听起来像单次录制而非拼接产物的关键。

## 生成音频的每分钟成本

官方按秒计价合理，但这里提供一个包含实际开销的每分钟成本参考：

| 模式 | 每分钟基础费用 | ffmpeg 后处理（可忽略） | 克隆附加费 |
|---|---|---|---|
| 预设声音，非流式 | ~0.6 RMB / min | +0 | 不适用 |
| 预设声音，流式 | ~0.6 RMB / min | +0 | 不适用 |
| 克隆语音 | 约 0.6 RMB/分钟 | +0 | 一次性克隆费用约 5 RMB |
| 使用长格式 SSML 标记 | 约 0.6 RMB/分钟 | +0 | 不适用 |

一部 60 分钟的有声书成本约 36 元，30 秒广告仅需 0.3 元。相比之下，中级中文配音演员的单项目报价通常在 2000–5000 元之间——API 方案的成本约为其 1/100 至 1/1000，且每次生成仅消耗零点几秒 CPU 时间，无需预约协调。只有当项目是“需要观众熟知的特定人声的品牌核心广告”时，才值得考虑真人配音；其他绝大多数场景，Qwen-TTS 都是更优解。

真正的隐藏成本在于后处理工程投入：搭建响度/修剪/拼接流水线约需一天时间，但此后可摊销至未来数年发布的每一个音频片段。若跳过后处理，无论模型多先进，最终音频听起来都会显得业余。

## 系列收尾

至此，五篇文章全部完结。简要回顾：

- **第一篇** —— Bailian / DashScope 平台概览与首次调用。
- **第二篇** —— Qwen LLM 家族及其细节（函数调用、JSON 模式、`enable_thinking`）。
- **第三篇** —— Qwen-Omni 多模态理解能力。
- **第四篇** —— Wanxiang 视频生成端到端流程。
- **第五篇** —— Qwen-TTS 语音合成（本文）。

配套的 **Aliyun PAI** 系列则聚焦 DSW / DLC / EAS —— 这是 *自管理* 的 GPU 层，用于训练和部署自有模型。我合作的大多数团队最终都采用混合策略：需要现成预训练模型时用 Bailian，需要掌控模型权重时用 PAI。选择依据应是你实际需要控制什么，而非哪种方案写在简历上更亮眼。
