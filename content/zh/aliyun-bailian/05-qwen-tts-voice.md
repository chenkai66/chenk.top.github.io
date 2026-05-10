---
title: "阿里云百炼实战（五）：Qwen-TTS 多语言语音合成"
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
我经手的中文产品最后都用 Qwen-TTS-Flash，真不是因为便宜——市面上有更便宜的 TTS API。真正的原因是只有它在同一个 SDK 里搞定了**大陆方言**（粤语、四川话、吴语）和英语，而且声音听起来不像 2019 年的海关广播。用了大概半年做营销视频配音 pipeline 后，有些坑我希望第一天就有人告诉我。

![Aliyun Bailian (5): Qwen-TTS for Multilingual Voice — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/illustration_1.png)

## 声音目录

模型卡片显示 Qwen-TTS-Flash 开放了 40+ 声音。我最常用的是：

![Qwen-TTS voice catalogue](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig1_tts_voice_map.png)

中文产品 narration 我默认用 `Cherry`（温暖，30 多岁女性）做营销内容，`Ethan`（稳重，40 多岁男性）做教程或解说内容。粤语广告 spot 选 `Sunny` 最稳。声音名字是稳定的，但新声音会定期增加——在生产代码里固定某个声音前，先去模型卡片拉一份 canonical list 确认。

## 只用原生 API

Qwen-TTS 不走 OpenAI 兼容层。得用 DashScope 原生 SDK 调用：

![Qwen-TTS native call structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig2_tts_request_flow.png)

最小化请求示例：

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

两点要注意：

- 默认输出是 *URL*，不是字节流。跟 Wanxiang 一样，**24 小时内下载**（我一般是立即下载并转存到自己的 OSS bucket）。
- `format` 默认是 `mp3`。也支持 WAV；如果是 downstream 拼接工作，我偏好 WAV，因为每个 chunk 没有 MP3 header 开销。

## 流式 TTS —— 延迟敏感场景

做语音机器人（实时对话 UI）得用 streaming。delta 是音频字节，可以直接写播放器或文件：

![Streaming TTS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig3_tts_streaming.png)

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

流式的首字节时间（Time-to-first-byte）在上海区域通常低于 400ms，快到让用户感觉是即时的。非流式模式下一条 30 秒的 utterance 墙钟时间接近 4-6 秒——做批量 narration 没问题，做聊天就太慢了。

## 多语言和方言细节

Qwen-TTS 会根据文本做语言 *检测*，但混合脚本时最好显式设置 `language`。我的生产规则：

- 纯中文文本 → `language="zh"`（默认）。
- 纯英文文本 → `language="en"`。像 `Eric` 这种声音在这里表现很好。
- 繁体中文粤语文本 → `language="zh-yue"`，声音选 `Sunny` 或 `Lily`。
- 混合 CJK + 英文（技术 narrations 的常见情况）→ 不设 language，模型处理 code-switch 出乎意料地好。

> **Tip:** 做方言工作，上线前**务必**跟母语者做 A/B 测试。Qwen-TTS 的粤语不错，但声调不完美——粤语里一个音节的声调错了，意思可能完全变样。

## SSML —— 哪些管用，哪些不管用

文档列了 SSML 支持，但没说哪些标签真管用。经验之谈：

- `<break time="500ms"/>` — 管用。营销文案句子间停顿用这个。
- `<emphasis level="strong">` — 管用。
- `<prosody rate="slow">` — 管用。`slow`, `medium`, `fast` 或数字百分比。
- `<prosody pitch="...">` — 相对变化管用（比如 `+10%`）。
- `<say-as interpret-as="digits">` — 电话号码、代码、日期管用。
- `<phoneme>` — 部分管用。中文上用带声调的拼音比 IPA 可靠。
- `<voice>` — **不管用**。不能在 utterance 中途切换声音。得用 separate calls 然后拼接。

## 拼接片段做 narration

营销脚本通常很长。60 秒配音的模式：

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

为什么要拆分？两个原因：(1) 短 utterance 的单次调用延迟低得多，并行合成更快；(2) 某一句坏了可以只重 roll 那一句，不用重做整个 take。我们做广告 spot 就用这个——`<break>` 标签处理句间停顿，并行合成意味着 60 秒的 clip 大概 4 秒就能 ready。

## 成本

按输出音频秒数计费。流式和非流式计费一样。60 秒广告 spot 也就几块钱——比真人 voice actor 的时薪便宜得多，而且快到让营销团队一个下午迭代几十个版本。

## 声音克隆 vs 预设声音 vs SSML 控制：选对旋钮

百炼给了三层声音控制，选哪个比想象中更重要。成本、复杂度、控制力跟“炫酷”程度成反比：

**预设声音。** 目录里的 40+ 声音。除非有特定理由，否则就用这些。它们覆盖了所有常见年龄段和 register 的中文男/女声，加上粤语、四川话、吴语、英语（美/英），还有一些用于 narration 的“角色”声音。每秒成本是标准费率；延迟是三层里最低的；质量是录音棚级。我 90% 的生产流量都是预设声音。

**预设声音上的 SSML 控制。** 声音不变，但微调 prosody、breaks、emphasis、发音。成本跟预设声音一样。以下情况用这个：
- 长篇幅 narration 需要 deliberate pacing（技术教程、有声书）。
- 特定短语需要强调，而模型默认会漏掉。
- 有专有名词或技术术语模型读错（特别是在 CJK + 英文 code-switch 场景："DSL" 经常读成 "D-S-L" 而不是 "deh-es-el"）。

我营销 pipeline 里的真实例子：

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

`<say-as>` 让 "DSL" 按字母读（缩略语正确读法）。`<break>` 在听众需要吸收信息的地方加 deliberate pause。`<prosody rate="95%">` 稍微放慢 URL 推荐部分，让听众来得及记下来。纯文本输入做不到这些。

**声音克隆（音色迁移）。** 百炼暴露了一个单独的 clone API，上传 10-30 秒参考音频，返回一个 voice id 用于后续调用。这是真的，但有两个警告：

- **质量波动。** 清晰、单人、录音棚参考音频的克隆效果极佳。电话音质、嘈杂或多人的参考音频克隆出来会诡异得吓人。生产使用前务必人工筛查克隆效果。
- **法律/伦理风险。** 未经同意克隆真人声音在很多司法辖区越来越违法，且明确被百炼 ToS 禁止。只克隆你拥有或获得书面许可的声音。平台可能会撤销看起来像 impersonations 公众人物的 voice ids。

我的营销 pipeline 里克隆了创始人的声音（经同意），用于一个产品线，他们想要几十种 30 秒 pitch 的变体。我们做一次，在 20 个 sample utterances 上验证质量，然后该产品线所有内容都用生成的 voice id。克隆创建后，成本跟预设声音一样。

决策树：
- 需要预设目录里没有的声音？→ 克隆，需 permission。
- 预设声音管用但发音/节奏不对？→ SSML。
- 预设声音直接管用？→ 预设，别想太多。

## 延迟预算：流式 chunk，口型同步窗口

![Aliyun Bailian (5): Qwen-TTS for Multilingual Voice — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/illustration_2.png)

交互式语音产品（语音 bot，实时 avatar），延迟比边际质量更重要。流式 TTS 的延迟分解：

1. **请求 setup**（TLS 握手，首个请求包）：国内 30-80ms，跨区 200-400ms。
2. **TTFT**（首个音频 chunk 到达）：上海区域新请求通常 200-400ms。
3. **每 chunk 节奏**：chunk 大约每 100-200ms 到达一个，每个包含 200-400ms 音频。
4. **尾部**：最后一个 chunk 在文本 spoken duration 后到达，加约 100ms。

对于“用户输入文本 → 播放音频”的 UX，端到端感知延迟落在首个声音的 250-500ms 左右。这是“感觉即时”的阈值——低于 500ms 读作 immediate，超过 1s 读作 laggy。

对于 **avatar 口型同步**，你需要音频 chunk 和每 phoneme 时间戳 track。基础 API 里 Qwen-TTS 不返回每 phoneme 时间戳。两个 workaround：

- **在输出上跑 forced-aligner。** 工具如 `whisper-timestamped` 或 `mfa` (Montreal Forced Aligner) 接收合成音频和源文本，输出每 word 或每 phoneme 时间戳。总 pipeline 延迟增加 200-500ms。非实时 avatar 工作我用这个。
- **从文本长度估算时间戳。** 实时工作负担不起 alignment pass 时，近似估算：默认速率下每个中文字符约 150ms 音频，每个英文单词约 250ms。误差会累积，但前 5-10 秒通常准确 enough， casual glance 下口型看起来是对的。

带 frame-aware avatar 的流式模式：

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

做营销 avatar 够用了。高保真 dubbing 不够用——那种情况渲染到文件然后跑 forced alignment。
## 音频后处理：响度标准化与静音修剪

原生 TTS 输出几乎没法直接用于生产环境。任何要交付给听众的流水线，这两步后处理都是必不可少的：

**响度标准化（Loudness normalization）**。不同的音色、不同的文本，哪怕是同一音色跑两次，生成的音频感知响度都不一样。不做标准化，听众前一秒被震得耳朵疼，后一句就听不见了。行业标准是 EBU R128，播客或长内容目标 -16 LUFS，流媒体平台通常 -14 LUFS。

```bash
ffmpeg -y -i in.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 out.wav
```

这一行 ffmpeg 命令能把所有 TTS 输出拉到同一感知响度。每个输出都要跑，没例外。CPU 开销也就是微秒级的事。

**静音修剪（Silence trimming）**。Qwen-TTS 经常会产出 100-300ms 的头部静音，尾部也差不多。如果是拼接 narration，这就麻烦了——20 句的脚本最后能多出 4-6 秒的空白。用这个命令修剪：

```bash
ffmpeg -y -i in.wav -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse" out.wav
```

先 trim 再 concatenation，然后用显式的 silence-padding 加回可控的句间停顿。这样出来的 narration 听起来是有意为之，而不是节奏尴尬。

**把响度 + 修剪 + 拼接做成流水线：**

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

句间 250ms 的静音填充，是让背靠背 TTS 听起来像单次录制而不是拼接怪物的关键。

## 生成音频的每分钟成本

官方按秒计价没错，但这里给个带实际 overhead 的每分钟版本：

| Mode | Per-minute base | With ffmpeg post (negligible) | With cloning surcharge |
|---|---|---|---|
| Preset voice, non-stream | ~0.6 RMB / min | +0 | n/a |
| Preset voice, stream | ~0.6 RMB / min | +0 | n/a |
| Cloned voice | ~0.6 RMB / min | +0 | one-time clone cost ~5 RMB |
| With long-form SSML markup | ~0.6 RMB / min | +0 | n/a |

60 分钟的有声书按 0.6 RMB/min 算是 36 RMB。30 秒广告位只要 0.3 RMB。跟真人配音 session fee 比（中级中文配音 talent：2000-5000 RMB per project），API 每个产出大概便宜 100-1000x。代价是每次生成只占几分之一秒 CPU 时间，而且零预订摩擦。 deciding "should I hire a voice actor or use Qwen-TTS" 的盈亏平衡点大概是“这是个需要观众熟知的特定人声的品牌英雄 spot"，除此之外几乎没别的情况。

隐藏成本是后处理 engineering time：花 1 天搭建响度/修剪/拼接流水线，然后摊销到你未来几年发布的每个 clip 上。跳过后处理，不管模型多好，音频听起来都很业余。


## 系列收尾

这就是五篇。回顾一下：

- **Article 1** — Bailian / DashScope 定位。
- **Article 2** — Qwen LLM 家族那些琐碎细节（function calling, JSON mode, `enable_thinking`）。
- **Article 3** — Qwen-Omni 多模态理解。
- **Article 4** — Wanxiang 视频生成。
- **Article 5** — Qwen-TTS 语音（本篇）。

配套的 **Aliyun PAI** 系列覆盖 DSW / DLC / EAS —— 这是 *self-managed* GPU 层，你自己训练和服务模型。我合作的大多数团队最后都是混用：想要别人的预训练模型就用 Bailian，需要控制 weights 就用 PAI。按你实际需要控制什么来选，别按什么写在简历上更 impress 来选。