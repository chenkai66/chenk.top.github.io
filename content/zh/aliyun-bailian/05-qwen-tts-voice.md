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
我参与开发的每个中文产品，最终都会选择调用 Qwen-TTS-Flash。这并不是因为它的价格有多便宜——市面上比它便宜的 TTS 服务并不少。真正的原因是，Qwen-TTS 是唯一一个能在同一个 SDK 中流畅支持**中国大陆方言**（如粤语、四川话、吴语）和英语的工具，而且音质听起来不像 2019 年那种生硬的海关广播。在将其用于某营销视频配音流水线大约半年后，我总结了一些经验，这些是我希望一开始就有人能告诉我的。

![阿里云百炼实战（五）：Qwen-TTS 多语言语音合成 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/illustration_1.png)
## 音色列表

根据模型卡片的说明，Qwen-TTS-Flash 提供了 40 多种不同的音色。以下是我在实际工作中最常用的几个：

![Qwen-TTS 音色目录](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig1_tts_voice_map.png)

在普通话的产品配音场景中，我通常会选择 `Cherry`（温暖亲切的 30 岁左右女性音色）来制作营销类内容，而使用 `Ethan`（沉稳有力的 40 岁左右男性音色）来录制教程或解说类内容。如果是粤语广告，则 `Sunny` 是一个非常稳妥的选择。虽然音色名称保持稳定，但官方会不定期新增音色——因此，在将某个音色写入生产代码之前，建议先到模型卡片中确认最新的音色列表。
## 仅支持原生 API

Qwen-TTS 不依赖 OpenAI 兼容层，而是通过 DashScope 的原生 SDK 进行调用：

![Qwen-TTS 原生调用结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig2_tts_request_flow.png)

一个最简请求示例如下：

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

- 默认情况下，返回的结果是一个 *URL*，而不是直接的音频字节流。这一点和万相类似，**必须在 24 小时内完成下载**（我的习惯是立即下载，并重新上传到自己的 OSS 存储桶中）。
- `format` 参数默认值为 `mp3`，同时也支持 WAV 格式。如果后续需要对音频片段进行拼接处理，建议使用 WAV 格式，因为每段 MP3 文件都会带有额外的头部开销，而 WAV 则没有这个问题。
## 流式 TTS——当延迟成为关键

在语音机器人等实时对话交互场景中，流式处理是必不可少的。通过流式 TTS，返回的数据是音频字节流，可以直接写入播放器或保存为文件：

![流式 TTS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/fig3_tts_streaming.png)

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

在上海区域，流式 TTS 的首字节延迟通常低于 400 毫秒，用户几乎感觉不到延迟，体验非常流畅。而如果是非流式合成，生成一段 30 秒的语音大约需要 4 到 6 秒的实际时间——这种速度对于批量旁白生成还可以接受，但在实时对话场景中就会显得迟缓。
## 多语言与方言的处理细节

Qwen-TTS 能够自动从文本中检测语言，但如果文本中混用了多种字符集，最好还是显式设置 `language` 参数。以下是我的实际使用经验总结：

- 如果是纯普通话内容，直接使用默认值 `language="zh"` 即可。
- 如果是纯英文内容，建议设置为 `language="en"`，此时像 `Eric` 这样的音色表现尤为出色。
- 对于用繁体中文书写的粤语内容，需要指定 `language="zh-yue"`，并选择 `Sunny` 或 `Lily` 作为发音人。
- 在科技类旁白中，常见的情况是中文、日文、韩文（CJK）与英文混用，这时可以不设置 `language` 参数，模型对这种代码切换（code-switch）的处理能力相当不错。

> **小贴士：** 在涉及方言的场景下，上线前一定要找母语者进行 A/B 测试。虽然 Qwen-TTS 的粤语表现已经很不错，但在声调上仍有改进空间——毕竟在粤语中，单字声调的错误可能会导致完全不同的意思。
## SSML——哪些功能可用，哪些不可用

官方文档提到支持 SSML，但并没有明确说明具体哪些标签能够正常工作。根据实际使用经验总结如下：

- `<break time="500ms"/>` — 可用。适合在营销文案中插入句子间的停顿。
- `<emphasis level="strong">` — 可用。可以用来加强语气。
- `<prosody rate="slow">` — 可用。支持 `slow`（慢）、`medium`（中）、`fast`（快）以及具体的百分比数值。
- `<prosody pitch="...">` — 可用。支持相对音高调整，比如 `+10%`。
- `<say-as interpret-as="digits">` — 可用。适用于电话号码、代码、日期等数字类内容的朗读。
- `<phoneme>` — 部分可用。对于中文场景，标注声调的拼音比国际音标（IPA）更可靠。
- `<voice>` — **不可用**。无法在一句话中间切换发音人。如果需要切换，建议分段合成后再拼接音频。
## 拼接长篇旁白

营销脚本通常篇幅较长。对于 60 秒的旁白合成，我的处理方式如下：

```python
def synthesize_long(script: str, voice: str = "Cherry") -> str:
    sentences = split_sentences(script)  # 使用简单的正则表达式分句即可
    parts = []
    for s in sentences:
        resp = SpeechSynthesizer.call(model="qwen3-tts-flash", text=s,
                                       voice=voice, format="wav")
        parts.append(download(resp.output.audio["url"]))
    return concat_wavs(parts, output="/tmp/full.wav")  # 使用 ffmpeg 拼接 WAV 文件
```

为什么要分句处理呢？主要有两点原因：  
1. 短句调用的延迟更低，并发合成可以显著提升效率；  
2. 如果某一句效果不理想，只需单独重新生成这一句，而不用整段重录。  

我们在广告配音中就是采用这种方式——通过 `<break>` 标签控制句子间的停顿时间，并发合成让一段 60 秒的音频在大约 4 秒内就能完成生成。
## 成本

按输出音频秒数计费。流式和非流式一个价。60 秒广告位几块钱——比配音演员小时费便宜得多，也快得让营销团队下午能试几十个版本。

## 音色克隆、预置音色与 SSML 控制：如何选择适合的工具？

在百炼平台中，语音合成提供了三种不同层次的控制方式。选择哪一种方式对项目的影响可能比你想象得更重要。从成本、复杂度到控制能力，这些选项的排序和它们的“高级感”恰好相反。

---

### **预置音色：简单高效的选择**
百炼内置了 40 多种预置音色，覆盖普通话男女声的不同年龄段和语域，还包括粤语、川语、吴语、美式/英式英语，以及一些适合旁白的“角色化”音色。除非有特殊需求，否则直接使用预置音色是最省心的选择。

- **成本**：按标准费率计费，每秒价格固定。
- **延迟**：三种方式中最低。
- **质量**：录音棚级别，稳定可靠。

在我的实际生产环境中，大约 90% 的流量都依赖预置音色完成任务。如果你的需求没有特别复杂的场景，直接用预置音色即可。

---

### **SSML 控制：微调预置音色的利器**
如果你需要更精细地调整发音、节奏或强调，可以结合 SSML（语音合成标记语言）来增强预置音色的表现力。这种方式的成本与直接使用预置音色相同，但能解决许多特定问题：

- **长篇内容需要节奏感**：例如技术教程或有声书，通过 SSML 调整停顿和语速可以让听感更自然。
- **关键短语需要强调**：模型默认生成的语音可能忽略某些重点，SSML 可以手动强化语气。
- **专有名词或术语的正确读法**：特别是在中日韩（CJK）与英文混排的场景下，比如“DSL”容易被读成逐字母发音“D-S-L”，而不是正确的“deh-es-el”。

以下是我营销流水线中的一个真实案例：

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

- `<say-as>`：确保“DSL”逐字母读出，符合缩写的正确发音。
- `<break>`：在适当位置加入停顿，给听众留出消化信息的时间。
- `<prosody rate="95%">`：稍微放慢语速，方便听众记录关键信息。

这些功能是纯文本输入无法实现的，尤其在需要精准控制的场景中，SSML 是不可或缺的工具。

---

### **音色克隆：定制专属声音**
如果预置音色无法满足需求，百炼还提供了音色克隆功能。通过上传 10-30 秒的参考音频，你可以生成一个专属的 `voice id`，用于后续调用。虽然这项功能非常强大，但需要注意以下两点：

1. **质量取决于参考音频的质量**  
   - 如果参考音频是清晰、单人录制、录音棚级别的，克隆效果通常非常好。
   - 如果参考音频来自手机录制、带有背景噪音或多说话人混合，克隆结果可能会显得怪异甚至难以接受。因此，在正式使用前一定要人工验证克隆音色的质量。

2. **法律与伦理问题**  
   - 在许多地区，未经许可克隆他人声音可能触犯法律，百炼的服务条款也明确禁止这种行为。
   - 只能克隆你自己拥有版权或获得书面授权的声音。如果平台检测到疑似冒充公众人物的行为，可能会撤销相关 `voice id`。

在我的营销项目中，我们曾克隆过创始人的声音（已获授权），用于一条产品线的几十个版本的 30 秒宣传语。具体流程如下：
- 先进行一次克隆，生成 `voice id`。
- 使用 20 条样本测试克隆音色的质量。
- 验证通过后，将该 `voice id` 应用于整条产品线的所有语音合成任务。

克隆完成后，其使用成本与预置音色相同。

---

### **决策树：如何选择合适的工具？**

1. **预置目录里没有需要的声音？**  
   → 使用音色克隆，但务必确保获得授权。

2. **预置音色可用，但发音或节奏需要调整？**  
   → 使用 SSML 微调。

3. **预置音色完全满足需求？**  
   → 直接使用，无需过度纠结。

通过合理选择工具，你可以在成本、效率和效果之间找到最佳平衡点。
## 延迟预算：流式分块与口型同步的时间窗口

![阿里云百炼实战（五）：Qwen-TTS 多语言语音合成 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-bailian/05-qwen-tts-voice/illustration_2.png)

在交互式语音产品（如语音机器人、实时数字人）中，延迟的重要性往往超过质量。尤其是在用户体验的边际上，哪怕几百毫秒的差异都会显著影响感知。以下是流式 TTS 的延迟分解：

1. **请求初始化**（TLS 握手、首个请求包传输）：国内通常为 30-80 毫秒，跨区域则可能达到 200-400 毫秒。
2. **TTFT**（首段音频到达时间）：在上海区域，对于全新的请求，通常需要 200-400 毫秒。
3. **每块间隔**：音频块每隔 100-200 毫秒到达一次，每块包含约 200-400 毫秒的音频内容。
4. **尾部延迟**：最后一块音频会在文本对应的实际发音时长基础上再延迟约 100 毫秒。

以“用户输入文字 → 音频播放”这种交互场景为例，端到端的首音延迟通常在 250-500 毫秒之间。这是用户感知“即时性”的关键阈值——低于 500 毫秒会被认为是即时响应，而超过 1 秒则会让人觉得明显卡顿。

### 数字人口型同步的需求

要让数字人的口型与语音同步，不仅需要音频块，还需要每个音素的时间戳。然而，Qwen-TTS 的基础 API 并不直接提供音素级时间戳。以下是两种常见的解决方案：

- **使用强制对齐工具处理输出**：像 `whisper-timestamped` 或 `mfa`（Montreal Forced Aligner）这样的工具可以将合成的音频与源文本结合，生成每个单词或音素的时间戳。这种方法会增加 200-500 毫秒的整体延迟，适合非实时的数字人制作场景。
- **根据文本长度估算时间戳**：在实时场景中，如果无法承受强制对齐的额外开销，可以通过估算来近似时间戳。例如，中文每个字符在默认语速下大约对应 150 毫秒的音频，英文每个单词则约为 250 毫秒。虽然误差会累积，但对于前 5-10 秒的内容来说，这种估算通常足够准确，能够让口型看起来大致匹配。

以下是一个支持音素同步的流式处理代码示例，适用于帧感知的数字人场景：

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
        # 粗略估算：16kHz 单声道 PCM 格式每秒约 32 KB
        chunk_duration_ms = len(chunk) / 32 * 1000
        elapsed_audio_ms += chunk_duration_ms
        on_phoneme_estimate(elapsed_audio_ms)
```

这套方案足以应对营销类数字人的需求。但如果目标是高保真配音，则需要将音频渲染为文件后，再通过强制对齐工具生成精确的时间戳。
## 音频后处理：响度标准化与静音修剪

直接从 TTS 模型生成的音频通常无法直接用于生产环境。为了让音频适合听众使用，任何交付流水线中都必须包含两个关键的后处理步骤：

**响度标准化。** 不同的语音、不同的文本内容，甚至同一语音模型在不同运行中的输出，其感知响度都会有所不同。如果不进行标准化处理，听众可能会被某些过大的声音“震到”，而另一些声音又可能小到听不清。业界的标准是 EBU R128，播客或长篇内容的目标响度为 -16 LUFS，流媒体平台则通常要求 -14 LUFS。

```bash
ffmpeg -y -i in.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 out.wav
```

通过这一行简单的 `ffmpeg` 命令，可以将所有 TTS 输出调整到相同的感知响度。建议对每一条音频都执行此操作，无一例外。计算成本极低，CPU 开销仅为微秒级别。

**静音修剪。** Qwen-TTS 生成的音频通常会在开头和结尾分别带有 100-300 毫秒的静音段。如果直接拼接多句旁白，这些静音会累积起来——例如，一个 20 句的脚本可能会额外增加 4-6 秒的空白时间。可以通过以下命令修剪掉多余的静音：

```bash
ffmpeg -y -i in.wav -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse" out.wav
```

在拼接之前先修剪静音，然后通过显式添加受控的句间停顿来保证节奏自然。这样处理后的旁白听起来会更加流畅，而不是显得断断续续或节奏怪异。

**完整的响度标准化 + 静音修剪 + 拼接流水线：**

```python
def post_process(parts: list[str], output: str):
    """标准化、修剪、拼接音频片段。"""
    cleaned = []
    for i, p in enumerate(parts):
        out = f"/tmp/clean_{i}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", p,
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,"
                               "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,"
                               "areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse",
                        out], check=True, capture_output=True)
        cleaned.append(out)
    # 在句间插入 250ms 的静音间隔
    with open("/tmp/concat.txt", "w") as f:
        for i, c in enumerate(cleaned):
            f.write(f"file '{c}'\n")
            if i < len(cleaned) - 1:
                f.write("file '/tmp/silence_250ms.wav'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", "/tmp/concat.txt", "-c", "copy", output], check=True)
```

句间插入 250 毫秒的静音间隔是让连续 TTS 输出听起来像“一气呵成”而非“东拼西凑”的关键所在。
## 生成音频的每分钟成本

官方按秒计费的方式是最合理的单位，但为了更直观，这里提供一个按分钟计算的成本估算，包含了一些实际使用中的额外开销：

| 模式 | 每分钟基础费用 | 加上 ffmpeg 后处理（可忽略不计） | 音色克隆附加费用 |
|---|---|---|---|
| 预置音色，非流式 | ~0.6 元/分钟 | +0 | 不适用 |
| 预置音色，流式 | ~0.6 元/分钟 | +0 | 不适用 |
| 克隆音色 | ~0.6 元/分钟 | +0 | 一次性克隆费用 ~5 元 |
| 带长 SSML 标记 | ~0.6 元/分钟 | +0 | 不适用 |

以一本 60 分钟的有声书为例，按 0.6 元/分钟计算，总成本为 36 元；而一条 30 秒的广告音频仅需 0.3 元。相比之下，聘请一位配音演员的单项目费用通常在 2000 至 5000 元之间（中端中文配音市场）。通过 API 生成音频的单条成本仅为人工配音的约 1/100 到 1/1000，代价是每次生成仅需零点几秒的 CPU 时间，且无需协调档期。只有在一种情况下值得选择配音演员：这是一条需要特定人声的品牌核心广告（hero brand spot），且目标受众对这个声音已经非常熟悉。除此之外，API 几乎总是更优的选择。

不过，隐藏的成本在于后期处理的工程时间。搭建一套响度调整、裁剪和拼接的流水线可能需要一天的时间，但一旦完成，这套流程可以为未来几年的所有音频输出服务。如果省略后期处理，无论模型生成的音频质量多高，最终效果都会显得业余而不专业。
## 系列总结

到这里，五篇文章就全部结束了。简单回顾一下：

- **第一篇** — 百炼和 DashScope 的入门指南。
- **第二篇** — Qwen LLM 系列模型的详细介绍，包括一些细节功能（如函数调用、JSON 模式、`enable_thinking`）。
- **第三篇** — Qwen-Omni 在多模态理解中的应用。
- **第四篇** — 万相在视频生成领域的使用。
- **第五篇** — Qwen-TTS 的语音合成能力（也就是本文的内容）。

与这个系列配套的还有**阿里云 PAI** 系列文章，主要讲解 DSW、DLC 和 EAS——这是你自己掌控 GPU 资源进行模型训练和推理的部分。在我接触的团队中，大部分都会根据实际需求灵活选择：如果需要快速使用预训练模型，会选择百炼；如果需要对模型权重进行精细控制，则会转向 PAI。记住，选择工具时要看你真正需要控制的是什么，而不是看哪个写在简历上更好看。