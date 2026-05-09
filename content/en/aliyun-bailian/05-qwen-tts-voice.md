---
title: "Aliyun Bailian (5): Qwen-TTS for Multilingual Voice"
date: 2026-03-01 09:00:00
tags:
  - Aliyun Bailian
  - Qwen-TTS
  - TTS
  - Voice
  - Audio
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 5
description: "Qwen-TTS-Flash for production: native-only API, the 40+ voice catalog (including Cantonese and Sichuanese), streaming synthesis, and the SSML quirks that the docs gloss over."
disableNunjucks: true
translationKey: "aliyun-bailian-5"
---

The reason every Chinese-language product I've worked on ends up calling Qwen-TTS-Flash isn't price — there are cheaper TTS APIs. It's that Qwen-TTS is the only one that handles **mainland Chinese dialects** (Cantonese, Sichuanese, Wu) and English in the same SDK, with voices that don't sound like a 2019 customs announcement. After about six months of using it for a marketing-video voice-over pipeline, this is what I wish someone had told me on day one.

![Aliyun Bailian (5): Qwen-TTS for Multilingual Voice — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/illustration_1.png)

## Voice catalog

Per the model card, Qwen-TTS-Flash exposes 40+ voices. The ones I use most:

![Qwen-TTS voice catalogue](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig1_tts_voice_map.png)

For Mandarin product narration my default is `Cherry` (warm, 30-something female) for marketing content and `Ethan` (steady, 40-something male) for tutorial / explainer content. For Cantonese ad spots `Sunny` is the safe choice. The voice names are stable but new voices are added regularly — fetch the canonical list from the model card before you pin one in production code.

## Native API only

Qwen-TTS does not go through the OpenAI compat layer. You call it via the DashScope native SDK:

![Qwen-TTS native call structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig2_tts_request_flow.png)

A minimal request:

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

Two things to underline:

- The output is a *URL* by default, not bytes. Same as Wanxiang, **download it within 24 hours** (I do it immediately and re-upload to my own OSS bucket).
- `format` defaults to `mp3`. WAV is also available; for downstream concatenation work I prefer WAV because there's no MP3 header overhead per chunk.

## Streaming TTS — when latency matters

For voice-bot use cases (real-time conversational UIs) you want streaming. The deltas are audio bytes you can write straight to a player or a file:

![Streaming TTS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/fig3_tts_streaming.png)

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

Time-to-first-byte on streaming is typically under 400ms in Shanghai region, which is fast enough that a user perceives it as immediate. Non-streaming for a 30-second utterance is closer to 4-6 seconds wall clock — fine for batch narration, sluggish for chat.

## Multi-language and dialect specifics

Qwen-TTS does language *detection* from the text, but if you mix scripts you should set `language` explicitly. My production rules:

- Pure Mandarin text → `language="zh"` (default).
- Pure English text → `language="en"`. Voices like `Eric` shine here.
- Cantonese text in Traditional Chinese → `language="zh-yue"`, voice `Sunny` or `Lily`.
- Mixed CJK + English (the common case for tech narration) → leave language unset, the model handles code-switch surprisingly well.

> **Tip:** For dialect work, **always** A/B against native speakers before launch. Qwen-TTS Cantonese is good but not perfect on tones — a one-syllable tone error in Cantonese can change the meaning entirely.

## SSML — what works, what doesn't

The docs list SSML support but are quiet about which tags actually behave. From experience:

- `<break time="500ms"/>` — works. Use for pauses between sentences in marketing copy.
- `<emphasis level="strong">` — works.
- `<prosody rate="slow">` — works. `slow`, `medium`, `fast`, or numeric percentage.
- `<prosody pitch="...">` — works for relative changes (e.g. `+10%`).
- `<say-as interpret-as="digits">` — works for phone numbers, codes, dates.
- `<phoneme>` — partial. Tone-marked pinyin is more reliable than IPA on Chinese.
- `<voice>` — does NOT work. You cannot mid-utterance switch voice. Use separate calls and concatenate.

## Concatenating clips for narration

Marketing scripts are long. The pattern for a 60-second voiceover:

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

Why split? Two reasons: (1) per-call latency is much lower for short utterances, so synthesizing in parallel is faster; (2) you can patch a single bad sentence by re-rolling just that one without redoing the whole take. We use this for ad spots — the `<break>` tags handle the inter-sentence pauses, and the parallel synth means a 60-second clip is ready in ~4 seconds.

## Cost

Per-second-of-output-audio billing. Streaming and non-streaming bill identically. A 60-second ad spot is in the few-RMB range — much cheaper than the cost of a voice actor's hourly rate, and fast enough for the marketing team to iterate dozens of variations in an afternoon.

## Voice cloning vs preset voices vs SSML control: pick the right knob

Bailian gives you three layers of voice control, and which one you reach for matters more than people think. The order of cost, complexity, and control is opposite to the order of "fancy":

**Preset voices.** The 40+ voices in the catalog. Use these unless you have a specific reason not to. They cover Mandarin male/female across all common ages and registers, plus Cantonese, Sichuanese, Wu, English (US/UK), and a handful of "character" voices for narration work. Per-second cost is the standard rate; latency is the lowest of the three; quality is studio-grade. About 90% of my production traffic is preset-voice.

**SSML control over preset voices.** Same voice, but you nudge prosody, breaks, emphasis, pronunciation. Costs the same as preset voices. Use this when:
- Long-form narration needs deliberate pacing (technical tutorials, audiobooks).
- A specific phrase needs emphasis the model misses by default.
- You have proper nouns or technical terms the model mispronounces (especially in CJK + English code-switch contexts: "DSL"  often comes out "D-S-L" instead of "deh-es-el").

A real example from my marketing pipeline:

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

The `<say-as>` makes "DSL" come out as letter-by-letter (correct for an acronym). The `<break>` adds a deliberate pause where listeners need to absorb. The `<prosody rate="95%">` slows the URL recommendation slightly so listeners can write it down. None of this is possible with plain text input.

**Voice cloning (timbre transfer).** Bailian exposes a separate clone API where you upload 10-30 seconds of reference audio and get back a voice id you can use in subsequent calls. This is real but hedge it with two warnings:

- **Quality varies.** Clones of clear, single-speaker, studio-recorded reference audio are excellent. Clones of phone-quality, noisy, or multi-speaker reference audio are uncanny in bad ways. Always hand-screen clones before production use.
- **Legal/ethical surface.** Cloning a real human's voice without consent is increasingly illegal in many jurisdictions and is explicitly banned in Bailian's ToS. Clone only voices you own or have written permission to clone. The platform may revoke voice ids that look like impersonations of public figures.

For my marketing pipeline I cloned the founder's voice (with their consent) for one product line where they wanted to do dozens of variations of a 30-second pitch. We did it once, validated quality across 20 sample utterances, then used the resulting voice id for everything in that line. Cost is identical to preset voices once the clone is created.

The decision tree:
- Need a voice that doesn't exist in the preset catalog? → cloning, with permission.
- Preset voice works but pronunciation/pacing is off? → SSML.
- Preset voice works as-is? → preset, don't overthink it.

## Latency budget: streaming chunks, mouth-shape sync window

![Aliyun Bailian (5): Qwen-TTS for Multilingual Voice — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/05-qwen-tts-voice/illustration_2.png)

For interactive voice products (voice bots, real-time avatars), latency matters more than quality at the margin. The latency breakdown for streaming TTS:

1. **Request setup** (TLS handshake, first request packet): 30-80ms within China, 200-400ms cross-region.
2. **TTFT** (first audio chunk arrives): typically 200-400ms in Shanghai region for a fresh request.
3. **Per-chunk cadence**: chunks arrive roughly every 100-200ms, each containing 200-400ms of audio.
4. **Tail**: the final chunk arrives after the spoken duration of the text, plus ~100ms.

For a "user types text → audio plays" UX, end-to-end perceived latency lands around 250-500ms for the first sound. That's the threshold for "feels instant" — anything under 500ms reads as immediate, anything over 1s reads as laggy.

For **mouth-shape synchronization** with an avatar, you need both the audio chunks and a per-phoneme timestamp track. Qwen-TTS doesn't return per-phoneme timestamps in the basic API. Two workarounds:

- **Run a forced-aligner on the output.** Tools like `whisper-timestamped` or `mfa` (Montreal Forced Aligner) take the synthesized audio and the source text, and emit per-word or per-phoneme timestamps. Adds 200-500ms to total pipeline latency. I use this for non-real-time avatar work.
- **Estimate timestamps from text length.** For real-time work where you can't afford the alignment pass, approximate: each Chinese character is ~150ms of audio at default rate, each English word is ~250ms. Errors compound but the first 5-10 seconds are usually accurate enough that mouth shapes look right at a casual glance.

A streaming pattern with a frame-aware avatar:

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

Good enough for marketing avatars. Not good enough for high-fidelity dubbing — for that, render to file and run forced alignment.

## Audio post-processing: loudness normalization, silence trimming

Raw TTS output is rarely production-ready. Two post-processing steps are mandatory in any pipeline that ships to listeners:

**Loudness normalization.** Different voices, different texts, even different runs of the same voice produce audio with different perceived loudness. Without normalization, your listener gets blasted by a loud sentence then can't hear a quiet one. The standard is EBU R128, target -16 LUFS for podcasts/long-form or -14 LUFS for streaming platforms.

```bash
ffmpeg -y -i in.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 out.wav
```

This single ffmpeg call brings every TTS output to the same perceived loudness. Run it on every output, no exceptions. The CPU cost is microseconds.

**Silence trimming.** Qwen-TTS often produces 100-300ms of leading silence and a similar trailing tail. For concatenated narration this adds up — a 20-sentence script ends up with 4-6 seconds of dead air. Trim with:

```bash
ffmpeg -y -i in.wav -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,areverse" out.wav
```

Trim before concatenation, then add controlled inter-sentence breaks back in with explicit silence-padding. The resulting narration sounds intentional rather than awkwardly paced.

**Loudness + trim + concat as a pipeline:**

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

The 250ms silence pad between sentences is what makes back-to-back TTS sound like a single take rather than a stitched-together chimera.

## Cost per minute of generated audio

The official per-second pricing is the right unit, but here's the per-minute version with realistic overhead:

| Mode | Per-minute base | With ffmpeg post (negligible) | With cloning surcharge |
|---|---|---|---|
| Preset voice, non-stream | ~0.6 RMB / min | +0 | n/a |
| Preset voice, stream | ~0.6 RMB / min | +0 | n/a |
| Cloned voice | ~0.6 RMB / min | +0 | one-time clone cost ~5 RMB |
| With long-form SSML markup | ~0.6 RMB / min | +0 | n/a |

A 60-minute audiobook at 0.6 RMB/min is 36 RMB. A 30-second ad spot is 0.3 RMB. Compared to a voice actor's session fee (mid-range Chinese voice talent: 2000-5000 RMB per project), the API is roughly 100-1000x cheaper *per output*, with the trade-off that each take is a fraction of a second of CPU time and zero booking friction. The break-even point for "should I hire a voice actor or use Qwen-TTS" sits around "this is a hero brand spot that needs a specific human voice the audience already knows", and almost nowhere else.

The hidden cost is post-processing engineering time: 1 day to set up the loudness/trim/concat pipeline once, then it amortizes across every clip you ship for years. Skip the post-processing and your audio sounds amateur regardless of how good the model is.


## Closing the series

That's the five. To recap:

- **Article 1** — Bailian / DashScope orientation.
- **Article 2** — the Qwen LLM family and the fiddly bits (function calling, JSON mode, `enable_thinking`).
- **Article 3** — Qwen-Omni for multimodal understanding.
- **Article 4** — Wanxiang for video generation.
- **Article 5** — Qwen-TTS for voice (this article).

The companion **Aliyun PAI** series covers DSW / DLC / EAS — the *self-managed* GPU layer where you train and serve your own models. Most teams I work with end up using both: Bailian when they want someone else's pre-trained model, PAI when they need control of the weights. Pick by what you actually need to control, not by what looks more impressive on a resume.
