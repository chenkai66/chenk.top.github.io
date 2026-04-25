---
title: "Aliyun Bailian (5): Qwen-TTS for Multilingual Voice"
date: 2026-03-01 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - TTS
  - Qwen-TTS
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 5
description: "qwen3-tts-flash, the 40+ voices, the dialect coverage that nothing else on the market matches, and the DashScope-native-only quirk that catches every team that tried OpenAI TTS first."
disableNunjucks: true
---

A merchant in Guangzhou wanted product narration in Cantonese for their video ads. Azure TTS speaks Cantonese in a customs-announcement monotone. ElevenLabs has no Cantonese voices in production. OpenAI TTS supports six voices, all in English-with-an-accent for everything else. Qwen-TTS-Flash has a native Cantonese voice that sounds like an actual Hong Kong native person reading copy. That moment, more than anything else in this series, was when I stopped reaching for non-Bailian options first. This article is how to use it without the most common mistake (assuming it's OpenAI-compatible) and how to pick voices.

## One model that matters: `qwen3-tts-flash`

There are older `sambert-*` and `cosyvoice-*` model families on Bailian. They still work and may be cheaper, but for new code use `qwen3-tts-flash`. It is faster, has more voices, has better dialect coverage, and is the one Alibaba is actively improving.

| Property | Value |
|---|---|
| model_id | `qwen3-tts-flash` |
| Endpoint | `https://dashscope.aliyuncs.com/api/v1/services/audio/tts` |
| Output formats | mp3, wav, pcm |
| Sample rates | 8k / 16k / 22050 / 24k / 44100 |
| Languages | Mandarin, English, Cantonese, Sichuanese, Shanghainese, Northeast, Japanese, Korean (varies by voice) |
| Voices | 40+ |
| Pricing | per-second-of-audio |

## The DashScope-native-only rule

This is the part that trips everyone up.

> **Qwen-TTS does NOT work via the OpenAI-compatible endpoint. It is DashScope-native only.**

You cannot point the `openai` SDK's `audio.speech.create` at the compat URL and have it work — there's no compat shim for TTS. Use the `dashscope` SDK or raw HTTP.

I have watched three engineers in three different companies make the same mistake of assuming "if Qwen LLM works through the OpenAI client, TTS must too." The error you get is a 404 or "model not found", which is unhelpful. Save yourself the search: install `dashscope`.

## The simplest call

```python
import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Cherry")
audio_bytes = synth.call("Welcome to the product demo. Today we'll show you three new features.")

with open("out.mp3", "wb") as f:
    f.write(audio_bytes)
```

That's it. `audio_bytes` is mp3 by default. For a 10-second sentence, expect ~1-2s of latency. The synthesizer object is reusable; reuse it for batches.

## Voice selection: don't browse the catalog, use this guide

The console has a voice gallery with samples. It is genuinely useful, but here is the shortlist I reach for, picked by use case from a year of production:

| Voice | Language(s) | Best for |
|---|---|---|
| `Cherry` | Mandarin, English | Friendly female, default for most consumer narration |
| `Ethan` | Mandarin, English | Warm male, good for explainer videos |
| `Serena` | English | Professional female, B2B content |
| `Chelsie` | Mandarin | Younger female, lifestyle/Xiaohongshu vibe |
| `Sunny` | Sichuanese | Sichuan dialect, strong character — use sparingly, very recognizable |
| `Lily` | Cantonese | Hong Kong-style female, the killer feature for southern markets |
| `Marcus` | Mandarin, English | Authoritative male, formal narration |
| `Jennifer` | Mandarin (Northeast) | Northeast accent, character voice |

Voice IDs are case-sensitive. `cherry` will fail; `Cherry` works. The full list is at the Bailian docs page for `qwen3-tts-flash`; treat the docs as the source of truth, but the eight above cover 90% of what I ship.

A voice handles multiple languages if it was trained on them. `Cherry` and `Ethan` are the bilingual workhorses. If you mix Chinese and English in one input string, those voices will switch naturally. Single-language voices will mangle the other language; pick correctly.

## Streaming for low first-byte latency

For a one-paragraph narration the non-streaming call is fine. For longer copy or voice-over-while-generating use cases, stream:

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
        print("error:", message)
        self.f.close()

callback = FileCallback("out_stream.mp3")
synth = SpeechSynthesizer(model="qwen3-tts-flash", voice="Cherry", callback=callback)

# You can call .streaming_call multiple times to feed text incrementally,
# then .streaming_complete() to flush.
synth.streaming_call("First sentence of the narration. ")
synth.streaming_call("Second sentence, possibly arriving from an LLM stream upstream.")
synth.streaming_complete()
```

The streaming pattern shines when you're piping LLM output straight into TTS — generate text with `qwen-plus` streaming, push each delta to TTS, and you get audio playback that starts before the LLM finishes thinking. Total perceived latency drops from "wait for LLM + wait for TTS" to "wait for LLM start."

## Format and sample-rate choices

```python
synth = SpeechSynthesizer(
    model="qwen3-tts-flash",
    voice="Cherry",
    format="wav",        # mp3 | wav | pcm
    sample_rate=24000,   # 8000 | 16000 | 22050 | 24000 | 44100
)
```

What I use:

- **mp3 / 24kHz** for anything that ends up on the web. Smallest file, indistinguishable from wav for narration.
- **wav / 24kHz** when I need to mix it with other audio in `ffmpeg` and don't want a re-encode round trip.
- **pcm / 16kHz** when piping to a real-time playback library that wants raw frames.

Don't go above 24kHz. The model's effective bandwidth doesn't benefit from higher rates and you're just paying for bigger files.

## Real use case: multilingual product narration

Here is the function that powers a "narrate this product description in five voices" feature. One product description in, five mp3 files out.

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
    """text_by_locale: {locale_id: text}, e.g. {'zh-mandarin': '...', 'en-pro': '...'}
    Returns {locale_id: file_path}."""
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

Two production notes:

1. **Translate before you synthesize.** Don't pass the same Chinese string and ask the Cantonese voice to "speak it in Cantonese" — it will read the written Chinese with a Cantonese accent, which is not the same thing. Translate to actual Cantonese script first (an `qwen-plus` call with the right prompt is fine) and then synthesize.
2. **Cache by hash of (text + voice + format).** Narrations are immutable; if the same product description goes through the pipeline twice, you should pay once. Hash the inputs, use it as the OSS key, check before synthesizing.

> **Tip:** SSML-style markup ("<break time='500ms'/>", emphasis tags, etc.) is partially supported but unevenly. For most narration work, just use punctuation. Commas give natural short pauses, periods give longer ones, and hyphens (—) give the dramatic pause you want before a feature beat. Save SSML for the rare case where punctuation isn't enough.

## Cost and latency

For a typical 8-second product narration:

- Latency: ~1-2 seconds non-streaming, ~300-500ms time-to-first-byte streaming
- Cost: small fractions of a yuan per call; full pricing is per-second on the Bailian pricing page
- Throughput: a single SDK client can comfortably do 10-20 syntheses/sec

The economics are forgiving. I have never had a TTS bill be the line item I worried about — the LLM and Wanxiang costs always dominated by 10x.

## What can go wrong

| Error | Cause | Fix |
|---|---|---|
| Model not found / 404 | Calling via OpenAI-compatible endpoint | Switch to `dashscope` SDK |
| `InvalidParameter.Voice` | Wrong voice ID, often case mismatch | Use exact ID from docs (`Cherry`, not `cherry`) |
| Robotic / bad pronunciation on a name | Out-of-vocabulary token | Spell phonetically (`艾迪琪` for "Aidge") in pinyin-friendly form, or wrap in SSML phoneme tag |
| Wrong language sounds | Voice not trained for that language | Pick a voice whose language matches the input |
| Truncated audio | Input exceeded model's max chars (~2000 chars per call) | Split on sentence boundaries, synthesize, concatenate |

The "spell phonetically" trick is worth knowing. Brand names in particular trip every TTS engine — for our product 艾迪琪 (Aidge), the model says it pretty much correctly when written in characters and butchers it when written as "Aidge." Test once for each brand name and stick with the form that sounds right.

## Closing the series

Five articles, one platform, the parts I actually use. The pattern across all of them is the same: pick the right model, know which endpoint it lives on (compat vs native vs async), respect the streaming requirements, and treat the API like any other vendor — with retry, caching, and a budget alert. Bailian has its quirks and so does every cloud AI platform; what makes this one worth the time is that for Chinese-language and dialect work it has no real competitor, and the rest of the catalog is a credible second opinion to OpenAI/Anthropic at meaningfully lower cost.

If you build something interesting on top of this series, I'd love to hear about it. The full demo code from these five articles will live in a small repo on my GitHub when I get a free weekend to clean it up.
