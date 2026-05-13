---
title: "Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding"
date: 2026-02-27 09:00:00
tags:
  - Aliyun Bailian
  - Qwen-Omni
  - Multimodal
  - Video Understanding
  - Streaming
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 3
description: "Qwen-Omni for production multimodal: the four input types, the streaming requirement that the docs do not warn you about, and a working video-understanding pipeline with sane pixel budgets."
disableNunjucks: true
translationKey: "aliyun-bailian-3"
---

Of all the Bailian models, Qwen-Omni has saved me the most from product-roadmap issues. "Can you tell me what's happening in this 2-minute promo video?" used to take 3 weeks, involving frame extraction, captioning each frame, and stitching them together. With Qwen-Omni, it's just one HTTP request. However, the documentation lacks details on some pitfalls, such as the requirement for streaming, which has cost more than one team a half-day. Let's avoid that for you.

![Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/03-qwen-omni-multimodal/illustration_1.png)

## What Qwen-Omni accepts

Per the Qwen API reference for multimodal models, a single user message can mix text, image, audio, and video parts in one `content` array. That is the headline capability — not "supports images", but "supports anything in any combination":

![Qwen-Omni inputs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/03-qwen-omni-multimodal/fig1_omni_inputs.png)

The structure for each type, drawn from the API reference:

| Type | Field | Notes |
|---|---|---|
| `text` | `text: "..."` | Plain string. |
| `image_url` | `image_url: {url}` | URL or base64 data URI. `min_pixels` / `max_pixels` control resize. |
| `input_audio` | `data, format` | `mp3`, `wav`, etc. URL or local base64. |
| `video_url` | `video_url: {url}` | URL or data URI. Or use `video` array of frame images. |

A real call:

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

## Streaming is not optional — this is the trap

The docs note streaming as a feature, but they bury the fact that **for Qwen-Omni it is required**. Set `stream=False` and you get a 400 with a message about the model requiring streaming.

![Streaming requirement](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/03-qwen-omni-multimodal/fig2_omni_streaming.png)

The reason makes sense when you think about it: the model processes large video files and generates a long response. The wire protocol assumes incremental delivery. Sending everything at once would block your client for tens of seconds without any progress signal.

If your downstream code expects a complete string, buffer the deltas yourself:

```python
def call_omni_buffered(messages):
    stream = client.chat.completions.create(
        model="qwen3-omni-flash", messages=messages, stream=True,
    )
    return "".join(c.choices[0].delta.content or "" for c in stream)
```

It is one extra function. You will write it once.

## Pixel and frame budgets — what costs you

The cost knob the docs are quiet about: `min_pixels` and `max_pixels` for images, and the equivalent fps / resize parameters for video. By default Qwen-Omni will process video at native resolution and a default fps. For a 2-minute 1080p clip that is a *lot* of token-equivalents, and the bill scales with it.

What I do in production:

- **Images for understanding tasks** — `max_pixels: 1280*720`. Almost no quality loss for "what's in this image" tasks, big cost savings. Set `min_pixels: 640*480` so the model never scales up tiny crops.
- **Video for description tasks** — pre-resize to 720p before upload, downsample fps to 4 for static-ish content (people talking) or 8 for action content (sports, fast cuts). Above 8 fps you are usually paying for redundant frames.
- **Long video** — chunk it. The model has a context limit. For anything over ~3 minutes, split into 90-second chunks, summarize each, then summarize-the-summaries with `qwen-plus`. Same pattern as long-document RAG.

## Sending a local video file

You have two options. The docs cover both.

![Sending a local video to Qwen-Omni](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/03-qwen-omni-multimodal/fig3_video_pipeline.png)

**Path 1 (preferred): upload to OSS, send signed URL.**

```python
import oss2, os, time
auth = oss2.Auth(os.environ["OSS_AK"], os.environ["OSS_SK"])
bucket = oss2.Bucket(auth, "https://oss-cn-shanghai.aliyuncs.com", "your-bucket")
bucket.put_object_from_file("clips/promo.mp4", "/tmp/promo.mp4")
signed = bucket.sign_url("GET", "clips/promo.mp4", 600)  # 10 min expiry
```

Then pass `signed` as the `url` field. This is the right answer for anything bigger than 30 seconds because base64 inflates the payload by 33%.

**Path 2: base64 inline.** Useful for short clips, avoids the round trip to OSS.

```python
import base64
with open("/tmp/short.mp4", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
content = [{
    "type": "video_url",
    "video_url": {"url": f"data:video/mp4;base64,{b64}"},
}]
```

> **Real-world tip:** When debugging a Qwen-Omni 400, check the URL is publicly fetchable from the open internet. The model service does not have access to your VPC. Signed URLs work; private OSS without signing does not.

## Audio understanding

Almost the same shape, but `type: "input_audio"`:

```python
content = [
    {"type": "text", "text": "Transcribe this and identify the speaker's mood."},
    {"type": "input_audio", "input_audio": {"data": signed_audio_url, "format": "mp3"}},
]
```

For pure transcription, Bailian also offers a dedicated **Paraformer** ASR model, which is cheaper. Use Paraformer for transcription only and Qwen-Omni when you need understanding (sentiment, summarization, "did the call mention pricing").

## A real product use case

The recurring pattern I ship for AI marketing: a creative team uploads a 60-second product video; we want a structured caption (`scene description`, `key product features visible`, `target audience guess`, `recommended music style`). One Qwen-Omni call, JSON mode (yes it works with multimodal), under 4 seconds wall clock for 720p input.

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

## Audio input encoding: sample rate, format, max duration

The Qwen-Omni audio path looks simple — pass an `input_audio` part with `data` and `format` — but the encoding requirements are strict in ways that don't surface as friendly errors. Get them wrong and you get a 400 with "unsupported audio format" and no further guidance.

What actually works in production:

- **Sample rate**: 16 kHz is the sweet spot. The model accepts 8 / 16 / 22.05 / 24 / 44.1 / 48 kHz, but anything other than 16 kHz gets resampled server-side, which costs you latency. For voice-over-IP recordings (8 kHz native) I upsample to 16 kHz client-side using `librosa.resample` — same quality, more predictable latency.
- **Format**: `wav` (PCM 16-bit), `mp3`, `m4a`, `flac`, `ogg`. I use WAV for short utterances (< 30s) where size doesn't matter and MP3 for anything longer. AAC inside MP4 *sometimes* works depending on the codec profile — easier to transcode to MP3 first than to debug.
- **Channels**: mono is preferred. Stereo gets downmixed; if your two channels are different speakers (interview-style content), downmix loses the speaker separation. Convert to mono yourself with whichever channel you care about, or split into two requests.
- **Bit depth**: 16-bit PCM is the safe default. 24-bit and 32-bit float work but sometimes get rejected — I've seen the same WAV file accepted as 16-bit and rejected as 32-bit in the same week.
- **Max duration**: 3 minutes per request for `qwen3-omni-flash`. Longer than that, you must chunk. The error is `Audio duration exceeds limit` with the exact cap in the message.
- **Max file size**: 10 MB. With 16 kHz / 16-bit mono PCM that's about 5 minutes of WAV — but MP3 at 64 kbps fits 20+ minutes in the same envelope. For long content, MP3 is mandatory.

A pre-flight transcoder I run before every audio call:

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

Run this on every upload before the API call. The 200ms transcode cost is negligible against API latency, and you eliminate the entire class of "format unsupported" 400s.

## Video frame sampling: 1 fps for talking heads vs 8 fps for action

![Aliyun Bailian (3): Qwen-Omni for Video, Audio, and Image Understanding — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-bailian/03-qwen-omni-multimodal/illustration_2.png)

The default video processing path inside Qwen-Omni samples frames at a model-internal rate, then encodes each frame as a vision token block. Token cost scales linearly with frame count, so frame rate is the single biggest cost knob you have on video.

The official documentation says the parameter is `fps` in the video URL part. What it does *not* say is the right value for different content types. After running maybe 2000 video calls in production, my rules:

- **Talking-head content (interviews, product demos, lectures)**: 1 fps is enough. The interesting signal is in the audio + sparse visual cues (slide changes, gestures). Higher fps wastes tokens on frames that look identical. I tested 1 / 4 / 8 / 16 fps on the same 2-minute interview clip; the description quality plateau is at 1 fps. You can sometimes drop to 0.5 fps and still get a reasonable summary.
- **Action content (sports highlights, ads with cuts, movement)**: 8 fps. Below this you miss motion-defined events. Above it the model is processing redundant intermediate frames. 8 fps is the sweet spot I've validated against three different ad-tech clients.
- **Mixed content (UGC, user uploads of unknown type)**: 4 fps as a default. It's a worse fit for both extremes but doesn't catastrophically fail on either.
- **Slideshow / static-with-overlay content**: 0.5 fps. The model wants to read each slide, not process the dissolve transitions.

To set the fps explicitly:

```python
content = [{
    "type": "video_url",
    "video_url": {
        "url": signed_url,
        "fps": 4,   # default ≈ 2 — set this when you know your content
    },
}]
```

For maximum control, *pre-extract* frames yourself and send them as a `video` array of image parts:

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

Pre-extracting gives you exact control over which frames the model sees, which is invaluable when the auto-sampler has skipped a critical moment (it has, for me, more than once).

## Token cost math for video

The cost surprise on Omni isn't the per-token rate — it's the per-second-of-video token count. Working it out from first principles:

A video frame at 1280×720, after Qwen-Omni's internal vision encoder, costs roughly **256 vision tokens per frame**. (The exact number depends on the model variant; `qwen3.5-omni-plus` is denser, `qwen3-omni-flash` is lighter, but 256 is a useful planning number.)

So a 30-second clip at 4 fps is `30 * 4 * 256 = 30,720 vision tokens`. At the `qwen3-omni-flash` vision rate of 12 RMB / million tokens, that's `30720 / 1e6 * 12 = 0.37 RMB` for the input alone, before the output tokens for the description.

Scale that up:

| Content | Duration | fps | Vision tokens | Input cost (RMB) |
|---|---|---|---|---|
| Talking head | 60s | 1 | 15,360 | 0.18 |
| Talking head | 60s | 4 | 61,440 | 0.74 |
| Ad spot | 15s | 8 | 30,720 | 0.37 |
| Long demo | 180s | 4 | 184,320 | 2.21 |
| Lecture, slides only | 600s | 0.5 | 76,800 | 0.92 |

Two takeaways:

- The fps choice can change cost by 4-8x. Pick it deliberately.
- A 10-minute lecture at 0.5 fps costs less than a 1-minute talking head at 4 fps. Long content is *cheap* if you sample sparsely.

For comparison, a pure audio call on the same content runs at maybe 50 tokens per second of audio (it's transcribed internally to text-token-equivalents). 10 minutes of audio is `600 * 50 = 30,000 audio tokens` — cheaper than 1 minute of dense video. **For talking-head content where the audio carries the meaning, send only audio**, not video — about 5x cheaper for equivalent understanding quality.

## Latency profile: TTFT for video understanding, batched mode

End-to-end latency on Qwen-Omni breaks down into three phases that are useful to measure separately:

1. **Upload + URL signing**: 200ms-2s depending on file size and OSS endpoint. Reuse a signed URL across multiple calls if you're iterating on prompts.
2. **Vision encoding (server-side)**: 1-4s for a 30-second 720p clip. This is the part the user experiences as "nothing is happening" before tokens start streaming. It scales roughly linearly with frame count.
3. **Streaming generation**: TTFT typically 1.5-3s after vision encoding finishes, then 30-60 tokens/sec.

A 30-second video clip → 60-word description, end-to-end, lands at **6-8 seconds wall clock** in production. Faster than humans, slower than text-only LLM. If your UX needs sub-second, you cannot use Omni — you have to pre-process videos at ingest time and cache the descriptions.

For batch workloads (process 1000 videos overnight), there's no native `batch` endpoint for Omni the way OpenAI has. The pattern I use:

- Async queue (Celery / SQS / your favorite) with 10 workers.
- Each worker holds an `OpenAI` client and calls Omni serially with a 10s soft timeout per task (long tail to 30s).
- Per-key concurrency cap of 10 (matches default workspace quota).
- Retry on 429 with exponential backoff up to 3 minutes.
- Throughput in production: about 600 videos/hour per workspace, ceiling set by the 60 RPM `qwen3-omni-flash` quota.

For higher throughput, request a quota bump (chapter 1 walks through the process) and fan out across multiple workspaces. Bailian doesn't have a native parallel-batch primitive, so you build it yourself.


## What's Next

Article 4 jumps to the *production* side — **Wanxiang text-to-video**. That is async-only, native-protocol-only, and the failure modes are completely different (queue depth, output URL expiry). It is also the API I have spent the most time tuning prompts for.
