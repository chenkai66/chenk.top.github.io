---
title: "Aliyun Bailian (4): Wanxiang Video Generation End-to-End"
date: 2026-04-24 09:00:00
tags:
  - Aliyun Bailian
  - DashScope
  - Wanxiang
  - Video Generation
categories: Aliyun Bailian
lang: en
mathjax: false
series: aliyun-bailian
series_title: "Aliyun Bailian Practical Guide"
series_order: 4
description: "The async task pattern for wan2.5-t2v-plus and wan2.5-i2v-plus, a robust polling loop with backoff, and the 24-hour result expiry that has burned every team I've talked to at least once."
disableNunjucks: true
---

A marketing team needed thirty 5-second product clips per day, each from a single hero image and a one-line prompt, delivered to OSS within an hour of the brief. We tried Sora (no Chinese billing), Kling (rate-limited and expensive), Pika (image-to-video quality lagged), and finally Wanxiang (`wan2.5-i2v-plus`). Wanxiang won not because the videos are obviously better, but because the API is async-first, rate limits are generous, and one engineer can build the whole pipeline in an afternoon. That afternoon, with hindsight, is what this article is.

## Two models, plus the older v2.1 family

| model_id | Mode | Max duration | Notes |
|---|---|---|---|
| `wan2.5-t2v-plus` | Text-to-video | 5s | Default for prompt-only generation |
| `wan2.5-i2v-plus` | Image-to-video | 5s | Anchors motion to a starting frame |
| `wanx2.1-t2v-turbo` | T2V, faster, lower quality | 5s | Use for previews / drafts |
| `wanx2.1-i2v-turbo` | I2V, faster, lower quality | 5s | Same |

The 2.5 plus models are what I ship. The 2.1 turbo models exist if you genuinely need 30%-faster, 50%-cheaper, lower-quality drafts; in my experience users would rather wait two more minutes for the better output. Both 2.5 models cap at **5 seconds**. If you need 10 seconds you make two clips and stitch them — same model can extend a clip by passing the last frame as `i2v` input.

## The async task pattern

Every Wanxiang call follows the same three-step dance:

1. **Create the task.** POST with header `X-DashScope-Async: enable`. You get a `task_id` immediately.
2. **Poll.** GET `/api/v1/tasks/{task_id}` until `task_status` is `SUCCEEDED` or `FAILED`.
3. **Download.** The success response includes `output.video_url`. **Download within 24 hours** — after that the URL 404s and your video is gone forever.

The 24-hour expiry is the single biggest operational footgun. I have seen multiple teams, mine included, lose work because they polled, logged the URL, then failed to download because of an unrelated bug, then noticed the next day. Treat the URL the way you would treat a one-time download link: download immediately, store on your own OSS, never assume it will be there tomorrow.

## Endpoints

```
POST https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis
GET  https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}
```

Auth is `Authorization: Bearer $DASHSCOPE_API_KEY`. The async header on the POST is **mandatory** — without it you get an immediate 400.

## Creating a text-to-video task

I'll show raw HTTP first because the SDK hides what's going on, then the SDK version.

```python
import os
import requests

def create_t2v_task(prompt: str, size: str = "1280*720") -> str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "wan2.5-t2v-plus",
        "input": {
            "prompt": prompt,
        },
        "parameters": {
            "size": size,           # "1280*720", "720*1280", "960*960", etc.
            "duration": 5,           # max 5 for 2.5-plus
            "prompt_extend": True,   # let the model rewrite your prompt for better results
            "seed": None,            # set for reproducibility
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    return body["output"]["task_id"]
```

A few parameters that matter:

- **`size`**: Use `width*height` (note the `*`, not `x`). 1280x720 is the sweet spot for cost vs quality. 960x960 for square Instagram/Xiaohongshu. The plus models support up to 1920x1080.
- **`prompt_extend`**: Defaults to `True`. The server expands your one-line prompt into a richer one. For most use cases leave it on; for tight artistic control turn it off.
- **`seed`**: Set it if you want reproducible outputs. Same prompt + same seed = same video, give or take server-side nondeterminism.
- **`duration`**: 1-5. Cost scales linearly. I default to 5 because the price difference is small.

## Image-to-video task

Same shape, different model and an extra `img_url` field:

```python
def create_i2v_task(image_url: str, prompt: str) -> str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "wan2.5-i2v-plus",
        "input": {
            "img_url": image_url,
            "prompt": prompt,        # describes the motion / camera, not the subject
        },
        "parameters": {
            "duration": 5,
            "prompt_extend": True,
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["output"]["task_id"]
```

Two non-obvious things about i2v:

1. **The prompt should describe motion, not subject.** "A young woman holding the bottle, smiling, soft natural light" is wrong — the woman is in your image already. Say "the woman lifts the bottle slightly and turns toward the camera, hair gently moving." If you describe the subject, the model may try to redraw it and lose fidelity.
2. **Aspect ratio of the output matches the image.** You don't pass `size` for i2v. If your image is 1080x1920, your video is 1080x1920.

## Polling, the right way

Naive polling looks like `while True: sleep(5); check()`. That works until you have 50 concurrent jobs and you're hitting `/tasks/{id}` 600 times a minute, which gets you rate-limited.

Here is the polling helper I actually use, with linear-then-exponential backoff:

```python
import time
import requests

def poll_task(task_id: str, timeout: int = 600) -> dict:
    """Poll until SUCCEEDED/FAILED or timeout. Returns the task body."""
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"}

    start = time.time()
    # Backoff schedule: tight at first (some jobs finish in 30s), then loosen
    delays = [5, 5, 5, 8, 10, 15, 20, 30, 30, 30]
    i = 0

    while time.time() - start < timeout:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        body = r.json()
        status = body["output"]["task_status"]

        if status == "SUCCEEDED":
            return body
        if status == "FAILED":
            raise RuntimeError(f"Task {task_id} failed: {body['output'].get('message', 'no message')}")
        if status == "UNKNOWN":
            raise RuntimeError(f"Task {task_id} unknown — likely expired")

        # PENDING or RUNNING — wait
        time.sleep(delays[min(i, len(delays) - 1)])
        i += 1

    raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")
```

A typical 5-second 720p clip on `wan2.5-t2v-plus` finishes in 60-180 seconds. I set my outer timeout to 600s and have never legitimately hit it.

## Putting it together: generate, poll, download, store

```python
import os
import time
import requests

API = "https://dashscope.aliyuncs.com/api/v1"
HEADERS_AUTH = {"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"}

def generate_video(prompt: str, out_path: str) -> dict:
    # 1. Create
    create_url = f"{API}/services/aigc/video-generation/video-synthesis"
    create_headers = {**HEADERS_AUTH, "X-DashScope-Async": "enable", "Content-Type": "application/json"}
    create_payload = {
        "model": "wan2.5-t2v-plus",
        "input": {"prompt": prompt},
        "parameters": {"size": "1280*720", "duration": 5, "prompt_extend": True},
    }
    r = requests.post(create_url, headers=create_headers, json=create_payload, timeout=30)
    r.raise_for_status()
    task_id = r.json()["output"]["task_id"]
    print(f"Created task {task_id}")

    # 2. Poll
    body = poll_task(task_id)
    video_url = body["output"]["video_url"]
    print(f"Got video URL (expires in 24h): {video_url[:80]}...")

    # 3. Download IMMEDIATELY
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    return {
        "task_id": task_id,
        "video_url": video_url,
        "local_path": out_path,
        "usage": body.get("usage"),
    }

if __name__ == "__main__":
    info = generate_video(
        prompt="A glass of iced lemon tea on a cafe table, condensation slowly forming, warm afternoon light",
        out_path="./out.mp4",
    )
    print(info)
```

This is essentially the production code, minus the OSS upload and DLQ handling. The shape is "create, poll, download" with the download happening in the same function, **not** queued for later. Every shortcut here has cost me a video.

## Real numbers

For a batch of 30 5-second 720p clips on `wan2.5-t2v-plus`, run sequentially:

- p50 generation time: ~95s
- p95 generation time: ~180s
- Failure rate: ~1% (mostly transient, retry once)
- Cost per clip: ~¥1.5

If you parallelize, your bottleneck becomes per-account concurrency limit, which is something like 5-10 concurrent video tasks per workspace by default. Ask support to raise it for production.

## Prompt patterns that work

After a few hundred clips:

- **Camera motion is the cheapest quality boost.** "slow dolly in", "static wide shot", "handheld over-the-shoulder" — adding one of these tightens the result a lot more than another adjective.
- **Specify lighting.** "golden hour", "softbox studio light", "cool overhead office light" anchor the look. Without it the model picks something average.
- **Avoid text.** Wanxiang, like every video model, can't render text reliably. If you need text, generate the clip and overlay text in post.
- **Keep the prompt under ~80 Chinese characters or 50 English words.** Longer prompts confuse the prompt extender. The tightest prompts give the most controllable outputs.

> **Tip:** Set `prompt_extend=False` and write your own enriched prompt the moment you care about consistency across a batch. The auto-extender is a black box that changes between model updates; controlling it yourself means a model update doesn't silently shift your brand aesthetic.

## Errors and what to do

| Error | Likely cause | Action |
|---|---|---|
| `task_status: FAILED, code: InvalidParameter.Prompt` | Prompt tripped content moderation | Rephrase, often a brand name or person |
| `task_status: FAILED, code: InternalError` | Server side | Recreate the task |
| 429 on create | Concurrent task limit | Queue, retry with backoff |
| 404 on download | URL expired (>24h) | The video is gone. Re-generate. |
| Polling returns same RUNNING for >5min | Stuck job, very rare | Cancel via task ID and recreate |

The content moderation failures are the annoying ones. Wanxiang is conservative with celebrity names, anything political, and certain trademark terms. If you're generating product content, name the product generically in the prompt and rely on the i2v model with the actual product image to keep brand consistency.

## A note on the SDK

`pip install dashscope` gives you `dashscope.VideoSynthesis.async_call(...)` which wraps the create-and-poll into one call:

```python
from dashscope import VideoSynthesis

rsp = VideoSynthesis.call(
    model="wan2.5-t2v-plus",
    prompt="...",
    size="1280*720",
    duration=5,
)
print(rsp.output.video_url)
```

It's fine for prototypes. For production I prefer raw HTTP because I want explicit control over the polling cadence, the timeout, the retry behavior, and the moment the URL is captured. The SDK's polling is reasonable but not configurable.

## Closing the loop

Wanxiang generates the videos; Qwen-Omni from article 3 understands them; Qwen-TTS from article 5 narrates them. The whole loop runs in one process under 60 seconds for short clips, and that pipeline is what powers the bulk of generative-marketing automation I've shipped on this stack. Article 5 is next: the TTS half of the story, which is — surprise — DashScope-native only.
