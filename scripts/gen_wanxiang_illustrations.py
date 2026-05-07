#!/usr/bin/env python3
"""Generate per-article Wanxiang editorial illustrations and insert into chenk.top articles.

Two illustrations per article: hero (before first ## heading) + mid (after a chosen narrative heading).
Same image used for EN and ZH. Idempotent.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import dashscope
from dashscope import ImageSynthesis

# ----- config -----
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
N_IMAGES = 1

CONTENT_ROOT = Path("/root/chenk-hugo/content")
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
LOG_FILE = SCRIPTS_DIR / "wanxiang_illustrations.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_illustrations_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_illustrations")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

WORKERS = 3
SUBMIT_MIN_INTERVAL_S = 1.5
MAX_SUBMIT_RETRIES = 6
POLL_TIMEOUT_S = 180

NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border"
)

dashscope.api_key = DASHSCOPE_API_KEY

# ----- per-article specs -----
# Each entry: stem -> (hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading)
# Mid headings are matched verbatim against `## <heading>`.

ARTICLES: dict[str, dict[str, list]] = {
    "llm-engineering": {
        "01-architectures": [
            "abstract editorial illustration of a layered transformer block dissolving into a mosaic of sparse expert pathways, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of mixture-of-experts gating: tokens flowing through translucent gates that light up selectively, slate and copper palette, editorial style, no text, 16:9",
            "Mixture of experts: more parameters, same FLOPs",
            "Mixture of Experts：参数更多，FLOPs 不变",
        ],
        "02-tokenization": [
            "abstract illustration of language being decomposed into geometric shards then reassembled into glowing strings, soft pastel palette of dusty rose and ivory, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of CJK characters fracturing into many tiny tokens while latin letters stay whole, contrasting amber and jade palette, editorial style, no text, 16:9",
            "The CJK token-bloat problem",
            "CJK token 膨胀",
        ],
        "03-pretraining": [
            "abstract editorial illustration of vast data rivers being filtered through translucent sieves into a luminous reservoir of training tokens, deep teal and parchment palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of synthetic data being woven from a generative loom, threads of plausibility and noise, muted plum and sand palette, editorial style, no text, 16:9",
            "Synthetic data: the dirty secret",
            "合成数据：脏秘密",
        ],
        "04-post-training": [
            "abstract editorial illustration of a base model being shaped by overlapping fields of preference, reward, and verification, warm coral and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of two preference signals on a balance, one pulling a model toward chosen responses, the other away from rejected, dusty olive and burgundy palette, editorial style, no text, 16:9",
            "DPO: preference optimization without a reward model",
            "DPO：不要奖励模型的偏好优化",
        ],
        "05-inference": [
            "abstract editorial illustration of a serving cluster pipelining tokens through paged memory blocks, muted teal and graphite palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of speculative decoding: a small fast draft model racing ahead of a large model that verifies in lockstep, copper and ink palette, editorial style, no text, 16:9",
            "Speculative decoding",
            "Speculative decoding",
        ],
        "06-long-context": [
            "abstract editorial illustration of position embeddings as concentric rotating rings of light around a sequence, muted violet and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a single needle of meaning suspended within a vast haystack of context tokens, soft amber and shadow palette, editorial style, no text, 16:9",
            "Needle in a haystack: the only honest benchmark",
            "Needle in a haystack：唯一诚实的基准测试",
        ],
        "07-function-calling": [
            "abstract editorial illustration of a model reaching out through translucent conduits to invoke external tools, then weaving results back, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of an agent loop as concentric feedback rings between thought, action, and observation, slate and copper palette, editorial style, no text, 16:9",
            "The agent loop",
            "Agent loop",
        ],
        "08-rag": [
            "abstract editorial illustration of fragmented documents being woven into a luminous answer thread, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a reranker as a fine sieve elevating the most relevant chunks above a sea of candidates, muted teal and ochre palette, editorial style, no text, 16:9",
            "Reranking is the unsung hero",
            "Reranking 是被忽略的英雄",
        ],
        "09-prompting": [
            "abstract editorial illustration of a chain of thought unfurling as a luminous filament through a dim mental space, muted plum and ivory palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of prompt cache as a warm reservoir of reusable context fragments lighting up on each request, soft amber and slate palette, editorial style, no text, 16:9",
            "Prompt caching changes the cost math",
            "Prompt caching 改写成本数学",
        ],
        "10-evaluation": [
            "abstract editorial illustration of a benchmark leaderboard dissolving as private holdout sets emerge from shadow, muted forest green and parchment palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of an LLM judge weighing two candidate responses on opposing scales of light, dusty rose and slate palette, editorial style, no text, 16:9",
            "LLM-as-judge: the dominant pattern and its failure modes",
            "LLM-as-judge：主导模式和它的失败模式",
        ],
        "11-safety": [
            "abstract editorial illustration of two opposing forces negotiating across a balance scale of glowing tokens, warm copper and cool indigo tension, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of red-team probes dissolving against a layered defense membrane around a model, deep crimson and slate palette, editorial style, no text, 16:9",
            "Red-teaming methodology",
            "红队方法学",
        ],
        "12-production": [
            "abstract editorial illustration of a production serving stack as a luminous cathedral of routing layers, caches, and observability streams, muted teal and graphite palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of multi-model routing as a junction where light streams diverge to specialist endpoints based on cost and capability, copper and slate palette, editorial style, no text, 16:9",
            "Multi-model routing and FrugalGPT",
            "多模型路由和 FrugalGPT",
        ],
    },
    "aliyun-bailian": {
        "01-platform-overview": [
            "abstract editorial illustration of a platform manifold revealing a catalog of model endpoints behind a translucent gateway, muted teal and warm sand palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of conversation tokens streaming through a translucent API conduit lit from within, warm amber and slate palette, editorial style, no text, 16:9",
            "A complete first request",
            "跑通的第一个请求",
        ],
        "02-qwen-llm-api": [
            "abstract editorial illustration of Qwen variants as concentric rings of capability, dense and MoE, around a luminous core, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a function-calling round trip as light traveling between a model, a tool, and back, muted copper and slate palette, editorial style, no text, 16:9",
            "Function calling deep dive: multi-round, parallel, the tool_choice=\"auto\" trap",
            "Function calling 深入：多轮、并行、`tool_choice=\"auto\"` 陷阱",
        ],
        "03-qwen-omni-multimodal": [
            "abstract editorial illustration of multimodal inputs — image, audio, video — converging as braided streams of light into a single understanding, muted violet and amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a video being sampled into discrete frames, each a luminous tile in a temporal grid, slate and copper palette, editorial style, no text, 16:9",
            "Video frame sampling: 1 fps for talking heads vs 8 fps for action",
            "视频帧采样：人头说话 1 fps，动作内容 8 fps",
        ],
        "04-wanxiang-video-generation": [
            "abstract editorial illustration of a still image unfolding into a temporal sequence of motion, muted plum and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of multi-clip stitching as overlapping luminous filmstrips chained by last-frame relays, dusty teal and copper palette, editorial style, no text, 16:9",
            "Multi-clip stitching: last-frame relay and continuity hacks",
            "多片拼接：末帧接力 + 连续性技巧",
        ],
        "05-qwen-tts-voice": [
            "abstract editorial illustration of a written sentence transforming into a flowing waveform of voice, muted rose and ivory palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of streaming TTS chunks arriving as overlapping ripples synchronized to a mouth-shape rhythm, soft amber and slate palette, editorial style, no text, 16:9",
            "Latency budget: streaming chunks, mouth-shape sync window",
            "延迟预算：流式分块、口型同步窗口",
        ],
    },
    "aliyun-pai": {
        "01-platform-overview": [
            "abstract editorial illustration of a machine learning platform as a layered city of sub-products — training, serving, notebook — connected by luminous infrastructure, muted teal and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a Designer pipeline as nodes connected by glowing edges, abstracted from drag-and-drop into pure flow, dusty cobalt and copper palette, editorial style, no text, 16:9",
            "What a Designer workflow really looks like under the hood",
            "Designer 工作流在引擎下面长什么样",
        ],
        "02-pai-dsw-notebook": [
            "abstract editorial illustration of a notebook environment as a workbench with code, GPU, and data streams converging, muted ochre and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of OSS-FUSE mount as a translucent bridge between local filesystem and a distant data lake, soft teal and sand palette, editorial style, no text, 16:9",
            "OSS-FUSE mount, latency profile, and when to copy instead",
            "OSS-FUSE 挂载：延迟剖面，以及什么时候改成拷贝",
        ],
        "03-pai-dlc-distributed-training": [
            "abstract editorial illustration of distributed training as many GPUs synchronizing gradients along a luminous all-reduce ring, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of multi-node NCCL communication as light traveling across an RDMA fabric in ring and tree topologies, slate and warm brass palette, editorial style, no text, 16:9",
            "Multi-node NCCL: RDMA vs TCP, ring vs tree",
            "多机 NCCL：RDMA vs TCP，ring vs tree",
        ],
        "04-pai-eas-model-serving": [
            "abstract editorial illustration of a model serving cluster as a layered hive humming with concurrent inference requests, muted teal and graphite palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of cold-start mitigation as warm reserve nodes lighting up just in time before a traffic surge crests, copper and slate palette, editorial style, no text, 16:9",
            "Cold start mitigation, in order of effectiveness",
            "冷启动缓解，按效果排序",
        ],
        "05-pai-designer-vs-quickstart": [
            "abstract editorial illustration of two divergent paths — a drag-and-drop pipeline composer and a zero-code model gallery — meeting at a single deployment outcome, muted plum and teal palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a decision tree branching across cost, latency, and team-skill axes toward a chosen platform path, dusty rose and slate palette, editorial style, no text, 16:9",
            "A concrete decision tree",
            "一棵具体的决策树",
        ],
    },
    "terraform-agents": {},
}

ARTICLES["terraform-agents"] = {
    "01-why-terraform-for-agents": [
        "abstract editorial illustration of an agent system blueprint emerging from console clicks, infrastructure pieces snapping into a coherent diagram, copper and slate palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of state files as a luminous bill of materials cataloging every piece of an agent stack, muted teal and parchment palette, editorial style, no text, 16:9",
        "State as the agent stack's bill of materials",
        "State 是 Agent 栈的物料清单",
    ],
    "02-provider-and-state-setup": [
        "abstract editorial illustration of provider authentication keys flowing into a remote state vault, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of workspaces as parallel realities of the same infrastructure, dev and prod glowing in different hues, dusty teal and copper palette, editorial style, no text, 16:9",
        "Step 6: workspaces for env isolation",
        "第 6 步：用 workspace 隔离环境",
    ],
    "03-vpc-and-security-baseline": [
        "abstract editorial illustration of a VPC as nested concentric rings of network isolation around an agent runtime core, muted slate and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of layered security groups as translucent membranes filtering traffic between zones, soft graphite and warm amber palette, editorial style, no text, 16:9",
        "Security groups, layered",
        "安全组按 tier 分层",
    ],
    "04-compute-for-agent-runtime": [
        "abstract editorial illustration of three compute patterns — ECS, ACK, function compute — as parallel architectural columns supporting an agent runtime, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of right-sizing as a precise balance between idle headroom and burst capacity on an instance, copper and slate palette, editorial style, no text, 16:9",
        "Right-sizing the instance",
        "实例规格选型",
    ],
    "05-storage-for-agent-memory": [
        "abstract editorial illustration of three memory layers — relational, vector, object — as stacked translucent strata of an agent's mind, muted plum and parchment palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a vector store as a constellation of embeddings clustering by semantic similarity, dusty indigo and warm copper palette, editorial style, no text, 16:9",
        "Layer 2: vector store",
        "第 2 层：向量库",
    ],
    "06-llm-gateway-and-secrets": [
        "abstract editorial illustration of an LLM gateway as a translucent conduit routing prompts across providers behind a wall of vaulted secrets, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of secret rotation as a clockwork mechanism quietly swapping keys without dropping a single in-flight request, slate and copper palette, editorial style, no text, 16:9",
        "Step 5: secret rotation flow",
        "第 5 步：密钥轮转流",
    ],
    "07-observability-and-cost-control": [
        "abstract editorial illustration of three telemetry pipelines — logs, traces, metrics — converging into an observability dashboard, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a cost dashboard as layered cost streams stratified by service, lit from within, copper and graphite palette, editorial style, no text, 16:9",
        "Step 5: the cost dashboard",
        "第 5 步：成本看板",
    ],
    "08-end-to-end-walkthrough": [
        "abstract editorial illustration of a full agent stack assembling itself from terraform modules into a coherent running system, muted slate and warm brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of promotion across dev, staging, prod as three concentric arenas linked by gated pipelines, dusty teal and copper palette, editorial style, no text, 16:9",
        "Promotion strategy: dev → staging → prod without surprises",
        "推进策略：dev → staging → prod 不出意外",
    ],
}

# ----- logging -----
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("wanxiang-illu")


# ----- OSS helpers -----
def oss_key(lang: str, series: str, stem: str, n: int) -> str:
    return f"posts/{lang}/{series}/{stem}/illustration_{n}.jpg"


def oss_url(lang: str, series: str, stem: str, n: int) -> str:
    return f"{OSS_PUBLIC_BASE}/{oss_key(lang, series, stem, n)}"


def oss_exists(lang: str, series: str, stem: str, n: int) -> bool:
    url = oss_url(lang, series, stem, n)
    req = Request(url, method="HEAD", headers={"User-Agent": "illu/1.0"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status == 200
    except HTTPError as e:
        return e.code != 404 and False
    except URLError:
        return False


# ----- generation -----
submit_lock = Lock()
_last_submit_ts = [0.0]


def _throttled_submit(prompt: str, tag: str):
    backoff = 4.0
    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        with submit_lock:
            now = time.time()
            wait = SUBMIT_MIN_INTERVAL_S - (now - _last_submit_ts[0])
            if wait > 0:
                time.sleep(wait)
            _last_submit_ts[0] = time.time()
        try:
            rsp = ImageSynthesis.async_call(
                model=MODEL, prompt=prompt, negative_prompt=NEG,
                n=N_IMAGES, size=SIZE,
            )
        except Exception as e:
            log.error("[%s] async_call exception attempt %d: %s", tag, attempt, e)
            time.sleep(backoff); backoff = min(backoff * 1.7, 30); continue
        if rsp.status_code == 200 and getattr(rsp, "output", None):
            return rsp
        code = getattr(rsp, "code", "") or ""
        msg = getattr(rsp, "message", "") or ""
        if rsp.status_code == 429 or "Throttl" in str(code) or "rate" in str(msg).lower():
            log.warning("[%s] 429 attempt %d, sleeping %.1f", tag, attempt, backoff)
            time.sleep(backoff); backoff = min(backoff * 1.7, 30); continue
        log.error("[%s] async_call failed: status=%s code=%s msg=%s", tag, rsp.status_code, code, msg)
        return None
    return None


def generate_image(prompt: str, tag: str) -> bytes | None:
    log.info("[%s] submit (%d chars)", tag, len(prompt))
    rsp = _throttled_submit(prompt, tag)
    if rsp is None:
        return None
    task_id = rsp.output.task_id
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        try:
            st = ImageSynthesis.fetch(task=task_id)
        except Exception as e:
            log.warning("[%s] fetch exception: %s", tag, e); continue
        status = st.output.task_status
        if status == "SUCCEEDED":
            results = st.output.results or []
            if not results:
                log.error("[%s] SUCCEEDED no results", tag); return None
            url = results[0].url
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception as e:
                log.error("[%s] download failed: %s", tag, e); return None
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            log.error("[%s] %s: %s", tag, status, st); return None
    log.error("[%s] poll timeout", tag); return None


def upload_oss(local: Path, lang: str, series: str, stem: str, n: int) -> bool:
    target = f"oss://{OSS_BUCKET}/{oss_key(lang, series, stem, n)}"
    cmd = [OSSUTIL, "cp", "-f", "--meta", "Cache-Control:public, max-age=300, must-revalidate", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing", str(local), target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%s/%d] ossutil rc=%d %s", series, stem, n, r.returncode, r.stderr[-200:])
        return False
    return True


# ----- worker: generate one image (used for both langs since same image) -----
def process_image(series: str, stem: str, n: int, prompt: str) -> dict:
    """Generate one image and upload to BOTH en and zh OSS paths."""
    tag = f"{series}/{stem}/illu{n}"
    result = {"series": series, "stem": stem, "n": n, "ok": False, "skipped": False, "url_en": None, "url_zh": None}

    # Idempotent: if both EN and ZH exist, skip
    if oss_exists("en", series, stem, n) and oss_exists("zh", series, stem, n):
        log.info("[%s] both langs exist, skipping", tag)
        result.update(ok=True, skipped=True,
                      url_en=oss_url("en", series, stem, n),
                      url_zh=oss_url("zh", series, stem, n))
        return result

    # Generate (or reuse if at least one lang exists — re-download from OSS to copy)
    data = generate_image(prompt, tag)
    if data is None:
        return result

    out_dir = TMP_DIR / series / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"illustration_{n}.jpg"
    local.write_bytes(data)
    log.info("[%s] saved %d bytes -> %s", tag, len(data), local)

    ok_en = upload_oss(local, "en", series, stem, n)
    ok_zh = upload_oss(local, "zh", series, stem, n)
    if ok_en and ok_zh:
        result.update(ok=True, url_en=oss_url("en", series, stem, n),
                      url_zh=oss_url("zh", series, stem, n))
    return result


# ----- markdown insertion -----
def insert_into_markdown(md_path: Path, lang: str, series: str, stem: str,
                         hero_url: str, mid_url: str, mid_heading: str, title: str) -> bool:
    if not md_path.exists():
        log.warning("missing md: %s", md_path)
        return False
    text = md_path.read_text(encoding="utf-8")

    # Idempotency: check for our markers
    hero_marker = f"illustration_1.jpg"
    mid_marker = f"illustration_2.jpg"
    has_hero = hero_marker in text
    has_mid = mid_marker in text

    if has_hero and has_mid:
        log.info("[%s/%s/%s] already has both, skipping insert", lang, series, stem)
        return True

    lines = text.split("\n")
    new_lines: list[str] = []

    hero_line = f"![{title} — visual]({hero_url})"
    mid_line = f"![{title} — visual]({mid_url})"

    inserted_hero = has_hero
    inserted_mid = has_mid

    for i, line in enumerate(lines):
        # Insert hero before first ## heading
        if not inserted_hero and line.startswith("## "):
            new_lines.append(hero_line)
            new_lines.append("")
            inserted_hero = True
        # Insert mid AFTER the chosen mid heading line
        new_lines.append(line)
        if not inserted_mid and line.strip() == f"## {mid_heading}":
            new_lines.append("")
            new_lines.append(mid_line)
            inserted_mid = True

    if not inserted_hero:
        log.warning("[%s/%s/%s] no ## heading found for hero", lang, series, stem)
    if not inserted_mid:
        log.warning("[%s/%s/%s] mid heading '%s' not found", lang, series, stem, mid_heading)

    new_text = "\n".join(new_lines)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        log.info("[%s/%s/%s] wrote %d bytes (hero=%s mid=%s)", lang, series, stem,
                 len(new_text), inserted_hero and not has_hero, inserted_mid and not has_mid)
    return inserted_hero and inserted_mid


def get_title(md_path: Path) -> str:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return md_path.stem
    m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else md_path.stem


# ----- main -----
def main():
    # Build job list
    image_jobs = []  # (series, stem, n, prompt)
    article_meta = []  # (series, stem, hero_prompt, mid_prompt, en_mid, zh_mid)
    for series, articles in ARTICLES.items():
        for stem, spec in articles.items():
            hero_prompt, mid_prompt, en_mid, zh_mid = spec
            image_jobs.append((series, stem, 1, hero_prompt))
            image_jobs.append((series, stem, 2, mid_prompt))
            article_meta.append((series, stem, hero_prompt, mid_prompt, en_mid, zh_mid))

    log.info("Total image jobs: %d (across %d articles)", len(image_jobs), len(article_meta))

    # Generate concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="wx") as ex:
        futs = {ex.submit(process_image, s, st, n, p): (s, st, n) for s, st, n, p in image_jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                results[(r["series"], r["stem"], r["n"])] = r
            except Exception as e:
                s, st, n = futs[f]
                log.error("[%s/%s/%d] worker exception: %s", s, st, n, e)
                results[(s, st, n)] = {"ok": False, "series": s, "stem": st, "n": n}

    # Save manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(image_jobs),
        "succeeded": sum(1 for r in results.values() if r.get("ok")),
        "skipped": sum(1 for r in results.values() if r.get("skipped")),
        "results": [r for r in results.values()],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest written: %s", MANIFEST_FILE)

    # Insert into markdown
    insert_summary = {"en_articles": 0, "zh_articles": 0, "failed": []}
    for series, stem, hero_prompt, mid_prompt, en_mid, zh_mid in article_meta:
        r1 = results.get((series, stem, 1), {})
        r2 = results.get((series, stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(f"{series}/{stem}")
            log.warning("[%s/%s] missing image, skip insert", series, stem)
            continue
        for lang, mid_heading in [("en", en_mid), ("zh", zh_mid)]:
            md_path = CONTENT_ROOT / lang / series / f"{stem}.md"
            title = get_title(md_path)
            hero_url = oss_url(lang, series, stem, 1)
            mid_url = oss_url(lang, series, stem, 2)
            ok = insert_into_markdown(md_path, lang, series, stem, hero_url, mid_url, mid_heading, title)
            if ok:
                insert_summary[f"{lang}_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "total_succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
