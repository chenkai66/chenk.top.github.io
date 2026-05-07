#!/usr/bin/env python3
"""Generate editorial-style cover images for chenk.top series via DashScope Wanxiang.

For each series in PROMPTS, generate a 1024x576 image, save to
  /root/chenk-hugo/static/covers/{slug}.jpg
and upload to OSS bucket blog-pic-ck under posts/covers/{slug}.jpg.

Idempotent: skips a series if the local file already exists and is >50KB.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request

import dashscope
from dashscope import ImageSynthesis

# ----- config -----
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"  # 16:9 banner
N_IMAGES = 1

OUT_DIR = Path("/root/chenk-hugo/static/covers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/root/.aliyun/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_PREFIX = "posts/covers"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"

dashscope.api_key = DASHSCOPE_API_KEY

# ----- prompts -----
NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered"
)

PROMPTS: dict[str, str] = {
    "recommendation-systems": (
        "Editorial illustration of a recommendation system as a constellation of interconnected nodes "
        "and user-item bipartite graph, glowing edges flowing between abstract avatars and product cards, "
        "warm coral and cream palette with soft cobalt accents, geometric Bauhaus composition, "
        "Stripe Press magazine aesthetic, soft gradients, depth, no text"
    ),
    "linear-algebra": (
        "Editorial illustration of linear algebra as overlapping translucent vector spaces and basis vectors, "
        "elegant orthogonal axes, eigenvector arrows piercing transparent planes, "
        "deep navy and indigo palette with cool ivory highlights, geometric minimalist composition, "
        "Quanta Magazine feature art aesthetic, subtle grid texture, soft gradients, no text"
    ),
    "ml-math-derivations": (
        "Editorial illustration of mathematical derivation as flowing chalkboard symbols dissolving into "
        "abstract gradient surfaces, layered translucent equations becoming geometric forms, "
        "muted aubergine, dusty rose, and parchment palette, scientific journal cover aesthetic, "
        "soft contemplative lighting, no readable text"
    ),
    "nlp": (
        "Editorial illustration of natural language processing as ribbons of words morphing into abstract "
        "token streams and attention matrices, flowing typographic forms dissolving into geometric vectors, "
        "warm violet and amber palette with soft cream highlights, Stripe Press book cover aesthetic, "
        "elegant composition, no readable text"
    ),
    "reinforcement-learning": (
        "Editorial illustration of a reinforcement learning agent and environment, abstract geometric shapes "
        "with flowing arrows representing policy and reward feedback loops, agent as glowing node moving "
        "through a gridded landscape, muted teal and forest green palette with bronze accents, "
        "Bauhaus-inspired clean composition, magazine cover aesthetic, soft gradients, no text"
    ),
    "transfer-learning": (
        "Editorial illustration of transfer learning as a bridge of light arcing between two distinct "
        "knowledge spheres, source and target domains as overlapping translucent membranes exchanging "
        "feature streams, warm sienna and slate-blue palette, scientific journal cover aesthetic, "
        "soft depth, no text"
    ),
    "ode": (
        "Editorial illustration of ordinary differential equations as flowing phase-space trajectories, "
        "elegant spiraling integral curves and vector fields, smooth flowing lines on a soft grid, "
        "deep indigo and twilight purple palette with pale gold accents, Quanta Magazine aesthetic, "
        "contemplative minimalist composition, no text"
    ),
    "pde-ml": (
        "Editorial illustration of partial differential equations meeting neural networks, abstract heat "
        "diffusion and wave fronts overlaid with translucent network layers, contour fields and gradient "
        "surfaces, muted emerald and steel-blue palette with parchment highlights, "
        "scientific magazine cover aesthetic, soft gradients, no text"
    ),
    "time-series": (
        "Editorial illustration of time series forecasting as elegant flowing waveforms layered through "
        "a translucent timeline, abstract seasonal cycles overlapping with trend ribbons, "
        "muted ochre and dusty teal palette with cream highlights, Stripe Press aesthetic, "
        "minimalist composition, soft depth, no text"
    ),
    "cloud-computing": (
        "Editorial illustration of cloud computing as abstract floating slate-grey cumulus forms with "
        "embedded geometric data centers, glowing fiber pathways connecting modular blocks, "
        "cool slate and pale azure palette with soft white highlights, isometric architectural minimalism, "
        "Quanta Magazine aesthetic, no text"
    ),
    "computer-fundamentals": (
        "Editorial illustration of computer fundamentals as cross-section of abstract layered hardware and "
        "operating system kernels, stacked translucent strata representing CPU, memory, and processes, "
        "muted graphite, copper, and cream palette, scientific cutaway diagram aesthetic, "
        "Stripe Press book cover style, soft lighting, no text"
    ),
    "leetcode": (
        "Editorial illustration of algorithm patterns as elegant abstract trees, graphs, and recursive "
        "spirals, geometric data structures rendered as translucent crystalline forms, "
        "muted plum and warm gold palette with ivory highlights, magazine cover aesthetic, "
        "minimalist composition, soft depth, no text"
    ),
    "aliyun-bailian": (
        "Editorial illustration of a large language model platform as a luminous central core radiating "
        "modular agent capabilities, abstract concentric rings of prompts and embeddings, "
        "warm orange-red and cream palette with deep ink accents, Stripe Press cover aesthetic, "
        "elegant geometric composition, soft glow, no text"
    ),
    "aliyun-pai": (
        "Editorial illustration of an end-to-end machine learning platform as connected modular pipelines, "
        "abstract data flowing through training, deployment, and serving stages, geometric cards linked by "
        "luminous threads, warm terracotta and ivory palette with deep navy accents, "
        "Stripe Press book cover aesthetic, no text"
    ),
    "terraform-agents": (
        "Editorial illustration of infrastructure-as-code agents as architectural blueprint with glowing "
        "modular building blocks being assembled by abstract intelligent forms, isometric infrastructure "
        "schematic, bronze and copper on deep slate-blue palette with parchment highlights, "
        "Quanta Magazine technical illustration aesthetic, no text"
    ),
    "linux": (
        "Editorial illustration of a Linux operating system as elegant terminal cursor radiating concentric "
        "process and kernel rings, abstract pipes and shells weaving through translucent layers, "
        "deep charcoal and emerald palette with warm amber accents, Stripe Press book cover aesthetic, "
        "minimalist composition, soft glow, no text"
    ),
    "openclaw-quickstart": (
        "Editorial scientific illustration of a sleek robotic claw mechanism gently grasping a glowing "
        "geometric token, abstract Bauhaus-inspired composition, muted teal and slate palette with copper "
        "accent, magazine-cover aesthetic, soft gradients, no text, 16:9"
    ),
    "llm-engineering": (
        "Editorial scientific illustration cover art for large language model engineering. "
        "Style: abstract geometric depiction of a layered transformer stack as overlapping translucent panels "
        "with glowing token paths flowing through attention rings and sparse expert routing nodes. "
        "Magazine-cover aesthetic inspired by Stripe Press and Quanta Magazine. "
        "Muted violet and electric magenta palette with cool steel accents and warm cream highlights. "
        "Clean composition, soft gradients, depth, no text, 16:9"
    ),
    "claude-code-learn": (
        "Editorial illustration of a developer terminal window opening into an abstract knowledge graph, "
        "geometric code-block shapes flowing into a thoughtful neural pattern, muted indigo and warm gold "
        "palette, clean Stripe Press aesthetic, magazine-cover style, no text, 16:9"
    ),
    "standalone": (
        "Editorial illustration of a standalone essay as a single elegant glowing book opening into "
        "abstract layered ideas and floating geometric thoughtforms, warm parchment and ink palette with "
        "muted gold accents, Stripe Press book cover aesthetic, contemplative minimalist composition, "
        "soft depth, no text"
    ),
}

# ----- logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("covers")


def already_done(slug: str) -> bool:
    p = OUT_DIR / f"{slug}.jpg"
    return p.exists() and p.stat().st_size > 50_000


def generate_one(slug: str, prompt: str) -> Path | None:
    """Submit async task, poll, download to local file. Returns local path or None."""
    out_path = OUT_DIR / f"{slug}.jpg"
    log.info("[%s] submitting prompt (%d chars)", slug, len(prompt))
    rsp = ImageSynthesis.async_call(
        model=MODEL,
        prompt=prompt,
        negative_prompt=NEG,
        n=N_IMAGES,
        size=SIZE,
    )
    if rsp.status_code != 200 or not getattr(rsp, "output", None):
        log.error("[%s] async_call failed: %s", slug, rsp)
        return None
    task_id = rsp.output.task_id
    log.info("[%s] task_id=%s", slug, task_id)

    # Poll
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(4)
        st = ImageSynthesis.fetch(task=task_id)
        status = st.output.task_status
        log.info("[%s] status=%s", slug, status)
        if status == "SUCCEEDED":
            results = st.output.results or []
            if not results:
                log.error("[%s] succeeded but no results", slug)
                return None
            url = results[0].url
            log.info("[%s] downloading %s", slug, url)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as r:
                data = r.read()
            out_path.write_bytes(data)
            log.info("[%s] saved %s (%d bytes)", slug, out_path, len(data))
            return out_path
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            log.error("[%s] task ended with %s: %s", slug, status, st)
            return None
    log.error("[%s] timed out", slug)
    return None


def upload_oss(local: Path, slug: str) -> str | None:
    key = f"{OSS_PREFIX}/{slug}.jpg"
    oss_url = f"oss://{OSS_BUCKET}/{key}"
    public = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{key}"
    cmd = [
        OSSUTIL, "cp", "-f", \"--cache-control\", \"public, max-age=300, must-revalidate\", \"--cache-control\", \"public, max-age=300, must-revalidate\",
        "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
        "--region", "cn-beijing",
        str(local), oss_url,
    ]
    log.info("[%s] uploading to %s", slug, public)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s] ossutil failed rc=%d stdout=%s stderr=%s",
                  slug, r.returncode, r.stdout[-400:], r.stderr[-400:])
        return None
    log.info("[%s] uploaded OK", slug)
    return public


def main(argv: list[str]) -> int:
    only = set(argv[1:]) if len(argv) > 1 else None
    summary: dict[str, dict] = {}
    for slug, prompt in PROMPTS.items():
        if only and slug not in only:
            continue
        if already_done(slug) and not only:
            log.info("[%s] skip — already exists", slug)
            summary[slug] = {
                "status": "skipped",
                "url": f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{OSS_PREFIX}/{slug}.jpg",
            }
            continue
        local = generate_one(slug, prompt)
        if not local:
            summary[slug] = {"status": "gen_failed"}
            continue
        if local.stat().st_size < 50_000:
            log.error("[%s] generated file too small (%d bytes)", slug, local.stat().st_size)
            summary[slug] = {"status": "too_small"}
            continue
        url = upload_oss(local, slug)
        summary[slug] = {"status": "ok" if url else "upload_failed", "url": url}

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    failed = [k for k, v in summary.items() if v["status"] not in ("ok", "skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
