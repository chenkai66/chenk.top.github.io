#!/usr/bin/env python3
"""Generate per-article Wanxiang cover images for chenk.top.

Walks /root/chenk-hugo/content/en/, parses front matter, and generates a unique
1024x576 cover image per article via DashScope wanx2.1-t2i-plus, uploading each
to oss://blog-pic-ck/posts/covers/articles/{series}/{stem}.jpg.

Idempotent: skips articles whose OSS cover already exists (HEAD check).
Concurrent: 5 workers. Per-article cost ~$0.30. Hard-stop at $200.
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

import yaml
import dashscope
from dashscope import ImageSynthesis

# ----- config -----
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
N_IMAGES = 1

CONTENT_DIR = Path("/root/chenk-hugo/content/en")
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
LOG_FILE = SCRIPTS_DIR / "article_covers.log"
MANIFEST_FILE = SCRIPTS_DIR / "article_covers_manifest.json"
TMP_DIR = Path("/tmp/article_covers")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/root/.aliyun/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_PREFIX = "posts/covers/articles"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

WORKERS = 3
COST_PER_IMAGE = 0.30  # USD, approximate for wanx2.1-t2i-plus
HARD_STOP_USD = 200.0
POLL_TIMEOUT_S = 180
SUBMIT_MIN_INTERVAL_S = 1.2  # global throttle between async_call submissions
MAX_SUBMIT_RETRIES = 6
SKIP_FRONT_MATTER_FILES = {"_index.md", "about.md", "archives.md", "projects.md", "series.md"}

dashscope.api_key = DASHSCOPE_API_KEY

# ----- series metadata: human-readable name + palette hint -----
SERIES_META: dict[str, dict] = {
    "recommendation-systems": {
        "name": "Recommendation Systems",
        "palette": "warm coral and cream with soft cobalt accents",
    },
    "linear-algebra": {
        "name": "Linear Algebra",
        "palette": "deep navy and indigo with cool ivory highlights",
    },
    "ml-math-derivations": {
        "name": "Machine Learning Mathematical Derivations",
        "palette": "muted aubergine, dusty rose, and parchment",
    },
    "nlp": {
        "name": "Natural Language Processing",
        "palette": "warm violet and amber with soft cream highlights",
    },
    "reinforcement-learning": {
        "name": "Reinforcement Learning",
        "palette": "muted teal and forest green with bronze accents",
    },
    "transfer-learning": {
        "name": "Transfer Learning",
        "palette": "warm sienna and slate-blue",
    },
    "ode": {
        "name": "Ordinary Differential Equations",
        "palette": "deep indigo and twilight purple with pale gold accents",
    },
    "pde-ml": {
        "name": "Partial Differential Equations and Machine Learning",
        "palette": "muted emerald, copper, and steel-blue with parchment highlights",
    },
    "time-series": {
        "name": "Time Series Forecasting",
        "palette": "muted ochre and dusty teal with cream highlights",
    },
    "cloud-computing": {
        "name": "Cloud Computing",
        "palette": "cool slate and pale azure with soft white highlights",
    },
    "computer-fundamentals": {
        "name": "Computer Fundamentals",
        "palette": "muted graphite, copper, and cream",
    },
    "leetcode": {
        "name": "Algorithms and Data Structures",
        "palette": "muted plum and warm gold with ivory highlights",
    },
    "aliyun-bailian": {
        "name": "Alibaba Cloud Bailian LLM Platform",
        "palette": "warm crimson, orange-red, and cream with deep ink accents",
    },
    "aliyun-pai": {
        "name": "Alibaba Cloud PAI Machine Learning Platform",
        "palette": "warm terracotta, amber, and ivory with deep navy accents",
    },
    "terraform-agents": {
        "name": "Terraform and Infrastructure-as-Code Agents",
        "palette": "bronze and copper on deep slate-blue with parchment blueprint highlights",
    },
    "linux": {
        "name": "Linux Operating System",
        "palette": "deep charcoal and emerald with warm amber accents",
    },
    "openclaw-quickstart": {
        "name": "OpenClaw Self-Hosted AI Agent Platform",
        "palette": "muted teal and slate with copper accents",
    },
    "llm-engineering": {
        "name": "LLM Engineering — End-to-End",
        "palette": "muted violet and electric magenta with cool steel accents and warm cream highlights",
    },
    "claude-code-learn": {
        "name": "Claude Code Hands-On Guide",
        "palette": "muted indigo and warm gold with cream highlights",
    },
    "standalone": {
        "name": "Essays and Notes",
        "palette": "warm parchment and ink with muted gold accents",
    },
    "abstract-algebra": {
        "name": "Abstract Algebra",
        "palette": "deep royal purple and warm gold with ivory parchment highlights",
    },
    "functional-analysis": {
        "name": "Functional Analysis",
        "palette": "cool cerulean and silver with warm pearl accents",
    },
    "differential-geometry": {
        "name": "Differential Geometry",
        "palette": "warm amber and forest green with deep mahogany accents",
    },
}

NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border"
)

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
log = logging.getLogger("article-covers")


# ----- front matter -----
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(md: Path) -> dict | None:
    try:
        text = md.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("read failed %s: %s", md, e)
        return None
    m = FM_RE.match(text)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception as e:
        log.warning("yaml parse failed %s: %s", md, e)
        return None


def collect_articles() -> list[dict]:
    """Walk content/en/ and return list of article job dicts."""
    jobs: list[dict] = []
    for md in sorted(CONTENT_DIR.rglob("*.md")):
        if md.name in SKIP_FRONT_MATTER_FILES:
            continue
        # series = parent directory name relative to content/en
        try:
            rel = md.relative_to(CONTENT_DIR)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            # top-level files (like about.md) — skip
            continue
        series_slug = parts[0]
        stem = md.stem
        fm = parse_front_matter(md) or {}
        title = fm.get("title") or stem
        description = fm.get("description") or ""
        tags = fm.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        jobs.append({
            "path": str(md),
            "series": series_slug,
            "stem": stem,
            "title": str(title),
            "description": str(description),
            "tags": [str(t) for t in tags],
        })
    return jobs


# ----- prompt -----
def build_prompt(job: dict) -> str:
    series = job["series"]
    meta = SERIES_META.get(series, {
        "name": series.replace("-", " ").title(),
        "palette": "muted neutral with soft accents",
    })
    tags = ", ".join(job["tags"][:6]) if job["tags"] else "general"
    return (
        f"Editorial scientific illustration cover art for an article titled "
        f"\"{job['title']}\". Topic: {meta['name']}. Tags: {tags}. "
        f"Style: abstract geometric, magazine-cover aesthetic inspired by Stripe Press "
        f"and Quanta Magazine. Muted color palette tied to series theme "
        f"({meta['palette']}). Clean composition, soft gradients, depth via subtle shadow, "
        f"no text, no figures of people, 16:9 horizontal composition."
    )


# ----- OSS helpers -----
def oss_key(job: dict) -> str:
    return f"{OSS_PREFIX}/{job['series']}/{job['stem']}.jpg"


def oss_url(job: dict) -> str:
    return f"{OSS_PUBLIC_BASE}/{oss_key(job)}"


def oss_exists(job: dict) -> bool:
    """HEAD check via HTTP — public bucket so a 200 means exists."""
    url = oss_url(job)
    req = Request(url, method="HEAD", headers={"User-Agent": "covergen/1.0"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status == 200
    except HTTPError as e:
        if e.code == 404:
            return False
        log.warning("[%s/%s] HEAD %s -> %d", job["series"], job["stem"], url, e.code)
        return False
    except URLError as e:
        log.warning("[%s/%s] HEAD %s -> %s", job["series"], job["stem"], url, e)
        return False


# ----- generation -----
submit_lock = Lock()
_last_submit_ts = [0.0]


def _throttled_submit(prompt: str, tag: str) -> object | None:
    """Submit async task with global throttle and 429 backoff."""
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
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue
        if rsp.status_code == 200 and getattr(rsp, "output", None):
            return rsp
        # Rate limit / transient
        code = getattr(rsp, "code", "") or ""
        msg = getattr(rsp, "message", "") or ""
        if rsp.status_code == 429 or "Throttl" in str(code) or "rate" in str(msg).lower():
            log.warning("[%s] 429 attempt %d/%d, sleeping %.1fs",
                        tag, attempt, MAX_SUBMIT_RETRIES, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30)
            continue
        log.error("[%s] async_call failed (no retry): status=%s code=%s msg=%s",
                  tag, rsp.status_code, code, msg)
        return None
    log.error("[%s] async_call exhausted retries", tag)
    return None


def generate_one(job: dict) -> Path | None:
    tag = f"{job['series']}/{job['stem']}"
    prompt = build_prompt(job)
    log.info("[%s] submit prompt (%d chars)", tag, len(prompt))
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
            log.warning("[%s] fetch exception: %s", tag, e)
            continue
        status = st.output.task_status
        if status == "SUCCEEDED":
            results = st.output.results or []
            if not results:
                log.error("[%s] SUCCEEDED but no results", tag)
                return None
            url = results[0].url
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception as e:
                log.error("[%s] download failed: %s", tag, e)
                return None
            out_dir = TMP_DIR / job["series"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{job['stem']}.jpg"
            out_path.write_bytes(data)
            log.info("[%s] saved %d bytes -> %s", tag, len(data), out_path)
            return out_path
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            log.error("[%s] task %s: %s", tag, status, st)
            return None
    log.error("[%s] poll timeout after %ds", tag, POLL_TIMEOUT_S)
    return None


def upload_oss(local: Path, job: dict) -> str | None:
    key = oss_key(job)
    target = f"oss://{OSS_BUCKET}/{key}"
    cmd = [
        OSSUTIL, "cp", "-f", "--cache-control", "public, max-age=300, must-revalidate",
        "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
        "--region", "cn-beijing",
        str(local), target,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%s] ossutil failed rc=%d stderr=%s",
                  job["series"], job["stem"], r.returncode, r.stderr[-300:])
        return None
    return oss_url(job)


# ----- worker -----
manifest_lock = Lock()
manifest: dict[str, dict] = {}
counter_lock = Lock()
counters = {"done": 0, "skipped": 0, "failed": 0, "generated": 0}


def process(job: dict) -> dict:
    tag = f"{job['series']}/{job['stem']}"
    url = oss_url(job)
    if oss_exists(job):
        log.info("[%s] skip — exists on OSS", tag)
        with counter_lock:
            counters["skipped"] += 1
            counters["done"] += 1
        result = {"status": "skipped", "url": url}
        with manifest_lock:
            manifest[job["path"]] = result
        return result

    local = generate_one(job)
    if not local:
        with counter_lock:
            counters["failed"] += 1
            counters["done"] += 1
        result = {"status": "gen_failed", "url": None}
        with manifest_lock:
            manifest[job["path"]] = result
        return result

    if local.stat().st_size < 50_000:
        log.error("[%s] too small (%d bytes)", tag, local.stat().st_size)
        with counter_lock:
            counters["failed"] += 1
            counters["done"] += 1
        result = {"status": "too_small", "url": None}
        with manifest_lock:
            manifest[job["path"]] = result
        return result

    uploaded = upload_oss(local, job)
    with counter_lock:
        counters["generated"] += 1
        counters["done"] += 1
        cur_gen = counters["generated"]
        cur_done = counters["done"]
    if uploaded:
        result = {"status": "ok", "url": uploaded}
    else:
        result = {"status": "upload_failed", "url": None}
    with manifest_lock:
        manifest[job["path"]] = result
    if cur_done % 20 == 0:
        cost = cur_gen * COST_PER_IMAGE
        log.info("PROGRESS: done=%d generated=%d skipped=%d failed=%d est_cost=$%.2f",
                 cur_done, cur_gen, counters["skipped"], counters["failed"], cost)
    return result


def write_manifest():
    with manifest_lock:
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> int:
    jobs = collect_articles()
    log.info("Discovered %d articles across %d series",
             len(jobs), len(set(j["series"] for j in jobs)))

    # Filter what is already done by HEAD checks (sequential, fast) so the
    # cost estimate is accurate before we start.
    log.info("Checking which covers already exist on OSS...")
    todo: list[dict] = []
    pre_skipped = 0
    for j in jobs:
        if oss_exists(j):
            counters["skipped"] += 1
            counters["done"] += 1
            with manifest_lock:
                manifest[j["path"]] = {"status": "skipped", "url": oss_url(j)}
            pre_skipped += 1
        else:
            todo.append(j)
    log.info("Pre-existing covers: %d. To generate: %d. Estimated cost: $%.2f",
             pre_skipped, len(todo), len(todo) * COST_PER_IMAGE)

    if len(todo) * COST_PER_IMAGE > HARD_STOP_USD:
        log.error("Aborting: estimated cost $%.2f exceeds hard stop $%.2f",
                  len(todo) * COST_PER_IMAGE, HARD_STOP_USD)
        return 2

    write_manifest()

    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="cover") as ex:
        futures = {ex.submit(process, j): j for j in todo}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.exception("[%s/%s] unhandled error: %s", j["series"], j["stem"], e)
            with counter_lock:
                cur_gen = counters["generated"]
            if cur_gen * COST_PER_IMAGE > HARD_STOP_USD:
                log.error("Hard cost stop reached ($%.2f). Cancelling remaining.",
                          cur_gen * COST_PER_IMAGE)
                for f in futures:
                    f.cancel()
                break
            # periodic flush
            if counters["done"] % 10 == 0:
                write_manifest()

    write_manifest()
    log.info("FINAL: done=%d generated=%d skipped=%d failed=%d cost=$%.2f",
             counters["done"], counters["generated"], counters["skipped"],
             counters["failed"], counters["generated"] * COST_PER_IMAGE)
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
