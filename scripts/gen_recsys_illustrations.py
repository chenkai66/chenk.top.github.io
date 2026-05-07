#!/usr/bin/env python3
"""Generate Wanxiang illustrations for recommendation-systems chapters.

Adapted from gen_wanxiang_illustrations.py to handle EN/ZH stem mismatch
(EN: 01-fundamentals.md vs ZH: 01-入门与基础概念.md).
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

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
N_IMAGES = 1

CONTENT_ROOT = Path("/root/chenk-hugo/content")
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
LOG_FILE = SCRIPTS_DIR / "wanxiang_recsys.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_recsys_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_recsys")
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

SERIES = "recommendation-systems"

# Per-article: en_stem -> {zh_stem, hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading}
ARTICLES = {
    "01-fundamentals": {
        "zh_stem": "01-入门与基础概念",
        "hero": "abstract editorial illustration of a recommendation funnel narrowing from millions of items to a personalized top-10, layers of filtering as concentric translucent membranes, muted teal and warm copper palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of a user-item matrix being decomposed into two latent factor matrices that glow with hidden taste dimensions, dusty indigo and amber palette, editorial style, no text, 16:9",
        "en_mid": "3. Matrix Factorization: The Workhorse",
        "zh_mid": "三、矩阵分解：协同过滤的主力",
    },
    "02-collaborative-filtering": {
        "zh_stem": "02-协同过滤与矩阵分解",
        "hero": "abstract editorial illustration of a user-item bipartite graph with glowing similarity edges connecting clusters of taste neighbors, muted plum and ivory palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of matrix factorization as a sparse user-item grid splitting into two dense latent factor blocks, soft teal and warm sand palette, editorial style, no text, 16:9",
        "en_mid": "4 · Matrix factorization",
        "zh_mid": "4 · 矩阵分解",
    },
    "03-deep-learning-basics": {
        "zh_stem": "03-深度学习基础模型",
        "hero": "abstract editorial illustration of a multilayer perceptron processing user and item embeddings into a luminous interaction score, muted slate and copper palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of sparse one-hot IDs flowing through an embedding lookup into a dense semantic vector space, dusty rose and ivory palette, editorial style, no text, 16:9",
        "en_mid": "3. Embeddings: the bridge from sparse IDs to learned semantics",
        "zh_mid": "三、Embedding：从稀疏 ID 到语义表征的桥梁",
    },
    "04-ctr-prediction": {
        "zh_stem": "04-CTR预估与点击率建模",
        "hero": "abstract editorial illustration of feature-cross interactions illuminated as a constellation of pairwise sparks above a CTR prediction surface, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of factorization machines as field embeddings interacting through inner products, lit pairwise lines forming a dense web, soft amber and slate palette, editorial style, no text, 16:9",
        "en_mid": "Factorization Machines (FM): Automatic Pairwise Interactions",
        "zh_mid": "Factorization Machines（FM）：自动二阶交互",
    },
    "05-embedding-techniques": {
        "zh_stem": "05-Embedding表示学习",
        "hero": "abstract editorial illustration of users and items as glowing points scattered in a continuous latent space, semantic clusters forming naturally, muted violet and amber palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of two-tower architecture with separate user and item encoders meeting at a dot-product space, dusty teal and copper palette, editorial style, no text, 16:9",
        "en_mid": "4. Two-tower models: separate user and item encoders",
        "zh_mid": "4. 双塔模型：用户和物品分开编码",
    },
    "06-sequential-recommendation": {
        "zh_stem": "06-序列推荐与会话建模",
        "hero": "abstract editorial illustration of a user behavior timeline as a flowing sequence of interaction tokens lit from within, muted plum and parchment palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of self-attention over a behavior sequence, every position attending to every other through translucent filaments, slate and warm brass palette, editorial style, no text, 16:9",
        "en_mid": "SASRec: self-attention for sequential recommendation",
        "zh_mid": "SASRec：自注意力的序列推荐",
    },
    "07-graph-neural-networks": {
        "zh_stem": "07-图神经网络与社交推荐",
        "hero": "abstract editorial illustration of graph nodes message-passing in propagating waves of light across a user-item graph, muted teal and copper palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of a graph convolution layer aggregating neighbor embeddings into a refined node representation, dusty cobalt and amber palette, editorial style, no text, 16:9",
        "en_mid": "Graph Convolutional Networks (GCN)",
        "zh_mid": "图卷积网络（GCN）",
    },
    "08-knowledge-graph": {
        "zh_stem": "08-知识图谱增强推荐系统",
        "hero": "abstract editorial illustration of knowledge triples weaving into a recommendation graph, head-relation-tail filaments lit by semantic meaning, muted forest green and parchment palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of preference propagation rippling outward from a user across knowledge graph hops, soft amber and slate palette, editorial style, no text, 16:9",
        "en_mid": "RippleNet: Preference Propagation",
        "zh_mid": "RippleNet：偏好的涟漪传播",
    },
    "09-multi-task-learning": {
        "zh_stem": "09-多任务学习与多目标优化",
        "hero": "abstract editorial illustration of a shared encoder branching into multiple objective heads — click, conversion, dwell — as parallel luminous outputs, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of mixture-of-experts gating: tokens routed by translucent gates to specialist expert towers, dusty teal and copper palette, editorial style, no text, 16:9",
        "en_mid": "Architecture 3: MMoE (Multi-gate Mixture-of-Experts)",
        "zh_mid": "架构三：MMoE（多门控混合专家）",
    },
    "10-deep-interest-networks": {
        "zh_stem": "10-深度兴趣网络与注意力机制",
        "hero": "abstract editorial illustration of attention weights illuminating the most relevant past behaviors of a user toward a candidate item, muted violet and gold palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of evolving user interest as a flowing GRU sequence with attention over hidden states, soft slate and copper palette, editorial style, no text, 16:9",
        "en_mid": "3. Deep Interest Evolution Network (DIEN)",
        "zh_mid": "3. Deep Interest Evolution Network (DIEN)",
    },
    "11-contrastive-learning": {
        "zh_stem": "11-对比学习与自监督学习",
        "hero": "abstract editorial illustration of two augmented views of the same node pulled together while negatives are pushed apart in a luminous embedding sphere, muted teal and warm sand palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of InfoNCE loss as a positive pair glowing brightly amid a cloud of dimmed negative pairs, dusty rose and slate palette, editorial style, no text, 16:9",
        "en_mid": "InfoNCE: the loss that does the work",
        "zh_mid": "InfoNCE：真正在干活的损失函数",
    },
    "12-llm-recommendation": {
        "zh_stem": "12-大语言模型与推荐系统",
        "hero": "abstract editorial illustration of an LLM as a semantic understanding overlay floating above a classical recommendation pipeline, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of a hybrid pipeline where LLM reasoning enhances retrieval and ranking stages, copper and slate palette, editorial style, no text, 16:9",
        "en_mid": "The Hybrid Pipeline (the architecture you actually ship)",
        "zh_mid": "混合管线（你最终一定会落到这个架构）",
    },
    "13-fairness-explainability": {
        "zh_stem": "13-公平性-去偏与可解释性",
        "hero": "abstract editorial illustration of bias forces being countered by causal debiasing layers in a recommendation system, muted plum and ivory palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of counterfactual reasoning as parallel ghost timelines diverging from observed outcomes, dusty teal and warm amber palette, editorial style, no text, 16:9",
        "en_mid": "Part 2 — Causal Inference for Recommenders",
        "zh_mid": "第二部分 —— 推荐系统的因果推断",
    },
    "14-cross-domain-cold-start": {
        "zh_stem": "14-跨域推荐与冷启动解决方案",
        "hero": "abstract editorial illustration of a cold sparse new user warming up as cross-domain signals and meta-learning prototypes flow in, muted slate and warm copper palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of multi-armed bandit exploration as glowing arms being sampled with uncertainty halos, soft indigo and brass palette, editorial style, no text, 16:9",
        "en_mid": "Bandits — Exploration vs Exploitation",
        "zh_mid": "Bandit —— 探索与利用",
    },
    "15-real-time-online": {
        "zh_stem": "15-实时推荐与在线学习",
        "hero": "abstract editorial illustration of a real-time event stream flowing through Kafka pipelines into Flink processors and back into a serving layer, muted teal and graphite palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of online learning as model weights updating continuously from a stream of fresh interactions, dusty copper and slate palette, editorial style, no text, 16:9",
        "en_mid": "5. Online learning: from SGD to FTRL",
        "zh_mid": "5. 在线学习：从 SGD 到 FTRL",
    },
    "16-industrial-practice": {
        "zh_stem": "16-工业级架构与最佳实践",
        "hero": "abstract editorial illustration of a production recommendation stack as a layered architectural cathedral — recall, ranking, reranking, feature store — humming with traffic, muted teal and warm brass palette, magazine-cover aesthetic, no text, 16:9",
        "mid": "abstract illustration of a multi-channel recall fan-in: parallel candidate streams from CF, content, graph, and LLM merging into a ranking funnel, copper and slate palette, editorial style, no text, 16:9",
        "en_mid": "Multi-Channel Recall",
        "zh_mid": "多路召回",
    },
}

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wx-recsys")


def oss_key(lang: str, stem: str, n: int) -> str:
    return f"posts/{lang}/{SERIES}/{stem}/illustration_{n}.jpg"


def oss_url(lang: str, stem: str, n: int) -> str:
    return f"{OSS_PUBLIC_BASE}/{oss_key(lang, stem, n)}"


def oss_exists(lang: str, stem: str, n: int) -> bool:
    url = oss_url(lang, stem, n)
    req = Request(url, method="HEAD", headers={"User-Agent": "illu/1.0"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status == 200
    except HTTPError:
        return False
    except URLError:
        return False


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
                model=MODEL, prompt=prompt, negative_prompt=NEG, n=N_IMAGES, size=SIZE,
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


def generate_image(prompt: str, tag: str):
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


def upload_oss(local: Path, lang: str, stem: str, n: int) -> bool:
    target = f"oss://{OSS_BUCKET}/{oss_key(lang, stem, n)}"
    cmd = [OSSUTIL, "cp", "-f", "--meta", "Cache-Control:public, max-age=300, must-revalidate", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing", str(local), target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%d] ossutil rc=%d %s", stem, n, r.returncode, r.stderr[-200:])
        return False
    return True


def process_image(en_stem: str, zh_stem: str, n: int, prompt: str) -> dict:
    """Generate one image, upload to BOTH en (en_stem) and zh (zh_stem) OSS paths."""
    tag = f"{en_stem}/illu{n}"
    result = {"en_stem": en_stem, "zh_stem": zh_stem, "n": n, "ok": False, "skipped": False}

    if oss_exists("en", en_stem, n) and oss_exists("zh", zh_stem, n):
        log.info("[%s] both langs exist, skipping", tag)
        result.update(ok=True, skipped=True)
        return result

    data = generate_image(prompt, tag)
    if data is None:
        return result

    out_dir = TMP_DIR / en_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"illustration_{n}.jpg"
    local.write_bytes(data)
    log.info("[%s] saved %d bytes -> %s", tag, len(data), local)

    ok_en = upload_oss(local, "en", en_stem, n)
    ok_zh = upload_oss(local, "zh", zh_stem, n)
    if ok_en and ok_zh:
        result.update(ok=True)
    return result


def insert_into_markdown(md_path: Path, hero_url: str, mid_url: str,
                         mid_heading: str, title: str, lang: str, stem: str) -> bool:
    if not md_path.exists():
        log.warning("missing md: %s", md_path)
        return False
    text = md_path.read_text(encoding="utf-8")

    has_hero = "illustration_1.jpg" in text
    has_mid = "illustration_2.jpg" in text

    if has_hero and has_mid:
        log.info("[%s/%s] already has both, skipping insert", lang, stem)
        return True

    lines = text.split("\n")
    new_lines: list[str] = []

    hero_line = f"![{title} — visual]({hero_url})"
    mid_line = f"![{title} — visual]({mid_url})"

    inserted_hero = has_hero
    inserted_mid = has_mid

    for line in lines:
        if not inserted_hero and line.startswith("## "):
            new_lines.append(hero_line)
            new_lines.append("")
            inserted_hero = True
        new_lines.append(line)
        if not inserted_mid and line.strip() == f"## {mid_heading}":
            new_lines.append("")
            new_lines.append(mid_line)
            inserted_mid = True

    if not inserted_hero:
        log.warning("[%s/%s] no ## heading found for hero", lang, stem)
    if not inserted_mid:
        log.warning("[%s/%s] mid heading '%s' not found", lang, stem, mid_heading)

    new_text = "\n".join(new_lines)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        log.info("[%s/%s] wrote %d bytes (hero=%s mid=%s)", lang, stem, len(new_text),
                 inserted_hero and not has_hero, inserted_mid and not has_mid)
    return inserted_hero and inserted_mid


def get_title(md_path: Path) -> str:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return md_path.stem
    m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else md_path.stem


def main():
    image_jobs = []
    for en_stem, spec in ARTICLES.items():
        image_jobs.append((en_stem, spec["zh_stem"], 1, spec["hero"]))
        image_jobs.append((en_stem, spec["zh_stem"], 2, spec["mid"]))
    log.info("Total image jobs: %d (across %d articles)", len(image_jobs), len(ARTICLES))

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="wx") as ex:
        futs = {ex.submit(process_image, en, zh, n, p): (en, n) for en, zh, n, p in image_jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                results[(r["en_stem"], r["n"])] = r
            except Exception as e:
                en, n = futs[f]
                log.error("[%s/%d] worker exception: %s", en, n, e)
                results[(en, n)] = {"ok": False, "en_stem": en, "n": n}

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(image_jobs),
        "succeeded": sum(1 for r in results.values() if r.get("ok")),
        "skipped": sum(1 for r in results.values() if r.get("skipped")),
        "results": list(results.values()),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest: %s", MANIFEST_FILE)

    insert_summary = {"en_articles": 0, "zh_articles": 0, "failed": []}
    for en_stem, spec in ARTICLES.items():
        zh_stem = spec["zh_stem"]
        r1 = results.get((en_stem, 1), {})
        r2 = results.get((en_stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(en_stem)
            log.warning("[%s] missing image, skip insert", en_stem)
            continue
        for lang, stem, mid_heading in [("en", en_stem, spec["en_mid"]), ("zh", zh_stem, spec["zh_mid"])]:
            md_path = CONTENT_ROOT / lang / SERIES / f"{stem}.md"
            title = get_title(md_path)
            hero_url = oss_url(lang, stem, 1)
            mid_url = oss_url(lang, stem, 2)
            ok = insert_into_markdown(md_path, hero_url, mid_url, mid_heading, title, lang, stem)
            if ok:
                insert_summary[f"{lang}_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "total_succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
