#!/usr/bin/env python3
"""Generate Wanxiang illustrations for NLP chapters and insert into EN+ZH markdown.

Two illustrations per chapter: hero (after intro paragraph) + mid (after a chosen ## heading).
Idempotent. HEAD-checks OSS first.
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
LOG_FILE = SCRIPTS_DIR / "wanxiang_nlp.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_nlp_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_nlp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

WORKERS = 3
SUBMIT_MIN_INTERVAL_S = 1.5
MAX_SUBMIT_RETRIES = 6
POLL_TIMEOUT_S = 240

NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border"
)

dashscope.api_key = DASHSCOPE_API_KEY

STYLE_SUFFIX = (
    ", editorial Stripe Press / Quanta Magazine style, soft muted pastel palette, "
    "abstract geometric, clean composition, no text, no letters, magazine cover aesthetic, "
    "soft gradients, generous negative space, 16:9"
)

# (en_stem, zh_stem, hero_prompt_core, mid_prompt_core, en_mid_h2, zh_mid_h2)
SPECS = [
    ("introduction-and-preprocessing", "01-NLP入门与文本预处理",
     "abstract editorial illustration of raw text being decomposed into clean tokens flowing through a pipeline, soft pastel palette",
     "tokenization concept: a stream of characters condensing into discrete subword pieces, gradient mesh",
     "What You Will Learn", "你将学到什么"),
    ("word-embeddings-lm", "02-词向量与语言模型",
     "words morphing into vectors floating in a luminous latent space, gradient mesh, dimensions visualized as soft bands",
     "skip-gram training: a center word radiating context predictions outward, abstract neural geometry",
     "What You Will Learn", "你将学到什么"),
    ("rnn-sequence-modeling", "03-RNN与序列建模",
     "memory cells passing context through sequential gates flowing left to right, soft glowing connections",
     "LSTM gates: forget, input, output gates as translucent valves controlling a memory river",
     "What You Will Learn", "你将学到什么"),
    ("attention-transformer", "04-注意力机制与Transformer",
     "translucent attention threads connecting tokens in parallel layers, multi-head visualization, soft pastel",
     "self-attention: every token reaching every other token with weighted glowing arrows in a stack of layers",
     "What You Will Learn", "你将学到什么"),
    ("bert-pretrained-models", "05-BERT与预训练模型",
     "masked tokens being filled in by bidirectional context arrows converging from both sides, abstract editorial",
     "pretraining and fine-tuning: a large frozen backbone with small task-specific heads attached",
     "What You Will Learn", "你将学到什么"),
    ("gpt-generative-models", "06-GPT与生成式语言模型",
     "tokens flowing left-to-right with glowing decoder layers, autoregressive generation as a luminous chain",
     "decoder stack generating one token at a time, causal mask visualized as a soft triangular gradient",
     "What You Will Learn", "你将学到什么"),
    ("prompt-engineering-icl", "07-提示工程与In-Context-Learning",
     "a carefully shaped prompt key unlocking hidden behavior inside a large abstract model silhouette",
     "few-shot in-context learning: example demonstrations stacked as cards feeding into a model query",
     "What You Will Learn", "你将学到什么"),
    ("fine-tuning-peft", "08-模型微调与PEFT",
     "small LoRA adapter weights inserted between frozen layers of a large transformer backbone diagram",
     "parameter-efficient tuning: a tiny colored module attached to a vast greyscale backbone",
     "What You Will Learn", "你将学到什么"),
    ("llm-architecture-deep-dive", "09-大语言模型架构深度解析",
     "stacked transformer blocks with attention and FFN sublayers visualized as luminous parallel rails",
     "mixture of experts: a router directing tokens to a subset of expert modules, abstract geometric",
     "What You Will Learn", "你将学到什么"),
    ("rag-knowledge-enhancement", "10-RAG与知识增强系统",
     "documents being retrieved from a vector store and woven into a generated answer thread, editorial",
     "retrieval pipeline: query embedding fetching top-k chunks from a glowing vector index",
     "What You Will Learn", "你将学到什么"),
    ("multimodal-nlp", "11-多模态大模型",
     "image, text, and audio modalities fusing in a shared embedding space as overlapping translucent layers",
     "vision-language alignment: an image patch grid mapped to text tokens through cross-attention",
     "What You Will Learn", "你将学到什么"),
    ("frontiers-applications", "12-前沿技术与实战应用",
     "future LLM evolution timeline with branching paths leading to diverse applications, abstract geometric",
     "agentic systems: multiple LLM agents collaborating through tools, glowing dataflow",
     "What You Will Learn", "你将学到什么"),
]

# ----- logging -----
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wnlp")

# ----- manifest -----
manifest_lock = Lock()
if MANIFEST_FILE.exists():
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
else:
    manifest = {}

def save_manifest():
    with manifest_lock:
        MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ----- OSS utils -----
def oss_url(stem_for_path: str, lang: str, idx: int) -> str:
    # use EN stem for both languages so URL is stable
    return f"{OSS_PUBLIC_BASE}/posts/{lang}/nlp/{stem_for_path}/illustration_{idx}.jpg"

def oss_head(url: str) -> bool:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=15) as r:
            return r.status == 200
    except HTTPError as e:
        return e.code == 200
    except URLError:
        return False

def oss_upload(local: Path, oss_path: str) -> bool:
    cmd = [OSSUTIL, "cp", "-f", str(local), f"oss://{OSS_BUCKET}/{oss_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ossutil cp failed: %s", r.stderr)
        return False
    return True

# ----- Wanxiang -----
submit_lock = Lock()
last_submit_t = [0.0]

def submit_with_retry(prompt: str):
    for attempt in range(MAX_SUBMIT_RETRIES):
        with submit_lock:
            now = time.time()
            wait = SUBMIT_MIN_INTERVAL_S - (now - last_submit_t[0])
            if wait > 0:
                time.sleep(wait)
            last_submit_t[0] = time.time()
        try:
            rsp = ImageSynthesis.async_call(
                model=MODEL, prompt=prompt, negative_prompt=NEG, n=N_IMAGES, size=SIZE,
            )
            if rsp.status_code == 200 and rsp.output and rsp.output.task_id:
                return rsp.output.task_id
            log.warning("submit non-200 (%s): %s", rsp.status_code, rsp.message)
        except Exception as e:
            log.warning("submit exc: %s", e)
        time.sleep(2 ** attempt)
    return None

def poll(task_id: str):
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT_S:
        try:
            rsp = ImageSynthesis.fetch(task=task_id)
            if rsp.status_code == 200:
                st = rsp.output.task_status
                if st == "SUCCEEDED":
                    return rsp.output.results
                if st == "FAILED":
                    log.error("task %s failed: %s", task_id, getattr(rsp.output, "message", ""))
                    return None
        except Exception as e:
            log.warning("poll exc: %s", e)
        time.sleep(3)
    log.error("task %s timed out", task_id)
    return None

def download(url: str, dst: Path) -> bool:
    try:
        with urlopen(url, timeout=60) as r:
            dst.write_bytes(r.read())
        return True
    except Exception as e:
        log.error("download fail: %s", e)
        return False

def gen_one(en_stem: str, idx: int, prompt_core: str):
    """Generate one image for both en and zh OSS paths (same image)."""
    en_url = oss_url(en_stem, "en", idx)
    zh_url = oss_url(en_stem, "zh", idx)  # use EN stem for zh too -> stable url

    # Wait — zh path should match the file's actual location. We're inserting into ZH md
    # but the zh file uses Chinese stem in directory. For OSS we use EN stem under both
    # /en/ and /zh/ paths. Markdown reference will just be the URL.
    if oss_head(en_url) and oss_head(zh_url):
        log.info("[skip] %s idx=%d already on OSS", en_stem, idx)
        return en_url, zh_url

    full_prompt = prompt_core + STYLE_SUFFIX
    task_id = submit_with_retry(full_prompt)
    if not task_id:
        log.error("submit failed for %s idx=%d", en_stem, idx)
        return None, None
    results = poll(task_id)
    if not results:
        return None, None
    img_url = results[0].url
    local = TMP_DIR / f"{en_stem}_{idx}.jpg"
    if not download(img_url, local):
        return None, None
    ok1 = oss_upload(local, f"posts/en/nlp/{en_stem}/illustration_{idx}.jpg")
    ok2 = oss_upload(local, f"posts/zh/nlp/{en_stem}/illustration_{idx}.jpg")
    if not (ok1 and ok2):
        return None, None
    log.info("[ok] %s idx=%d -> %s", en_stem, idx, en_url)
    return en_url, zh_url

# ----- markdown insertion -----
HERO_MARKER = "<!-- wanx-hero -->"
MID_MARKER = "<!-- wanx-mid -->"

def insert_hero(md_text: str, url: str, alt: str) -> tuple[str, bool]:
    if HERO_MARKER in md_text:
        return md_text, False
    # Find front matter end
    m = re.match(r"^---\n.*?\n---\n", md_text, re.DOTALL)
    if not m:
        return md_text, False
    fm_end = m.end()
    rest = md_text[fm_end:]
    # find first ## heading
    h2 = re.search(r"^## ", rest, re.MULTILINE)
    if not h2:
        return md_text, False
    insert_pos = fm_end + h2.start()
    # back up to keep blank line before heading
    block = f"\n{HERO_MARKER}\n![{alt}]({url})\n\n"
    new = md_text[:insert_pos] + block + md_text[insert_pos:]
    return new, True

def insert_mid(md_text: str, url: str, alt: str, h2_title: str) -> tuple[str, bool]:
    if MID_MARKER in md_text:
        return md_text, False
    pat = re.compile(r"^## " + re.escape(h2_title) + r"\s*$", re.MULTILINE)
    m = pat.search(md_text)
    if not m:
        # fallback: insert after second ## heading
        all_h2 = list(re.finditer(r"^## .*$", md_text, re.MULTILINE))
        if len(all_h2) < 2:
            return md_text, False
        m = all_h2[1]
    # find end of that heading line
    line_end = md_text.find("\n", m.end())
    if line_end == -1:
        return md_text, False
    # find the next blank line after some content (skip directly to after one paragraph)
    rest_start = line_end + 1
    # insert image right after heading line (with blank line separation)
    block = f"\n{MID_MARKER}\n![{alt}]({url})\n\n"
    new = md_text[:rest_start] + block + md_text[rest_start:]
    return new, True

def find_zh_md(zh_stem: str) -> Path:
    return CONTENT_ROOT / "zh" / "nlp" / f"{zh_stem}.md"

def find_en_md(en_stem: str) -> Path:
    return CONTENT_ROOT / "en" / "nlp" / f"{en_stem}.md"

def main():
    stats = {"new": 0, "skipped": 0, "failed": 0, "inserted": 0}
    sample_urls = []
    tasks = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for spec in SPECS:
            en_stem, zh_stem, hero_p, mid_p, en_h2, zh_h2 = spec
            for idx, prompt in [(1, hero_p), (2, mid_p)]:
                f = ex.submit(gen_one, en_stem, idx, prompt)
                futs[f] = (spec, idx)
        for f in as_completed(futs):
            spec, idx = futs[f]
            en_url, zh_url = f.result()
            en_stem, zh_stem, hero_p, mid_p, en_h2, zh_h2 = spec
            if en_url is None:
                stats["failed"] += 1
                continue
            key = f"{en_stem}|{idx}"
            was_new = key not in manifest
            manifest[key] = {"en_url": en_url, "zh_url": zh_url}
            if was_new:
                stats["new"] += 1
                if len(sample_urls) < 3:
                    sample_urls.append(en_url)
            else:
                stats["skipped"] += 1
    save_manifest()

    # Now insert into markdown files
    for spec in SPECS:
        en_stem, zh_stem, hero_p, mid_p, en_h2, zh_h2 = spec
        en_md = find_en_md(en_stem)
        zh_md = find_zh_md(zh_stem)

        for idx, h2_en, h2_zh, marker_role in [(1, None, None, "hero"), (2, en_h2, zh_h2, "mid")]:
            key = f"{en_stem}|{idx}"
            if key not in manifest:
                continue
            en_url = manifest[key]["en_url"]
            zh_url = manifest[key]["zh_url"]

            # EN
            if en_md.exists():
                txt = en_md.read_text(encoding="utf-8")
                title_match = re.search(r'^title:\s*"?([^"\n]+)"?', txt, re.MULTILINE)
                title = title_match.group(1) if title_match else en_stem
                alt = f"{title} — visual"
                if marker_role == "hero":
                    new, ok = insert_hero(txt, en_url, alt)
                else:
                    new, ok = insert_mid(txt, en_url, alt, h2_en)
                if ok:
                    en_md.write_text(new, encoding="utf-8")
                    stats["inserted"] += 1
                    log.info("inserted %s into %s", marker_role, en_md.name)
            # ZH
            if zh_md.exists():
                txt = zh_md.read_text(encoding="utf-8")
                title_match = re.search(r'^title:\s*"?([^"\n]+)"?', txt, re.MULTILINE)
                title = title_match.group(1) if title_match else zh_stem
                alt = f"{title} — 配图"
                if marker_role == "hero":
                    new, ok = insert_hero(txt, zh_url, alt)
                else:
                    new, ok = insert_mid(txt, zh_url, alt, h2_zh)
                if ok:
                    zh_md.write_text(new, encoding="utf-8")
                    stats["inserted"] += 1
                    log.info("inserted %s into %s", marker_role, zh_md.name)

    log.info("DONE stats=%s sample=%s", stats, sample_urls)
    print(json.dumps({"stats": stats, "sample_urls": sample_urls}, ensure_ascii=False))

if __name__ == "__main__":
    main()
