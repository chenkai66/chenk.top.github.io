#!/usr/bin/env python3
"""Audit all Wanxiang covers on OSS for unwanted text/typography.

Phase 1: download all covers, OCR each, flag those with >5 alphanumeric/CJK chars.
Phase 2: for flagged ones, re-build prompt with hard no-text constraints, regen,
         re-OCR to verify, then upload to OSS replacing original. Up to 3 retries.

Saves manifest to /root/chenk-hugo/scripts/cover_text_audit.json.
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

OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_PREFIX = "posts/covers"
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
MANIFEST_FILE = SCRIPTS_DIR / "cover_text_audit.json"
LOG_FILE = SCRIPTS_DIR / "cover_text_audit.log"

TMP_DIR = Path("/tmp/cover_audit")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OCR_WORKERS = 8
GEN_WORKERS = 3

CHAR_THRESHOLD = 5            # >5 alphanumeric/CJK chars => flagged
MAX_REGEN_RETRIES = 3
COST_PER_IMAGE = 0.30         # USD
HARD_STOP_USD = 30.0          # cap per spec
SUBMIT_MIN_INTERVAL_S = 1.2

dashscope.api_key = DASHSCOPE_API_KEY

# ----- logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("audit")


# ----- helpers -----
def list_all_covers() -> list[str]:
    """Return list of all cover OSS keys (relative to bucket)."""
    out = subprocess.run(
        [OSSUTIL, "ls", f"oss://{OSS_BUCKET}/{OSS_PREFIX}/", "-s"],
        capture_output=True, text=True, check=True,
    )
    keys = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith(f"oss://{OSS_BUCKET}/"):
            continue
        if not (line.endswith(".jpg") or line.endswith(".png") or line.endswith(".webp")):
            continue
        key = line[len(f"oss://{OSS_BUCKET}/"):]
        keys.append(key)
    return keys


def local_path(key: str) -> Path:
    safe = key.replace("/", "__")
    return TMP_DIR / safe


def download(key: str) -> Path | None:
    """Download OSS object via public HTTP."""
    p = local_path(key)
    if p.exists() and p.stat().st_size > 0:
        return p
    url = f"{OSS_PUBLIC_BASE}/{key}"
    try:
        req = Request(url, headers={"User-Agent": "audit/1.0"})
        with urlopen(req, timeout=30) as r:
            data = r.read()
        p.write_bytes(data)
        return p
    except (HTTPError, URLError) as e:
        log.warning("download failed %s: %s", key, e)
        return None


# count alphanumeric (ascii letters + digits) + CJK ideographs
CJK_RE = re.compile(r"[A-Za-z0-9一-鿿]")


def ocr_text(p: Path) -> str:
    """Run tesseract eng+chi_sim, return raw extracted text."""
    out = subprocess.run(
        ["tesseract", str(p), "-", "-l", "eng+chi_sim", "--psm", "6"],
        capture_output=True, text=True,
    )
    return out.stdout or ""


def text_score(text: str) -> int:
    """Count alphanumeric + CJK characters."""
    return len(CJK_RE.findall(text))


# ----- prompt builder for regen (hard no-text) -----
def build_regen_prompt(key: str) -> str:
    """Build a generic-but-themed abstract prompt with HARD no-text constraints.

    We don't have original prompts; infer theme from path components.
    """
    # key like posts/covers/articles/{series}/{stem}.jpg or posts/covers/{series}.jpg
    parts = key.split("/")
    # strip ext
    stem = Path(parts[-1]).stem
    # series
    series = parts[-2] if "articles" in parts else stem

    series_name = series.replace("-", " ").replace("_", " ")
    topic = stem.replace("-", " ").replace("_", " ")

    return (
        f"Editorial abstract scientific illustration cover art on the theme of "
        f"\"{series_name}\" / \"{topic}\". "
        f"Style: abstract geometric, magazine-cover aesthetic inspired by Stripe Press "
        f"and Quanta Magazine. Muted color palette, clean composition, soft gradients, "
        f"depth via subtle shadow, 16:9 horizontal composition. "
        f"STRICTLY no text, no letters, no characters, no labels, no captions, "
        f"no numbers, no logos, no watermarks, no signatures, no glyphs, no symbols "
        f"of writing systems, no UI mockups. NO TYPOGRAPHY OF ANY KIND. "
        f"Pure abstract visual only."
    )


NEG_PROMPT = (
    "text, letters, words, numbers, characters, glyphs, watermark, logo, signature, "
    "captions, labels, typography, fonts, ascii, alphabet, kanji, hanzi, kana, latin, "
    "ugly, low quality, blurry, distorted, photorealistic faces, stock photo aesthetic, "
    "cluttered, frame, border, ui mockup, screenshot, document, paper page"
)

# ----- gen throttle -----
submit_lock = Lock()
_last_submit_ts = [0.0]


def _throttled_call(prompt: str, tag: str):
    with submit_lock:
        now = time.time()
        delta = now - _last_submit_ts[0]
        if delta < SUBMIT_MIN_INTERVAL_S:
            time.sleep(SUBMIT_MIN_INTERVAL_S - delta)
        _last_submit_ts[0] = time.time()
    try:
        resp = ImageSynthesis.call(
            model=MODEL,
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            n=1,
            size=SIZE,
        )
        return resp
    except Exception as e:
        log.warning("[%s] dashscope call failed: %s", tag, e)
        return None


def regen_one(key: str) -> tuple[bool, int, str]:
    """Try to regen a text-free cover. Returns (success, char_count, last_text)."""
    prompt = build_regen_prompt(key)
    last_text = ""
    last_score = 999
    for attempt in range(1, MAX_REGEN_RETRIES + 1):
        log.info("[%s] regen attempt %d/%d", key, attempt, MAX_REGEN_RETRIES)
        resp = _throttled_call(prompt, key)
        if resp is None or not getattr(resp, "output", None) or not resp.output.results:
            log.warning("[%s] empty resp", key)
            continue
        url = resp.output.results[0].url
        # download new image
        new_p = TMP_DIR / f"new__{key.replace('/', '__')}"
        try:
            req = Request(url, headers={"User-Agent": "audit/1.0"})
            with urlopen(req, timeout=60) as r:
                new_p.write_bytes(r.read())
        except Exception as e:
            log.warning("[%s] dl new failed: %s", key, e)
            continue
        # OCR new image
        text = ocr_text(new_p)
        score = text_score(text)
        last_text = text.strip().replace("\n", " ")[:120]
        last_score = score
        log.info("[%s] new image score=%d sample=%r", key, score, last_text[:60])
        if score <= CHAR_THRESHOLD:
            # upload replace
            up = subprocess.run(
                [OSSUTIL, "cp", "-f", "--cache-control", "public, max-age=300, must-revalidate", str(new_p), f"oss://{OSS_BUCKET}/{key}"],
                capture_output=True, text=True,
            )
            if up.returncode == 0:
                log.info("[%s] uploaded text-free replacement", key)
                return True, score, last_text
            log.warning("[%s] upload failed: %s", key, up.stderr.strip())
    return False, last_score, last_text


# ----- main -----
def main():
    log.info("=== cover text audit start ===")
    keys = list_all_covers()
    log.info("found %d covers", len(keys))

    manifest: dict = {}
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
            log.info("loaded existing manifest with %d entries", len(manifest))
        except Exception:
            manifest = {}

    def save():
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # ---- Phase 1: OCR all ----
    def audit_one(key: str):
        if key in manifest and "text_score" in manifest[key]:
            return key, manifest[key]
        p = download(key)
        if p is None:
            return key, {"download": "failed"}
        text = ocr_text(p)
        score = text_score(text)
        sample = text.strip().replace("\n", " ")[:200]
        return key, {
            "text_score": score,
            "has_text": score > CHAR_THRESHOLD,
            "ocr_sample": sample,
            "regen_status": "pending" if score > CHAR_THRESHOLD else "not_needed",
        }

    log.info("--- Phase 1: OCR audit ---")
    done = 0
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
        futs = [ex.submit(audit_one, k) for k in keys]
        for fut in as_completed(futs):
            key, info = fut.result()
            manifest[key] = {**manifest.get(key, {}), **info}
            done += 1
            if done % 25 == 0:
                log.info("audited %d/%d", done, len(keys))
                save()
    save()

    flagged = [k for k, v in manifest.items() if v.get("has_text")]
    log.info("Phase 1 done: %d flagged with text", len(flagged))

    # ---- Phase 2: regen ----
    log.info("--- Phase 2: regen flagged ---")
    cost = 0.0
    regen_done = 0
    regen_fail = 0

    # serialize through pool but cost-cap globally
    def regen_wrap(key: str):
        return key, regen_one(key)

    pending = [k for k in flagged if manifest[k].get("regen_status") not in ("done", "failed_persistent")]
    log.info("regen queue: %d", len(pending))

    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {}
        for k in pending:
            if cost + (COST_PER_IMAGE * MAX_REGEN_RETRIES) > HARD_STOP_USD:
                log.warning("cost cap reached (proj=%.2f), stopping submission", cost)
                manifest[k]["regen_status"] = "skipped_cost_cap"
                continue
            cost += COST_PER_IMAGE  # optimistic cost (1 try); add for retries lazily
            futs[ex.submit(regen_wrap, k)] = k
        for fut in as_completed(futs):
            key, (ok, score, sample) = fut.result()
            manifest[key]["regen_attempts_text_score"] = score
            manifest[key]["regen_sample"] = sample
            if ok:
                manifest[key]["regen_status"] = "done"
                regen_done += 1
            else:
                manifest[key]["regen_status"] = "failed_persistent"
                regen_fail += 1
            save()

    log.info("=== summary ===")
    log.info("total covers: %d", len(keys))
    log.info("flagged with text: %d", len(flagged))
    log.info("regen successful: %d", regen_done)
    log.info("regen failed: %d", regen_fail)
    log.info("approx cost: $%.2f", cost)


if __name__ == "__main__":
    main()
