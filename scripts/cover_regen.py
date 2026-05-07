#!/usr/bin/env python3
"""Regenerate flagged covers with hard no-text prompt. Cost-capped at $30."""
import os
from __future__ import annotations
import json, logging, subprocess, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

import dashscope
from dashscope import ImageSynthesis

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.oss-cn-beijing.aliyuncs.com"
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
MANIFEST_FILE = SCRIPTS_DIR / "cover_text_audit.json"
LOG_FILE = SCRIPTS_DIR / "cover_text_audit.log"
TMP_DIR = Path("/tmp/cover_audit")
TMP_DIR.mkdir(parents=True, exist_ok=True)

SCORE_THRESHOLD = 25       # regen anything score >= this
CHAR_OK_THRESHOLD = 5      # new image is ok if score <= this
MAX_RETRIES = 3
COST_PER_CALL = 0.30
HARD_STOP_USD = 30.0
GEN_WORKERS = 3
SUBMIT_INTERVAL = 1.2

dashscope.api_key = DASHSCOPE_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("regen")

CJK_RE = re.compile(r"[A-Za-z0-9一-鿿]")

def ocr_score(p: Path):
    out = subprocess.run(["tesseract", str(p), "-", "-l", "eng+chi_sim", "--psm", "6"],
                         capture_output=True, text=True)
    t = out.stdout or ""
    return len(CJK_RE.findall(t)), t.strip().replace("\n", " ")[:120]

def build_prompt(key: str) -> str:
    parts = key.split("/")
    stem = Path(parts[-1]).stem
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

NEG = ("text, letters, words, numbers, characters, glyphs, watermark, logo, signature, "
       "captions, labels, typography, fonts, alphabet, kanji, hanzi, latin, "
       "ugly, low quality, blurry, distorted, cluttered, frame, border, "
       "ui mockup, screenshot, document, paper page")

submit_lock = Lock()
last_ts = [0.0]
cost_lock = Lock()
total_cost = [0.0]

def throttled_call(prompt: str):
    with submit_lock:
        now = time.time()
        d = now - last_ts[0]
        if d < SUBMIT_INTERVAL:
            time.sleep(SUBMIT_INTERVAL - d)
        last_ts[0] = time.time()
    try:
        return ImageSynthesis.call(model=MODEL, prompt=prompt, negative_prompt=NEG, n=1, size=SIZE)
    except Exception as e:
        log.warning("dashscope call failed: %s", e)
        return None

def regen_one(key: str):
    prompt = build_prompt(key)
    last_score = 999
    last_sample = ""
    attempts = 0
    for attempt in range(1, MAX_RETRIES + 1):
        with cost_lock:
            if total_cost[0] + COST_PER_CALL > HARD_STOP_USD:
                log.warning("[%s] cost cap reached at $%.2f, stopping", key, total_cost[0])
                return False, last_score, last_sample, attempts, "cost_cap"
            total_cost[0] += COST_PER_CALL
        attempts += 1
        log.info("[%s] attempt %d/%d (running cost $%.2f)", key, attempt, MAX_RETRIES, total_cost[0])
        resp = throttled_call(prompt)
        if not resp or not getattr(resp, "output", None) or not resp.output.results:
            log.warning("[%s] empty resp", key)
            continue
        url = resp.output.results[0].url
        new_p = TMP_DIR / f"new__{key.replace('/', '__')}"
        try:
            req = Request(url, headers={"User-Agent": "audit/1.0"})
            with urlopen(req, timeout=60) as r:
                new_p.write_bytes(r.read())
        except Exception as e:
            log.warning("[%s] dl failed: %s", key, e)
            continue
        score, sample = ocr_score(new_p)
        last_score, last_sample = score, sample
        log.info("[%s] new score=%d sample=%r", key, score, sample[:60])
        if score <= CHAR_OK_THRESHOLD:
            up = subprocess.run([OSSUTIL, "cp", "-f", "--meta", "Cache-Control:public, max-age=300, must-revalidate", str(new_p), f"oss://{OSS_BUCKET}/{key}"],
                                capture_output=True, text=True)
            if up.returncode == 0:
                log.info("[%s] uploaded text-free replacement (score=%d)", key, score)
                return True, score, sample, attempts, "ok"
            log.warning("[%s] upload failed: %s", key, up.stderr.strip())
            return False, score, sample, attempts, "upload_fail"
    return False, last_score, last_sample, attempts, "still_text"

def main():
    with MANIFEST_FILE.open() as f:
        manifest = json.load(f)
    # Pick covers
    candidates = sorted(
        [(k, v["score"]) for k, v in manifest.items() if v.get("score", 0) >= SCORE_THRESHOLD],
        key=lambda x: -x[1]
    )
    log.info("regen queue: %d covers (score >= %d)", len(candidates), SCORE_THRESHOLD)
    log.info("budget: $%.2f, per-call: $%.2f, max retries: %d", HARD_STOP_USD, COST_PER_CALL, MAX_RETRIES)

    save_lock = Lock()
    def save():
        with save_lock:
            MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    done = 0
    failed = 0
    skipped_cap = 0

    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {}
        for k, _ in candidates:
            # skip already done
            if manifest[k].get("regen_status") == "done":
                continue
            with cost_lock:
                if total_cost[0] + COST_PER_CALL > HARD_STOP_USD:
                    manifest[k]["regen_status"] = "skipped_cost_cap"
                    skipped_cap += 1
                    continue
            futs[ex.submit(regen_one, k)] = k
        for fut in as_completed(futs):
            k = futs[fut]
            ok, score, sample, attempts, reason = fut.result()
            manifest[k]["regen_attempts"] = attempts
            manifest[k]["regen_new_score"] = score
            manifest[k]["regen_new_sample"] = sample
            manifest[k]["regen_reason"] = reason
            manifest[k]["regen_status"] = "done" if ok else "failed"
            if ok:
                done += 1
            else:
                failed += 1
            save()
    save()

    log.info("=== regen summary ===")
    log.info("regen successful: %d", done)
    log.info("regen failed:     %d", failed)
    log.info("skipped (cap):    %d", skipped_cap)
    log.info("total cost:       $%.2f", total_cost[0])

if __name__ == "__main__":
    main()
