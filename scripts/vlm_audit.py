#!/usr/bin/env python3
"""VLM-driven audit of OSS-hosted figures on chenk.top.

Scans all PNG figures via qwen-vl-max, classifies as broken/clean,
records to /tmp/vlm_audit/results.jsonl with checkpointing.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env
ENV_PATH = Path("/root/chenk-hugo/.env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

AUDIT_DIR = Path("/tmp/vlm_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ALL_FIGS = AUDIT_DIR / "all_figs.txt"
RESULTS = AUDIT_DIR / "results.jsonl"
LOG = AUDIT_DIR / "scan.log"

VLM_PROMPT = (
    "Is this technical diagram free of bugs? Reply ONLY in JSON:\n"
    '{"broken": true/false, "severity": "high|medium|low|none", "issue": "<one short sentence>"}\n'
    "Look specifically for: text overlapping other text or shapes; text clipped by box edges; "
    "labels colliding (t-SNE/scatter); garbled/unreadable characters from positioning bugs.\n"
    "IGNORE: math notation, foreign-language characters, intentional artistic style. "
    "Only flag REAL rendering bugs."
)

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-vl-max-latest"
MAX_WORKERS = 8


def already_done() -> set:
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add(d["url"])
            except Exception:
                pass
    return done


def call_vlm(url: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ],
            timeout=60,
        )
        text = resp.choices[0].message.content.strip()
        # Strip code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        # find first { ... }
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            data = json.loads(text[i : j + 1])
        else:
            data = {"broken": False, "severity": "none", "issue": "PARSE_FAIL: " + text[:120]}
        return {"url": url, **data}
    except Exception as e:
        return {"url": url, "broken": False, "severity": "none", "issue": f"VLM_ERROR: {e}"}


def main():
    urls = [u.strip() for u in ALL_FIGS.read_text().splitlines() if u.strip()]
    done = already_done()
    pending = [u for u in urls if u not in done]
    total = len(urls)
    print(f"[init] total={total} done={len(done)} pending={len(pending)}", flush=True)

    if not pending:
        print("[done] no pending", flush=True)
        return

    written = 0
    t0 = time.time()
    fout = RESULTS.open("a")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(call_vlm, u): u for u in pending}
        for fut in as_completed(futs):
            res = fut.result()
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 50 == 0:
                rate = written / (time.time() - t0)
                eta = (len(pending) - written) / max(rate, 0.01)
                print(
                    f"[progress] {written}/{len(pending)} rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
    fout.close()
    print(f"[done] wrote {written} new results in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
