#!/usr/bin/env python3
"""topic_to_outline.py — turn a free-form topic into a structured series outline.

Usage:  python3 topic_to_outline.py "<topic>" <n_chapters>
Output: JSON to stdout (and /tmp/outline.json)

Calls qwen3-max via the ecs-run keys pool. Validates the output schema
before saving — rejects malformed JSON or chapter-count mismatch.
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

POOL_PATHS = ["/usr/local/lib/ecs-run", os.path.expanduser("~/.local/share/ecs-run")]
for p in POOL_PATHS:
    sys.path.insert(0, p)
try:
    from ecs_run_keys import KeyPool  # noqa: E402
except ImportError:
    print("ERROR: ecs_run_keys loader not found. Run `ecs-run keys sync` first.", file=sys.stderr)
    sys.exit(2)

SYSTEM = """你是一名资深技术博客架构师。给定一个题材，规划 N 章的 series。

强制输出 JSON（不要 markdown fence、不要解释）：
{
  "series_slug": "kebab-case-slug",
  "series_title_en": "...",
  "series_title_zh": "...",
  "hue": 1-4,
  "chapters": [
    {
      "n": 1,
      "slug_en": "01-foundations",
      "slug_zh": "01-基础",
      "title_en": "...",
      "title_zh": "...",
      "h2_sequence": ["Intro hook", "What it is", "Hands-on", "Edge cases", "What's next"],
      "code_focus": "what concrete code reader writes",
      "depends_on_chapters": []
    },
    ...
  ]
}

规则:
- 4 ≤ N ≤ 12
- 每章独立可读（开头 1 段交代背景）
- "What's next" 必须是每章最后一个 H2
- 章节之间要递进，不要并列堆砌
- 每章 H2 序列 4-7 个
- slug_en 必须是 NN-kebab-case（NN 是两位序号）
- depends_on_chapters 只能引用更早的 chapter
"""


def call_qwen(user_prompt, model="qwen3-max"):
    pool = KeyPool.load()
    last_err = None
    for attempt in range(3):
        key, base_url = pool.next(provider="dashscope", model=model)
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 8000,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:
            last_err = str(e)
        pool.report_failure(key)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"all retries failed; last: {last_err}")


def validate_outline(d, n_expected):
    """Check schema. Raise AssertionError with hint if anything wrong."""
    for k in ["series_slug", "series_title_en", "series_title_zh", "hue", "chapters"]:
        assert k in d, f"missing top-level key: {k}"
    assert re.match(r"^[a-z0-9-]+$", d["series_slug"]), "series_slug not kebab-case"
    assert d["hue"] in (1, 2, 3, 4), "hue must be 1..4"
    assert len(d["chapters"]) == n_expected, f"got {len(d['chapters'])} chapters, want {n_expected}"
    seen_slugs = set()
    for i, ch in enumerate(d["chapters"], 1):
        for k in ["n", "slug_en", "slug_zh", "title_en", "title_zh", "h2_sequence", "code_focus"]:
            assert k in ch, f"chapter {i} missing {k}"
        assert ch["n"] == i, f"chapter {i} has n={ch['n']} (want {i})"
        assert re.match(r"^\d{2}-[a-z0-9-]+$", ch["slug_en"]), f"chapter {i} slug_en bad: {ch['slug_en']}"
        assert ch["slug_en"] not in seen_slugs, f"duplicate slug_en: {ch['slug_en']}"
        seen_slugs.add(ch["slug_en"])
        h2s = ch["h2_sequence"]
        assert isinstance(h2s, list) and 4 <= len(h2s) <= 7, f"chapter {i} h2_sequence len={len(h2s)}"
        assert "What's next" in h2s[-1] or "Where to go" in h2s[-1], \
            f"chapter {i} last H2 not 'What's next': {h2s[-1]}"


def main():
    if len(sys.argv) != 3:
        print("usage: topic_to_outline.py '<topic>' <n_chapters>", file=sys.stderr)
        sys.exit(1)
    topic = sys.argv[1]
    n = int(sys.argv[2])
    assert 4 <= n <= 12, f"n must be 4..12, got {n}"
    user = f"题材: {topic}\nN: {n}\n请输出 outline JSON。"

    raw = call_qwen(user)
    # Strip optional ```json fences
    raw = re.sub(r"^\s*```(?:json)?\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)
    d = json.loads(raw)
    validate_outline(d, n)

    out_path = "/tmp/outline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"\n[saved to {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
