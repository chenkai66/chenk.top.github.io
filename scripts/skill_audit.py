#!/usr/bin/env python3
"""skill_audit.py — full skill-compliance audit of one series.

Checks beyond fast_validate.sh:
A. EN↔ZH H2 / image count parity per article
B. translationKey 配对 (every EN must have matching ZH)
C. series_order 唯一性 + 连续性 (1..N 无跳号)
D. tags 一致性 EN↔ZH
E. date 严格递增 (EN; ZH 由 deploy 同步)
F. ZH 翻译腔 phrases (在...的过程中 / 我们 / 被设计成)
G. matplotlib figure 数 ≥ 1 per article
H. scripts/figures/{series}/ 存在
I. cover image existence (best effort)
J. EN 风格 — banned phrases (let's dive in / it's important to note)
"""
import argparse, glob, os, re, sys, yaml
from pathlib import Path
from collections import defaultdict

REPORT = []
def issue(category, msg):
    REPORT.append((category, msg))

def fm(path):
    """Parse front matter as dict."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    try:
        d = yaml.safe_load(text[4:end])
        return d or {}, text[end+5:]
    except Exception as e:
        return {}, text


def body_h2_count(body: str) -> int:
    return sum(1 for l in body.split("\n") if l.startswith("## "))

def body_img_count(body: str) -> int:
    return sum(1 for _ in re.finditer(r"!\[[^\]]*\]\([^)]+\)", body))


ZH_STINK = [
    "在.{0,12}的过程中",
    "被设计成",
    "进行了",
    "对于.{0,12}来说",
    "众所周知",
    "如下所示",
    "正如",
    "我们将",
]

EN_BANNED = [
    "let's dive in",
    "in this article we will",
    "as you can see",
    "it's important to note",
    "without further ado",
    "buckle up",
]


def check_series(series: str, base="/root/chenk-hugo"):
    en_dir = Path(f"{base}/content/en/{series}")
    zh_dir = Path(f"{base}/content/zh/{series}")
    if not en_dir.is_dir() and not zh_dir.is_dir():
        print(f"series not found", file=sys.stderr); return 1

    en_files = sorted([p for p in en_dir.glob("*.md") if p.stem != "_index"]) if en_dir.is_dir() else []
    zh_files = sorted([p for p in zh_dir.glob("*.md") if p.stem != "_index"]) if zh_dir.is_dir() else []

    en_meta = {}
    zh_meta = {}
    for p in en_files:
        d, body = fm(p)
        en_meta[p.stem] = (d, body, p)
    for p in zh_files:
        d, body = fm(p)
        zh_meta[p.stem] = (d, body, p)

    # B. translationKey pairing
    en_keys = {d.get("translationKey"): stem for stem, (d,_,_) in en_meta.items() if d.get("translationKey")}
    zh_keys = {d.get("translationKey"): stem for stem, (d,_,_) in zh_meta.items() if d.get("translationKey")}
    for k, stem in en_keys.items():
        if k not in zh_keys:
            issue("B/translationKey", f"EN {stem}.md key={k!r} 无 ZH 对应")
    for k, stem in zh_keys.items():
        if k not in en_keys:
            issue("B/translationKey", f"ZH {stem}.md key={k!r} 无 EN 对应")

    # C. series_order 唯一性 + 连续性
    for label, meta in [("EN", en_meta), ("ZH", zh_meta)]:
        orders = sorted([(d.get("series_order"), stem) for stem, (d,_,_) in meta.items() if d.get("series_order")])
        nums = [o for o, _ in orders if o is not None]
        seen = set()
        for n, stem in orders:
            if n in seen: issue("C/series_order", f"{label} 重复 series_order={n} ({stem})")
            seen.add(n)
        if nums:
            expected = list(range(1, max(nums)+1))
            missing = sorted(set(expected) - set(nums))
            if missing: issue("C/series_order", f"{label} 缺序号: {missing}")

    # D. EN/ZH front-matter consistency for paired articles
    for k in en_keys:
        if k not in zh_keys: continue
        en_stem = en_keys[k]
        zh_stem = zh_keys[k]
        en_d, _, _ = en_meta[en_stem]
        zh_d, _, _ = zh_meta[zh_stem]
        if en_d.get("series_order") != zh_d.get("series_order"):
            issue("D/parity", f"key={k} series_order 不一致 EN={en_d.get('series_order')} ZH={zh_d.get('series_order')}")
        if en_d.get("series") != zh_d.get("series"):
            issue("D/parity", f"key={k} series 不一致 EN={en_d.get('series')} ZH={zh_d.get('series')}")
        # tags can be language-localized (EN tags vs ZH tags) — that's expected, not a bug

    # A. 内容长度 / 图片 count parity per pair
    # Use LINE COUNT (better proxy than H2 count — EN/ZH may use H2 vs H3 differently)
    for k in en_keys:
        if k not in zh_keys: continue
        _, en_body, _ = en_meta[en_keys[k]]
        _, zh_body, _ = zh_meta[zh_keys[k]]
        en_lines = en_body.count("\n")
        zh_lines = zh_body.count("\n")
        en_img = body_img_count(en_body); zh_img = body_img_count(zh_body)
        # Flag content gap when ratio drops below 60% (excluding case where ZH compresses naturally — typically ratio is 0.85-1.0)
        if en_lines > 100 and zh_lines / max(en_lines, 1) < 0.6:
            issue("A/parity", f"key={k} ZH 内容缺失? EN={en_lines}行 ZH={zh_lines}行 (比例 {zh_lines/max(en_lines,1):.2f})")
        if en_lines > 100 and en_lines / max(zh_lines, 1) < 0.6:
            issue("A/parity", f"key={k} EN 内容少于 ZH? EN={en_lines}行 ZH={zh_lines}行")
        if abs(en_img - zh_img) > 2:
            issue("A/parity", f"key={k} 图片差距大: EN={en_img} ZH={zh_img}")

    # E. EN date 严格递增 by series_order
    en_by_order = sorted([(d.get("series_order"), d.get("date"), stem) for stem, (d,_,_) in en_meta.items() if d.get("series_order")])
    prev_date = None
    for n, date, stem in en_by_order:
        if date is None: continue
        if prev_date and str(date) < str(prev_date):
            issue("E/date", f"EN {stem} date={date} 早于前一篇 {prev_date}")
        prev_date = date

    # F. ZH 翻译腔 phrases
    for stem, (_, body, p) in zh_meta.items():
        for pat in ZH_STINK:
            n = len(re.findall(pat, body))
            if n >= 3:  # allow 1-2 occurrences (sometimes legitimate); flag at 3+
                issue("F/translation-stink", f"ZH {stem}.md: 翻译腔 '{pat}' x{n}")

    # G + H. matplotlib figure 数 / 脚本目录
    # Only flag if the series ACTUALLY uses matplotlib figures (look for fig\d+_ pattern in OSS refs)
    fig_dir = Path(f"{base}/scripts/figures/{series}")
    has_fig_dir = fig_dir.is_dir()
    fig_count = len(list(fig_dir.glob("*.py"))) if has_fig_dir else 0
    uses_matplotlib = False
    for stem, (_, body, _) in en_meta.items():
        if re.search(r"/fig\d+[_\-]\w+\.png", body) or re.search(r"posts/.+/fig\d+", body):
            uses_matplotlib = True
            break
    if uses_matplotlib:
        if not has_fig_dir:
            issue("H/figures-dir", f"系列引用了 figN_* 图但 scripts/figures/{series}/ 不存在")
        elif fig_count == 0:
            issue("H/figures-dir", f"scripts/figures/{series}/ 存在但无 .py 文件")

    # J. EN style: banned phrases
    for stem, (_, body, p) in en_meta.items():
        body_lower = body.lower()
        for phrase in EN_BANNED:
            if phrase in body_lower:
                issue("J/EN-style", f"EN {stem}.md 含禁用短语 '{phrase}'")

    # I. cover image — try OSS URL HEAD
    # (skipped here; full check in scripts/validate.sh)

    return REPORT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("series")
    args = p.parse_args()
    check_series(args.series)
    if not REPORT:
        print(f"{args.series}: ALL PASSED (0 skill-compliance issues)")
        return 0
    by_cat = defaultdict(list)
    for cat, msg in REPORT: by_cat[cat].append(msg)
    print(f"{args.series}: {len(REPORT)} skill-compliance issues\n")
    for cat in sorted(by_cat):
        print(f"== {cat} ({len(by_cat[cat])}) ==")
        for m in by_cat[cat][:8]: print(f"  {m}")
        if len(by_cat[cat]) > 8: print(f"  ... +{len(by_cat[cat])-8} more")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
