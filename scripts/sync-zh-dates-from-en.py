#!/usr/bin/env python3
"""
Sync date field from EN articles to their ZH counterparts using translationKey.
EN is the single source of truth for publish dates.
Run before every Hugo build (deploy.sh hooks this).
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def scan(lang):
    by_key = {}
    base = os.path.join(ROOT, "content", lang)
    for r, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith(".md"): continue
            if fn == "_index.md": continue
            p = os.path.join(r, fn)
            with open(p, encoding="utf-8") as f: head = f.read(3000)
            tk = re.search(r"^translationKey:\s*[\"]?([^\"\n]+)", head, re.M)
            d = re.search(r"^date:\s*[\"]?(\d{4}-\d{2}-\d{2})", head, re.M)
            if not tk: continue
            by_key[tk.group(1).strip().strip("\"")] = (p, d.group(1) if d else None)
    return by_key

en = scan("en")
zh = scan("zh")
synced, missing_en_date, no_zh_pair = 0, [], []
for tk, (en_path, en_date) in en.items():
    if not en_date:
        missing_en_date.append((tk, en_path))
        continue
    if tk not in zh:
        no_zh_pair.append((tk, en_path))
        continue
    zh_path, zh_date = zh[tk]
    if zh_date == en_date: continue
    with open(zh_path, encoding="utf-8") as f: text = f.read()
    new_text, n = re.subn(
        r"^(date:\s*[\"]?)\d{4}-\d{2}-\d{2}",
        lambda m: m.group(1) + en_date,
        text, count=1, flags=re.M
    )
    if n == 0:
        # No date field at all in front-matter — inject after opening ---
        new_text = re.sub(r"^---\n", f"---\ndate: {en_date}\n", text, count=1)
    if new_text != text:
        with open(zh_path, "w", encoding="utf-8") as f: f.write(new_text)
        synced += 1
        print("  "+tk.ljust(55)+" "+(zh_date or "(none)")+" -> "+en_date)

print()
print(f"Synced: {synced} ZH files")
print(f"EN articles missing date: {len(missing_en_date)}")
for tk, p in missing_en_date: print(f"  ! {tk}  {p}")
print(f"EN articles with no ZH pair: {len(no_zh_pair)}")
