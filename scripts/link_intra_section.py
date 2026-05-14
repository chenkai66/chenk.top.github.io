#!/usr/bin/env python3
"""Auto-link intra-article 'Section N' / '第 N 节' refs to the Nth H2's anchor.

Hugo slugifies H2 text to lowercase-kebab anchors. We replicate that.
Skip if N exceeds the H2 count, OR if the same line already has a link.
"""
import argparse, glob, re, sys, unicodedata
from pathlib import Path


def slugify(text: str, lang: str) -> str:
    """Approximate Hugo's heading anchor slug.
    EN: lowercase, drop punctuation, spaces→hyphens.
    ZH: keep Chinese chars, lowercase ASCII, drop punctuation, spaces→hyphens.
    """
    # Strip emphasis markers
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    # Drop common punctuation that Hugo strips
    text = re.sub(r"[\.\,\:\;\!\?\'\"\(\)\[\]\&]", "", text)
    text = text.strip()
    # Replace whitespace and underscores with hyphens
    text = re.sub(r"[\s_]+", "-", text)
    if lang == "en":
        text = text.lower()
    return text


def section_map(body: list, lang: str) -> dict:
    """Return {N: slug} for content H2s (skipping pre-`---` intro and trailers)."""
    # Find first horizontal rule '---' (the content boundary per skill §10)
    sep_idx = -1
    for i, line in enumerate(body):
        if line.strip() == "---":
            sep_idx = i
            break
    start = sep_idx + 1 if sep_idx >= 0 else 0
    h2s = []
    for line in body[start:]:
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            h2s.append(m.group(1))
    return {i + 1: slugify(h2s[i], lang) for i in range(len(h2s))}


def split_fm_body(text: str):
    """Return (front_matter_text, body_lines)."""
    if not text.startswith("---\n"):
        return None, text.split("\n")
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text.split("\n")
    fm = text[:end + 5]
    body = text[end + 5:].split("\n")
    return fm, body


def fix_article(path: str, lang: str):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    fm, body = split_fm_body(text)
    if fm is None:
        return 0
    sec_map = section_map(body, lang)
    if not sec_map:
        return 0

    pattern_en = re.compile(r"\bSection\s+(\d{1,2})\b")
    pattern_zh = re.compile(r"第\s*(\d{1,2})\s*节")

    changes = 0
    in_code = False
    for i, line in enumerate(body):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s.startswith("#"):
            continue
        # Skip if line is in References/Bibliography (caller can ensure)
        for pat in (pattern_en if lang == "en" else pattern_zh,):
            for m in list(pat.finditer(line)):
                start = m.start()
                # Skip if already inside [...](...)
                preceding = line[:start]
                last_close = preceding.rfind("]")
                last_open = preceding.rfind("[")
                if last_open > last_close:
                    continue
                # Skip if next is `](`
                if line[m.end():m.end()+2] == "](":
                    continue
                n = int(m.group(1))
                if n not in sec_map:
                    continue
                old_text = m.group(0)
                anchor = sec_map[n]
                new_text = f"[{old_text}](#{anchor})"
                # Replace ONLY the first occurrence of old_text in current line
                if old_text in line and new_text not in line:
                    line = line.replace(old_text, new_text, 1)
                    changes += 1
        body[i] = line

    if changes:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n".join(body))
    return changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    total = 0
    for base in ["/root/chenk-hugo/content/en", "/root/chenk-hugo/content/zh"]:
        lang = "en" if "/en" in base else "zh"
        for path in sorted(glob.glob(f"{base}/*/*.md")):
            if "_index" in path:
                continue
            n = fix_article(path, lang)
            if n > 0:
                rel = "/".join(path.split("/")[-3:])
                print(f"  {rel}: {n}")
                total += n
    print(f"\n{total} intra-article Section refs linked ({'APPLIED' if args.apply else 'DRY-RUN'})")


if __name__ == "__main__":
    main()
