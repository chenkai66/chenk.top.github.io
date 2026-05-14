#!/usr/bin/env python3
"""Auto-link cross-chapter "Part N" / "Chapter N" / "第 N 章" / "第 N 部分" refs.

Rules (conservative):
1. Only link when target chapter exists in the same series.
2. Skip "Section N" / "第 N 节" (those are intra-article anchors, harder to map).
3. Skip if already inside a markdown link [text](url).
4. Skip if line starts with #/code/blockquote.
5. Print proposed changes per file; apply only when --apply passed.

Usage:
  python3 link_cross_chapter.py <series> [--apply]
"""
import argparse
import glob
import re
import sys
from pathlib import Path


def chapter_map(content_dir: str, lang: str) -> dict:
    """Return {N: slug} for the series in lang.
    Tries filename NN-prefix first, falls back to front-matter `series_order:`.
    """
    out = {}
    for p in sorted(glob.glob(f"{content_dir}/*.md")):
        bn = Path(p).stem
        if bn == "_index":
            continue
        # Try filename NN-prefix
        m = re.match(r"(\d{1,2})[-_]", bn)
        if m:
            out[int(m.group(1))] = bn
            continue
        # Fall back to front matter
        try:
            with open(p, encoding="utf-8") as f:
                head = f.read(2048)
            mo = re.search(r"^series_order:\s*(\d+)", head, re.M)
            if mo:
                out[int(mo.group(1))] = bn
        except Exception:
            pass
    return out


def link_url(lang: str, series: str, slug: str) -> str:
    # Hugo lowercases slugs in URLs; keep original case for ZH (Hugo passes through)
    if lang == "en":
        return f"/{lang}/{series}/{slug.lower()}/"
    else:
        return f"/{lang}/{series}/{slug}/"


def find_unlinked_refs(content: str, lang: str):
    """Return list of (line_idx, match_obj, replacement_text)."""
    findings = []
    # Patterns:
    #  EN: "Part N" / "Chapter N"  (followed by . , ) - or end of line)
    #  ZH: "第 N 章" / "第N章" / "第 N 部分" (note: spaces optional)
    if lang == "en":
        # Match "Part N" / "Chapter N" with N=1..20, where the surrounding text is plain
        # Skip if preceded by `[` or followed by `](`
        pattern = re.compile(r"\b(Part|Chapter)\s+(\d{1,2})\b")
    else:
        pattern = re.compile(r"第 ?(\d{1,2})\s*(章|部分)")
    in_code = False
    in_table = False
    for i, line in enumerate(content.split("\n")):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s.startswith("#"):
            continue
        # Allow tables — markdown link [text](url) is fine inside table cells
        # Allow blockquotes — series-nav footers often live there
        for m in pattern.finditer(line):
            # Skip if already inside [...](...)
            start = m.start()
            # Look backward for unclosed `[`
            preceding = line[:start]
            # If there's a `[` after the most recent `]`, we're inside link text
            last_close = preceding.rfind("]")
            last_open = preceding.rfind("[")
            if last_open > last_close:
                continue
            # Skip if next chars are `](` (the match is the visible text of an existing link)
            after = line[m.end():m.end()+2]
            if after == "](":
                continue
            # Capture the part number
            if lang == "en":
                n = int(m.group(2))
            else:
                n = int(m.group(1))
            findings.append((i, m, n))
    return findings


def process_series(series: str, apply: bool):
    base = Path("/root/chenk-hugo/content")
    en_dir = base / "en" / series
    zh_dir = base / "zh" / series
    if not en_dir.exists() and not zh_dir.exists():
        print(f"series not found: {series}", file=sys.stderr)
        return 1

    en_chapters = chapter_map(str(en_dir), "en") if en_dir.exists() else {}
    zh_chapters = chapter_map(str(zh_dir), "zh") if zh_dir.exists() else {}

    total_changes = 0
    for path in sorted(en_dir.glob("*.md")) if en_dir.exists() else []:
        if path.stem == "_index":
            continue
        # find the article's own chapter number to skip self-references
        m = re.match(r"(\d{1,2})", path.stem)
        own_n = int(m.group(1)) if m else None

        with open(path) as f:
            content = f.read()
        findings = find_unlinked_refs(content, "en")
        if not findings:
            continue
        new_lines = content.split("\n")
        line_changes = {}  # i → new_line
        for i, m, n in findings:
            if n == own_n:
                continue  # self-ref
            if n not in en_chapters:
                continue  # target doesn't exist
            url = link_url("en", series, en_chapters[n])
            old_text = m.group(0)
            new_text = f"[{old_text}]({url})"
            cur = line_changes.get(i, new_lines[i])
            if old_text in cur and new_text not in cur:
                # Replace only the FIRST occurrence to avoid double-replacing
                cur_new = cur.replace(old_text, new_text, 1)
                line_changes[i] = cur_new

        if line_changes:
            for i, new_line in line_changes.items():
                old = new_lines[i]
                new_lines[i] = new_line
                print(f"  EN {path.name}:{i+1}")
                print(f"    -- {old.strip()[:100]}")
                print(f"    ++ {new_line.strip()[:100]}")
                total_changes += 1
            if apply:
                with open(path, "w") as f:
                    f.write("\n".join(new_lines))

    for path in sorted(zh_dir.glob("*.md")) if zh_dir.exists() else []:
        if path.stem == "_index":
            continue
        m = re.match(r"(\d{1,2})", path.stem)
        own_n = int(m.group(1)) if m else None

        with open(path) as f:
            content = f.read()
        findings = find_unlinked_refs(content, "zh")
        if not findings:
            continue
        new_lines = content.split("\n")
        line_changes = {}
        for i, m, n in findings:
            if n == own_n:
                continue
            if n not in zh_chapters:
                continue
            url = link_url("zh", series, zh_chapters[n])
            old_text = m.group(0)
            new_text = f"[{old_text}]({url})"
            cur = line_changes.get(i, new_lines[i])
            if old_text in cur and new_text not in cur:
                cur_new = cur.replace(old_text, new_text, 1)
                line_changes[i] = cur_new

        if line_changes:
            for i, new_line in line_changes.items():
                old = new_lines[i]
                new_lines[i] = new_line
                print(f"  ZH {path.name}:{i+1}")
                print(f"    -- {old.strip()[:100]}")
                print(f"    ++ {new_line.strip()[:100]}")
                total_changes += 1
            if apply:
                with open(path, "w") as f:
                    f.write("\n".join(new_lines))

    print(f"\n{series}: {total_changes} changes ({'APPLIED' if apply else 'DRY-RUN'})")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("series")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    sys.exit(process_series(args.series, args.apply))
