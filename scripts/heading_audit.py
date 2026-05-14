#!/usr/bin/env python3
"""Scan all H2 across the blog. Find case/synonym variants of common section names.
Run with --apply to mass-rename to canonical form. Run with --survey to show
top un-mapped H2 occurrences (so we can extend the canonical map).
"""
import argparse, glob, re
from collections import Counter, defaultdict

H2 = re.compile(r"^##\s+(.+?)\s*$")

CANONICAL_EN = {
    "what's next":"What's next", "whats next":"What's next", "what next":"What's next",
    "next steps":"What's next", "what is next":"What's next",
    "where to go next":"Where to go from here",
    "where to go from here":"Where to go from here",
    "what you'll learn":"What You Will Learn",
    "what you will learn":"What You Will Learn",
    "learning objectives":"What You Will Learn",
    "in this chapter":"What You Will Learn",
    "prerequisites":"Prerequisites", "prereqs":"Prerequisites",
    "before you start":"Prerequisites", "before we begin":"Prerequisites",
    "frequently asked questions":"FAQ", "common questions":"FAQ",
    "q&a":"FAQ", "questions":"FAQ", "faqs":"FAQ",
    "references":"References", "bibliography":"References",
    "further reading":"Further Reading", "further references":"Further Reading",
    "additional reading":"Further Reading",
    "summary":"Summary", "tl;dr":"Summary", "in summary":"Summary",
    "key takeaways":"Summary", "wrap-up":"Summary", "wrap up":"Summary",
    "conclusion":"Conclusion",
}

CANONICAL_ZH = {
    "接下来":"接下来", "下一步":"接下来", "下一篇":"接下来", "后续":"接下来", "下一章":"接下来",
    "你将学到什么":"你将学到什么", "本文你将学到":"你将学到什么",
    "本章学习目标":"你将学到什么", "本文内容概览":"你将学到什么", "本文将讲什么":"你将学到什么",
    "前置知识":"前置知识", "预备知识":"前置知识", "先决条件":"前置知识", "前提条件":"前置知识",
    "常见问题":"常见问题", "常见疑问":"常见问题", "常见问题与解答":"常见问题",
    "实操问答":"常见问题", "问答":"常见问题",
    "参考文献":"参考文献", "参考资料":"参考文献", "参考":"参考文献",
    "延伸阅读":"延伸阅读", "深入阅读":"延伸阅读", "进一步阅读":"延伸阅读",
    "总结":"总结", "小结":"总结", "总览":"总结", "本章小结":"总结",
    "全文小结":"总结", "结语":"总结",
}


def normalize(text, lang):
    if lang == "en":
        return text.lower().strip().rstrip(":：。")
    return text.strip().rstrip(":：。")


def find_canonical(text, lang):
    norm = normalize(text, lang)
    table = CANONICAL_EN if lang == "en" else CANONICAL_ZH
    return table.get(norm)


def scan_all():
    results = []
    for base in ["/root/chenk-hugo/content/en", "/root/chenk-hugo/content/zh"]:
        lang = "en" if "/en/" in base + "/" else "zh"
        for path in sorted(glob.glob(f"{base}/*/*.md")):
            if "_index" in path: continue
            series = path.split("/")[-2]
            with open(path, encoding="utf-8") as f:
                lines = f.read().split("\n")
            in_code = False; fm_count = 0
            for i, line in enumerate(lines, 1):
                if line.strip() == "---":
                    fm_count += 1; continue
                if fm_count < 2: continue
                if line.strip().startswith("```"):
                    in_code = not in_code; continue
                if in_code: continue
                m = H2.match(line)
                if not m: continue
                text = m.group(1)
                canon = find_canonical(text, lang)
                results.append((lang, series, path, i, text, canon))
    return results


def report_variants(results):
    groups = defaultdict(Counter)
    for lang, _, _, _, text, canon in results:
        if canon and text != canon:
            groups[(lang, canon)][text] += 1
    print("=== Heading variants needing standardization ===\n")
    total = 0
    for (lang, canon), variants in sorted(groups.items()):
        if not variants: continue
        print(f"[{lang}] '{canon}':")
        for v, n in variants.most_common(8):
            print(f"    '{v}'  x{n}")
            total += n
        print()
    print(f"Total renames possible: {total}")


def apply_renames(results):
    by_path = defaultdict(list)
    for lang, _, path, lineno, text, canon in results:
        if canon and text != canon:
            by_path[path].append((lineno, text, canon))
    n_files = 0; n_renames = 0
    for path, changes in by_path.items():
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        for lineno, text, canon in changes:
            old = lines[lineno - 1]
            new = old.replace(f"## {text}", f"## {canon}", 1)
            if new != old:
                lines[lineno - 1] = new
                n_renames += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_files += 1
    print(f"Renamed {n_renames} headings across {n_files} files")


def report_survey(results):
    en_counts = Counter(); zh_counts = Counter()
    for lang, _, _, _, text, canon in results:
        if not canon:
            (en_counts if lang == "en" else zh_counts)[text] += 1
    print("\n=== Top un-mapped H2 (potential additions to canonical map) ===")
    print("\n[EN]")
    for t, n in en_counts.most_common(40):
        if n >= 3: print(f"   x{n:3d}  {t}")
    print("\n[ZH]")
    for t, n in zh_counts.most_common(40):
        if n >= 3: print(f"   x{n:3d}  {t}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--survey", action="store_true")
    args = p.parse_args()
    results = scan_all()
    print(f"Scanned {len(results)} H2 occurrences\n")
    if args.survey:
        report_survey(results)
    else:
        report_variants(results)
        if args.apply:
            apply_renames(results)
