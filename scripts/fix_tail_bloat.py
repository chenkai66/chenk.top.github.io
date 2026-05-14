#!/usr/bin/env python3
"""Auto-fix tail-bloat: move anchor block (What's next / Summary / 接下来 / 总结)
to end of content, before any trailer (References / FAQ / 参考文献 / 常见问题)."""
import re, glob, sys

# Anchor patterns (last H2 at end is the article wrap-up bridge)
ANCHOR_EN = re.compile(r"^## (.*(?:What'?s [Nn]ext|Where to go from here|[Ss]ummary|[Cc]onclusion).*)$")
ANCHOR_ZH = re.compile(r"^## (.*(?:接下来|下一步|总结|小结|结语|结论).*)$")
TRAILER_EN = re.compile(r"^## (References|Summary|FAQ|Frequently Asked|Acknowled|Conclusion|Bibliography|Appendix|Exercises|Further [Rr]eading|Practice|Resources|Notes|Glossary)")
TRAILER_ZH = re.compile(r"^## (参考文献|参考资料|致谢|常见问题|FAQ|总结|小结|附录|结语|结论|练习题|延伸阅读|深入阅读|进一步阅读|资源|备注|术语表)")
H2 = re.compile(r"^## ")


def fix_article(path, lang):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    anchor_re = ANCHOR_EN if lang == "en" else ANCHOR_ZH
    trailer_re = TRAILER_EN if lang == "en" else TRAILER_ZH

    # Find all H2 line indices
    h2_indices = [i for i, l in enumerate(lines) if H2.match(l)]
    if not h2_indices:
        return False

    # Find LAST anchor H2
    anchor_idx = -1
    for i in h2_indices:
        if anchor_re.match(lines[i]):
            anchor_idx = i
    if anchor_idx == -1:
        return False  # no anchor, nothing to fix

    # Check if there are non-trailer H2s AFTER the anchor
    h2s_after = [i for i in h2_indices if i > anchor_idx]
    non_trailer_after = [i for i in h2s_after if not trailer_re.match(lines[i])]
    if not non_trailer_after:
        return False  # no tail-bloat

    # Find the anchor block end: from anchor_idx to next H2 (exclusive)
    next_h2_after_anchor = next((i for i in h2_indices if i > anchor_idx), len(lines))
    anchor_block = lines[anchor_idx:next_h2_after_anchor]

    # Find insertion point: walk backwards from end of file, find the FIRST non-trailer
    # H2; insert anchor block right after that one (= right before the final trailer block).
    h2s_after_anchor = [i for i in h2_indices if i > anchor_idx]
    last_non_trailer = -1
    for i in h2s_after_anchor:
        if not trailer_re.match(lines[i]):
            last_non_trailer = i
    if last_non_trailer == -1:
        return False  # all post-anchor H2s are trailers — already correct
    # Insertion point = the H2 right after last_non_trailer (which is the first trailer of the final block)
    # OR end of file if no such H2
    insert_at = next((i for i in h2_indices if i > last_non_trailer), len(lines))

    # Remove old anchor block, insert before insert_at
    new_lines = (lines[:anchor_idx] +
                 lines[next_h2_after_anchor:insert_at] +
                 anchor_block +
                 lines[insert_at:])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    return True


fixed = 0
for base in ["/root/chenk-hugo/content/en", "/root/chenk-hugo/content/zh"]:
    lang = "en" if "/en" in base else "zh"
    for path in glob.glob(f"{base}/*/*.md"):
        if "_index" in path: continue
        try:
            if fix_article(path, lang):
                rel = "/".join(path.split("/")[-3:])
                print(f"  {rel}")
                fixed += 1
        except Exception as e:
            print(f"  ERR {path}: {e}", file=sys.stderr)
print(f"\nReordered {fixed} articles with tail-bloat")
