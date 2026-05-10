#!/usr/bin/env python3
"""Post-process translated Chinese articles to fix common issues."""

import sys
import os
import re
import glob

def fix_article(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    original = text
    fixes = 0

    # Fix internal links: /en/ -> /zh/
    text, n = re.subn(r'\]\(/en/', '](/zh/', text)
    fixes += n

    # Fix stray English words that should be Chinese
    replacements = {
        ' half ': ' 一半 ',
        ' consistently ': ' 一致地 ',
        ' settled': ' 定型了',
        'not 是': '不是',
    }
    for eng, zh in replacements.items():
        if eng in text:
            text = text.replace(eng, zh)
            fixes += 1

    # Fix image alt text - keep as-is since it's not user-visible
    # Fix double newlines (more than 3 in a row)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    if text != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Fixed {fixes} issues in {filepath}")
    else:
        print(f"  No fixes needed in {filepath}")

    return fixes

def main():
    if len(sys.argv) < 2:
        print("Usage: postprocess.py <directory_or_file>")
        sys.exit(1)

    target = sys.argv[1]
    total_fixes = 0

    if os.path.isfile(target):
        total_fixes += fix_article(target)
    elif os.path.isdir(target):
        for f in sorted(glob.glob(os.path.join(target, "[0-9]*.md"))):
            total_fixes += fix_article(f)
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    print(f"\nTotal fixes: {total_fixes}")

if __name__ == "__main__":
    main()
