#!/usr/bin/env python3
"""Bulk-fix \\| in markdown table cells across all content."""
import re, glob

dirs = ["/root/chenk-hugo/content/en", "/root/chenk-hugo/content/zh"]
total_files = 0
total_subs = 0
for base in dirs:
    for path in glob.glob(f"{base}/*/*.md"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new_lines = []
        n_subs = 0
        for line in text.split("\n"):
            # Identify table rows: starts with optional whitespace then `|`, has at least 2 more `|`
            if re.match(r"^\s*\|", line) and line.count("|") >= 3:
                # Inside math $...$, replace \| with \mid
                def fix(m):
                    return re.sub(r"\\\|", r"\\mid", m.group(0))
                new = re.sub(r"\$[^$]*\$", fix, line)
                if new != line:
                    n_subs += 1
                new_lines.append(new)
            else:
                new_lines.append(line)
        if n_subs > 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
            total_files += 1
            total_subs += n_subs
            print(f"  {path.split('/')[-2]}/{path.split('/')[-1]}: {n_subs} fixes")
print(f"\n{total_subs} table-cell fixes across {total_files} files")
