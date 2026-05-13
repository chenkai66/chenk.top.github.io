#!/usr/bin/env python3
"""Auto-insert space between CJK and ASCII letter/digit (Pangu-style spacing)."""
import glob, re

def pangu_line(line):
    """Insert space at CJK<->ASCII boundary, except in masked regions."""
    # Mask: inline code, math, links, URLs
    placeholders = []
    def stash(m):
        placeholders.append(m.group(0))
        return f"\x00{len(placeholders)-1}\x00"
    masked = re.sub(r"`[^`]+`", stash, line)
    masked = re.sub(r"\$\$[^$]+\$\$", stash, masked)
    masked = re.sub(r"\$[^$]+\$", stash, masked)
    masked = re.sub(r"!\[[^\]]*\]\([^)]+\)", stash, masked)
    masked = re.sub(r"\[[^\]]*\]\([^)]+\)", stash, masked)
    masked = re.sub(r"https?://\S+", stash, masked)
    # Apply Pangu rule
    masked = re.sub(r"([一-鿿])([A-Za-z0-9])", r"\1 \2", masked)
    masked = re.sub(r"([A-Za-z0-9])([一-鿿])", r"\1 \2", masked)
    # Restore
    masked = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], masked)
    return masked


total_files = 0
total_fixes = 0
for path in glob.glob("/root/chenk-hugo/content/zh/**/*.md", recursive=True):
    if "_index" in path: continue
    with open(path) as f: c = f.read()
    parts = c.split("---", 2)
    if len(parts) < 3: continue
    fm_with_delim = "---" + parts[1] + "---"
    body = parts[2]
    in_code = False
    new_lines = []
    file_fixes = 0
    for line in body.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            new_lines.append(line); continue
        if in_code:
            new_lines.append(line); continue
        new_line = pangu_line(line)
        if new_line != line:
            file_fixes += new_line.count(" ") - line.count(" ")
        new_lines.append(new_line)
    new_body = "\n".join(new_lines)
    if file_fixes > 0:
        with open(path, "w") as f:
            f.write(fm_with_delim + new_body)
        total_files += 1
        total_fixes += file_fixes
        print(f"  +{file_fixes:3d}  {path.replace('/root/chenk-hugo/content/zh/', '')}")

print(f"\nTotal: {total_fixes} spaces inserted in {total_files} files")
