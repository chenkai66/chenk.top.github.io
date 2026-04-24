#!/usr/bin/env python3
"""Audit image references in all posts. Find broken links and fix paths."""

import re
from pathlib import Path

POSTS_DIR = Path(__file__).parent.parent / "source" / "_posts"

def audit_images():
    broken = []
    fixed = 0
    total_refs = 0

    for lang_dir in ['en', 'zh']:
        dir_path = POSTS_DIR / lang_dir
        if not dir_path.exists():
            continue

        for md_file in sorted(dir_path.glob('*.md')):
            content = md_file.read_text(encoding='utf-8')
            # Find all image references
            img_refs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)

            for alt, src in img_refs:
                total_refs += 1
                # Check if image exists
                if src.startswith('http'):
                    continue  # Skip external URLs

                # Decode URL-encoded paths
                from urllib.parse import unquote
                decoded_src = unquote(src)

                # Try relative to post
                post_stem = md_file.stem
                img_path = md_file.parent / decoded_src.lstrip('./')

                # Try relative to source root
                source_root = POSTS_DIR.parent
                img_path2 = source_root / decoded_src.lstrip('/')

                # Try in post asset folder
                img_path3 = md_file.parent / post_stem / Path(decoded_src).name

                found = False
                for p in [img_path, img_path2, img_path3]:
                    if p.exists():
                        found = True
                        break

                if not found:
                    broken.append({
                        'file': f'{lang_dir}/{md_file.name}',
                        'alt': alt,
                        'src': src[:80],
                    })

    print(f"Total image references: {total_refs}")
    print(f"Broken references: {len(broken)}")
    print(f"\nBroken images by file:")

    by_file = {}
    for b in broken:
        by_file.setdefault(b['file'], []).append(b['src'])

    for f, srcs in sorted(by_file.items()):
        print(f"\n  {f}: {len(srcs)} broken")
        for s in srcs[:3]:
            print(f"    - {s}")
        if len(srcs) > 3:
            print(f"    ... and {len(srcs)-3} more")

if __name__ == '__main__':
    audit_images()
