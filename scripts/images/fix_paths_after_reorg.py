#!/usr/bin/env python3
"""
After reorganizing files into series folders, fix image paths in markdown files.

Old reference: ![alt](./old-article-slug/fig1.png)
New location: source/_posts/en/series/01-slug/fig1.png

The image refs need to be updated to relative paths.
After reorganization, the markdown is at source/_posts/en/series/01-slug.md
and the asset folder is source/_posts/en/series/01-slug/
So the relative ref is ./01-slug/fig1.png
"""

import re
from pathlib import Path
from urllib.parse import unquote

REPO_DIR = Path(__file__).parent.parent.parent
POSTS_DIR = REPO_DIR / "source" / "_posts"


def fix_md_image_refs(md_file):
    """Update image refs in a markdown file to point to the new asset folder."""
    content = md_file.read_text(encoding='utf-8')
    new_stem = md_file.stem  # e.g. "01-fundamentals"

    def fix_ref(match):
        alt = match.group(1)
        src = match.group(2)
        if src.startswith('http'):
            return match.group(0)

        # Get just the filename
        filename = Path(unquote(src)).name
        if not filename or '.' not in filename:
            return match.group(0)

        # Check if image exists in new asset folder
        new_asset_dir = md_file.parent / new_stem
        target = new_asset_dir / filename
        if target.exists():
            return f'![{alt}](./{new_stem}/{filename})'

        # Otherwise leave it (will be fixed by separate audit)
        return match.group(0)

    new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_ref, content)
    if new_content != content:
        md_file.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    fixed = 0
    for md_file in POSTS_DIR.rglob('*.md'):
        if not md_file.exists():
            continue
        try:
            if fix_md_image_refs(md_file):
                fixed += 1
        except Exception as e:
            print(f'  Skip {md_file.name}: {e}')

    print(f'Updated image refs in {fixed} files')


if __name__ == '__main__':
    main()
