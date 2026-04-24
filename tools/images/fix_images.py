#!/usr/bin/env python3
"""Fix broken image references by finding the actual image files and updating paths."""

import re
import os
from pathlib import Path
from urllib.parse import unquote

POSTS_DIR = Path(__file__).parent.parent / "source" / "_posts"

def find_image_file(filename, search_dirs):
    """Search for an image file across all post asset directories."""
    for d in search_dirs:
        for img in d.rglob(filename):
            return img
    return None

def fix_images():
    fixed_count = 0
    removed_count = 0

    # Build image index: filename -> path
    image_index = {}
    for img_file in POSTS_DIR.rglob('*'):
        if img_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'):
            image_index[img_file.name] = img_file

    for lang_dir in ['en', 'zh']:
        dir_path = POSTS_DIR / lang_dir
        if not dir_path.exists():
            continue

        for md_file in sorted(dir_path.glob('*.md')):
            content = md_file.read_text(encoding='utf-8')
            new_content = content
            post_stem = md_file.stem
            post_asset_dir = md_file.parent / post_stem

            def fix_img_ref(match):
                nonlocal fixed_count, removed_count
                alt = match.group(1)
                src = match.group(2)

                if src.startswith('http'):
                    return match.group(0)  # Keep external URLs

                decoded = unquote(src)
                filename = Path(decoded).name

                # Try to find the image
                if filename in image_index:
                    img_path = image_index[filename]
                    # Copy to post asset dir if not already there
                    if post_asset_dir.exists():
                        dest = post_asset_dir / filename
                        if not dest.exists():
                            import shutil
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(img_path, dest)
                    elif img_path.parent.name != post_stem:
                        # Create asset dir and copy
                        post_asset_dir.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(img_path, post_asset_dir / filename)

                    fixed_count += 1
                    return f'![{alt}](./{post_stem}/{filename})'
                else:
                    # Image truly missing - remove the reference to avoid broken display
                    removed_count += 1
                    if alt:
                        return f'*[Figure: {alt}]*'
                    return ''

            new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_img_ref, content)

            if new_content != content:
                md_file.write_text(new_content, encoding='utf-8')
                print(f'  Fixed: {lang_dir}/{md_file.name}')

    print(f'\n{"="*50}')
    print(f'Image paths fixed: {fixed_count}')
    print(f'Missing images removed: {removed_count}')

if __name__ == '__main__':
    fix_images()
