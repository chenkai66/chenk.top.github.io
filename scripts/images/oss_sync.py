#!/usr/bin/env python3
"""
Upload all images from source/_posts to OSS bucket blog-pic-ck.
Then update markdown files to use OSS URLs instead of relative paths.

OSS structure mirrors local:
  Local:  source/_posts/en/recommendation-systems/01-fundamentals/fig1.png
  OSS:    {prefix}/en/recommendation-systems/01-fundamentals/fig1.png
  URL:    https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/{prefix}/en/...

Markdown change:
  ![alt](./01-fundamentals/fig1.png)
  →
  ![alt](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig1.png)
"""

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, quote

REPO_DIR = Path(__file__).parent.parent.parent
POSTS_DIR = REPO_DIR / "source" / "_posts"

OSS_BUCKET = "blog-pic-ck"
OSS_REGION = "cn-beijing"
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PREFIX = "posts"  # Top-level prefix in bucket
OSS_AK = os.environ.get("OSS_AK", "")
OSS_SK = os.environ.get("OSS_SK", "")

URL_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{OSS_PREFIX}"


def oss_cp_recursive(local_dir, oss_path):
    """Use ossutil cp -r to upload a directory recursively."""
    cmd = [
        "ossutil", "cp", "-r", "-u",
        str(local_dir),
        f"oss://{OSS_BUCKET}/{oss_path}",
        "-i", OSS_AK,
        "-k", OSS_SK,
        "-e", OSS_ENDPOINT,
        "--region", OSS_REGION,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def upload_all():
    """Upload all post images to OSS."""
    print(f"Uploading {POSTS_DIR}/en → oss://{OSS_BUCKET}/{OSS_PREFIX}/en/")
    ok, out = oss_cp_recursive(POSTS_DIR / "en", f"{OSS_PREFIX}/en/")
    print(out[-1500:] if not ok else "EN upload OK")

    print(f"\nUploading {POSTS_DIR}/zh → oss://{OSS_BUCKET}/{OSS_PREFIX}/zh/")
    ok, out = oss_cp_recursive(POSTS_DIR / "zh", f"{OSS_PREFIX}/zh/")
    print(out[-1500:] if not ok else "ZH upload OK")


def rewrite_md_to_oss():
    """Rewrite all markdown image refs from relative paths to OSS URLs."""
    fixed = 0

    for md_file in POSTS_DIR.rglob('*.md'):
        if not md_file.exists():
            continue

        # Determine the post's URL path (en/series/article-slug)
        rel = md_file.relative_to(POSTS_DIR)  # e.g. en/recommendation-systems/01-fundamentals.md
        url_dir = str(rel.parent / md_file.stem)  # en/recommendation-systems/01-fundamentals

        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception:
            continue

        original = content

        def fix_ref(match):
            alt = match.group(1)
            src = match.group(2)
            if src.startswith('http'):
                return match.group(0)

            decoded = unquote(src)
            # Strip leading ./ or /
            decoded = decoded.lstrip('./').lstrip('/')
            # Get just the filename
            filename = Path(decoded).name
            if not filename:
                return match.group(0)

            # URL-encode the path components individually (preserves /)
            url_path_parts = url_dir.split('/') + [filename]
            url_encoded = '/'.join(quote(p) for p in url_path_parts)
            new_url = f'{URL_BASE}/{url_encoded}'
            return f'![{alt}]({new_url})'

        new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_ref, content)

        if new_content != original:
            md_file.write_text(new_content, encoding='utf-8')
            fixed += 1

    print(f'\nRewrote image refs in {fixed} markdown files')


if __name__ == '__main__':
    import sys
    if '--rewrite-only' in sys.argv:
        rewrite_md_to_oss()
    elif '--upload-only' in sys.argv:
        upload_all()
    else:
        upload_all()
        rewrite_md_to_oss()
