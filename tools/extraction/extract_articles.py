#!/usr/bin/env python3
"""
Extract articles from Hexo-generated HTML (master branch) back to Markdown source.
Recovers ~300 articles that exist only as HTML on the master branch.
"""

import os
import re
import subprocess
import html
import json
import sys
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.parent
OUTPUT_DIR = REPO_DIR / "source" / "_posts"
EXCLUDE_DIRS = {
    '.git', 'assets', 'css', 'js', 'images', 'webfonts', 'lib',
    'categories', 'tags', 'archives', 'all-archives', 'all-categories',
    'all-tags', 'about', 'page', 'scaffolds', 'source', 'themes',
    'node_modules 2', 'en'
}


def decode_git_path(path):
    """Decode git's octal-escaped paths for non-ASCII characters."""
    path = path.strip().strip('"')
    # Replace octal sequences like \345\256\214 with actual bytes
    def replace_octal(m):
        return bytes([int(m.group(1), 8)])
    try:
        encoded = path.encode('raw_unicode_escape')
        decoded = re.sub(rb'\\(\d{3})', replace_octal, encoded)
        return decoded.decode('utf-8')
    except Exception:
        return path


def git_show(path):
    """Read a file from the master branch."""
    try:
        result = subprocess.run(
            ['git', 'show', f'master:{path}'],
            capture_output=True, text=True, cwd=REPO_DIR,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def git_show_binary(path):
    """Read a binary file from the master branch."""
    try:
        result = subprocess.run(
            ['git', 'show', f'master:{path}'],
            capture_output=True, cwd=REPO_DIR,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def list_master_dirs():
    """List all content directories on master branch."""
    result = subprocess.run(
        ['git', 'ls-tree', '--name-only', 'master'],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    dirs = []
    for raw_name in result.stdout.strip().split('\n'):
        name = decode_git_path(raw_name)
        if name and name not in EXCLUDE_DIRS:
            # Check if it's a directory with index.html
            check = subprocess.run(
                ['git', 'cat-file', '-t', f'master:{name}/index.html'],
                capture_output=True, text=True, cwd=REPO_DIR
            )
            if check.returncode == 0:
                dirs.append(name)
    return dirs


def list_en_dirs():
    """List English content directories under en/ on master."""
    result = subprocess.run(
        ['git', 'ls-tree', '--name-only', 'master', 'en/'],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    dirs = []
    for raw_name in result.stdout.strip().split('\n'):
        name = decode_git_path(raw_name)
        if name and name not in EXCLUDE_DIRS:
            basename = name.split('/')[-1] if '/' in name else name
            if basename not in EXCLUDE_DIRS:
                check = subprocess.run(
                    ['git', 'cat-file', '-t', f'master:{name}/index.html'],
                    capture_output=True, text=True, cwd=REPO_DIR
                )
                if check.returncode == 0:
                    dirs.append(name)
    return dirs


def extract_frontmatter(html_content):
    """Extract front-matter metadata from HTML."""
    fm = {}

    # Title
    m = re.search(r'<title>\s*(.*?)(?:\s*\|.*?)?\s*</title>', html_content, re.DOTALL)
    if m:
        fm['title'] = html.unescape(m.group(1).strip())

    # Date
    m = re.search(r'"datePublished":\s*"([^"]+)"', html_content)
    if not m:
        m = re.search(r'<time[^>]*datetime="([^"]+)"', html_content)
    if not m:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', html_content[:5000])
    if m:
        date_str = m.group(1)
        try:
            if 'T' in date_str:
                date_str = date_str.split('T')[0] + ' ' + date_str.split('T')[1][:8]
            fm['date'] = date_str
        except Exception:
            fm['date'] = date_str

    # Tags
    tags = re.findall(r'<a[^>]*href="/(?:en/)?tags/[^"]*"[^>]*>([^<]+)</a>', html_content)
    if tags:
        fm['tags'] = list(set(tags))

    # Categories
    cats = re.findall(r'<a[^>]*href="/(?:en/)?categories/[^"]*"[^>]*>([^<]+)</a>', html_content)
    if cats:
        fm['categories'] = cats[0] if len(cats) == 1 else list(set(cats))

    # Check if it's English
    if '"language":"en"' in html_content[:3000] or '/en/' in html_content[:1000]:
        fm['lang'] = 'en'
    else:
        fm['lang'] = 'zh-CN'

    # MathJax detection
    if 'mjx-container' in html_content or 'MathJax' in html_content or 'katex' in html_content.lower():
        fm['mathjax'] = True

    return fm


def html_to_markdown(html_content):
    """Convert article HTML body to markdown."""
    # Find the article content
    m = re.search(
        r'class="article-content\s+markdown-body">\s*(.*?)\s*(?:<section\s+class="post-copyright|<div\s+class="post-copyright|<div\s+class="post-tags|<ul\s+class="post-tags)',
        html_content, re.DOTALL
    )
    if not m:
        # Fallback: try to find any markdown-body content
        m = re.search(r'markdown-body">\s*(.*?)</div>\s*(?:</article>|<section|<footer)', html_content, re.DOTALL)
    if not m:
        return None

    text = m.group(1)

    # Pre-process: handle code blocks first (preserve them)
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f'__CODE_BLOCK_{len(code_blocks)-1}__'

    text = re.sub(r'<pre[^>]*>.*?</pre>', save_code_block, text, flags=re.DOTALL)

    # Headings
    for level in range(1, 7):
        text = re.sub(
            rf'<h{level}[^>]*>(.*?)</h{level}>',
            lambda m, l=level: f'\n{"#" * l} {re.sub(r"<[^>]+>", "", m.group(1)).strip()}\n',
            text, flags=re.DOTALL
        )

    # Bold and italic
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)

    # Inline code
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)

    # Links
    text = re.sub(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)

    # Images
    text = re.sub(
        r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*/?>',
        r'![\2](\1)',
        text, flags=re.DOTALL
    )
    text = re.sub(
        r'<img[^>]*src="([^"]+)"[^>]*/?>',
        r'![](\1)',
        text, flags=re.DOTALL
    )

    # Lists
    text = re.sub(r'<ul[^>]*>', '\n', text)
    text = re.sub(r'</ul>', '\n', text)
    text = re.sub(r'<ol[^>]*>', '\n', text)
    text = re.sub(r'</ol>', '\n', text)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', text, flags=re.DOTALL)

    # Blockquotes
    text = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>',
                  lambda m: '\n' + '\n'.join('> ' + line for line in m.group(1).strip().split('\n')) + '\n',
                  text, flags=re.DOTALL)

    # Paragraphs
    text = re.sub(r'<p[^>]*>', '\n\n', text)
    text = re.sub(r'</p>', '\n', text)

    # Line breaks
    text = re.sub(r'<br\s*/?>', '\n', text)

    # Tables (basic)
    text = re.sub(r'<table[^>]*>', '\n', text)
    text = re.sub(r'</table>', '\n', text)
    text = re.sub(r'<thead[^>]*>', '', text)
    text = re.sub(r'</thead>', '', text)
    text = re.sub(r'<tbody[^>]*>', '', text)
    text = re.sub(r'</tbody>', '', text)
    text = re.sub(r'<tr[^>]*>', '\n|', text)
    text = re.sub(r'</tr>', '|', text)
    text = re.sub(r'<t[hd][^>]*>(.*?)</t[hd]>', r' \1 |', text, flags=re.DOTALL)

    # MathJax/KaTeX - try to preserve
    # Display math
    text = re.sub(r'<mjx-container[^>]*display="true"[^>]*>(.*?)</mjx-container>',
                  lambda m: '\n$$\n' + re.sub(r'<[^>]+>', '', m.group(1)).strip() + '\n$$\n',
                  text, flags=re.DOTALL)
    # Inline math
    text = re.sub(r'<mjx-container[^>]*>(.*?)</mjx-container>',
                  lambda m: '$' + re.sub(r'<[^>]+>', '', m.group(1)).strip() + '$',
                  text, flags=re.DOTALL)

    # Remove remaining HTML tags
    text = re.sub(r'<div[^>]*>', '\n', text)
    text = re.sub(r'</div>', '\n', text)
    text = re.sub(r'<span[^>]*>', '', text)
    text = re.sub(r'</span>', '', text)
    text = re.sub(r'</?(?:section|article|nav|aside|figure|figcaption|details|summary)[^>]*>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        # Extract language and code from the pre block
        lang_match = re.search(r'class="[^"]*language-(\w+)', block)
        lang = lang_match.group(1) if lang_match else ''
        code_match = re.search(r'<code[^>]*>(.*?)</code>', block, re.DOTALL)
        if code_match:
            code = html.unescape(re.sub(r'<[^>]+>', '', code_match.group(1)))
        else:
            code = html.unescape(re.sub(r'<[^>]+>', '', block))
        text = text.replace(f'__CODE_BLOCK_{i}__', f'\n```{lang}\n{code}\n```\n')

    # Unescape HTML entities
    text = html.unescape(text)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    return text


def dir_to_slug(dirname):
    """Convert a directory name to a clean kebab-case slug."""
    slug = dirname
    # Chinese to pinyin is too complex; use the dirname as-is for Chinese
    # but clean up special chars
    slug = slug.replace('——', '-')
    slug = slug.replace('—', '-')
    slug = slug.replace(' ', '-')
    slug = slug.replace('（', '-')
    slug = slug.replace('）', '-')
    slug = slug.replace('：', '-')
    slug = slug.replace('、', '-')
    slug = slug.replace(',', '-')
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    return slug


def format_frontmatter(fm):
    """Format front-matter as YAML."""
    lines = ['---']
    if 'title' in fm:
        # Escape quotes in title
        title = fm['title'].replace('"', '\\"')
        lines.append(f'title: "{title}"')
    if 'date' in fm:
        lines.append(f'date: {fm["date"]}')
    if 'tags' in fm:
        lines.append('tags:')
        for tag in fm['tags']:
            lines.append(f'  - {tag}')
    if 'categories' in fm:
        cat = fm['categories']
        if isinstance(cat, list):
            lines.append('categories:')
            for c in cat:
                lines.append(f'  - {c}')
        else:
            lines.append(f'categories: {cat}')
    if fm.get('lang'):
        lines.append(f'lang: {fm["lang"]}')
    if fm.get('mathjax'):
        lines.append('mathjax: true')
    lines.append('---')
    return '\n'.join(lines)


def process_article(dirname, is_en_subdir=False):
    """Process a single article directory."""
    path = f'en/{dirname}' if is_en_subdir else dirname
    html_path = f'{path}/index.html'

    content = git_show(html_path)
    if not content:
        return None

    # Extract metadata
    fm = extract_frontmatter(content)
    if not fm.get('title'):
        fm['title'] = dirname.replace('-', ' ').replace('——', ' - ')

    if is_en_subdir:
        fm['lang'] = 'en'

    # Extract body
    body = html_to_markdown(content)
    if not body:
        return None

    # Build filename
    slug = dir_to_slug(dirname)
    lang_prefix = 'en/' if fm.get('lang') == 'en' else 'zh/'
    filename = f'{slug}.md'

    # Combine
    frontmatter = format_frontmatter(fm)
    full_content = f'{frontmatter}\n\n{body}\n'

    return {
        'filename': filename,
        'lang': fm.get('lang', 'zh-CN'),
        'content': full_content,
        'title': fm.get('title', dirname),
        'lines': len(body.split('\n'))
    }


def list_images_in_dir(dirname):
    """List image files in a master branch directory."""
    result = subprocess.run(
        ['git', 'ls-tree', '--name-only', 'master', f'{dirname}/'],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    images = []
    for raw_name in result.stdout.strip().split('\n'):
        name = decode_git_path(raw_name)
        if name and name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
            images.append(name)
    return images


def copy_image(src_path, dest_dir):
    """Copy an image from master branch to the local filesystem."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(src_path).name
    dest_path = dest_dir / filename

    data = git_show_binary(src_path)
    if data:
        dest_path.write_bytes(data)
        return True
    return False


def main():
    os.chdir(REPO_DIR)

    # Create output directories
    zh_dir = OUTPUT_DIR / "zh"
    en_dir = OUTPUT_DIR / "en"
    zh_dir.mkdir(parents=True, exist_ok=True)
    en_dir.mkdir(parents=True, exist_ok=True)

    # Get article directories
    print("Scanning master branch for content directories...")
    root_dirs = list_master_dirs()
    en_dirs = list_en_dirs()

    print(f"Found {len(root_dirs)} root directories, {len(en_dirs)} EN directories")

    # Track results
    stats = {'success': 0, 'failed': 0, 'skipped': 0, 'images': 0}
    results = []

    # Process root directories
    print("\nProcessing root directories...")
    for i, dirname in enumerate(sorted(root_dirs)):
        sys.stdout.write(f"\r  [{i+1}/{len(root_dirs)}] {dirname[:60]}...")
        sys.stdout.flush()

        try:
            result = process_article(dirname)
            if result:
                # Determine output path
                if result['lang'] == 'en':
                    outpath = en_dir / result['filename']
                else:
                    outpath = zh_dir / result['filename']

                outpath.write_text(result['content'], encoding='utf-8')
                stats['success'] += 1
                results.append(result)

                # Copy images
                images = list_images_in_dir(dirname)
                if images:
                    img_dir = outpath.parent / outpath.stem
                    for img in images:
                        if copy_image(img, img_dir):
                            stats['images'] += 1
            else:
                stats['failed'] += 1
        except Exception as e:
            print(f"\n  ERROR processing {dirname}: {e}")
            stats['failed'] += 1

    # Process EN subdirectory
    print(f"\n\nProcessing EN subdirectory...")
    for i, dirpath in enumerate(sorted(en_dirs)):
        dirname = dirpath.split('/')[-1] if '/' in dirpath else dirpath
        sys.stdout.write(f"\r  [{i+1}/{len(en_dirs)}] {dirname[:60]}...")
        sys.stdout.flush()

        # Skip if we already have this from root
        slug = dir_to_slug(dirname)
        existing = en_dir / f'{slug}.md'
        if existing.exists():
            stats['skipped'] += 1
            continue

        try:
            result = process_article(dirname, is_en_subdir=True)
            if result:
                outpath = en_dir / result['filename']
                outpath.write_text(result['content'], encoding='utf-8')
                stats['success'] += 1
                results.append(result)
            else:
                stats['failed'] += 1
        except Exception as e:
            print(f"\n  ERROR processing en/{dirname}: {e}")
            stats['failed'] += 1

    # Summary
    print(f"\n\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful: {stats['success']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Images:     {stats['images']}")
    print(f"\nOutput directories:")
    print(f"  Chinese: {zh_dir} ({len(list(zh_dir.glob('*.md')))} files)")
    print(f"  English: {en_dir} ({len(list(en_dir.glob('*.md')))} files)")

    # Write extraction report
    report = {
        'timestamp': datetime.now().isoformat(),
        'stats': stats,
        'articles': [
            {'title': r['title'], 'lang': r['lang'], 'filename': r['filename'], 'lines': r['lines']}
            for r in sorted(results, key=lambda x: x['filename'])
        ]
    }
    report_path = REPO_DIR / 'scripts' / 'extraction_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
