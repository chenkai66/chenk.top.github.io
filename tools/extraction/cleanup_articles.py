#!/usr/bin/env python3
"""
Post-optimization cleanup for all extracted/optimized blog articles.
Fixes common issues left by extraction and agent optimization.
"""

import os
import re
from pathlib import Path

POSTS_DIR = Path(__file__).parent.parent / "source" / "_posts"


def clean_tags(tags_lines):
    """Remove # prefixes from tags and deduplicate."""
    clean = []
    seen = set()
    for tag in tags_lines:
        tag = tag.strip().lstrip('-').strip()
        tag = tag.lstrip('#').strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            clean.append(tag)
    return clean


def fix_frontmatter(content):
    """Fix common front-matter issues."""
    # Split front-matter and body
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not m:
        return content, False

    fm_text = m.group(1)
    body = content[m.end():]
    changed = False

    # Fix tags with # prefix
    if re.search(r'^\s+-\s+#', fm_text, re.MULTILINE):
        fm_text = re.sub(r'^(\s+-\s+)#\s*', r'\1', fm_text, flags=re.MULTILINE)
        changed = True

    # Deduplicate tags
    tag_match = re.search(r'^tags:\s*\n((?:\s+-\s+.+\n)+)', fm_text, re.MULTILINE)
    if tag_match:
        tag_block = tag_match.group(1)
        tags = [line.strip().lstrip('-').strip() for line in tag_block.strip().split('\n')]
        clean = []
        seen = set()
        for tag in tags:
            key = tag.lower().lstrip('#').strip()
            if key and key not in seen:
                seen.add(key)
                clean.append(tag.lstrip('#').strip())

        if len(clean) < len(tags):
            new_tags = 'tags:\n' + '\n'.join(f'  - {t}' for t in clean) + '\n'
            fm_text = fm_text[:tag_match.start()] + new_tags + fm_text[tag_match.end():]
            changed = True

    # Fix date format (ensure it has time)
    date_match = re.search(r'^date:\s*(\d{4}-\d{2}-\d{2})\s*$', fm_text, re.MULTILINE)
    if date_match:
        fm_text = fm_text[:date_match.start()] + f'date: {date_match.group(1)} 09:00:00' + fm_text[date_match.end():]
        changed = True

    return f'---\n{fm_text}\n---\n{body}', changed


def fix_body(content):
    """Fix common body issues from HTML extraction."""
    changed = False

    # Fix headings with line breaks in the middle (from HTML extraction)
    # e.g., "## The Information Overload\nProblem" -> "## The Information Overload Problem"
    heading_pattern = r'^(#{1,6}\s+[^\n]+)\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
    new_content = re.sub(heading_pattern, r'\1 \2', content, flags=re.MULTILINE)
    if new_content != content:
        content = new_content
        changed = True

    # Fix extra spaces around inline math
    new_content = re.sub(r'\$\s+', '$', content)
    new_content = re.sub(r'\s+\$', '$', new_content)
    # Be careful not to break display math
    new_content = re.sub(r'\$\$', '$$', new_content)
    if new_content != content:
        content = new_content
        changed = True

    # Fix list items that start with empty "- \n"
    new_content = re.sub(r'^-\s*\n\n', '', content, flags=re.MULTILINE)
    if new_content != content:
        content = new_content
        changed = True

    # Fix double blank lines (more than 2 consecutive)
    new_content = re.sub(r'\n{4,}', '\n\n\n', content)
    if new_content != content:
        content = new_content
        changed = True

    # Fix broken URLs with spaces (from extraction)
    new_content = re.sub(r'\]\(\s+', '](', content)
    if new_content != content:
        content = new_content
        changed = True

    return content, changed


def process_file(filepath):
    """Process a single file."""
    content = filepath.read_text(encoding='utf-8')
    original = content

    content, fm_changed = fix_frontmatter(content)
    content, body_changed = fix_body(content)

    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False


def main():
    fixed = 0
    total = 0

    for lang_dir in ['en', 'zh']:
        dir_path = POSTS_DIR / lang_dir
        if not dir_path.exists():
            continue

        for md_file in sorted(dir_path.glob('*.md')):
            total += 1
            if process_file(md_file):
                fixed += 1
                print(f'  Fixed: {md_file.name}')

    # Also process root-level posts (skip broken symlinks/missing files)
    for md_file in sorted(POSTS_DIR.glob('*.md')):
        if not md_file.exists():
            continue
        total += 1
        try:
            if process_file(md_file):
                fixed += 1
                print(f'  Fixed: {md_file.name}')
        except Exception as e:
            print(f'  Skip: {md_file.name} ({e})')

    print(f'\n{"="*50}')
    print(f'Cleanup complete: {fixed}/{total} files modified')


if __name__ == '__main__':
    main()
