#!/usr/bin/env python3
"""
Reorganize the chenk-site blog into a clean series-based folder structure.

Before:
  source/_posts/en/recommendation-systems-1-fundamentals.md
  source/_posts/en/recommendation-systems-1-fundamentals/fig1.png

After:
  source/_posts/en/recommendation-systems/01-fundamentals.md
  source/_posts/en/recommendation-systems/01-fundamentals/fig1.png
"""

import os
import re
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent
POSTS_DIR = REPO_DIR / "source" / "_posts"

# Series classification rules
# Pattern: regex matching filename → (series_slug, part_number_extractor or None)

CN_NUM_MAP = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18,
    '十九': 19, '二十': 20,
}

def cn_to_int(s):
    """Convert Chinese numerals to int."""
    return CN_NUM_MAP.get(s, None)

# (regex, series_slug, part_extractor_func)
# part_extractor takes the regex match and returns int or None (no number = standalone)

EN_SERIES = [
    # Recommendation Systems
    (r'^recommendation-systems-(\d+)-(.+)\.md$', 'recommendation-systems', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # Linear Algebra
    (r'^chapter-(\d+)-(.+)\.md$', 'linear-algebra', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # ML Math Derivations
    (r'^Machine-Learning-Mathematical-Derivations-(\d+)-(.+)\.md$', 'ml-math-derivations', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # Old variants of ML Math (treat as standalone or merge later)
    (r'^mathematical-derivations-in-machine-learning-0?(\d+)-(.+)\.md$', 'ml-math-derivations', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^ml-math-0?(\d+)-(.+)\.md$', 'ml-math-derivations', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # NLP
    (r'^nlp-(.+)\.md$', 'nlp', None, lambda m: m.group(1)),
    # Reinforcement Learning
    (r'^reinforcement-learning-(\d+)-(.+)\.md$', 'reinforcement-learning', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # Transfer Learning
    (r'^transfer-learning-(\d+)-(.+)\.md$', 'transfer-learning', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # ODE
    (r'^ode-chapter-(\d+)-(.+)\.md$', 'ode', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # PDE+ML
    (r'^PDE-and-Machine-Learning-(\d+)-(.+)\.md$', 'pde-ml', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^pde-ml-(\d+)-(.+)\.md$', 'pde-ml', lambda m: int(m.group(1)), lambda m: m.group(2)),
    # Time Series
    (r'^time-series-(\d+)-(.+)\.md$', 'time-series', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^time-series-(.+)\.md$', 'time-series', None, lambda m: m.group(1)),
    # Cloud Computing
    (r'^cloud-computing-(.+)\.md$', 'cloud-computing', None, lambda m: m.group(1)),
    # Computer Fundamentals
    (r'^computer-fundamentals-(\d+)-(.+)\.md$', 'computer-fundamentals', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^computer-fundamentals-(.+)\.md$', 'computer-fundamentals', None, lambda m: m.group(1)),
    # LeetCode
    (r'^leetcode-(\d+)-(.+)\.md$', 'leetcode', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^leetcode-(.+)\.md$', 'leetcode', None, lambda m: m.group(1)),
    # Linux
    (r'^linux-(.+)\.md$', 'linux', None, lambda m: m.group(1)),
]

ZH_SERIES = [
    # 推荐系统
    (r'^推荐系统-(.+?)-(.+)\.md$', 'recommendation-systems', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 线性代数
    (r'^线性代数-(.+?)-(.+)\.md$', 'linear-algebra', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 机器学习数学推导
    (r'^机器学习数学推导-(.+?)-(.+)\.md$', 'ml-math-derivations', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 自然语言处理
    (r'^自然语言处理-(.+?)-(.+)\.md$', 'nlp', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 强化学习
    (r'^强化学习-(.+?)-(.+)\.md$', 'reinforcement-learning', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 迁移学习
    (r'^迁移学习-(.+?)-(.+)\.md$', 'transfer-learning', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 常微分方程
    (r'^常微分方程-(.+?)-(.+)\.md$', 'ode', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    (r'^常微分方程-(.+?)\.md$', 'ode', lambda m: cn_to_int(m.group(1)), lambda m: m.group(1)),
    # PDE与机器学习
    (r'^PDE与机器学习-(.+?)-(.+)\.md$', 'pde-ml', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # 时间序列模型
    (r'^时间序列模型-(.+?)-(.+)\.md$', 'time-series', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # LeetCode
    (r'^LeetCode-(.+?)-(.+)\.md$', 'leetcode', lambda m: cn_to_int(m.group(1)), lambda m: m.group(2)),
    # Linux
    (r'^Linux-(.+)\.md$', 'linux', None, lambda m: m.group(1)),
    # Cloud Computing (Chinese articles still have English filename)
    (r'^cloud-computing-(.+)\.md$', 'cloud-computing', None, lambda m: m.group(1)),
    # Computer Fundamentals (Chinese articles use English filename)
    (r'^computer-fundamentals-(\d+)-(.+)\.md$', 'computer-fundamentals', lambda m: int(m.group(1)), lambda m: m.group(2)),
    (r'^computer-fundamentals-(.+)\.md$', 'computer-fundamentals', None, lambda m: m.group(1)),
]


def slugify(s, max_len=60):
    """Make a clean filesystem-safe slug."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9一-鿿]+', '-', s)
    s = re.sub(r'-+', '-', s)
    s = s.strip('-')
    return s[:max_len]


def classify(filename, series_rules):
    """Classify a filename. Returns (series_slug, part_num, slug_part) or None."""
    for pattern, series_slug, part_func, slug_func in series_rules:
        m = re.match(pattern, filename)
        if m:
            try:
                part = part_func(m) if part_func else None
                slug = slug_func(m).rstrip('.md').rstrip('-')
                # Strip .md if it slipped in
                if slug.endswith('.md'):
                    slug = slug[:-3]
                return series_slug, part, slug
            except Exception:
                continue
    return None


def reorganize_lang(lang_dir, series_rules):
    """Reorganize one language folder."""
    moved = 0
    standalone = []

    md_files = sorted(lang_dir.glob('*.md'))

    for md_file in md_files:
        result = classify(md_file.name, series_rules)
        if result is None:
            standalone.append(md_file)
            continue

        series_slug, part, slug = result

        # Build new filename
        if part is not None:
            new_name = f'{part:02d}-{slug}.md' if slug else f'{part:02d}.md'
        else:
            new_name = f'{slug}.md'

        # Target directory
        target_dir = lang_dir / series_slug
        target_dir.mkdir(parents=True, exist_ok=True)

        target_md = target_dir / new_name

        # Move .md file
        if md_file.exists():
            shutil.move(str(md_file), str(target_md))
            moved += 1

            # Move asset folder if exists (named same as md without .md)
            asset_dir = lang_dir / md_file.stem
            if asset_dir.exists() and asset_dir.is_dir():
                target_asset_dir = target_dir / target_md.stem
                if target_asset_dir.exists():
                    # Merge contents
                    for f in asset_dir.iterdir():
                        shutil.move(str(f), str(target_asset_dir / f.name))
                    asset_dir.rmdir()
                else:
                    shutil.move(str(asset_dir), str(target_asset_dir))

    # Move standalone articles
    if standalone:
        standalone_dir = lang_dir / 'standalone'
        standalone_dir.mkdir(exist_ok=True)
        for md_file in standalone:
            new_name = slugify(md_file.stem) + '.md'
            target_md = standalone_dir / new_name
            shutil.move(str(md_file), str(target_md))
            moved += 1

            # Move asset folder
            asset_dir = lang_dir / md_file.stem
            if asset_dir.exists() and asset_dir.is_dir():
                target_asset_dir = standalone_dir / target_md.stem
                if not target_asset_dir.exists():
                    shutil.move(str(asset_dir), str(target_asset_dir))

    # Move any leftover asset folders that lost their .md
    for d in list(lang_dir.iterdir()):
        if d.is_dir() and d.name not in ('recommendation-systems', 'linear-algebra', 'ml-math-derivations',
                                          'nlp', 'reinforcement-learning', 'transfer-learning', 'ode',
                                          'pde-ml', 'time-series', 'cloud-computing', 'computer-fundamentals',
                                          'leetcode', 'linux', 'standalone'):
            # Orphan asset folder, move to standalone
            standalone_dir = lang_dir / 'standalone'
            standalone_dir.mkdir(exist_ok=True)
            target = standalone_dir / d.name
            if not target.exists():
                shutil.move(str(d), str(target))

    return moved


def main():
    print("=" * 60)
    print("REORGANIZING BLOG STRUCTURE")
    print("=" * 60)

    print("\n--- English ---")
    en_moved = reorganize_lang(POSTS_DIR / 'en', EN_SERIES)
    print(f"Moved {en_moved} EN files")

    print("\n--- Chinese ---")
    zh_moved = reorganize_lang(POSTS_DIR / 'zh', ZH_SERIES)
    print(f"Moved {zh_moved} ZH files")

    print("\n--- Final structure ---")
    for lang in ['en', 'zh']:
        lang_dir = POSTS_DIR / lang
        print(f"\n{lang}/")
        for series_dir in sorted(lang_dir.iterdir()):
            if series_dir.is_dir():
                count = len(list(series_dir.glob('*.md')))
                assets = len([d for d in series_dir.iterdir() if d.is_dir()])
                print(f"  {series_dir.name}/  ({count} articles, {assets} asset folders)")


if __name__ == '__main__':
    main()
