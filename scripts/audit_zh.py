#!/usr/bin/env python3
"""Qwen-powered Chinese quality audit for all ZH blog articles."""
import os
import re
import json
import time
import glob
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = '/root/chenk-hugo/content/zh'
LOG_DIR = '/root/chenk-hugo/scripts/audit_logs'

API_KEYS = [
    'sk-6407a4292fd94f24aecd2fcfdaaa7567',
    'sk-27210a1ca9e74b9796638942da67de1d',
    'sk-96ab453901c84e4cb802bb38bb15af61',
    'sk-312d19df5072411492f51b32023ce94e',
    'sk-b45ff56bcadf4a77a51fbf71e4eb2ecd',
    'sk-b77b4c7520174aca9e39b1cb0ef415f0',
    'sk-a28750cc69674a22b7b603e5ef6f92ad',
    'sk-555ed573299a477d823e994cab356fb8',
    'sk-f58b74dd85884cffb81e1fd4777ef908',
]

API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

SYSTEM_PROMPT = """你是资深中文技术写作编辑，专门修复机器翻译的英译腔。用户会给你一篇技术博客的中文正文（已去除代码和公式）。

**最重要的问题（重点修复）：**
1. **句号过密、短句过多** —— 英文常用 "X. Y. Z." 的并列短句风格，直译成中文后变成 "X。Y。Z。" 读起来像机器人。要合并为一句自然中文，用顿号、逗号、破折号衔接。
   反例（机器味）："十一篇文章。几十条 CLI 命令。上百个手动步骤。现在我们把它们全部扔掉。"
   正例（地道）："十一篇文章、几十条 CLI 命令、上百个手动步骤——现在全部扔掉。"
2. **明显的英文句法直译** —— 如"这是 X 的核心价值所在"、"我们将把 ... 统一封装为 ..."、"读完本文，你将获得 ..."等八股翻译腔。
3. **冗余的连接词和虚词** —— "我们将"、"我们把"、"它们" 等过度使用代词，中文重视主语省略。
4. **技术术语应该保留英文**（如 embedding、token、BPE、LSTM、JSON、API 等）—— 不要意译。

**修改原则：**
- 优先合并 "X。Y。Z。" 为 "X、Y、Z" 或 "X，Y，Z"
- 主动语态优先于被动语态
- 短句优先于长句，但避免过短产生机器味
- 中文重意合，可大胆省略冗余主语
- 保持术语原文和数字精确性
- 不要改变技术含义、引用出处、代码引用
- 不要过度润色已经流畅的段落

**输出格式（严格 JSON）：**
{"issues": [{"original": "原文片段（20-80字，足够定位）", "fixed": "修改后的完整片段", "reason": "简短说明（如：句号过密合并、被动改主动等）"}]}

如果段落已经流畅自然，返回 {"issues": []}
只输出 JSON，不要其他内容。"""

key_lock = threading.Lock()
key_index = [0]


def get_next_key():
    with key_lock:
        key = API_KEYS[key_index[0] % len(API_KEYS)]
        key_index[0] += 1
        return key


def extract_prose(content):
    """Extract readable Chinese prose from markdown, stripping code/math/frontmatter."""
    lines = content.split('\n')
    result = []
    in_frontmatter = False
    in_code_block = False
    fm_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped == '---':
            fm_count += 1
            if fm_count <= 2:
                in_frontmatter = not in_frontmatter if fm_count == 1 else False
                continue
            if fm_count == 2:
                in_frontmatter = False
                continue

        if in_frontmatter:
            continue

        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Skip pure math display blocks
        if stripped.startswith('$$'):
            continue

        # Skip image lines
        if re.match(r'^!\[.*\]\(.*\)$', stripped):
            continue

        # Skip HTML tags
        if re.match(r'^<.*>$', stripped):
            continue

        # Skip empty lines but keep as separators
        if not stripped:
            if result and result[-1] != '':
                result.append('')
            continue

        # Remove inline math for readability but keep surrounding text
        line_clean = re.sub(r'\$[^$]+\$', '[公式]', stripped)
        # Remove inline code
        line_clean = re.sub(r'`[^`]+`', '[代码]', line_clean)
        # Remove markdown link syntax, keep text
        line_clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line_clean)
        # Remove image references
        line_clean = re.sub(r'!\[.*?\]\(.*?\)', '', line_clean)

        if line_clean.strip():
            result.append(line_clean)

    return '\n'.join(result).strip()


def call_qwen(prose, retries=3):
    """Call Qwen API to review Chinese text quality."""
    for attempt in range(retries):
        key = get_next_key()
        try:
            resp = requests.post(API_URL, headers={
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json',
            }, json={
                'model': 'qwen-max',
                'temperature': 0.2,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'请审查以下中文技术文章：\n\n{prose}'},
                ],
            }, timeout=120)

            data = resp.json()
            if 'error' in data:
                print(f'    API error (key ...{key[-6:]}): {data["error"]["message"]}')
                time.sleep(2)
                continue

            text = data['choices'][0]['message']['content'].strip()
            # Try to parse JSON from response
            # Handle markdown code blocks
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            result = json.loads(text)
            return result

        except json.JSONDecodeError:
            print(f'    JSON parse error, retrying...')
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f'    Request error: {e}')
            if attempt < retries - 1:
                time.sleep(3)

    return {'issues': []}


def apply_fixes(filepath, issues):
    """Apply fixes to the original markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()

    applied = 0
    for issue in issues:
        original = issue.get('original', '')
        fixed = issue.get('fixed', '')
        if not original or not fixed or original == fixed:
            continue

        if original in content:
            content = content.replace(original, fixed, 1)
            applied += 1
        else:
            # Try fuzzy match - sometimes Qwen adds/removes spaces
            original_normalized = re.sub(r'\s+', ' ', original.strip())
            # Search in content with flexible whitespace
            pattern = re.escape(original_normalized).replace(r'\ ', r'\s+')
            m = re.search(pattern, content)
            if m:
                content = content[:m.start()] + fixed + content[m.end():]
                applied += 1

    if applied > 0:
        with open(filepath, 'w') as f:
            f.write(content)

    return applied


def process_article(filepath):
    """Process a single article: extract, review, fix."""
    basename = os.path.basename(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    prose = extract_prose(content)

    # Skip very short articles or those with mostly code/math
    if len(prose) < 200:
        return basename, 0, []

    # Truncate very long prose to avoid API limits (keep first ~6000 chars)
    if len(prose) > 6000:
        prose = prose[:6000] + '\n\n[文章截断，仅审查前半部分]'

    result = call_qwen(prose)
    issues = result.get('issues', [])

    if not issues:
        return basename, 0, []

    applied = apply_fixes(filepath, issues)
    return basename, applied, issues


def process_series(series_name):
    """Process all articles in a series."""
    series_dir = os.path.join(BASE, series_name)
    if not os.path.isdir(series_dir):
        print(f'  SKIP: {series_name} (not found)')
        return 0

    articles = sorted(glob.glob(os.path.join(series_dir, '*.md')))
    articles = [a for a in articles if os.path.basename(a) != '_index.md']

    if not articles:
        print(f'  SKIP: {series_name} (no articles)')
        return 0

    print(f'\n{"="*60}')
    print(f'  Series: {series_name} ({len(articles)} articles)')
    print(f'{"="*60}')

    total_fixes = 0
    all_issues = []
    print_lock = threading.Lock()
    completed = [0]

    def worker(idx_article):
        i, article = idx_article
        basename = os.path.basename(article)
        try:
            name, applied, issues = process_article(article)
        except Exception as e:
            with print_lock:
                completed[0] += 1
                print(f'  [{completed[0]}/{len(articles)}] {basename}... ERROR: {e}', flush=True)
            return 0, []
        with print_lock:
            completed[0] += 1
            if applied > 0:
                print(f'  [{completed[0]}/{len(articles)}] {basename}... {applied} fixes', flush=True)
            else:
                print(f'  [{completed[0]}/{len(articles)}] {basename}... OK', flush=True)
        return applied, [{**iss, 'file': basename} for iss in issues]

    # 9 API keys → 9 parallel workers (one per key roughly)
    with ThreadPoolExecutor(max_workers=30) as ex:
        for applied, file_issues in ex.map(worker, list(enumerate(articles))):
            total_fixes += applied
            all_issues.extend(file_issues)

    # Save log
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'{series_name}.json')
    with open(log_path, 'w') as f:
        json.dump({'series': series_name, 'total_fixes': total_fixes,
                   'issues': all_issues}, f, ensure_ascii=False, indent=2)

    print(f'\n  Result: {total_fixes} fixes applied to {series_name}')
    return total_fixes


# Series order (worst quality first)
SERIES_ORDER = [
    # Batch 1: 5 new series
    'databases', 'docker-containers', 'python-engineering',
    'probability-statistics', 'system-design',
    # Batch 2: Previously batch-translated
    'llm-engineering', 'claude-code-learn', 'openclaw-quickstart',
    'terraform-agents', 'aliyun-fullstack', 'aliyun-bailian', 'aliyun-pai',
    # Batch 3: Older translated
    'nlp', 'recommendation-systems', 'reinforcement-learning',
    'transfer-learning', 'time-series', 'cloud-computing',
    'computer-fundamentals', 'linux', 'leetcode',
    # Batch 4: Math-heavy
    'linear-algebra', 'ml-math-derivations', 'ode', 'pde-ml', 'standalone',
]


def main():
    if len(sys.argv) > 1:
        # Process specific series
        series_list = sys.argv[1:]
    else:
        series_list = SERIES_ORDER

    print(f'Chinese Quality Audit — {len(series_list)} series to process')
    print(f'API keys: {len(API_KEYS)}')

    grand_total = 0
    for series in series_list:
        fixes = process_series(series)
        grand_total += fixes

    print(f'\n{"="*60}')
    print(f'GRAND TOTAL: {grand_total} fixes applied')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
