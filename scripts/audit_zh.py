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
    # Domestic (cn) keys -> dashscope.aliyuncs.com
    ('sk-6407a4292fd94f24aecd2fcfdaaa7567', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-27210a1ca9e74b9796638942da67de1d', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-96ab453901c84e4cb802bb38bb15af61', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-312d19df5072411492f51b32023ce94e', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-b45ff56bcadf4a77a51fbf71e4eb2ecd', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-b77b4c7520174aca9e39b1cb0ef415f0', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-a28750cc69674a22b7b603e5ef6f92ad', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-555ed573299a477d823e994cab356fb8', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-f58b74dd85884cffb81e1fd4777ef908', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    # International keys -> dashscope-intl.aliyuncs.com
    ('sk-329ee3abadff4192bdafa2f23d145f51', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-e6798c99da7e4fe1a9468bdc95bc2245', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-3682116aa6f74580a5b159b074798b2f', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-1ad1ec7c647b4bd4970604f406c8a8e6', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-52abc92c45004ca48bd8624cfba41966', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-3817926c65c44520b723e184eae42d0a', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-e329a8a6241c456592b944bb2f8b4ba9', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
]


SYSTEM_PROMPT = """你是中文技术写作编辑，专攻"机器翻译式中文"修复。给你一段中文技术博客正文（已去除代码与公式），请大胆合并短句。

**核心规则（必须执行）：**

1. **三句以上紧邻的短句必须合并**——典型病灶："X 很 Y。它 Z。默认 W；你可以 V。"  这种由 3-4 个短句串联、每句独立成段的写法是英译腔铁证。要合并为一两个自然中文长句，用"，""——""；""，"或"："连接。

2. **inline code 周围的短句尤其要合并**——例："`exec` 很危险。它运行任意 Shell。默认每次调用都需要确认；你可以在 `openclaw.json` 里标记受信任的模式。" 改为："`exec` 很危险——它运行任意 Shell，默认每次调用都需要确认，但可以在 `openclaw.json` 里把某些模式标记为受信任。"

3. **冗余主语必须省略**——"我们将"、"我们可以"、"它会"、"你需要" 频繁出现是英译腔，中文重视主语隐去。

4. **被动语态尽量改主动**。

5. **主动找形容词冗余**："非常"、"特别"、"明显地"、"显著地" 多余时删掉。

**保留：**
- LaTeX 公式（$...$、$$...$$）
- 代码块、inline code 反引号
- Markdown 结构、表格、引用
- 通用英文术语（embedding、token、Hugo、Shell 等）
- 引用、出处

**输出（严格 JSON）：**
{"issues": [{"original": "原文片段（30-100字，足够定位）", "fixed": "修改后的完整片段", "reason": "短句合并/去冗余/被动改主动"}]}

如果段落已自然，返回 {"issues": []}。**只输出 JSON**。"""

key_lock = threading.Lock()
key_index = [0]


def get_next_key():
    """Return (api_key, url) round-robin from the combined CN+INTL pool."""
    with key_lock:
        entry = API_KEYS[key_index[0] % len(API_KEYS)]
        key_index[0] += 1
        return entry


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
        key, api_url = get_next_key()
        try:
            resp = requests.post(api_url, headers={
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
    with ThreadPoolExecutor(max_workers=50) as ex:
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
