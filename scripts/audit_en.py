#!/usr/bin/env python3
"""EN article audit via Qwen-Max. Mirrors audit_zh.py but for English content,
targeting common machine-translation artifacts and engineering writing issues."""
import os
import re
import json
import time
import glob
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = '/root/chenk-hugo/content/en'
LOG_DIR = '/root/chenk-hugo/scripts/audit_en_logs'

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

SYSTEM_PROMPT = """You are a senior English technical editor for an engineering blog. The author is bilingual; some passages may have subtle non-native phrasings, redundancies, or awkward constructions.

**Issues to fix:**
1. **Awkward phrasing or non-native idioms** (e.g. "the same situation occurs" → "the same thing happens"; "doing the operation" → "doing it")
2. **Redundant words** (e.g. "in order to" → "to"; "due to the fact that" → "because"; "at this point in time" → "now")
3. **Passive voice when active is clearer**
4. **Wordy or imprecise verbs** (e.g. "make use of" → "use"; "perform a calculation" → "calculate")
5. **Overuse of "we" / "you"** when the actor is implied
6. **Comma splices and run-on sentences**
7. **Inconsistent capitalization of technical terms** (Python ✓, python ✗ when standalone; HTTP ✓; etc.)

**Do NOT:**
- Change technical content or meaning
- Touch math, code, citations, or proper names
- Touch image/link markdown
- Over-edit prose that is already clean and direct
- Make the prose more formal — keep the conversational engineer's voice

**Output (strict JSON):**
{"issues": [{"original": "exact 20-80 word excerpt for locating", "fixed": "improved version", "reason": "short note (e.g. 'wordy', 'passive', 'redundant')"}]}

If the chunk is already clean, return {"issues": []}. Output only JSON."""

key_lock = threading.Lock()
key_index = [0]


def get_next_key():
    with key_lock:
        key = API_KEYS[key_index[0] % len(API_KEYS)]
        key_index[0] += 1
        return key


def extract_prose(content):
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
        if in_frontmatter:
            continue
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith('$$'):
            continue
        if re.match(r'^!\[.*\]\(.*\)$', stripped):
            continue
        if re.match(r'^<.*>$', stripped):
            continue
        if not stripped:
            if result and result[-1] != '':
                result.append('')
            continue
        line_clean = re.sub(r'\$[^$]+\$', '[FORMULA]', stripped)
        line_clean = re.sub(r'`[^`]+`', '[CODE]', line_clean)
        line_clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line_clean)
        line_clean = re.sub(r'!\[.*?\]\(.*?\)', '', line_clean)
        if line_clean.strip():
            result.append(line_clean)
    return '\n'.join(result).strip()


def call_qwen(prose, retries=3):
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
                    {'role': 'user', 'content': f'Please review the following English passage:\n\n{prose}'},
                ],
            }, timeout=120)
            data = resp.json()
            if 'error' in data:
                time.sleep(2)
                continue
            text = data['choices'][0]['message']['content'].strip()
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f'    Error (attempt {attempt+1}): {e}')
            if attempt < retries - 1:
                time.sleep(3)
    return {'issues': []}


def apply_fixes(filepath, issues):
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
    if applied > 0:
        with open(filepath, 'w') as f:
            f.write(content)
    return applied


def process_article(filepath):
    basename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    prose = extract_prose(content)
    if len(prose) < 200:
        return basename, 0, []
    if len(prose) > 6000:
        prose = prose[:6000] + '\n\n[truncated for review]'
    result = call_qwen(prose)
    issues = result.get('issues', [])
    if not issues:
        return basename, 0, []
    applied = apply_fixes(filepath, issues)
    return basename, applied, issues


def process_series(series_name):
    series_dir = os.path.join(BASE, series_name)
    if not os.path.isdir(series_dir):
        return 0
    articles = sorted(glob.glob(os.path.join(series_dir, '*.md')))
    articles = [a for a in articles if os.path.basename(a) != '_index.md']
    if not articles:
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
            tag = f'{applied} fixes' if applied > 0 else 'OK'
            print(f'  [{completed[0]}/{len(articles)}] {basename}... {tag}', flush=True)
        return applied, [{**iss, 'file': basename} for iss in issues]

    with ThreadPoolExecutor(max_workers=30) as ex:
        for applied, file_issues in ex.map(worker, list(enumerate(articles))):
            total_fixes += applied
            all_issues.extend(file_issues)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'{series_name}.json')
    with open(log_path, 'w') as f:
        json.dump({'series': series_name, 'total_fixes': total_fixes, 'issues': all_issues}, f, ensure_ascii=False, indent=2)
    print(f'\n  Result: {total_fixes} fixes applied to {series_name}')
    return total_fixes


def main():
    series_dirs = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])
    if len(sys.argv) > 1:
        series_dirs = sys.argv[1:]
    print(f'EN Audit — {len(series_dirs)} series')
    grand_total = 0
    for s in series_dirs:
        grand_total += process_series(s)
    print(f'\n{"="*60}\nGRAND TOTAL: {grand_total} fixes\n{"="*60}')


if __name__ == '__main__':
    main()
