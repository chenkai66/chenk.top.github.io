#!/usr/bin/env python3
"""Translate optim series new articles EN -> ZH using Qwen-Plus.
Splits articles into chunks at section boundaries to fit within max_tokens.
Preserves frontmatter (with title localized) + math/code blocks unchanged.
"""
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = '/root/chenk-hugo/content'

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

key_lock = threading.Lock()
key_idx = [0]


def next_key():
    with key_lock:
        k = API_KEYS[key_idx[0] % len(API_KEYS)]
        key_idx[0] += 1
        return k


SYSTEM_PROMPT = """你是一位精通机器学习与凸优化的中文技术作家。任务：将一篇英文技术博客（已经过严谨的数学证明）翻译为流畅、专业、地道的中文。

**绝对规则：**
1. 保持所有 LaTeX 数学公式（$...$ 和 $$...$$）原封不动
2. 保持所有代码块 (``` 包裹) 原封不动
3. 保持所有 Markdown 表格的对齐和列分隔符
4. 保持所有 Markdown 标题层级（#、##、### 等）和编号
5. 保持所有引用块（> ）格式

**翻译指南：**
- 保留通用英文术语：Lipschitz、convex、Nesterov、Newton、Hessian、Gradient、SGD、KKT、Lagrangian、PSD、Banach 等数学/算法人名和专有名词
- 中文术语优先：「光滑」、「强凸」、「次梯度」、「对偶」、「鞍点」、「方差缩减」、「内点法」等
- 行文要专业、简洁、地道，不要英译腔；不要加任何注释或解释；只输出翻译后的中文文本

**关键：直接输出翻译结果，不要包含任何 markdown 代码块标记（```）或解释性文字。**"""


def translate_chunk(text, retries=3):
    """Translate one chunk via Qwen API."""
    for attempt in range(retries):
        key = next_key()
        try:
            resp = requests.post(API_URL, headers={
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json',
            }, json={
                'model': 'qwen-plus',
                'temperature': 0.2,
                'max_tokens': 8000,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text},
                ],
            }, timeout=180)

            data = resp.json()
            if 'error' in data:
                msg = data['error'].get('message', str(data['error']))
                print(f'    API err (attempt {attempt+1}, key ...{key[-6:]}): {msg[:80]}', flush=True)
                time.sleep(3)
                continue

            out = data['choices'][0]['message']['content'].strip()
            # Strip wrapping code blocks if present
            if out.startswith('```'):
                out = re.sub(r'^```(?:\w+)?\n?', '', out)
                out = re.sub(r'\n?```$', '', out)
            return out
        except Exception as e:
            print(f'    Exception (attempt {attempt+1}): {e}', flush=True)
            time.sleep(5)
    raise RuntimeError('Translation failed after retries')


def split_into_chunks(body, max_chars=4500):
    """Split markdown body at H2 boundaries, packing chunks up to max_chars."""
    sections = re.split(r'(?=^## )', body, flags=re.MULTILINE)
    chunks = []
    current = ''
    for sec in sections:
        if len(current) + len(sec) > max_chars and current:
            chunks.append(current)
            current = sec
        else:
            current += sec
    if current:
        chunks.append(current)
    return chunks


def translate_article(en_path, zh_path, zh_title):
    """Translate a single article EN -> ZH, write to zh_path."""
    print(f'\n== {os.path.basename(en_path)} ==', flush=True)

    with open(en_path) as f:
        content = f.read()

    # Split frontmatter from body
    lines = content.split('\n')
    if lines[0] != '---':
        raise ValueError('No frontmatter')
    fm_end = -1
    for i in range(1, len(lines)):
        if lines[i] == '---':
            fm_end = i
            break
    fm_lines = lines[1:fm_end]
    body = '\n'.join(lines[fm_end+1:])

    # Build localized frontmatter (replace title, change lang to zh)
    new_fm = []
    for line in fm_lines:
        if line.startswith('title:'):
            new_fm.append(f'title: "{zh_title}"')
        elif line.startswith('lang:'):
            new_fm.append('lang: zh')
        elif line.startswith('description:'):
            # Translate description separately (short)
            desc_match = re.match(r'^description:\s*"(.*)"$', line)
            if desc_match:
                en_desc = desc_match.group(1)
                print(f'  Translating description...', flush=True)
                zh_desc = translate_chunk(f'请翻译这段元数据描述（保持简洁专业，单行不换行）：\n\n{en_desc}')
                zh_desc = zh_desc.replace('"', "'").replace('\n', ' ').strip()
                new_fm.append(f'description: "{zh_desc}"')
            else:
                new_fm.append(line)
        else:
            new_fm.append(line)

    # Split body into chunks for translation
    chunks = split_into_chunks(body)
    print(f'  {len(chunks)} chunks to translate', flush=True)

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f'  [{i+1}/{len(chunks)}] {len(chunk)} chars...', end=' ', flush=True)
        zh = translate_chunk(chunk)
        print('OK', flush=True)
        translated_chunks.append(zh)
        time.sleep(1)

    zh_body = '\n'.join(translated_chunks)
    zh_content = '---\n' + '\n'.join(new_fm) + '\n---\n' + zh_body

    with open(zh_path, 'w') as f:
        f.write(zh_content)
    print(f'  Wrote {len(zh_content)} bytes -> {os.path.basename(zh_path)}', flush=True)


# Articles to translate
ARTICLES = [
    ('01-convex-analysis-foundations.md', '优化理论（一）：凸分析基础'),
    ('05-acceleration-beyond-nesterov.md', '优化理论（五）：Nesterov 之外的加速——Heavy-Ball、下界与 Catalyst'),
    ('07-second-order-methods.md', '优化理论（七）：二阶方法——Newton、BFGS、L-BFGS、信赖域'),
    ('08-lagrangian-duality-kkt.md', '优化理论（八）：Lagrangian 对偶与 KKT 条件'),
    ('09-interior-point-barrier.md', '优化理论（九）：内点法与自和谐障碍函数'),
    ('10-stochastic-variance-reduction.md', '优化理论（十）：随机优化与方差缩减'),
    ('11-nonconvex-saddle-escape.md', '优化理论（十一）：非凸优化——鞍点逃逸、PL 条件与神经网络损失景观'),
    # 12 needs full re-translate since we added IP/B&B/heuristics theory
    ('12-discrete-global-optimization.md', '优化理论（十二）：离散与全局优化——IP、分支定界、启发式与 Portfolio 案例'),
]


def main():
    if len(sys.argv) > 1:
        # Specific article(s)
        targets = sys.argv[1:]
        articles = [a for a in ARTICLES if any(t in a[0] for t in targets)]
    else:
        articles = ARTICLES

    print(f'Translating {len(articles)} articles...', flush=True)

    for fname, zh_title in articles:
        en_path = os.path.join(BASE, 'en', 'optimization-theory', fname)
        zh_path = os.path.join(BASE, 'zh', 'optimization-theory', fname)
        try:
            translate_article(en_path, zh_path, zh_title)
        except Exception as e:
            print(f'  FAIL: {e}', flush=True)
            continue

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
