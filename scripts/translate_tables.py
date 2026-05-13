#!/usr/bin/env python3
"""Translate EN-only table rows in ZH articles via Qwen-max, batched per file."""
import os, glob, re, json, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests

API_KEYS = [
    ('sk-6407a4292fd94f24aecd2fcfdaaa7567', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-27210a1ca9e74b9796638942da67de1d', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-96ab453901c84e4cb802bb38bb15af61', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-312d19df5072411492f51b32023ce94e', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-b45ff56bcadf4a77a51fbf71e4eb2ecd', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-b77b4c7520174aca9e39b1cb0ef415f0', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-329ee3abadff4192bdafa2f23d145f51', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-1ad1ec7c647b4bd4970604f406c8a8e6', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-52abc92c45004ca48bd8624cfba41966', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-3817926c65c44520b723e184eae42d0a', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
]

key_lock = threading.Lock(); idx = [0]
def next_key():
    with key_lock:
        e = API_KEYS[idx[0] % len(API_KEYS)]; idx[0] += 1; return e


SYS = """你是中文技术写作编辑。我会给你一组英文 markdown 表格行，请翻译每个单元格的内容成自然简洁的中文。

规则（重要）：
- 必须保留 markdown 表格的 `|` 分隔结构。每行起始和结尾都是 `|`。
- 单元格中的反引号代码 `code` 原样保留
- VPC, RDS, ECS, OSS, RAM, MySQL, BGP, NAT, REST, SLS, ARMS, CloudMonitor, RAG, LLM, CoT, BERT, GPU, vCPU, EIP, IOPS, ESSD, PolarDB, Lindorm 等技术术语保留英文
- AWS 服务名（SageMaker, CloudWatch, RDS, X-Ray 等）保留英文
- 数字、单位、价格保留原样
- 简洁清晰，不要冗长

输入格式（JSON）：
{"rows": ["| col1 | col2 | col3 |", "| ...|"]}

输出格式（严格 JSON，行数与输入一致）：
{"translated": ["| ... |", "| ... |"]}

只输出 JSON。"""


def call(prompt, retries=3):
    for _ in range(retries):
        k, u = next_key()
        try:
            r = requests.post(u, headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
                json={"model": "qwen-max", "temperature": 0.2,
                    "messages": [{"role":"system","content":SYS},{"role":"user","content":prompt}]},
                timeout=180)
            t = r.json()["choices"][0]["message"]["content"].strip()
            if t.startswith("```"):
                t = re.sub(r"^```(?:json)?\n?", "", t); t = re.sub(r"\n?```$", "", t)
            return json.loads(t)
        except Exception as e:
            print(f"  err: {e}"); time.sleep(2)
    return None


def is_en_table_row(line):
    if not line.startswith("|"):
        return False
    if re.match(r"^\|[\s\-:|]+\|$", line):
        return False
    stripped = re.sub(r"`[^`]+`", "", line)
    stripped = re.sub(r"\$[^$]+\$", "", stripped)
    cjk = sum(1 for ch in stripped if "一" <= ch <= "鿿")
    letters = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    return letters >= 30 and cjk == 0


def process_file(path):
    with open(path) as f: c = f.read()
    body = c
    en_rows = []  # (line_idx, original_line)
    in_code = False
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        if is_en_table_row(line):
            en_rows.append((i, line))
    if not en_rows:
        return 0
    rows_only = [r for _, r in en_rows]
    payload = json.dumps({"rows": rows_only}, ensure_ascii=False)
    result = call(payload)
    if not result or "translated" not in result:
        print(f"  ✗ {os.path.basename(path)}: API failed")
        return 0
    translated = result["translated"]
    if len(translated) != len(rows_only):
        print(f"  ✗ {os.path.basename(path)}: count mismatch ({len(translated)} vs {len(rows_only)})")
        return 0
    fixed = 0
    for (i, orig), zh in zip(en_rows, translated):
        if lines[i] == orig:
            lines[i] = zh
            fixed += 1
    if fixed > 0:
        with open(path, "w") as f: f.write("\n".join(lines))
        print(f"  ✓ {os.path.basename(path)}: {fixed} rows translated")
    return fixed


# Find files
files_to_process = []
for path in glob.glob("/root/chenk-hugo/content/zh/**/*.md", recursive=True):
    if "_index" in path: continue
    with open(path) as f: c = f.read()
    in_code = False
    for line in c.split("\n"):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        if is_en_table_row(line):
            files_to_process.append(path); break

print(f"Processing {len(files_to_process)} files...")
total = 0
with ThreadPoolExecutor(max_workers=10) as ex:
    for n in ex.map(process_file, files_to_process):
        total += n
print(f"\nDone. Total table rows translated: {total}")
