#!/usr/bin/env python3
"""Translate EN-only figure captions in ZH articles via Qwen-max, batched per file."""
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
    ('sk-a28750cc69674a22b7b603e5ef6f92ad', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-555ed573299a477d823e994cab356fb8', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-f58b74dd85884cffb81e1fd4777ef908', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-329ee3abadff4192bdafa2f23d145f51', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-e6798c99da7e4fe1a9468bdc95bc2245', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-3682116aa6f74580a5b159b074798b2f', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-1ad1ec7c647b4bd4970604f406c8a8e6', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-52abc92c45004ca48bd8624cfba41966', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-3817926c65c44520b723e184eae42d0a', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
    ('sk-e329a8a6241c456592b944bb2f8b4ba9', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'),
]

key_lock = threading.Lock()
key_idx = [0]

def next_key():
    with key_lock:
        e = API_KEYS[key_idx[0] % len(API_KEYS)]
        key_idx[0] += 1
        return e


SYSTEM = """你是中文技术写作编辑。我会给你一组英文图片说明（figure captions），请翻译成自然简洁的中文。

规则：
- 保留通用英文术语（VPC, RDS, ECS, NAT, SLB, KMS, MySQL, BGP, REST, gRPC, GraphQL, CLT, LLN, EM 等技术术语保留英文）
- 保留缩写和首字母词
- 图说要简洁清晰，不要冗长
- 保留 "fig1:" "图1：" 这类前缀格式（fig1: → 图1：）
- 数学符号保持原样

输入格式（JSON）：
{"captions": ["caption 1", "caption 2", ...]}

输出格式（严格 JSON，与输入对应）：
{"translated": ["中文 1", "中文 2", ...]}

只输出 JSON，不要解释。"""


def call_qwen(prompt, retries=3):
    for attempt in range(retries):
        key, url = next_key()
        try:
            r = requests.post(url, headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }, json={
                "model": "qwen-max",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            }, timeout=120)
            data = r.json()
            if "error" in data:
                time.sleep(2); continue
            text = data["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return json.loads(text)
        except Exception as e:
            print(f"    err: {e}")
            time.sleep(2)
    return None


CAPTION_RE = re.compile(r"!\[([^\]]+)\]\(([^)]+)\)")


def is_en_only(cap):
    cap = cap.strip()
    if len(cap) < 10:
        return False
    cjk = sum(1 for ch in cap if "一" <= ch <= "鿿")
    letters = sum(1 for ch in cap if ch.isascii() and ch.isalpha())
    return letters >= 10 and cjk == 0


def process_file(path):
    with open(path) as f:
        c = f.read()
    # Avoid touching code blocks
    body = c
    en_caps = []  # list of (caption_text, full_match_string, url)
    seen = set()
    # Track inside code blocks
    in_code = False
    pos = 0
    for line in c.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code:
            for m in CAPTION_RE.finditer(line):
                cap = m.group(1).strip()
                url = m.group(2)
                if is_en_only(cap) and cap not in seen:
                    seen.add(cap)
                    en_caps.append((cap, url))
    if not en_caps:
        return 0

    captions_only = [cap for cap, _ in en_caps]
    payload = json.dumps({"captions": captions_only}, ensure_ascii=False)
    result = call_qwen(payload)
    if not result or "translated" not in result:
        print(f"  ✗ {os.path.basename(path)}: API failed")
        return 0
    translated = result["translated"]
    if len(translated) != len(captions_only):
        print(f"  ✗ {os.path.basename(path)}: count mismatch ({len(translated)} vs {len(captions_only)})")
        return 0

    new_c = c
    fixed = 0
    for (en_cap, url), zh_cap in zip(en_caps, translated):
        old_pat = f"![{en_cap}]({url})"
        new_pat = f"![{zh_cap}]({url})"
        if old_pat in new_c:
            new_c = new_c.replace(old_pat, new_pat)
            fixed += 1
    if fixed > 0:
        with open(path, "w") as f:
            f.write(new_c)
        print(f"  ✓ {os.path.basename(path)}: {fixed} captions translated")
    return fixed


# Find files with EN-only captions
files_to_process = []
for path in glob.glob("/root/chenk-hugo/content/zh/**/*.md", recursive=True):
    if "_index" in path:
        continue
    with open(path) as f:
        c = f.read()
    in_code = False
    has_en_cap = False
    for line in c.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code:
            for m in CAPTION_RE.finditer(line):
                if is_en_only(m.group(1).strip()):
                    has_en_cap = True; break
        if has_en_cap:
            break
    if has_en_cap:
        files_to_process.append(path)

print(f"Processing {len(files_to_process)} files...")
total_fixed = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for n in ex.map(process_file, files_to_process):
        total_fixed += n
print(f"\nDone. Total captions translated: {total_fixed}")
