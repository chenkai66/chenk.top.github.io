#!/usr/bin/env python3
"""Find paragraphs in ZH articles with mid-CJK English words AND rewrite them via Qwen.

Approach:
- For each ZH article, find paragraphs containing suspect EN words mid-sentence
- Send the paragraph to Qwen-max with explicit instructions to produce natural Chinese
- Replace the paragraph in place
"""
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

# Words frequently leaked through translation - mid-sentence connectives/adverbs that should be Chinese
# Important: don't flag words that are legitimate proper nouns / technical terms in technical context
SUSPECT_WORDS = {
    'broadly', 'loosely', 'mostly', 'essentially', 'actually', 'obviously',
    'eventually', 'usually', 'typically', 'basically', 'generally',
    'specifically', 'literally', 'frankly', 'honestly', 'arguably',
    'roughly', 'approximately', 'precisely',
    'particularly', 'especially', 'primarily', 'mainly',
    'meanwhile', 'however', 'therefore', 'although',
    'somewhat', 'somehow', 'rather', 'fairly',
    'consequently', 'subsequently', 'previously',
    'aggressive', 'aggressively',
    'increasingly', 'gradually', 'rapidly',
    'simply', 'fully', 'completely',
}

def has_suspect_word(line):
    """Check if line has a suspect EN word mid-CJK (excluding code/math regions)."""
    masked = re.sub(r"`[^`]+`", "X", line)
    masked = re.sub(r"\$[^$]+\$", "X", masked)
    masked = re.sub(r"!\[[^\]]*\]\([^)]+\)", "X", masked)
    masked = re.sub(r"\[[^\]]*\]\([^)]+\)", "X", masked)
    masked = re.sub(r"https?://\S+", "X", masked)
    for m in re.finditer(r'\b([A-Za-z]+)\b', masked):
        word = m.group(1).lower()
        if word not in SUSPECT_WORDS: continue
        s, e = m.span()
        left = masked[max(0,s-2):s]; right = masked[e:e+2]
        if re.search(r"[一-鿿]", left) or re.search(r"[一-鿿]", right):
            return True
    return False


SYS = """你是中文技术写作编辑。我会给你一段中文技术博客的段落，里面夹杂了一些没翻译干净的英文副词或形容词（如 broadly, loosely, mostly, essentially, particularly 等），有时候还会有生硬的直译（如"它测的是东西"这种）。

请你把段落改写成自然、流畅、地道的中文，保持原意不变。

规则：
- 把所有夹在中文里的英文副词/连接词翻译成自然的中文表达（broadly → 大体上 / 大致上；loosely → 松散地 / 大致；mostly → 大多 / 主要；particularly → 尤其 / 特别）
- 修正生硬的直译（"它测的是东西" → "它确实测出了某些东西" / "它确实有所衡量"）
- 保留所有 inline code（用反引号包裹的）原样
- 保留 LaTeX 数学公式 $...$ 和 $$...$$ 原样
- 保留所有 markdown 格式（**粗体**、*斜体*、链接、列表标记 - * 1. 等）
- 保留专有名词、模型名、库名英文（MMLU, GSM8K, GPT, Phi-3, Qwen, BERT, CommonCrawl, HuggingFace 等）
- 保留作者引用（如 Brown et al. (2020), *Don't Make Your...*）原样
- 不要添加任何解释性文字，不要改变段落的整体结构和长度
- 输出严格 JSON：{"rewritten": "改写后的段落"}

只输出 JSON，不要其他内容。"""


def call(prompt, retries=3):
    for _ in range(retries):
        k, u = next_key()
        try:
            r = requests.post(u,
                headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
                json={"model": "qwen-max", "temperature": 0.2,
                    "messages": [{"role":"system","content":SYS},{"role":"user","content":prompt}]},
                timeout=120)
            t = r.json()["choices"][0]["message"]["content"].strip()
            if t.startswith("```"):
                t = re.sub(r"^```(?:json)?\n?", "", t); t = re.sub(r"\n?```$", "", t)
            return json.loads(t)
        except Exception as e:
            print(f"  err: {e}"); time.sleep(2)
    return None


def process_file(path):
    with open(path) as f: c = f.read()
    parts = c.split("---", 2)
    if len(parts) < 3: return 0
    fm = "---" + parts[1] + "---"
    body = parts[2]
    lines = body.split("\n")
    in_code = False
    fixes = 0
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        # Skip table rows, headings, bullets-only
        if line.startswith("|") or line.startswith("#"):
            continue
        if not has_suspect_word(line):
            continue
        if len(line.strip()) < 20:
            continue
        # Send the line to Qwen
        result = call(json.dumps({"para": line}, ensure_ascii=False))
        if not result or "rewritten" not in result:
            continue
        new_line = result["rewritten"]
        # Verify no English suspect words remain
        if has_suspect_word(new_line):
            continue
        if new_line != line and len(new_line) > 5:
            lines[i] = new_line
            fixes += 1
    if fixes > 0:
        with open(path, "w") as f:
            f.write(fm + "\n".join(lines))
        print(f"  ✓ {os.path.basename(path)}: {fixes} paras rewritten")
    return fixes


# Find files
files = []
for path in glob.glob("/root/chenk-hugo/content/zh/**/*.md", recursive=True):
    if "_index" in path: continue
    with open(path) as f: c = f.read()
    body = c.split("---", 2)[-1]
    in_code = False
    found = False
    for line in body.split("\n"):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        if line.startswith("|") or line.startswith("#"): continue
        if has_suspect_word(line):
            found = True; break
    if found:
        files.append(path)

print(f"Processing {len(files)} files...")
total = 0
with ThreadPoolExecutor(max_workers=10) as ex:
    for n in ex.map(process_file, files):
        total += n
print(f"\nDone. Total paragraphs rewritten: {total}")
