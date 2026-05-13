#!/usr/bin/env python3
"""full_retranslate.py v2 — Qwen3-max powered ZH article rewriter.
Changes from v1: JSON output format with LaTeX-safe parser, 128 workers."""

import os, sys, glob, re, json, time, threading
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
key_lock = threading.Lock(); idx = [0]
def next_key():
    with key_lock:
        e = API_KEYS[idx[0] % len(API_KEYS)]; idx[0] += 1; return e


def pretest_keys():
    print(f"Pre-testing {len(API_KEYS)} API keys...")
    ok_count = 0
    for i, (key, url) in enumerate(API_KEYS):
        region = "INTL" if "intl" in url else "CN"
        try:
            r = requests.post(url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "qwen3-max", "max_tokens": 20, "temperature": 0.1,
                      "messages": [{"role": "user", "content": "say OK"}]},
                timeout=30)
            if r.status_code == 200:
                print(f"  [{i+1:2d}] {region} {key[:10]}... OK")
                ok_count += 1
            else:
                print(f"  [{i+1:2d}] {region} {key[:10]}... FAIL (HTTP {r.status_code})")
        except Exception as e:
            print(f"  [{i+1:2d}] {region} {key[:10]}... FAIL ({e})")
    print(f"Pre-test: {ok_count}/{len(API_KEYS)} keys working")
    if ok_count == 0:
        print("FATAL: no working keys!"); sys.exit(1)
    return ok_count


REPO = "/root/chenk-hugo"

def parse_fm(content):
    parts = content.split("---", 2)
    if len(parts) < 3: return None, content
    return parts[1], parts[2]

def get_field(fm, field):
    m = re.search(rf'^{field}:\s*"([^"]+)"', fm, re.MULTILINE)
    if m: return m.group(1).strip()
    m = re.search(rf'^{field}:\s*(\S.*?)$', fm, re.MULTILINE)
    return m.group(1).strip() if m else None


def build_series_context(series):
    en_articles = []
    zh_articles = []
    for path in sorted(glob.glob(f"{REPO}/content/en/{series}/*.md")):
        if "_index" in path: continue
        with open(path) as f: c = f.read()
        fm, _ = parse_fm(c)
        if not fm: continue
        title = get_field(fm, "title")
        order = get_field(fm, "series_order") or "?"
        if title:
            en_articles.append({"order": order, "title": title})
    for path in sorted(glob.glob(f"{REPO}/content/zh/{series}/*.md")):
        if "_index" in path: continue
        with open(path) as f: c = f.read()
        fm, _ = parse_fm(c)
        if not fm: continue
        title = get_field(fm, "title")
        order = get_field(fm, "series_order") or "?"
        if title:
            zh_articles.append({"order": order, "title": title})
    series_zh = series_en = series
    try:
        with open(f"{REPO}/themes/chenk/data/series.toml") as f:
            tc = f.read()
        block_re = re.search(rf'\[\[series\]\]\s+id\s*=\s*"{re.escape(series)}"(.*?)(?=\[\[series\]\]|$)', tc, re.DOTALL)
        if block_re:
            block = block_re.group(1)
            zm = re.search(r'name_zh\s*=\s*"([^"]+)"', block)
            em = re.search(r'name_en\s*=\s*"([^"]+)"', block)
            if zm: series_zh = zm.group(1)
            if em: series_en = em.group(1)
    except: pass
    return {
        "series_slug": series,
        "series_zh": series_zh,
        "series_en": series_en,
        "en_articles": en_articles,
        "zh_articles": zh_articles,
    }


SYS = r"""你是一名顶尖的中文技术写作编辑。你的任务是把一篇技术博客的中文版本改写得自然、流畅、地道。

你会收到：
1. **系列上下文**：系列名（中英）、本系列所有文章的标题
2. **本文英文版**：作为语义权威——中文版的意思必须忠实于此
3. **本文当前中文版**：经过初步翻译，但可能有：
   - 夹杂未翻译的英文副词/连接词（broadly, mostly, loosely, particularly 等）
   - 生硬的直译（如"它测的是东西"应该是"它确实测出了某些东西"）
   - 翻译腔（"被认为""作为...的存在""不可避免地"等）
   - 句号过密、短句串成机械感（"X 很 Y。它 Z。默认 W；你可以 V。"）
   - 不通顺的语序

请重写中文版，要求：

**保留不变**：
- 所有 markdown 结构（H2/H3/H4 标题层级、列表、表格、引用、加粗、斜体）
- 所有 §§数字§§ 形式的占位符（包括相对位置）—— 后处理时会被替换回原始的图片、链接、代码块
- 数字、版本号、URL 不动

**风格要求**：
- 自然中文，不要翻译腔
- 长短句搭配，避免机械感（不要每句一个句号）
- 中英文/数字之间有空格（"使用 Qwen 模型"，不是"使用Qwen模型"）
- 中文标点用全角（，。；："" '' ！？），英文上下文除外
- 技术准确性优先于文学性
- 保留原文的语气（轻松、严肃、调侃）
- 保留专有名词、模型名、库名、缩写英文（VPC, RDS, ECS, OSS, MMLU, GSM8K, BERT, MySQL, PolarDB 等）
- 论文引用（Brown et al. (2020), *Title*）原样

**输出格式**（严格遵守）：
输出一个 JSON 对象，只包含一个 body 字段：

{"body": "改写后的完整中文正文..."}

规则：
- 只输出 JSON，不要解释、不要有其他文字
- body 值中的换行输出为实际换行（不需要转义为 \n）
- 如果 JSON 解析器要求，可以用 markdown code fence 包裹
- LaTeX 公式保持原样（$\frac{1}{2}$ 等）"""


def fix_json_backslashes(s):
    """Fix LaTeX backslashes that break JSON parsing.
    \frac -> \\frac, \mathbb -> \\mathbb, etc.
    But keep real JSON escapes: \n \t \r \\ \" \/ \b \f (when not followed by alpha)."""
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '\\' and i + 1 < n:
            nxt = s[i+1]
            if nxt == '\\' or nxt == '"' or nxt == '/':
                result.append(s[i:i+2]); i += 2
            elif nxt == 'u' and i+5 < n and all(c in '0123456789abcdefABCDEF' for c in s[i+2:i+6]):
                result.append(s[i:i+6]); i += 6
            elif nxt in 'nrtbf':
                # \n, \t, etc. could be real JSON escapes OR LaTeX (\nabla, \text, \frac, \boldsymbol)
                # If followed by another alpha char => LaTeX command => double the backslash
                if i+2 < n and s[i+2].isalpha():
                    result.append('\\\\'); result.append(nxt); i += 2
                else:
                    result.append(s[i:i+2]); i += 2
            else:
                # Not a valid JSON escape => double the backslash
                result.append('\\\\'); result.append(nxt); i += 2
        else:
            result.append(s[i]); i += 1
    return ''.join(result)


def parse_json_body(raw):
    """Extract body from Qwen's JSON output with robust LaTeX handling."""
    raw = raw.strip()
    # Strip code fences
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*\n?', '', raw)
        raw = re.sub(r'\n?```\s*$', '', raw)
        raw = raw.strip()

    # Method 1: Try clean JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "body" in obj:
            return obj["body"]
    except json.JSONDecodeError:
        pass

    # Method 2: Fix LaTeX backslashes then retry
    fixed = fix_json_backslashes(raw)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict) and "body" in obj:
            return obj["body"]
    except json.JSONDecodeError:
        pass

    # Method 3: Regex extraction — find "body" : " ... last "
    m = re.search(r'"body"\s*:\s*"', raw)
    if m:
        start = m.end()
        # Find matching close: scan for unescaped " followed by optional whitespace and }
        # Simple approach: last " before final }
        last_brace = raw.rfind('}')
        if last_brace > start:
            end = raw.rfind('"', start, last_brace)
            if end > start:
                body = raw[start:end]
                # Manual unescape (order matters: \\ first)
                body = body.replace('\\\\', '\x00BKSL\x00')
                body = body.replace('\\n', '\n')
                body = body.replace('\\t', '\t')
                body = body.replace('\\"', '"')
                body = body.replace('\\/', '/')
                body = body.replace('\x00BKSL\x00', '\\')
                return body

    # Method 4: Fall back to delimiter format if model ignored JSON instruction
    dm = re.search(r"<<<REWRITE>>>\s*(.*?)\s*<<<END>>>", raw, re.DOTALL)
    if dm:
        return dm.group(1).strip()
    dm2 = re.search(r"<<<REWRITE>>>\s*(.*)", raw, re.DOTALL)
    if dm2:
        return dm2.group(1).strip()

    return None


def call(prompt, retries=3):
    for attempt in range(retries):
        k, u = next_key()
        try:
            r = requests.post(u,
                headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
                json={"model": "qwen3-max", "temperature": 0.2,
                    "messages": [{"role":"system","content":SYS},{"role":"user","content":prompt}],
                    "max_tokens": 16000,
                },
                timeout=600)
            data = r.json()
            if "error" in data:
                err = data.get("error", {}).get("message", "?")
                if "rate" in err.lower():
                    time.sleep(5); continue
                print(f"  api error: {err[:100]}", flush=True); time.sleep(2); continue
            t = data["choices"][0]["message"]["content"]
            body = parse_json_body(t)
            if body:
                return {"rewritten": body}
            print(f"  json parse failed, raw start: {t[:200]}", flush=True)
            time.sleep(2); continue
        except Exception as e:
            print(f"  err: {str(e)[:120]}", flush=True); time.sleep(2)
    return None


def split_into_chunks(body, max_chars=16000):
    sections = []
    cur = []
    for line in body.split("\n"):
        if re.match(r"^## ", line) and cur:
            sections.append("\n".join(cur)); cur = [line]
        else:
            cur.append(line)
    if cur:
        sections.append("\n".join(cur))
    chunks = []
    cur_chunk = []
    cur_size = 0
    for sec in sections:
        if cur_size + len(sec) > max_chars and cur_chunk:
            chunks.append("\n".join(cur_chunk))
            cur_chunk = [sec]; cur_size = len(sec)
        else:
            cur_chunk.append(sec); cur_size += len(sec)
    if cur_chunk:
        chunks.append("\n".join(cur_chunk))
    return chunks


def mask_preserve(body):
    placeholders = []
    def stash(m):
        placeholders.append(m.group(0))
        return f"§§{len(placeholders)-1}§§"
    masked = re.sub(r"```[\s\S]*?```", stash, body)
    masked = re.sub(r"!\[[^\]]*\]\([^)]+\)", stash, masked)
    masked = re.sub(r"\(\s*(/[^)\s]+|https?://[^)\s]+)\s*\)", stash, masked)
    return masked, placeholders


def strip_images_for_context(body):
    body = re.sub(r"```[\s\S]*?```", "[code block]", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body)
    body = re.sub(r"\n\n\n+", "\n\n", body)
    return body


def unmask(text, placeholders):
    def repl(m):
        i = int(m.group(1))
        return placeholders[i] if 0 <= i < len(placeholders) else m.group(0)
    return re.sub(r"§§(\d+)§§", repl, text)


def process_article(zh_path, en_path, ctx):
    with open(zh_path) as f: zh_full = f.read()
    with open(en_path) as f: en_full = f.read()
    zh_fm, zh_body = parse_fm(zh_full)
    en_fm, en_body = parse_fm(en_full)
    if not zh_fm or not en_fm: return False, "no frontmatter"

    zh_title = get_field(zh_fm, "title") or "?"
    en_title = get_field(en_fm, "title") or "?"

    zh_masked, zh_placeholders = mask_preserve(zh_body)
    en_clean = strip_images_for_context(en_body)

    total_size = len(en_clean) + len(zh_masked)
    if total_size < 50000:
        en_chunks = [en_clean]; zh_chunks = [zh_masked]
    else:
        en_chunks = split_into_chunks(en_clean, max_chars=16000)
        zh_chunks = split_into_chunks(zh_masked, max_chars=16000)
        if len(en_chunks) != len(zh_chunks):
            n = max(1, min(len(en_chunks), len(zh_chunks)))
            en_chunks = split_into_chunks(en_clean, max_chars=len(en_clean)//n + 100)
            zh_chunks = split_into_chunks(zh_masked, max_chars=len(zh_masked)//n + 100)
        if len(en_chunks) != len(zh_chunks):
            return False, f"chunk count mismatch {len(en_chunks)} vs {len(zh_chunks)}"

    series_titles = "\n".join(
        f"  {a['order']}. EN: {a['title']}" for a in ctx['en_articles']
    ) + "\n" + "\n".join(
        f"  {a['order']}. ZH: {a['title']}" for a in ctx['zh_articles']
    )

    new_chunks = []
    for i, (en_chunk, zh_chunk) in enumerate(zip(en_chunks, zh_chunks)):
        prompt = f"""## 系列上下文

系列：{ctx['series_zh']} / {ctx['series_en']}
本文 EN 标题：{en_title}
本文 ZH 标题：{zh_title}

系列内所有文章：
{series_titles}

## 本文英文版（仅供语义参考，已剥离图片/代码）

{en_chunk}

## 本文当前中文版（待改写）

{zh_chunk}

请输出改写后的中文版（仅本部分），以 JSON 格式。

**关键规则**（违反则输出无效）：
- 所有 §§数字§§ 形式的占位符必须**原封不动**保留在输出中（包括相对位置）
- 不要在输出中创造任何 ![alt](url) 形式的新图片标签
- 不要在输出中创造任何 ```...``` 形式的代码块
- 输出仅基于"本文当前中文版"改写，**不要**直接抄写英文版的图片/链接/代码"""
        result = call(prompt)
        if not result or "rewritten" not in result:
            return False, f"chunk {i+1}/{len(en_chunks)} failed"
        rewritten_masked = result["rewritten"].strip()
        original_placeholders = set(re.findall(r"§§(\d+)§§", zh_chunk))
        rewritten_placeholders = set(re.findall(r"§§(\d+)§§", rewritten_masked))
        if original_placeholders - rewritten_placeholders:
            missing = original_placeholders - rewritten_placeholders
            return False, f"chunk {i+1}: placeholders dropped ({sorted(list(missing))[:5]})"
        if re.search(r"!\[[^\]]*\]\([^)]+\)", rewritten_masked):
            return False, f"chunk {i+1}: fresh image markdown injected"
        if "```" in rewritten_masked:
            return False, f"chunk {i+1}: fresh code fence injected"
        new_chunks.append(rewritten_masked)

    new_body_masked = "\n\n".join(new_chunks)
    new_body = unmask(new_body_masked, zh_placeholders)
    new_full = "---" + zh_fm + "---\n" + new_body + ("\n" if not new_body.endswith("\n") else "")
    if len(new_body) < len(zh_body) * 0.4 or len(new_body) > len(zh_body) * 3.0:
        return False, f"length anomaly (old={len(zh_body)} new={len(new_body)})"

    with open(zh_path, "w") as f:
        f.write(new_full)
    return True, f"rewritten ({len(zh_body)} -> {len(new_body)} chars)"


def find_pair(zh_path):
    with open(zh_path) as f: c = f.read()
    fm, _ = parse_fm(c)
    if not fm: return None
    series_m = re.search(r'^series:\s*(\S.*?)$', fm, re.MULTILINE)
    order_m = re.search(r'^series_order:\s*(\d+)', fm, re.MULTILINE)
    if not series_m or not order_m: return None
    series = series_m.group(1).strip().strip('"')
    order = order_m.group(1)
    for ep in glob.glob(f"{REPO}/content/en/{series}/*.md"):
        if "_index" in ep: continue
        with open(ep) as f: ec = f.read()
        efm, _ = parse_fm(ec)
        if not efm: continue
        eo = re.search(r'^series_order:\s*(\d+)', efm, re.MULTILINE)
        if eo and eo.group(1) == order:
            return ep
    return None


def process_series(series):
    print(f"\n=== {series} ===")
    ctx = build_series_context(series)
    files = sorted(glob.glob(f"{REPO}/content/zh/{series}/*.md"))
    files = [f for f in files if "_index" not in f]
    print(f"  {len(files)} articles")

    def worker(zh_path):
        en_path = find_pair(zh_path)
        if not en_path:
            return (zh_path, False, "no EN pair")
        ok, msg = process_article(zh_path, en_path, ctx)
        return (zh_path, ok, msg)

    results = []
    with ThreadPoolExecutor(max_workers=128) as ex:
        for zh_path, ok, msg in ex.map(worker, files):
            mark = "✓" if ok else "✗"
            print(f"  {mark} {os.path.basename(zh_path)}: {msg}")
            results.append((zh_path, ok))
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  Series {series}: {n_ok}/{len(results)} succeeded")
    return n_ok


def process_files(file_list):
    """Process a list of specific file paths (for retrying failures)."""
    print(f"\n=== Retry {len(file_list)} files ===")
    series_ctx_cache = {}
    def worker(zh_path):
        # Extract series from path
        parts = zh_path.replace(f"{REPO}/content/zh/", "").split("/")
        series = parts[0] if parts else "unknown"
        if series not in series_ctx_cache:
            series_ctx_cache[series] = build_series_context(series)
        ctx = series_ctx_cache[series]
        en_path = find_pair(zh_path)
        if not en_path:
            return (zh_path, False, "no EN pair")
        ok, msg = process_article(zh_path, en_path, ctx)
        return (zh_path, ok, msg)

    results = []
    with ThreadPoolExecutor(max_workers=128) as ex:
        for zh_path, ok, msg in ex.map(worker, file_list):
            mark = "✓" if ok else "✗"
            print(f"  {mark} {os.path.basename(zh_path)}: {msg}")
            results.append((zh_path, ok))
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  Retry: {n_ok}/{len(results)} succeeded")
    return n_ok


if __name__ == "__main__":
    if "--pretest" in sys.argv:
        pretest_keys()
        sys.exit(0)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Check for --files mode (specific file paths)
    if "--files" in sys.argv:
        file_idx = sys.argv.index("--files")
        file_list = [f"{REPO}/{f}" if not f.startswith("/") else f for f in sys.argv[file_idx+1:]]
        file_list = [f for f in file_list if os.path.exists(f)]
        if file_list:
            process_files(file_list)
        else:
            print("No valid files found")
        sys.exit(0)

    series_list = args if args else ["llm-engineering"]
    total = 0
    for s in series_list:
        total += process_series(s)
    print(f"\n=== TOTAL: {total} articles rewritten ===")
