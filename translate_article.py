#!/usr/bin/env python3
"""Translate an English blog article to natural, idiomatic Chinese using Qwen.
Handles long articles by splitting on ## headings and translating each section."""

import sys
import os
import json
import re
import http.client
import time

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL = "qwen3.5-plus"

SYSTEM_PROMPT = """你是一位资深技术博主，正在将自己的英文博客文章改写为中文版本。

关键要求：
1. 不是翻译，是改写。用你自己的话重新表达，像是你本来就用中文写的一样。
2. 口语化但专业。该用术语时用术语（不翻译专有名词如 Transformer、MoE、RoPE、vLLM、KV cache、softmax 等），但连接语句要自然流畅，像跟同事聊天。
3. 避免翻译腔。不要出现"在本文中""值得注意的是""需要指出的是"这类套话。不要用"的"堆叠长定语。少用被动句。
4. 保持作者的声音。原文有观点、有态度、有经验之谈。中文版也要有同样的锐度和个性。可以加入"我""我们"等第一人称视角。
5. 技术准确。公式（$$...$$和$...$）、代码块、数字、引用出处必须一字不差地保留。LaTeX公式不要改动任何内容。
6. 保留 Markdown 格式。标题层级、代码块（含语言标识）、表格、列表、链接、图片引用全部保留原始格式。
7. 图片路径中的 /en/ 替换为 /zh/，文件名不变。
8. 不要添加任何前言、后记、总结或"以上就是"之类的收尾。直接输出改写后的正文。

/no_think"""

def call_qwen(content: str) -> str:
    conn = http.client.HTTPSConnection("dashscope.aliyuncs.com")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "max_tokens": 16000,
        "temperature": 0.3,
        "stream": False,
    })
    conn.request("POST", "/compatible-mode/v1/chat/completions", payload, headers)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    else:
        print(f"API Error: {json.dumps(data, indent=2)}", file=sys.stderr)
        return None

def split_by_sections(text: str, max_chars: int = 12000):
    """Split markdown text by ## headings into chunks under max_chars."""
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        if line.startswith("## ") and current_len > 500:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    # Merge small chunks
    merged = []
    buf = ""
    for c in chunks:
        if len(buf) + len(c) < max_chars:
            buf = buf + "\n" + c if buf else c
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)

    return merged

def split_frontmatter(text: str):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return "---" + parts[1] + "---", parts[2].lstrip("\n")
    return None, text

def main():
    if len(sys.argv) < 3:
        print("Usage: translate_article.py <en_file> <zh_file>")
        sys.exit(1)

    en_file = sys.argv[1]
    zh_file = sys.argv[2]

    with open(en_file, "r", encoding="utf-8") as f:
        en_text = f.read()

    # Keep existing ZH front matter
    zh_frontmatter = None
    if os.path.exists(zh_file):
        with open(zh_file, "r", encoding="utf-8") as f:
            zh_text = f.read()
        zh_frontmatter, _ = split_frontmatter(zh_text)

    _, en_body = split_frontmatter(en_text)
    print(f"Translating {en_file} ({len(en_body)} chars)", file=sys.stderr)

    chunks = split_by_sections(en_body)
    print(f"  Split into {len(chunks)} chunks", file=sys.stderr)

    translated_parts = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", file=sys.stderr)
        result = call_qwen(chunk)
        if result is None:
            print(f"  FAILED on chunk {i+1}, aborting", file=sys.stderr)
            sys.exit(1)
        translated_parts.append(result)
        if i < len(chunks) - 1:
            time.sleep(1)

    zh_body = "\n".join(translated_parts)
    print(f"  Total Chinese: {len(zh_body)} chars", file=sys.stderr)

    if zh_frontmatter:
        result = zh_frontmatter + "\n" + zh_body
    else:
        en_fm, _ = split_frontmatter(en_text)
        result = (en_fm + "\n" if en_fm else "") + zh_body

    with open(zh_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"  Done: {zh_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
