#!/usr/bin/env python3
"""Polish ZH NLP chapters per-section using qwen-max-2025-01-25.

Splits ZH by ## H2 headings, calls Qwen to rewrite each section with EN reference,
preserves code/math/images, skips References and code-only sections.
"""
import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from openai import OpenAI

API_KEY = os.environ["DASHSCOPE_API_KEY"]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-max-2025-01-25"

EN_DIR = Path("/root/chenk-hugo/content/en/nlp")
ZH_DIR = Path("/root/chenk-hugo/content/zh/nlp")
BACKUP_DIR = Path("/tmp/zh_nlp_polished")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("/root/chenk-hugo/scripts/polish_nlp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("polish")

# (en_stem, zh_stem)
PAIRS = [
    ("introduction-and-preprocessing", "01-NLP入门与文本预处理"),
    ("word-embeddings-lm", "02-词向量与语言模型"),
    ("rnn-sequence-modeling", "03-RNN与序列建模"),
    ("attention-transformer", "04-注意力机制与Transformer"),
    ("bert-pretrained-models", "05-BERT与预训练模型"),
    ("gpt-generative-models", "06-GPT与生成式语言模型"),
    ("prompt-engineering-icl", "07-提示工程与In-Context-Learning"),
    ("fine-tuning-peft", "08-模型微调与PEFT"),
    ("llm-architecture-deep-dive", "09-大语言模型架构深度解析"),
    ("rag-knowledge-enhancement", "10-RAG与知识增强系统"),
    ("multimodal-nlp", "11-多模态大模型"),
    ("frontiers-applications", "12-前沿技术与实战应用"),
]

PROMPT = """你是中文技术写作编辑。我会给你一段技术文章的英文原文和现有中文译文。中文译文写得不够地道，有翻译腔。请你重写中文版本，使其：
1. 完全地道：像中国资深工程师/作者写的，不是翻译。短句优先，主动语态。
2. 保留技术准确性：所有数字、模型名、API 名、论文引用、代码片段保持不变
3. 保留英文专名（BERT、GPT、Transformer、LoRA、RAG、tokenizer 等）不译
4. 全角中文标点 但 URL 和 code 内保持半角
5. 阿拉伯数字
6. 第一人称"我"自然出现，不要"我们"
7. 不要"在...的过程中"、"通过...的方式"、"对...而言"、"换句话说" 这种翻译腔
8. 数学公式 $...$ 和 $$...$$ 保留不动
9. 图片引用 ![...](url) 保留不动
10. markdown 结构（headings、lists、code blocks）保留

只输出重写后的中文，不要解释，不要 wrap 在 markdown code block 里。

## 英文原文
{en_section}

## 现有中文（需要重写）
{zh_section}
"""

SKIP_HEADINGS_PATTERNS = [
    r"^references?$", r"^参考(文献|资料)?$", r"^延伸阅读$", r"^reading list$",
]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def split_by_h2(text: str):
    """Returns (front_matter, intro_before_first_h2, [(h2_line, body), ...])"""
    m = re.match(r"^(---\n.*?\n---\n)", text, re.DOTALL)
    if m:
        fm = m.group(1)
        rest = text[m.end():]
    else:
        fm = ""
        rest = text
    # split on ## (but not ###)
    parts = re.split(r"(?m)^(## [^\n]+\n)", rest)
    # parts[0] = intro, then alternating heading,body,heading,body
    intro = parts[0]
    sections = []
    for i in range(1, len(parts), 2):
        h2 = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        sections.append((h2, body))
    return fm, intro, sections

def is_skip_section(h2_line: str, body: str) -> bool:
    title = h2_line.strip().lstrip("#").strip().lower()
    for pat in SKIP_HEADINGS_PATTERNS:
        if re.match(pat, title, re.I):
            return True
    # code-only: body is mostly code
    no_code = re.sub(r"```.*?```", "", body, flags=re.DOTALL).strip()
    if len(no_code) < 80:
        return True
    return False

def heading_match(en_h2: str, zh_h2: str) -> bool:
    """Loose match by position only — we'll pair by index."""
    return True

def call_qwen(prompt: str, retries=4) -> str:
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            return r.choices[0].message.content
        except Exception as e:
            log.warning("qwen call failed (attempt %d): %s", i, e)
            time.sleep(2 ** i)
    return ""

def polish_chapter(en_stem: str, zh_stem: str, stats: dict, samples: list):
    en_path = EN_DIR / f"{en_stem}.md"
    zh_path = ZH_DIR / f"{zh_stem}.md"
    if not en_path.exists() or not zh_path.exists():
        log.warning("missing pair: %s / %s", en_path, zh_path)
        return

    en_text = en_path.read_text(encoding="utf-8")
    zh_text = zh_path.read_text(encoding="utf-8")

    # backup
    (BACKUP_DIR / f"{zh_stem}.before.md").write_text(zh_text, encoding="utf-8")

    en_fm, en_intro, en_secs = split_by_h2(en_text)
    zh_fm, zh_intro, zh_secs = split_by_h2(zh_text)

    n_min = min(len(en_secs), len(zh_secs))
    new_secs = []
    polished_count = 0

    for i in range(len(zh_secs)):
        zh_h2, zh_body = zh_secs[i]
        if i >= len(en_secs):
            new_secs.append((zh_h2, zh_body))
            continue
        en_h2, en_body = en_secs[i]
        if is_skip_section(zh_h2, zh_body):
            log.info("[skip section] %s :: %s", zh_stem, zh_h2.strip())
            new_secs.append((zh_h2, zh_body))
            continue
        prompt = PROMPT.format(en_section=(en_h2 + en_body).strip(),
                                zh_section=(zh_h2 + zh_body).strip())
        result = call_qwen(prompt)
        if not result or len(result) < 50:
            log.warning("[empty] keep original: %s :: %s", zh_stem, zh_h2.strip())
            new_secs.append((zh_h2, zh_body))
            continue
        # strip if model wrapped in code fence
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n", "", result)
            result = re.sub(r"\n```\s*$", "", result)
        # ensure starts with the H2
        if not result.startswith("## "):
            result = zh_h2 + result + "\n"
        else:
            if not result.endswith("\n"):
                result += "\n"
        # split out h2 from result
        m = re.match(r"^(## [^\n]+\n)(.*)$", result, re.DOTALL)
        if m:
            new_h2 = m.group(1)
            new_body = m.group(2)
        else:
            new_h2 = zh_h2
            new_body = result + "\n"
        new_secs.append((new_h2, new_body))
        polished_count += 1
        stats["polished"] += 1
        if len(samples) < 3:
            samples.append({
                "chapter": zh_stem,
                "section": zh_h2.strip(),
                "before": zh_body[:200],
                "after": new_body[:200],
            })
        time.sleep(0.3)

    # reassemble
    out = zh_fm + zh_intro
    for h2, body in new_secs:
        out += h2 + body
    zh_path.write_text(out, encoding="utf-8")
    log.info("[done] %s polished %d sections", zh_stem, polished_count)

def main():
    stats = {"polished": 0, "chapters": 0}
    samples = []
    for en_stem, zh_stem in PAIRS:
        polish_chapter(en_stem, zh_stem, stats, samples)
        stats["chapters"] += 1
    log.info("DONE %s", stats)
    print(json.dumps({"stats": stats, "samples": samples}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
