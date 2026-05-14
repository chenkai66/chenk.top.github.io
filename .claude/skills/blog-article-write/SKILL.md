# Blog Article Writing Skill

## Purpose
Codified workflow for writing high-quality, bilingual technical blog articles on chenk.top (Hugo + custom theme). Covers the complete pipeline from research to deployment.

---

## 1. Article Structure

### Front Matter Template
```yaml
---
title: "Series Name (N): Subtitle"
date: YYYY-MM-DD 09:00:00
tags:
  - Tag1
  - Tag2
categories: Category Name
lang: en
mathjax: false
series: series-slug
series_title: "Full Series Title"
series_order: N
description: "One-paragraph summary (2-3 sentences, what the reader will learn)."
disableNunjucks: true
translationKey: "series-slug-N"
---
```

### Opening (first 2-3 paragraphs)
- Start with a personal hook: a real problem you faced, a mistake you made, or why this matters.
- NEVER open with "In this article we will..." or "This guide covers..." or any variant.
- State what the reader will walk away with by the end.
- Include the cover illustration image right after the opening.

### Body
- Use `##` headings for major sections, `###` for subsections.
- Every `##` section must have real substance (minimum 20-30 lines). No 3-line stub sections.
- Solution-oriented: explain the "why" before the "how", then give detailed steps.
- Include: real CLI commands, real config files, real API responses, tables for comparisons.
- Beginner-friendly: define every term before using it. Build concepts incrementally.
- Code blocks must specify the language: ```bash, ```python, ```hcl, ```json, ```yaml, etc.

### Closing
- Key takeaways as bullet points (3-5 items).
- What is next in the series (link to next article if available).
- NEVER use "In conclusion...", "To summarize...", "以上就是..." or similar filler.

---

## 2. Length & Quality Requirements

| Metric | Minimum | Target |
|--------|---------|--------|
| Lines per article | 500 | 600-800 |
| `##` sections per article | 5 | 7-10 |
| Code blocks per article | 3 | 5-10 |
| Tables per article | 1 | 2-4 |
| Images/diagrams per article | 3 | 5-6 |

### Quality Checklist
- [ ] Every concept explained before first use
- [ ] No wall-of-text paragraphs (max 5 lines per paragraph)
- [ ] All CLI commands are copy-paste-able (no pseudo-code)
- [ ] All config files are complete (not fragments)
- [ ] Comparison tables for any "A vs B" decision
- [ ] At least one real-world example or scenario per major section
- [ ] No broken markdown (check code block fences, table alignment)

---

## 3. Image Requirements

### Cover Image (1 per article)
- Generated via **Wanxiang** (万象) text-to-image API or **matplotlib**
- Style: Professional technical illustration, clean design
- Upload to OSS: `blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/{lang}/{series}/{slug}/illustration_1.png`

### Technical Diagrams (3-5 per article)
- Generated via **matplotlib** Python scripts
- Types: architecture diagrams, flowcharts, comparison charts, process flows
- Style: Dark background (#1a1a2e or #0d1117), accent colors, clean labels
- Naming: `illustration_1.png`, `illustration_2.png`, etc.
- Upload to same OSS path as cover image

### Matplotlib Diagram Guidelines
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
# ... diagram code ...
ax.axis("off")
plt.tight_layout()
plt.savefig("diagram.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117", edgecolor="none")
```

### Image Reference Format in Markdown
```markdown
![Description](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/{series}/{slug}/illustration_N.png)
```

### OSS Upload Command
```bash
aliyun oss cp diagram.png oss://blog-pic-ck/posts/en/{series}/{slug}/illustration_N.png \
  --access-key-id $AK --access-key-secret $SK --region cn-beijing
```


### Script Preservation
All diagram generation scripts MUST be saved to `/root/chenk-hugo/scripts/figures/{series}/`.
- Scripts are the source of truth for reproducibility — if an image needs updating, re-run the script.
- Each script should be self-contained: imports, data generation, plotting, and savefig in one file.
- Name convention: `gen_{topic}.py` for static PNGs, `anim_{topic}.py` for GIF animations.
- GIF animations use `matplotlib.animation.FuncAnimation` + `PillowWriter` (no imageio dependency).

---

## 4. Bilingual Workflow

### Step 1: Write English (Source of Truth)
- Write the complete English article first.
- English version defines structure, content depth, and technical accuracy.

### Step 2: Qwen Chinese Rewrite
- Use `qwen3.5-plus` via DashScope API (OpenAI-compatible endpoint).
- API endpoint: `dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- **CRITICAL**: This is a REWRITE, not a translation. The Chinese version should read like it was written by a Chinese tech blogger.

#### System Prompt for Qwen
```
你是一位资深技术博主，正在将自己的英文博客文章改写为中文版本。

关键要求：
1. 不是翻译，是改写。用你自己的话重新表达，像是你本来就用中文写的一样。
2. 口语化但专业。该用术语时用术语（不翻译专有名词），但连接语句要自然流畅，像跟同事聊天。
3. 避免翻译腔。不要出现"在本文中""值得注意的是""需要指出的是"这类套话。不要用"的"堆叠长定语。少用被动句。
4. 保持作者的声音。原文有观点、有态度、有经验之谈。中文版也要有同样的锐度和个性。
5. 技术准确。公式、代码块、数字、引用出处必须一字不差地保留。
6. 保留 Markdown 格式。标题层级、代码块（含语言标识）、表格、列表、链接、图片引用全部保留原始格式。
7. 图片路径中的 /en/ 替换为 /zh/，文件名不变。
8. 不要添加任何前言、后记、总结或"以上就是"之类的收尾。直接输出改写后的正文。

/no_think
```

#### Chunking Strategy
- Split long articles by `##` headings into chunks under 12,000 characters.
- Translate each chunk separately.
- Merge chunks, joining with `\n`.

### Step 3: Qwen Chinese Title Design
- Send all English titles to Qwen with a prompt requesting natural Chinese titles.
- Format: `系列名（中文数字）：副标题` (subtitle ≤ 15 characters).
- Series name must be consistent across all articles.

### Step 4: Post-Processing
Run a post-processing script on Chinese articles:
1. Fix internal links: `/en/` → `/zh/`
2. Fix stray English words left untranslated (common culprits: specifically, effectively, enough, beyond, shipped, halve)
3. Clean excessive blank lines: `\n{4,}` → `\n\n\n`
4. Verify code blocks are intact (same number of ``` pairs in EN and ZH)

---

## 5. Date Requirements

- Articles in a series are spaced **2 days apart**.
- Time is always `09:00:00`.
- Format: `YYYY-MM-DD 09:00:00`
- New series starts after the latest existing article date on the blog.
- Check latest date: `grep -r "^date:" content/en/ | sort -t: -k2 | tail -5`

---

## 6. Series Setup

### Create Series Directories
```bash
mkdir -p content/en/{series-slug}
mkdir -p content/zh/{series-slug}
```

### Create _index.md for each language
EN:
```yaml
---
title: "Series Title"
description: "One-line description."
---
```

ZH:
```yaml
---
title: "中文系列名"
description: "一行描述。"
---
```

### Add to themes/chenk/data/series.toml
```toml
[series-slug]
  name = "Series Title"
  name_zh = "中文系列名"
  description = "English description."
  description_zh = "中文描述。"
  hue = N  # 0-4, pick unused color
```

---

## 7. Research from alidocs

Reference documentation lives at `/Users/kchen/Desktop/Project/alidocs/`.

### Reading xdita Files
xdita files are HTML-based. Extract text content by stripping tags:
```bash
cat docs/filename.xdita | sed "s/<[^>]*>//g" | head -100
```

### Available Products
| Product | Directory | Doc Count |
|---------|-----------|-----------|
| ECS | ECS_77930_en-US | 186 |
| OSS | OSS_153_en-US | 4,986 |
| RDS | RDS_2184_en-US | 6,533 |
| PAI | PAI_16768_en-US | 2,795 |
| DashScope | DashScope_85185_en-US | 1,696 |
| Terraform | Terraform_2189_en-US | 443 |
| OpenSearch | OpenSearch_93167_en-US | ~500 |
| EventBridge | EventBridge_58874_en-US | ~200 |
| CAS (Certs) | CAS_17104_en-US | ~100 |
| IDaaS | IDaaS_56554_en-US | ~150 |

### Pull Fresh Docs
```bash
cd /Users/kchen/Desktop/Project/alidocs/{product_dir}
adoc icms pull
```

---

## 8. Deployment

### Build
```bash
cd /root/chenk-hugo && hugo --minify
```

### Deploy
```bash
cd /root/chenk-hugo && bash deploy.sh
```

### Commit Source
```bash
cd /root/chenk-hugo && git add content/ themes/ && git commit -m "add: series-name articles" && git push origin source
```

### Verify
- Check series listing page in browser
- Spot-check 2-3 articles (EN and ZH)
- Verify images load correctly
- Verify code blocks render properly
- Check mobile layout

---

## 9. Common Pitfalls

1. **Qwen model name**: Use `qwen3.5-plus` (NOT `qwen3-plus` which does not exist)
2. **Long articles**: Split by `##` headings for Qwen translation (max 12K chars per chunk)
3. **ecs-run-ai4m timeout**: For long operations, use `nohup ... &` and poll log files
4. **Stray English in Chinese**: Always run post-processing script after Qwen translation
5. **Image paths**: EN uses `/en/`, ZH uses `/zh/` — Qwen prompt handles this but verify
6. **Front matter**: Both EN and ZH must have matching `translationKey` for language switcher
7. **Hugo build**: Check for `WARNING` in build output — usually indicates broken refs



---

## 10. Heading Format Rules

### Auto-numbering (CSS + JS)
The site uses JavaScript-based auto-numbering for headings. A numbering.js script assigns section numbers to all h2 and h3 headings AFTER the first horizontal rule in the article body.

Rules:
1. NEVER add manual numbers to headings. No "## 1. Title", no "### 3 Subtitle".
2. Content separator (---) required before numbered content.
3. Headings before the first --- are NOT numbered (intro, prerequisites).
4. Multiple --- between sections are fine for visual separation.

### Heading Hierarchy
- ## = major sections (auto-numbered 1., 2., 3.)
- ### = sub-sections (auto-numbered 1.1, 1.2, 2.1)
- #### = rare, deep nesting only, NOT auto-numbered
- NEVER use flat ## for everything. Sub-topics must use ###.

### When a number IS part of the title (keep it)
- "7 ms budget" (measurement)
- "18 chapters overview" (count)
- "2023 onwards" (year)
- "Top 10 Algorithms" (content)
- "0.0.0 any network" (IP address)

### What to avoid (will double with auto-numbering)
- "### 1 SVRG algorithm" renders as "6.1 1 SVRG"
- "## 3. Convex analysis" renders as "1. 3. Convex"

---

## 11. Math (LaTeX) Rules

### Passthrough Extension
Hugo uses Goldmark passthrough with $ (inline) and $$ (block). KaTeX renders client-side.

### Critical: nabla corruption
The backslash-n in \nabla can be consumed as a newline during Python file writes or Qwen translation. After any batch processing:
- Grep for bare "abla" in math contexts
- Use raw strings (r"\nabla") in Python
- Same risk applies to: \newcommand, \neq, \nu, \newline

### Blockquote Math
Hugo cannot handle multi-line $$ blocks inside blockquotes. Always use single-line: > $$content$$
