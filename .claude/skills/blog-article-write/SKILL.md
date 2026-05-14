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
8. **Markdown table cells with `\|` in math**: breaks column count — use `\mid` instead. See §12.2.
9. **Tables looking visually broken**: ensure `layouts/_default/_markup/render-table.html` exists. See §12.1.
10. **Plain-text "Part N" / "第 N 章" mentions**: must be hyperlinks to the target chapter. See §13.
11. **OSS image-path slug variants** (Title-Case vs lowercase, truncated vs full): pick one per article, verify with HEAD-check. See §14.
12. **Wrong Part numbers in series text**: after any series re-shuffle, grep every "Part N" / "第 N 章" and verify against current outline. See §13.3.



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

---

## 12. Markdown Tables

### 12.1 Required render hook (CRITICAL — root-cause fix for "tables look weird")

**Symptom**: 2- or 3-column tables render with rows the wrong width, columns overflow the article column on desktop, ZH tables in particular get pushed out beyond the layout.

**Root cause**: Goldmark emits a raw `<table>` element. The site's CSS only applies `overflow-x: auto` to `.prose .table-wrap > table`, which requires the table to be wrapped in a `<div class="table-wrap">` element. Without a Hugo render hook, raw tables get no scroll container and overflow.

**Fix (already installed at `layouts/_default/_markup/render-table.html`)**: this hook wraps every markdown table in `<div class="table-wrap">` so the existing CSS kicks in.

```html
<div class="table-wrap">
<table>
  <thead>
    {{- range .THead }}
    <tr>{{- range . }}<th{{ with .Alignment }} style="text-align: {{ . }}"{{ end }}>{{ .Text | safeHTML }}</th>{{- end }}</tr>
    {{- end }}
  </thead>
  <tbody>
    {{- range .TBody }}
    <tr>{{- range . }}<td{{ with .Alignment }} style="text-align: {{ . }}"{{ end }}>{{ .Text | safeHTML }}</td>{{- end }}</tr>
    {{- end }}
  </tbody>
</table>
</div>
```

If this file is ever deleted, every table on the site loses its scroll container. After editing the hook, run `hugo --minify` to verify the output contains `class="table-wrap"`.

### 12.2 Cell escaping — the `\|` pipe trap

Markdown table parsers split rows on every `|`, including `\|` written inside `$...$` math. So a cell like:

```
| Distribution | $P_S(Y\|X)$ | $P_T(Y\|X)$ |
```

…is parsed as a 5-cell row, not 3, and the table renders with broken column widths.

**Rule**: never write `\|` inside a table cell. Always use `\mid`:

```
| Distribution | $P_S(Y\mid X)$ | $P_T(Y\mid X)$ |
```

`\mid` renders identically as `|` in KaTeX but doesn't trip the markdown table parser. This applies to BOTH EN and ZH copies — fix in pairs.

Detection script (catches column-count mismatches in tables):

```python
# Run from the repo root after edits
import re, glob
for path in glob.glob("content/{en,zh}/*/*.md"):
    with open(path) as f: lines = f.read().split("\n")
    in_table = False; hdr_n = 0
    for i, line in enumerate(lines, 1):
        if not re.match(r"^\s*\|.*\|\s*$", line): in_table = False; continue
        n = len(line.strip().strip("|").split("|"))
        if not in_table: in_table = True; hdr_n = n
        elif n != hdr_n and not re.match(r"^\s*\|?\s*:?-{2,}", line):
            print(f"{path}:{i}  cols={n} hdr={hdr_n}")
```

### 12.3 Wide-table best practice

Even with the wrap, tables wider than ~6 columns are painful on mobile. When a comparison naturally has 7+ columns:
- **Split** into two side-by-side smaller tables; OR
- **Transpose** so columns become rows; OR
- **Move detail to prose** below the table.

Don't rely on horizontal scroll alone — readers often miss off-screen columns.

---

## 13. Cross-chapter Links (mandatory in series)

Whenever a sentence references another chapter / part / section of the same series, it must be a hyperlink, not plain text. Plain "Part 9" or "第 9 章" mentions render as inert grey text and don't help navigation.

### 13.1 The patterns to grep for

After any expansion or audit, run these greps and link every hit (allow only the ones already inside `[...](...)` to pass):

```bash
# EN
grep -nE '(Chapter|Part|Section) [0-9]+' content/en/<series>/*.md \
  | grep -vE '\]\(|http|^[^:]*:\s*#'

# ZH
grep -nE '第 ?[0-9０-９一二三四五六七八九十]+ ?[章节部]' content/zh/<series>/*.md \
  | grep -vE '\]\(|http|^[^:]*:\s*#'
```

### 13.2 Link form

```markdown
EN intra-series:    [Part 9](/en/<series>/09-<slug>/)
EN intra-article:   [Section 4](#section-anchor)        ← auto-slugified Hugo anchor
ZH intra-series:    [第 9 章](/zh/<series>/09-<slug>/)
```

Use the **URL slug** Hugo serves the article at (lowercase, kebab-case for EN; the original Chinese filename for ZH). Verify by visiting the link in the dev server before committing.

### 13.3 Factual numbering check

Every "Part N" / "第 N 章" claim names a specific chapter — a stale series outline introduces off-by-one errors that misdirect readers. After any series re-shuffle:
1. List the actual chapter order from the `_index.md` / `series.toml`.
2. Grep every "Part N" / "第 N 章" in the prose.
3. Verify N matches the current series numbering, not an old draft.

E.g., during the transfer-learning expansion an article said "distillation (Part 6)" when distillation is actually Part 5; "continual learning, Part 9" when it's Part 10. These survive Qwen translation untouched, so they're easy to miss.

---

## 14. OSS image-path slug consistency

Each article has TWO independent slugs:
1. **Hugo URL slug** — used in `[link](/en/<series>/<slug>/)` and the article's URL on chenk.top.
2. **OSS image directory** — used in `posts/{en,zh}/<series>/<dir>/<file>.png`.

These don't have to match, but they MUST be consistent across every image reference in a given article. Mixing case-variants causes 404s on case-sensitive OSS:

```
✗ posts/en/transfer-learning/12-Industrial-Applications/foo.png       (Title-Case)
✗ posts/en/transfer-learning/12-industrial-applications/foo.png       (lowercase, truncated)
✓ posts/en/transfer-learning/12-industrial-applications-and-best-practices/foo.png
```

When uploading new figures, pick one canonical OSS dir name per article and stick with it. Recommended: lowercase-kebab, full slug. Document the chosen name at the top of the figure-generation script:

```python
# OSS dir for this article: 12-industrial-applications-and-best-practices
OSS_PATH = "posts/en/transfer-learning/12-industrial-applications-and-best-practices"
```

After bulk-uploading, run a HEAD-check on every image URL referenced from the article's markdown (run from a server with healthy DNS — local Mac → blog-pic-ck OSS sometimes has timeouts, use ai4m or the deploy server).

### 14.1 Mistake-prone alt + path mapping in ZH copies

When duplicating an EN article's image references into ZH, it's easy to leave the path pointing at `/posts/en/...`. The image will still render (because the EN bucket has it), but a future re-render of localized labels won't appear in ZH. Always flip both the alt-text language AND the URL path to `/posts/zh/...`.


---

## 23. 端到端流水线（题材 → 上线，每步一条命令）

弱模型按这 11 步顺序执行就能从题目走到上线。每步如果失败，跳到 §28 故障恢复表对症处理。

### 步骤总览

```
┌─[1] 解析 ──┐    ┌─[3] 写 EN ──┐    ┌─[6] 配图 ──┐    ┌─[9] validate ──┐
│ 题材→outline│ → │ 1..N 篇    │ → │ matplotlib │ → │ exit code 0    │
└───────────┘    └────────────┘    └────────────┘    └────────────────┘
                                                             ↓
┌─[2] 注册 ──┐    ┌─[4] 写 ZH ──┐    ┌─[7] 上传 ──┐    ┌─[10] deploy ──┐
│series.toml │    │ 不机翻      │    │ EN+ZH 两路  │    │ deploy.sh      │
└───────────┘    └────────────┘    └────────────┘    └────────────────┘
                                          ↓
                                   ┌─[5] 封面 ──┐    ┌─[11] verify ──┐
                                   │ Wanxiang  │    │ curl HEAD 200  │
                                   └────────────┘    └────────────────┘
```

### 每步的具体动作（COPY-EXEC 友好）

| # | 操作 | 命令 / 产出 |
|---|---|---|
| 1 | 解析题材 → outline JSON | `python3 scripts/topic_to_outline.py "用 Terraform 部署 AI Agent" 8` → `/tmp/outline.json` |
| 2 | 注册 series（仅新建时） | 编辑 `themes/chenk/data/series.toml` 添加条目（参考 §6） |
| 3 | 写 EN 全部 N 篇 | 按 outline.json 每章 600-1500 字（教程） / 2500 字（深技），结构：intro → H2 序列 → "What's next" |
| 4 | 写 ZH 全部 N 篇 | 调 §24 的 Qwen 翻译模板，**逐章**翻译，过 §11 翻译腔黑名单 |
| 5 | 生成封面 | `python3 scripts/gen_article_covers.py <series> 1..N` |
| 6 | 跑 matplotlib | `python3 scripts/figures/<series>/gen_*.py` → `/tmp/figs/*.png` |
| 7 | 上传 OSS | `python3 scripts/upload_oss.py <series>`（同时传 EN+ZH 两路径） |
| 8 | 插入 image refs | 在每篇相应 H2 后加 `![caption](https://blog-pic-ck.../posts/{lang}/{series}/{slug}/figX.png)` |
| 9 | **自动校验** | `bash scripts/validate.sh <series>`（exit 0 才能继续，见 §26） |
| 10 | 部署 | `cd /root/chenk-hugo && bash deploy.sh "Add <series> (N articles)"` |
| 11 | live 校验 | `for n in $(seq 1 N); do curl -sI https://www.chenk.top/{en,zh}/<series>/<NN-slug>/ -o /dev/null -w '%{http_code}\n'; done` 应全 200 |

### 中断与恢复

每步**完成后**写一个 marker 文件 `/tmp/<series>.step{N}.done`，下次进入直接 `ls /tmp/<series>.step*.done | wc -l` 判断断点，从 N+1 开始。让弱模型即使 session 中断也能续上。

---

## 24. Qwen prompt 模板库（弱模型直接调用）

下面所有 prompt 走 `ecs-run keys` 配的池（CN/INTL DashScope，见全局 `~/.claude/CLAUDE.md`）。模型选 `qwen3-max`（翻译质量明显高于 plus）。

### 24.1 Outline 生成（题材 → JSON）

```python
SYSTEM = """你是一名资深技术博客架构师。给定一个题材，规划 N 章的 series。

强制输出 JSON：
{
  "series_slug": "kebab-case",
  "series_title_en": "...",
  "series_title_zh": "...",
  "hue": 1-4,
  "chapters": [
    {
      "n": 1,
      "slug_en": "01-foundations",        # NN-kebab
      "slug_zh": "01-基础",
      "title_en": "...",
      "title_zh": "...",
      "h2_sequence": ["Intro hook", "What it is", "Hands-on", "Edge cases", "What's next"],
      "code_focus": "what concrete code reader writes",
      "depends_on_chapters": []           # 仅引用更早的 chapter
    },
    ...
  ]
}

规则:
- 4 ≤ N ≤ 12
- 每章必须独立可读（开头 1 段交代背景）
- "What's next" 必须是每章最后一个 H2
- 章节之间要递进，不要并列堆砌
- 每章 H2 序列 4-7 个

只输出 JSON，不要解释。"""

USER = f"题材: {topic}\nN: {n_chapters}\n请输出 outline JSON。"
```

### 24.2 EN 写作（每章一次调用）

```python
SYSTEM = """你是 chenk.top 的英文技术博客作者。Voice: first-person, dry, wry, "I tried X, Y broke, Z fixed it".

强制规则:
- 600-1500 字（教程）/ 2500 字（深技）
- 短句优先，段落 3-5 句
- 每个论断配证据：代码片段、数字、具体案例
- 禁止: "In this article we will...", "As you can see", "Let's dive in", "It's important to note"
- 允许: "I've been bitten by this twice", "the docs are thinnest here"
- 只用 ## H2 / ### H3，不用 H4
- 代码 fenced + lang: ```python / ```bash / ```hcl
- 数学用 $...$ inline / $$...$$ display；表格里数学竖线用 \\mid 不用 \\|
- 末尾 H2 是 "What's next"（series 终篇用 "Where to go from here"）

输出: 完整 markdown，含 front matter（YAML）。
front matter 必填字段: title, date, tags, categories, lang, series, series_order, series_title, mathjax, disableNunjucks, description, translationKey
"""

USER = f"""series: {outline['series_slug']}
order: {chapter['n']}
title: {chapter['title_en']}
H2 sequence: {chapter['h2_sequence']}
code focus: {chapter['code_focus']}
date: {chapter_date}  # YYYY-MM-DD ≤ today
translationKey: {series_slug}-{chapter['n']}

写完整文章。"""
```

### 24.3 ZH 写作（不是翻译，独立写）

```python
SYSTEM = """你是 chenk.top 的中文技术博客作者。下面给你 EN 版本作为参考，但你**不是翻译**——你要重新组织语言、用中文母语者的思维表达同一组技术内容。

强制规则:
- 中文母语写作，短句、口语化
- **翻译腔黑名单**（每条都不能出现）:
  * "在 X 的过程中" → "做 X 时"
  * "我们将" → "本文" / 直接动词
  * "被设计成" → "用来" / "目的是"
  * "进行 X" → "做 X" / 直接动词
  * "对于 X 来说" → "X 这个东西"
- 数学符号、专业术语保留英文（LoRA, MMD, transfer learning, fine-tune）
- LaTeX / 代码 / markdown 结构（H2/H3/表格/链接）一字不差保留
- 表格里 \\| 改 \\mid（防破坏列）
- front matter 单独翻 title 和 description；series/series_order/tags/translationKey 保持与 EN 一致
- 末尾 H2 用 "接下来" 或 "总结"
"""

USER = f"""EN 参考:
---
{en_content}
---

date: {chapter_date}
请输出 ZH 版本完整 markdown（含 front matter）。"""
```

### 24.4 调用模板（带重试+key 池）

```python
import sys, json, urllib.request, time
sys.path.insert(0, "/usr/local/lib/ecs-run")
from ecs_run_keys import KeyPool

pool = KeyPool.load()

def call_qwen(system, user, model="qwen3-max", max_retries=3):
    for attempt in range(max_retries):
        key, base_url = pool.next(provider="dashscope", model=model)
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": 32000,
            "temperature": 0.3,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"attempt {attempt+1} failed: {e}", file=sys.stderr)
            pool.report_failure(key)
            time.sleep(2 ** attempt)
    raise RuntimeError("all retries exhausted")
```

### 24.5 输出验证（每次 Qwen 调用后必跑）

```python
def assert_well_formed(md, expected_h2_count, lang):
    # 1. front matter 边界
    assert md.startswith("---\n"), "missing front matter open"
    fm_end = md.find("\n---\n", 4)
    assert fm_end > 0, "missing front matter close"

    # 2. front matter 必填字段
    fm = md[4:fm_end]
    for field in ["title:", "date:", "lang:", "series:", "series_order:", "translationKey:"]:
        assert field in fm, f"missing {field}"

    # 3. H2 数量
    h2_count = sum(1 for line in md.split("\n") if line.startswith("## "))
    assert h2_count == expected_h2_count, f"got {h2_count} H2, want {expected_h2_count}"

    # 4. 末尾 H2 是 What's next / 接下来
    last_h2 = [l for l in md.split("\n") if l.startswith("## ")][-1]
    expected_tail = ("What's next", "Where to go from here") if lang == "en" else ("接下来", "总结")
    assert any(t in last_h2 for t in expected_tail), f"last H2 wrong: {last_h2}"

    # 5. 表格里 \| 不能存在（用 \mid）
    for line in md.split("\n"):
        if line.strip().startswith("|") and "\\|" in line:
            raise AssertionError(f"table cell uses \\|, change to \\mid: {line[:80]}")

    return True
```

每次 Qwen 写出一篇就跑这个 assert；fail 就重试一次（重新 prompt + 加上"上次输出错在 X，请修正"）。

---

## 25. Series 连续性规范（合并 §14/15/16/21 并补完）

一个 series 的 N 篇必须满足以下所有连续性规则。这是新模型最容易翻车的地方。

### 25.1 强制不变量

| 不变量 | 检查 |
|---|---|
| `series:` 字段 | N 篇 markdown 全部用同一个 slug（=子目录名） |
| `series_order:` | 1, 2, 3, ..., N，无跳号、无重复 |
| `series_title:` | N 篇全部相同 |
| `tags:` | 共享至少 1 个核心 tag（series 本身的标签） |
| `translationKey:` | EN/ZH 同一篇必须相同；不同篇必须不同 |
| `date:` | EN 严格递增（间隔 1-2 天）；ZH 由 deploy 自动同步，**不要手动写** |
| H2 数量 | EN 与 ZH 同篇 H2 数量必须一致（`grep -c '^## '`） |
| 图片数 | EN 与 ZH 同篇图片数必须一致 |
| 末尾 H2 | EN: "What's next" / "Where to go from here"；ZH: "接下来" / "总结" |
| 末尾 footer 段 | series-nav: 上一篇/下一篇 链接（见 25.3 模板） |

### 25.2 章节衔接桥（每章开头/结尾）

**第 N 篇开头**（除第 1 篇）必须有 1-2 句"承上"：

EN: `> Previously, we set up the cluster (Part 2). Now we wire in the autoscaler.`
ZH: `> 上一篇我们搭好了集群（第 2 篇），这一篇接上 autoscaler。`

**第 N 篇结尾**（除最后一篇）必须有 1-2 句"启下"：

EN: `Next, we'll deal with the cold-start latency this approach introduces.`
ZH: `下一篇会处理这个方案带来的冷启动延迟问题。`

避免每章独立到读者不知道前后顺序。

### 25.3 series-nav footer 模板（必装）

每篇文章 markdown 最末尾插入一段（在所有 H2 之后，文档结束之前）：

```markdown
---

*This is Part {N} of [{series_title}](/en/series/{series-slug}/) ({total} parts).
Previous: [Part {N-1} — {prev_title}](/en/{series-slug}/{NN-1}-{prev_slug}/) ·
Next: [Part {N+1} — {next_title}](/en/{series-slug}/{NN+1}-{next_slug}/)*
```

ZH 版：

```markdown
---

*本文是 [{series_title_zh}](/zh/series/{series-slug}/) 系列的第 {N} 篇，共 {total} 篇。
上一篇: [第 {N-1} 篇 — {prev_title_zh}](/zh/{series-slug}/{NN-1}-{prev_slug_zh}/) ·
下一篇: [第 {N+1} 篇 — {next_title_zh}](/zh/{series-slug}/{NN+1}-{next_slug_zh}/)*
```

第 1 篇省略 "Previous"，第 N 篇省略 "Next"。

### 25.4 跨章引用必须超链接

任何 inline 引用 `Part 9 / 第 9 章 / Section 4` 必须是 `[Part 9](/en/...)` 形式。检测见 §21.1。重排过的 series 要再核对编号是否还匹配（off-by-one 是最常见错误）。

### 25.5 series 全篇校验（一次性跑）

```bash
SERIES=transfer-learning
EN_DIR=/root/chenk-hugo/content/en/$SERIES
ZH_DIR=/root/chenk-hugo/content/zh/$SERIES

# 1. 篇数对齐
en=$(ls $EN_DIR/*.md | grep -v _index | wc -l)
zh=$(ls $ZH_DIR/*.md | grep -v _index | wc -l)
[ "$en" = "$zh" ] || echo "MISMATCH: EN=$en ZH=$zh"

# 2. series_order 无跳号
for f in $EN_DIR/*.md; do grep -E '^series_order:' $f; done | sort -t: -k2 -n

# 3. translationKey 配对
diff <(grep -h translationKey $EN_DIR/*.md | sort) <(grep -h translationKey $ZH_DIR/*.md | sort)

# 4. 每篇 H2 数量对齐
for n in 01 02 03 04 05 06 07 08 09 10 11 12; do
  en_h2=$(grep -c '^## ' $EN_DIR/${n}-*.md 2>/dev/null)
  zh_h2=$(grep -c '^## ' $ZH_DIR/${n}-*.md 2>/dev/null)
  [ "$en_h2" = "$zh_h2" ] || echo "Art $n H2 mismatch: EN=$en_h2 ZH=$zh_h2"
done

# 5. series-nav footer 存在
grep -L 'Part [0-9]\+ of\|本文是.*系列' $EN_DIR/*.md $ZH_DIR/*.md
```

---

## 26. validate.sh 自动校验（部署前必跑）

放 `scripts/validate.sh`。弱模型只要 `bash scripts/validate.sh <series>` 拿 exit 0 就允许 deploy；非 0 则按输出修。

### 脚本骨架

```bash
#!/bin/bash
# scripts/validate.sh — gate before deploy
set -u
SERIES="${1:?usage: validate.sh <series-slug>}"
EN=/root/chenk-hugo/content/en/$SERIES
ZH=/root/chenk-hugo/content/zh/$SERIES
ERR=0

fail() { echo "✗ $*"; ERR=$((ERR+1)); }
ok()   { echo "✓ $*"; }

# A. 篇数对齐
en=$(ls $EN/*.md 2>/dev/null | grep -v _index | wc -l)
zh=$(ls $ZH/*.md 2>/dev/null | grep -v _index | wc -l)
[ "$en" = "$zh" ] && ok "篇数对齐 ($en)" || fail "EN=$en ZH=$zh"

# B. front matter 必填字段
for f in $EN/*.md $ZH/*.md; do
  for field in "title:" "date:" "lang:" "series:" "series_order:" "translationKey:"; do
    grep -q "^$field" $f || fail "$f 缺 $field"
  done
done

# C. H2 末尾合规
for f in $EN/*.md; do
  last=$(grep -E '^## ' $f | tail -1)
  [[ "$last" == *"What's next"* || "$last" == *"Where to go from here"* ]] \
    || fail "$f 末尾 H2 不合规: $last"
done
for f in $ZH/*.md; do
  last=$(grep -E '^## ' $f | tail -1)
  [[ "$last" == *"接下来"* || "$last" == *"总结"* || "$last" == *"展望"* ]] \
    || fail "$f 末尾 H2 不合规: $last"
done

# D. 表格里没有 \|（应改 \mid）
for f in $EN/*.md $ZH/*.md; do
  if grep -nE '^\|.*\\\|.*\|' $f > /dev/null; then
    fail "$f 表格 cell 含 \\|（改用 \\mid）"
  fi
done

# E. 未来日期
today=$(date +%Y-%m-%d)
for f in $EN/*.md $ZH/*.md; do
  d=$(grep -m1 '^date:' $f | awk '{print $2}')
  [[ "$d" > "$today" ]] && fail "$f date=$d 在未来"
done

# F. 图片 URL HEAD 检查（在 ai4m 网络环境跑）
imgs=$(grep -hoE 'https://blog-pic-ck[^)]+' $EN/*.md $ZH/*.md | sort -u)
broken=0
for u in $imgs; do
  code=$(curl -sI -o /dev/null -w '%{http_code}' "$u")
  [ "$code" != "200" ] && fail "img $code: $u" && broken=$((broken+1))
done
[ "$broken" = "0" ] && ok "$(echo $imgs | wc -w) 张图全部 200"

# G. Hugo 构建无 warning
cd /root/chenk-hugo
if hugo --minify 2>&1 | grep -iE 'WARN|ERROR|broken'; then
  fail "Hugo 构建有 warning/error"
else
  ok "Hugo 构建无 warning"
fi

# H. 跨章引用超链接化
for f in $EN/*.md; do
  unlinked=$(grep -nE '(Chapter|Part|Section) [0-9]+' $f | grep -vE '\]\(|http|^[^:]*:\s*#' | head -3)
  [ -n "$unlinked" ] && fail "$f 有未链接的 Part/Section 引用:\n$unlinked"
done
for f in $ZH/*.md; do
  unlinked=$(grep -nE '第 ?[0-9]+ ?[章节部]' $f | grep -vE '\]\(|http|^[^:]*:\s*#' | head -3)
  [ -n "$unlinked" ] && fail "$f 有未链接的 第N章 引用:\n$unlinked"
done

echo "==="
[ "$ERR" = "0" ] && { echo "ALL PASSED — safe to deploy"; exit 0; } \
                 || { echo "FAILED $ERR check(s) — fix before deploy"; exit 1; }
```

### 用法

```bash
bash /root/chenk-hugo/scripts/validate.sh transfer-learning
echo "exit: $?"   # 必须是 0
# 失败时按输出逐项修，再跑直到 exit 0
```

---

## 27. 最小示范文章（hello-world，每条规则各满足一次）

新模型如果没见过 chenk.top 文章长什么样，先抄这一篇当模板再改。

### EN: `content/en/standalone/example-article.md`

```markdown
---
title: "How I Stopped Hating YAML Anchors"
date: 2026-04-15 09:00:00
tags:
  - yaml
  - workflow
  - editor-tools
categories: tooling
lang: en
series: ""              # standalone 留空
series_order: 0
mathjax: false
disableNunjucks: true
description: "YAML anchors looked like a footgun until I tried them on a 200-line CI config. Then they earned their keep."
translationKey: "example-article"
---

YAML anchors have a reputation. The syntax (`&name` / `*name`) looks alien, the docs are thinnest where the cases get interesting, and "merge keys" sound like an obscure category-theory operation. I dodged them for years.

Then I rewrote a 200-line GitHub Actions config that had three near-identical jobs, and I gave anchors a real chance. They paid off in 30 minutes.

---

## What an anchor actually is

An anchor is a label on any YAML node. A reference (`*name`) substitutes that node's value. That's the whole feature.

```yaml
defaults: &job_defaults
  runs-on: ubuntu-latest
  timeout-minutes: 10

build:
  <<: *job_defaults
  steps: [...]

test:
  <<: *job_defaults
  steps: [...]
```

The `<<:` is the **merge key** — it splices the referenced map into the current map. Three jobs, one canonical block.

## The footgun I was afraid of

I assumed anchors were lexical macros. They're not — they're parsed at YAML-load time, and the resolver only sees one big tree. So you can't reference an anchor from a *different file* unless your loader supports it (most don't).

What this meant in practice: my "shared anchors" lived inside the same file. That's fine for one CI config, awkward across a project.

## When I'd skip them

- Config under 50 lines: not worth the cognitive cost
- Multi-file split where you can't merge anchors anyway
- Audience unfamiliar with YAML: anchors look strange, reviewers will pause

## What's next

Next time I'll cover **multi-document YAML** (`---` separators) and how Kubernetes uses them deliberately for resource lists.
```

### ZH: `content/zh/standalone/example-article.md`

```markdown
---
title: "我是怎么不再讨厌 YAML 锚点的"
date: 2026-04-15 09:00:00
tags:
  - yaml
  - workflow
  - editor-tools
categories: tooling
lang: zh
series: ""
series_order: 0
mathjax: false
disableNunjucks: true
description: "YAML 锚点看着像踩坑利器,直到我把它用在一份 200 行的 CI 配置上,瞬间真香。"
translationKey: "example-article"
---

YAML 锚点声誉不太好。语法 (`&name` / `*name`) 看着像外星人,文档在关键的地方又最薄,merge key 听起来像 category theory 里冷僻的概念。我躲了好几年。

后来我重写了一份 200 行的 GitHub Actions,里面有三个几乎一样的 job——给锚点一个真正的机会,半小时回本。

---

## 锚点到底是什么

锚点就是给 YAML 节点打个标签,引用 (`*name`) 取这个节点的值。功能就这么点。

```yaml
defaults: &job_defaults
  runs-on: ubuntu-latest
  timeout-minutes: 10

build:
  <<: *job_defaults
  steps: [...]

test:
  <<: *job_defaults
  steps: [...]
```

`<<:` 是 **merge key**,把引用的 map 拼进当前 map。三个 job,一份 canonical 配置。

## 我以前怕的那个坑

我以为锚点是 lexical macro,其实不是——它在 YAML 解析期处理,resolver 只看到一棵完整的树。结果是:跨文件引用基本不行(除非 loader 专门支持,大多不支持)。

实际意味着:"共享锚点"只能放同一个文件里。一份 CI 配置足够,跨工程就不太顺手。

## 什么时候我会跳过它

- 50 行以下的配置:认知成本不值
- 多文件拆分,跨文件 merge 用不上
- 团队不熟 YAML:锚点写法陌生,review 会卡

## 接下来

下一篇我打算讲 **multi-document YAML** (`---` 分隔符) ,以及 Kubernetes 怎么用它来描述资源列表。
```

每条规则在这两份文件里都有体现:front matter 11 字段、第一人称 dry voice、3-5 句段落、码块带 lang、末尾 H2 是 "What's next"/"接下来"、ZH 不机翻、`description:` 各自独立写。新模型先看这两份,再产新内容时按结构平移。

---

## 28. 故障恢复决策树（弱模型遇错查表）

| 现象 | 原因 | 修法 | 后续 |
|---|---|---|---|
| Qwen 输出缺 H2(数量不对) | 上下文太长被截断 | 把章节拆 2 次调用,每次只写一半;或换 `qwen3-max-preview`(更大上下文) | §24.5 assert 重跑 |
| Qwen 输出缺 front matter | prompt 没强调或被吃掉 | 重 prompt,把 "front matter 必填字段" 移到 SYSTEM 顶部 | §24.5 assert 重跑 |
| Qwen 输出有翻译腔(ZH) | model 退化到机翻模式 | 在 USER prompt 末尾追加"你的输出不能出现以下短语:在...的过程中、我们将、被设计成。每出现一次重写整段。" | 抽样人工验 |
| Qwen 调用 HTTP 429 | 该 key 撞 rate limit | `pool.report_failure(key)` + 1s 退避后重试 | 见 §24.4 |
| Qwen 调用 HTTP 400 max_tokens | 该模型最大 max_tokens 不一致 | 先用 `qwen3.6-flash` 试一遍,确认 max_tokens 合法 | 长输出用 32000 |
| Hugo build WARN broken refs | 某 markdown 引用了不存在的页面 | `hugo --minify 2>&1 \| grep -i broken` 看哪个文件,核对 link target | re-build |
| Hugo build WARN translationKey duplicate | 两篇文章用了同一 translationKey | grep `translationKey:` 找到双胞胎,改其中一个 | re-build |
| validate.sh 报"H2 mismatch" | EN 写得比 ZH 多/少一个 H2 | 回看 EN 结构,把缺的 H2 补回 ZH(或反向);保持两边对齐 | 重跑 validate |
| validate.sh 报"img 404" | OSS 路径写错或没传 | 跑 §22 的 HEAD 检查找到错位,要么重传要么改 markdown URL | 重跑 validate |
| `bash deploy.sh` push 被拒 | 历史 commit 含真 secret | `git filter-repo` 抹掉 + 旋转 secret;或 deploy.sh 容忍 source push 失败 | 重 push |
| `bash deploy.sh` 卡住 | hugo 编译 OOM 或某个文章死循环 shortcode | `kill -9` hugo;`hugo -v` 找慢章节;移走逐一二分 | 修后 re-deploy |
| live URL 200 但内容旧 | GitHub Pages CDN 缓存 | 等 60-90 秒再试;不要硬刷 deploy 多次 | curl 重检 |
| 表格列宽崩了 | cell 含 `\|` 数学竖线 | 改 `\mid`(见 §20.2) | re-build |
| pre-commit hook 拦了 deploy | render 出来的 HTML 含示例 sk-key | `git commit --no-verify` 单次绕过(确认确实是 fake);或更新 hook placeholder list | deploy |

每行都是一个"症状 → 命令 → 期望"的迷你决策树,弱模型不需要 reasoning,直接按表执行。

---
