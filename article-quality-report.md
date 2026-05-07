# chenk.top Article Quality Report

_Generated: 2026-05-05T20:29:09_  
_Read-only heuristic scan. No articles modified._

## Aggregate Stats

- Total articles scanned: **442** (EN: 221, ZH: 221)
- EN articles < 400 words: **0** (very short < 200w: 0)
- ZH articles < 800 chars: **13** (very short < 400ch: 0)
- Articles with TODO/FIXME/placeholder markers: **7**
- Articles with abrupt/non-sentence endings: **53**
- Articles with heading-level skips (h1->h3 etc): **82**
- Articles with >2 h1s in body: **194** _(common in Hugo, low-severity)_
- Articles with no/short description front-matter: **20**
- Translation orphans (matched by series+series_order): **1**

### External resource sampling (10% uniform)

- Images sampled: 50 of 2404; 4xx/5xx/error: **0**
- External links sampled: 30 of 1849; 4xx/5xx/error: **1**

Bad/blocked external links (note: DOI/many sites refuse HEAD):
  - [403] `zh/recommendation-systems/02-协同过滤与矩阵分解.md` -> https://doi.org/10.1145/1401890.1401944

## Top 30 Worst-Quality Articles

Ranked by composite heuristic score (higher = worse). Score weights:
placeholder=3, very-short=4, short=2, abrupt=2, code-heavy>70%=2, no-desc=1, no-date=1, future-date=2, heading-skip=1, multi-h1>2=1, translation-orphan=2.

| # | Score | Path | Reasons |
|---|------:|------|---------|
| 1 | 5 | `zh/time-series/05-Transformer架构.md` | placeholder x2, abrupt ending |
| 2 | 5 | `zh/time-series/08-Informer长序列预测.md` | placeholder x4, abrupt ending |
| 3 | 4 | `en/linux/basics.md` | placeholder x1, 26 h1s |
| 4 | 4 | `en/ode/11-numerical-methods.md` | abrupt ending, heading skip, 3 h1s |
| 5 | 4 | `zh/linux/使用基础.md` | placeholder x1, 26 h1s |
| 6 | 4 | `zh/linux/文件操作深入解析.md` | placeholder x1, 58 h1s |
| 7 | 4 | `zh/linux/用户管理.md` | placeholder x1, 31 h1s |
| 8 | 3 | `en/computer-fundamentals/04-motherboard-gpu.md` | abrupt ending, 12 h1s |
| 9 | 3 | `en/computer-fundamentals/06-deep-dive.md` | abrupt ending, 8 h1s |
| 10 | 3 | `en/ode/05-laplace-transform.md` | abrupt ending, 4 h1s |
| 11 | 3 | `en/time-series/01-traditional-models.md` | abrupt ending, 5 h1s |
| 12 | 3 | `en/time-series/n-beats.md` | abrupt ending, 3 h1s |
| 13 | 3 | `zh/computer-fundamentals/04-motherboard-gpu.md` | abrupt ending, 12 h1s |
| 14 | 3 | `zh/computer-fundamentals/06-deep-dive.md` | abrupt ending, 8 h1s |
| 15 | 3 | `zh/leetcode/06-二叉树遍历与构造.md` | no/short description, heading skip, 9 h1s |
| 16 | 3 | `zh/openclaw-quickstart/02-install-and-first-chat.md` | short (591zh), 4 h1s |
| 17 | 3 | `zh/openclaw-quickstart/06-skills-and-mcp.md` | short (708zh), 5 h1s |
| 18 | 3 | `zh/openclaw-quickstart/10-production-deploy.md` | short (672zh), 4 h1s |
| 19 | 3 | `zh/recommendation-systems/04-CTR预估与点击率建模.md` | abrupt ending, heading skip |
| 20 | 3 | `zh/recommendation-systems/06-序列推荐与会话建模.md` | abrupt ending, heading skip |
| 21 | 3 | `zh/recommendation-systems/07-图神经网络与社交推荐.md` | abrupt ending, 3 h1s |
| 22 | 3 | `zh/recommendation-systems/08-知识图谱增强推荐系统.md` | abrupt ending, 3 h1s |
| 23 | 3 | `zh/time-series/01-传统模型.md` | abrupt ending, 5 h1s |
| 24 | 3 | `zh/time-series/07-N-BEATS深度架构.md` | abrupt ending, 3 h1s |
| 25 | 3 | `zh/transfer-learning/08-多模态迁移.md` | placeholder x1 |
| 26 | 2 | `en/cloud-computing/cloud-native-containers.md` | heading skip, 13 h1s |
| 27 | 2 | `en/cloud-computing/networking-sdn.md` | heading skip, 13 h1s |
| 28 | 2 | `en/cloud-computing/operations-devops.md` | heading skip, 6 h1s |
| 29 | 2 | `en/cloud-computing/storage-systems.md` | heading skip, 7 h1s |
| 30 | 2 | `en/cloud-computing/virtualization.md` | heading skip, 29 h1s |

## Detailed Notes (Top 10)

### 1. `zh/time-series/05-Transformer架构.md` (score 5)

- Title: 时间序列模型（五）：时间序列的 Transformer 架构
- Length: 2915 (CJK chars)
- Date: 2024-10-31 09:00:00
- Heuristics fired: placeholder x2, abrupt ending
- h1 count: 0; heading skip: False; placeholders: 2; abrupt: True

### 2. `zh/time-series/08-Informer长序列预测.md` (score 5)

- Title: 时间序列模型（八）：Informer -- 高效长序列预测
- Length: 2856 (CJK chars)
- Date: 2024-12-15 09:00:00
- Heuristics fired: placeholder x4, abrupt ending
- h1 count: 0; heading skip: False; placeholders: 4; abrupt: True

### 3. `en/linux/basics.md` (score 4)

- Title: Linux Basics: Core Concepts and Essential Commands
- Length: 2178 (words)
- Date: 2022-01-01 09:00:00
- Heuristics fired: placeholder x1, 26 h1s
- h1 count: 26; heading skip: False; placeholders: 1; abrupt: False

### 4. `en/ode/11-numerical-methods.md` (score 4)

- Title: Ordinary Differential Equations (11): Numerical Methods
- Length: 2283 (words)
- Date: 2023-12-18 09:00:00
- Heuristics fired: abrupt ending, heading skip, 3 h1s
- h1 count: 3; heading skip: True; placeholders: 0; abrupt: True

### 5. `zh/linux/使用基础.md` (score 4)

- Title: Linux 使用基础
- Length: 3151 (CJK chars)
- Date: 2022-01-01 09:00:00
- Heuristics fired: placeholder x1, 26 h1s
- h1 count: 26; heading skip: False; placeholders: 1; abrupt: False

### 6. `zh/linux/文件操作深入解析.md` (score 4)

- Title: Linux 文件操作深入解析
- Length: 3588 (CJK chars)
- Date: 2022-04-02 09:00:00
- Heuristics fired: placeholder x1, 58 h1s
- h1 count: 58; heading skip: False; placeholders: 1; abrupt: False

### 7. `zh/linux/用户管理.md` (score 4)

- Title: Linux 用户管理
- Length: 3193 (CJK chars)
- Date: 2022-02-22 09:00:00
- Heuristics fired: placeholder x1, 31 h1s
- h1 count: 31; heading skip: False; placeholders: 1; abrupt: False

### 8. `en/computer-fundamentals/04-motherboard-gpu.md` (score 3)

- Title: Computer Fundamentals: Motherboard, Graphics, and Expansion
- Length: 3642 (words)
- Date: 2022-12-03 09:00:00
- Heuristics fired: abrupt ending, 12 h1s
- h1 count: 12; heading skip: False; placeholders: 0; abrupt: True

### 9. `en/computer-fundamentals/06-deep-dive.md` (score 3)

- Title: Computer Fundamentals: Deep Dive and System Integration
- Length: 2904 (words)
- Date: 2023-01-14 09:00:00
- Heuristics fired: abrupt ending, 8 h1s
- h1 count: 8; heading skip: False; placeholders: 0; abrupt: True

### 10. `en/ode/05-laplace-transform.md` (score 3)

- Title: ODE Chapter 5: Power Series and Special Functions
- Length: 2398 (words)
- Date: 2023-09-07 09:00:00
- Heuristics fired: abrupt ending, 4 h1s
- h1 count: 4; heading skip: False; placeholders: 0; abrupt: True

## Findings & Themes

1. **time-series ZH series** is the worst cluster. `05-Transformer架构.md` and `08-Informer长序列预测.md` both contain placeholder markers AND end abruptly — likely never finished. `02-LSTM`, `04-Attention`, `06-TCN`, `07-N-BEATS`, `01-传统模型` also end mid-sentence.
2. **linux series** (both EN and ZH) has placeholder markers in 4 articles AND huge h1 inflation (26-58 h1s/file). Indicates raw-import or copy-paste structure.
3. **leetcode ZH** has multiple articles missing description front-matter and using h1 for every problem — no concept of body hierarchy.
4. **openclaw-quickstart ZH** — three articles under 800 CJK chars (591, 672, 708). Likely stub pages.
5. **recommendation-systems ZH** — 4 articles (CTR, sequential, GNN, KG) end abruptly. Possible truncation during writing.
6. **Naming divergence** EN/ZH counterparts use different filename schemes (EN slug vs ZH `01-中文`). Cross-language matching depends on `series` + `series_order` front-matter, which is consistent today but fragile.
7. **Image health is excellent** — 50/50 sampled OSS images return 200.
8. **External link health is essentially fine** — only 1/30 returned 4xx, and that one (doi.org) is a known HEAD-blocker, not a real broken link.

## Recommended Investment Priority

1. **time-series ZH series** — finish `05-Transformer架构.md` and `08-Informer长序列预测.md` (both have explicit TODO markers and incomplete endings). Then audit endings of `02/04/06/07`.
2. **linux series** — replace placeholder content; collapse h1-everywhere structure into proper h2/h3 hierarchy.
3. **leetcode ZH** — add front-matter descriptions for all 10 chapters; restructure h1 sprawl.
4. **openclaw-quickstart ZH** stubs (02/06/10) — expand to match EN counterparts.
5. **recommendation-systems ZH** abrupt endings — verify content was not truncated.

