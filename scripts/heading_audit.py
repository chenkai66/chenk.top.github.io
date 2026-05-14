#!/usr/bin/env python3
"""
Exhaustive heading re-audit for chenk.top Hugo blog.
Checks:
1. Exercise subheading patterns (### under ## Exercises/练习题)
2. Standard ## section name consistency
3. Within-series consistency
4. Misc issues (trailing colons, ALL-CAPS, capitalization)
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CONTENT_ROOT = "/root/chenk-hugo/content"
OUTPUT_FILE = "/tmp/heading_reaudit.txt"

# ── Standard section names ──
ZH_STANDARD = {
    "你将学到什么", "前置知识", "总结", "练习题", "常见问题",
    "下一步", "参考文献", "常见陷阱", "系列导航"
}

EN_STANDARD = {
    "What You Will Learn", "Prerequisites", "Summary", "Exercises", "FAQ",
    "What's Next", "References", "Common Pitfalls", "Series Navigation"
}

# Variants map: variant -> canonical form
ZH_VARIANTS = {
    "你将学到": "你将学到什么",
    "你将学到的": "你将学到什么",
    "学习目标": "你将学到什么",
    "本章目标": "你将学到什么",
    "学习目标与预期收获": "你将学到什么",
    "本章学习目标": "你将学到什么",
    "本篇目标": "你将学到什么",
    "学习成果": "你将学到什么",

    "前提知识": "前置知识",
    "先修知识": "前置知识",
    "预备知识": "前置知识",
    "必备知识": "前置知识",
    "基础要求": "前置知识",
    "前置条件": "前置知识",

    "小结": "总结",
    "本章总结": "总结",
    "章节总结": "总结",
    "本章小结": "总结",
    "总结与回顾": "总结",
    "总结与展望": "总结",
    "全篇总结": "总结",

    "习题": "练习题",
    "课后习题": "练习题",
    "练习": "练习题",
    "思考题": "练习题",
    "课后练习": "练习题",
    "实践练习": "练习题",
    "动手练习": "练习题",
    "练习与思考": "练习题",

    "FAQ": "常见问题",
    "Q&A": "常见问题",
    "常见疑问": "常见问题",
    "疑难解答": "常见问题",
    "问题解答": "常见问题",
    "常见问题解答": "常见问题",
    "常见问题与解答": "常见问题",

    "接下来": "下一步",
    "下一篇": "下一步",
    "后续学习": "下一步",
    "进一步学习": "下一步",
    "延伸阅读": "下一步",
    "拓展阅读": "下一步",
    "下一步学习": "下一步",

    "参考资料": "参考文献",
    "参考链接": "参考文献",
    "引用": "参考文献",
    "参考": "参考文献",
    "文献参考": "参考文献",

    "易错点": "常见陷阱",
    "注意事项": "常见陷阱",
    "常见错误": "常见陷阱",
    "易犯错误": "常见陷阱",
    "踩坑指南": "常见陷阱",

    "导航": "系列导航",
    "章节导航": "系列导航",
    "文章导航": "系列导航",
}

EN_VARIANTS = {
    "What You'll Learn": "What You Will Learn",
    "What you will learn": "What You Will Learn",
    "What You Will Learn in This Chapter": "What You Will Learn",
    "Learning Objectives": "What You Will Learn",
    "Objectives": "What You Will Learn",
    "What You'll Learn": "What You Will Learn",
    "What you'll learn": "What You Will Learn",
    "Goals": "What You Will Learn",

    "Prerequisite": "Prerequisites",
    "Pre-requisites": "Prerequisites",
    "Background": "Prerequisites",
    "Required Knowledge": "Prerequisites",

    "Conclusion": "Summary",
    "Chapter Summary": "Summary",
    "Key Takeaways": "Summary",
    "Takeaways": "Summary",
    "Recap": "Summary",
    "Wrap-Up": "Summary",
    "Wrap Up": "Summary",

    "Exercise": "Exercises",
    "Practice": "Exercises",
    "Practice Problems": "Exercises",
    "Problems": "Exercises",
    "Practice Exercises": "Exercises",
    "Hands-on Exercises": "Exercises",

    "Frequently Asked Questions": "FAQ",
    "Common Questions": "FAQ",
    "Q&A": "FAQ",

    "Next Steps": "What's Next",
    "What Next": "What's Next",
    "Where to Go Next": "What's Next",
    "Further Reading": "What's Next",
    "Next": "What's Next",
    "What's next": "What's Next",

    "Reference": "References",
    "Bibliography": "References",
    "Sources": "References",
    "Further References": "References",

    "Pitfalls": "Common Pitfalls",
    "Common Mistakes": "Common Pitfalls",
    "Gotchas": "Common Pitfalls",
    "Watch Out": "Common Pitfalls",

    "Navigation": "Series Navigation",
    "Chapter Navigation": "Series Navigation",
}

# Known acronym-only headings that are OK in ALL-CAPS
KNOWN_ACRONYMS = {
    "FAQ", "API", "SQL", "CSS", "HTML", "JSON", "XML", "REST", "GPU", "CPU",
    "RAM", "ROM", "SSD", "HDD", "TCP", "UDP", "HTTP", "HTTPS", "SSH", "DNS",
    "IP", "URL", "URI", "CLI", "GUI", "IDE", "ORM", "MVC", "CRUD", "ACID",
    "CAP", "CDN", "VPN", "VLAN", "LAN", "WAN", "NAS", "SAN", "RAID", "VM",
    "OS", "BIOS", "UEFI", "GRUB", "DHCP", "NAT", "SMTP", "IMAP", "POP",
    "FTP", "NFS", "SMB", "LDAP", "OAuth", "JWT", "TLS", "SSL", "PGP",
    "RSA", "AES", "SHA", "MD5", "CRC", "HMAC", "KMS", "IAM", "RBAC",
    "ABAC", "ACL", "CORS", "CSRF", "XSS", "MITM", "DDoS", "WAF", "IDS",
    "IPS", "SIEM", "SOC", "NOC", "SRE", "SLA", "SLO", "SLI", "KPI",
    "ETL", "ELT", "OLAP", "OLTP", "DWH", "BI", "ML", "AI", "DL", "NLP",
    "NLU", "NLG", "CV", "RL", "GAN", "VAE", "CNN", "RNN", "LSTM", "GRU",
    "LLM", "GPT", "BERT", "T5", "PCA", "SVD", "ICA", "LDA", "HMM", "CRF",
    "SVM", "KNN", "GMM", "EM", "MCMC", "MAP", "MLE", "MSE", "MAE", "RMSE",
    "AUC", "ROC", "PR", "FPR", "TPR", "BLEU", "ROUGE", "PPO", "DQN",
    "A2C", "A3C", "SAC", "DDPG", "TD3", "TRPO", "MCTS", "RLHF",
    "GNN", "GAT", "GCN", "MPP", "WL", "PDE", "ODE", "SDE", "FEM", "BVP",
    "IVP", "RK4", "AB", "AM", "BDF", "FFT", "DFT", "DCT", "DST",
    "CTR", "CVR", "DIEN", "DIN", "NCF", "BPR", "PINN",
    "CUDA", "MPI", "NCCL", "RDMA", "NVLINK", "TPU", "FPGA", "ASIC",
    "ECS", "OSS", "RDS", "SLB", "VPC", "EIP", "NAT", "CDN",
    "K8S", "CI", "CD", "TDD", "BDD",
    "YAML", "TOML", "INI", "CSV", "TSV",
    "PAI", "DSW", "DLC", "EAS",
    "ARIMA", "SARIMA", "AR", "MA", "GARCH",
    "GRU", "TCN",
    "LASSO", "SDE",
    "AWGN", "SNR", "BER",
}


def strip_number_prefix(heading_text):
    """Strip number prefixes like '1. ', '一、', '第一章 ' etc."""
    # Arabic number prefix: "1. ", "12. "
    text = re.sub(r'^\d+[\.\)]\s*', '', heading_text)
    # Chinese number prefix: "一、", "十二、"
    text = re.sub(r'^[一二三四五六七八九十百千万]+[、\.]\s*', '', text)
    # "第X章" or "第X节"
    text = re.sub(r'^第[一二三四五六七八九十百千万\d]+[章节篇]\s*', '', text)
    # "Step N:" or "Step N."
    text = re.sub(r'^Step\s+\d+[:\.\)]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def is_all_caps_word(word):
    """Check if a word is ALL-CAPS (and not an acronym)."""
    if len(word) <= 1:
        return False
    if not word.isalpha():
        return False
    if word.isupper() and word not in KNOWN_ACRONYMS:
        return True
    return False


def parse_headings(filepath):
    """Parse all headings from a markdown file, returning list of (level, raw_text, line_number)."""
    headings = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            in_frontmatter = False
            in_code_block = False
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                # Track frontmatter
                if stripped == '---':
                    in_frontmatter = not in_frontmatter
                    continue
                if in_frontmatter:
                    continue
                # Track code blocks
                if stripped.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue
                # Match headings
                m = re.match(r'^(#{1,6})\s+(.+)$', stripped)
                if m:
                    level = len(m.group(1))
                    text = m.group(2).strip()
                    headings.append((level, text, i))
    except Exception as e:
        pass
    return headings


def get_series_and_article(filepath):
    """Extract series name and article name from filepath."""
    parts = Path(filepath).parts
    # Find en/ or zh/
    try:
        lang_idx = next(i for i, p in enumerate(parts) if p in ('en', 'zh'))
    except StopIteration:
        return None, None, None
    lang = parts[lang_idx]
    if lang_idx + 1 < len(parts):
        series = parts[lang_idx + 1]
    else:
        return lang, None, None
    if lang_idx + 2 < len(parts):
        article = parts[lang_idx + 2]
    else:
        article = None
    return lang, series, article


def find_all_md_files(root):
    """Find all .md files under root."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.md'):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def main():
    out_lines = []
    def out(s=""):
        out_lines.append(s)
        print(s)

    out("=" * 80)
    out("HEADING RE-AUDIT REPORT")
    out("=" * 80)
    out()

    all_files = find_all_md_files(CONTENT_ROOT)
    out(f"Total .md files scanned: {len(all_files)}")
    out()

    # Collect all data
    # file -> headings
    file_headings = {}
    # series -> lang -> [files]
    series_files = defaultdict(lambda: defaultdict(list))

    for fp in all_files:
        headings = parse_headings(fp)
        file_headings[fp] = headings
        lang, series, article = get_series_and_article(fp)
        if lang and series:
            series_files[series][lang].append(fp)

    # ════════════════════════════════════════════════════════════════
    # CHECK 1: Exercise subheading patterns
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("CHECK 1: EXERCISE SUBHEADING PATTERNS (### under ## Exercises/练习题)")
    out("=" * 80)
    out()

    issues_found_1 = False
    # series -> lang -> { file: [h3 texts under exercises] }
    exercise_subheadings = defaultdict(lambda: defaultdict(dict))

    for fp, headings in file_headings.items():
        lang, series, article = get_series_and_article(fp)
        if not series:
            continue

        in_exercises = False
        exercise_h3s = []
        for level, text, lineno in headings:
            stripped = strip_number_prefix(text)
            if level == 2:
                if stripped in ("练习题", "Exercises", "练习", "习题", "Exercise"):
                    in_exercises = True
                    exercise_h3s = []
                else:
                    if in_exercises and exercise_h3s:
                        exercise_subheadings[series][lang][fp] = list(exercise_h3s)
                    in_exercises = False
                    exercise_h3s = []
            elif level == 3 and in_exercises:
                exercise_h3s.append(text)

        # End of file
        if in_exercises and exercise_h3s:
            exercise_subheadings[series][lang][fp] = list(exercise_h3s)

    for series in sorted(exercise_subheadings):
        for lang in sorted(exercise_subheadings[series]):
            file_patterns = exercise_subheadings[series][lang]
            if len(file_patterns) < 2:
                # Still report what pattern is used
                for fp, h3s in file_patterns.items():
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  [{lang}/{series}] {rel}")
                    out(f"    H3 pattern: {h3s}")
                continue

            # Check if all files use the same pattern
            all_patterns = set()
            for fp, h3s in file_patterns.items():
                all_patterns.add(tuple(h3s))

            if len(all_patterns) > 1:
                issues_found_1 = True
                out(f"  *** INCONSISTENCY in [{lang}/{series}] ***")
                for fp, h3s in sorted(file_patterns.items()):
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"    {rel}")
                    out(f"      H3s: {h3s}")
                out()
            else:
                for fp, h3s in sorted(file_patterns.items()):
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  [{lang}/{series}] {rel}")
                    out(f"    H3 pattern: {h3s}")

    if not issues_found_1:
        out("  No inconsistencies found in exercise subheadings.")
    out()

    # ════════════════════════════════════════════════════════════════
    # CHECK 2: Standard ## section name variants
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("CHECK 2: STANDARD ## SECTION NAME VARIANTS")
    out("=" * 80)
    out()

    issues_found_2 = False

    for fp, headings in sorted(file_headings.items()):
        lang, series, article = get_series_and_article(fp)
        if not lang:
            continue

        for level, text, lineno in headings:
            if level != 2:
                continue
            stripped = strip_number_prefix(text)

            # Check ZH variants
            if lang == "zh":
                if stripped in ZH_VARIANTS:
                    issues_found_2 = True
                    canonical = ZH_VARIANTS[stripped]
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  VARIANT: {rel}:{lineno}")
                    out(f"    Found:    '## {text}'")
                    out(f"    Expected: '## {canonical}'")
                    out()

            # Check EN variants
            if lang == "en":
                if stripped in EN_VARIANTS:
                    issues_found_2 = True
                    canonical = EN_VARIANTS[stripped]
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  VARIANT: {rel}:{lineno}")
                    out(f"    Found:    '## {text}'")
                    out(f"    Expected: '## {canonical}'")
                    out()

    # Also do a fuzzy search: find any ## heading that CONTAINS a standard name word but doesn't match exactly
    out("  --- Fuzzy matches (headings that partially match standard names) ---")
    out()
    fuzzy_found = False

    zh_keywords = ["学到", "前置", "先修", "总结", "小结", "练习", "习题", "常见问题", "问题解答",
                    "下一步", "参考文献", "参考资料", "陷阱", "导航", "目标"]
    en_keywords = ["prerequisite", "summary", "conclusion", "exercise", "pitfall",
                    "takeaway", "recap", "reference", "navigation", "next step",
                    "what you", "further reading", "wrap"]

    for fp, headings in sorted(file_headings.items()):
        lang, series, article = get_series_and_article(fp)
        if not lang:
            continue

        for level, text, lineno in headings:
            if level != 2:
                continue
            stripped = strip_number_prefix(text)

            if lang == "zh" and stripped not in ZH_STANDARD and stripped not in ZH_VARIANTS:
                for kw in zh_keywords:
                    if kw in stripped and stripped not in ZH_STANDARD:
                        # Check it's not already flagged as variant
                        if stripped not in ZH_VARIANTS:
                            fuzzy_found = True
                            rel = fp.replace(CONTENT_ROOT + "/", "")
                            out(f"  FUZZY: {rel}:{lineno}")
                            out(f"    Found: '## {text}'")
                            out(f"    Might be a variant of a standard section")
                            out()
                        break

            if lang == "en" and stripped not in EN_STANDARD and stripped not in EN_VARIANTS:
                for kw in en_keywords:
                    if kw.lower() in stripped.lower() and stripped not in EN_STANDARD:
                        if stripped not in EN_VARIANTS:
                            fuzzy_found = True
                            rel = fp.replace(CONTENT_ROOT + "/", "")
                            out(f"  FUZZY: {rel}:{lineno}")
                            out(f"    Found: '## {text}'")
                            out(f"    Might be a variant of a standard section")
                            out()
                        break

    if not fuzzy_found:
        out("  No fuzzy matches found.")
    if not issues_found_2:
        out("  No exact variant matches found.")
    out()

    # ════════════════════════════════════════════════════════════════
    # CHECK 3: Within-series consistency
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("CHECK 3: WITHIN-SERIES CONSISTENCY")
    out("=" * 80)
    out()

    issues_found_3 = False

    # For each series+lang, collect the set of standard sections used by each article
    for series in sorted(series_files):
        for lang in sorted(series_files[series]):
            files = series_files[series][lang]
            if len(files) < 2:
                continue

            # Collect standard sections per file
            # file -> set of standard section names found
            file_sections = {}
            # Also track ALL ## headings for each file
            file_all_h2 = {}

            for fp in files:
                headings = file_headings.get(fp, [])
                std_sections = set()
                all_h2 = []
                for level, text, lineno in headings:
                    if level != 2:
                        continue
                    stripped = strip_number_prefix(text)
                    all_h2.append(stripped)

                    # Check if it's a standard section
                    standards = ZH_STANDARD if lang == "zh" else EN_STANDARD
                    variants = ZH_VARIANTS if lang == "zh" else EN_VARIANTS

                    if stripped in standards:
                        std_sections.add(stripped)
                    elif stripped in variants:
                        std_sections.add(variants[stripped])

                file_sections[fp] = std_sections
                file_all_h2[fp] = all_h2

            # Find sections that appear in SOME but not ALL articles
            all_std_sections = set()
            for s in file_sections.values():
                all_std_sections |= s

            for section in sorted(all_std_sections):
                has_it = [fp for fp in files if section in file_sections[fp]]
                missing = [fp for fp in files if section not in file_sections[fp]]

                # Only flag if it's a majority pattern (>50% have it but some don't)
                if missing and len(has_it) > len(missing):
                    issues_found_3 = True
                    out(f"  [{lang}/{series}] '{section}' present in {len(has_it)}/{len(files)} articles")
                    out(f"    MISSING from:")
                    for fp in sorted(missing):
                        rel = fp.replace(CONTENT_ROOT + "/", "")
                        out(f"      - {rel}")
                    out()

            # Also check: within this series, are there ## headings that look like they're
            # trying to be the same thing but are named differently?
            # E.g., one article says "## 核心概念" and another says "## 基本概念"
            all_unique_h2 = defaultdict(list)
            for fp in files:
                for h2 in file_all_h2.get(fp, []):
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    all_unique_h2[h2].append(rel)

    if not issues_found_3:
        out("  No within-series inconsistencies found.")
    out()

    # ════════════════════════════════════════════════════════════════
    # CHECK 4: Misc issues
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("CHECK 4: MISC ISSUES")
    out("=" * 80)
    out()

    # 4a: Trailing colons
    out("  --- 4a: Headings with trailing colons ---")
    out()
    found_4a = False
    for fp, headings in sorted(file_headings.items()):
        for level, text, lineno in headings:
            if text.rstrip().endswith(':') or text.rstrip().endswith(':'):
                found_4a = True
                rel = fp.replace(CONTENT_ROOT + "/", "")
                out(f"  TRAILING COLON: {rel}:{lineno}")
                out(f"    '{'#' * level} {text}'")
                out()
    if not found_4a:
        out("  None found.")
    out()

    # 4b: ALL-CAPS headings (non-acronym)
    out("  --- 4b: ALL-CAPS headings (non-acronym) ---")
    out()
    found_4b = False
    for fp, headings in sorted(file_headings.items()):
        lang, series, article = get_series_and_article(fp)
        if lang != "en":
            continue  # Only check EN
        for level, text, lineno in headings:
            words = text.split()
            for word in words:
                # Strip punctuation
                clean = re.sub(r'[^\w]', '', word)
                if clean and is_all_caps_word(clean):
                    found_4b = True
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  ALL-CAPS: {rel}:{lineno}")
                    out(f"    '{'#' * level} {text}'")
                    out(f"    Offending word: '{clean}'")
                    out()
                    break  # One flag per heading
    if not found_4b:
        out("  None found.")
    out()

    # 4c: Inconsistent capitalization within same series (EN only)
    out("  --- 4c: Inconsistent capitalization within same series ---")
    out()
    found_4c = False

    for series in sorted(series_files):
        if "en" not in series_files[series]:
            continue
        files = series_files[series]["en"]
        if len(files) < 2:
            continue

        # Collect all ## headings, normalized to lowercase, tracking the exact form
        heading_forms = defaultdict(list)  # lowercase -> [(file, exact, lineno)]
        for fp in files:
            headings = file_headings.get(fp, [])
            for level, text, lineno in headings:
                if level != 2:
                    continue
                stripped = strip_number_prefix(text)
                key = stripped.lower()
                rel = fp.replace(CONTENT_ROOT + "/", "")
                heading_forms[key].append((rel, stripped, lineno))

        for key, occurrences in sorted(heading_forms.items()):
            unique_forms = set(exact for _, exact, _ in occurrences)
            if len(unique_forms) > 1:
                found_4c = True
                out(f"  CAPITALIZATION INCONSISTENCY in [en/{series}]:")
                out(f"    Same heading appears as:")
                for form in sorted(unique_forms):
                    files_with_form = [f for f, e, _ in occurrences if e == form]
                    out(f"      '{form}' in: {', '.join(files_with_form)}")
                out()

    if not found_4c:
        out("  None found.")
    out()

    # 4d: Trailing/leading whitespace in headings
    out("  --- 4d: Headings with unusual whitespace ---")
    out()
    found_4d = False
    for fp, headings in sorted(file_headings.items()):
        for level, text, lineno in headings:
            if text != text.strip():
                found_4d = True
                rel = fp.replace(CONTENT_ROOT + "/", "")
                out(f"  WHITESPACE: {rel}:{lineno}")
                out(f"    '{'#' * level} {repr(text)}'")
                out()
            # Check for double spaces
            if '  ' in text:
                found_4d = True
                rel = fp.replace(CONTENT_ROOT + "/", "")
                out(f"  DOUBLE SPACE: {rel}:{lineno}")
                out(f"    '{'#' * level} {text}'")
                out()
    if not found_4d:
        out("  None found.")
    out()

    # 4e: ## headings that contain parenthetical translations
    out("  --- 4e: Headings with mixed language (parenthetical translations) ---")
    out()
    found_4e = False
    for fp, headings in sorted(file_headings.items()):
        lang, series, article = get_series_and_article(fp)
        if not lang:
            continue
        for level, text, lineno in headings:
            if level not in (2, 3):
                continue
            # ZH heading with English in parentheses, or EN with Chinese
            if lang == "zh":
                # Check for patterns like "总结（Summary）" or "总结(Summary)"
                if re.search(r'[（(][A-Za-z\s]+[)）]', text):
                    found_4e = True
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  MIXED LANG: {rel}:{lineno}")
                    out(f"    '{'#' * level} {text}'")
                    out()
            elif lang == "en":
                # Check for Chinese characters in EN headings
                if re.search(r'[一-鿿]', text):
                    found_4e = True
                    rel = fp.replace(CONTENT_ROOT + "/", "")
                    out(f"  MIXED LANG: {rel}:{lineno}")
                    out(f"    '{'#' * level} {text}'")
                    out()
    if not found_4e:
        out("  None found.")
    out()

    # 4f: _index.md files with non-standard headings
    out("  --- 4f: _index.md files heading check ---")
    out()
    found_4f = False
    for fp, headings in sorted(file_headings.items()):
        if not fp.endswith('_index.md'):
            continue
        for level, text, lineno in headings:
            if level == 2:
                rel = fp.replace(CONTENT_ROOT + "/", "")
                out(f"  INDEX H2: {rel}:{lineno}")
                out(f"    '## {text}'")
                found_4f = True
    if not found_4f:
        out("  No _index.md files with ## headings found.")
    out()

    # ════════════════════════════════════════════════════════════════
    # CHECK 5: Cross-language consistency
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("CHECK 5: CROSS-LANGUAGE CONSISTENCY (ZH vs EN standard sections)")
    out("=" * 80)
    out()

    issues_found_5 = False

    # Map ZH standard -> EN standard
    zh_en_map = {
        "你将学到什么": "What You Will Learn",
        "前置知识": "Prerequisites",
        "总结": "Summary",
        "练习题": "Exercises",
        "常见问题": "FAQ",
        "下一步": "What's Next",
        "参考文献": "References",
        "常见陷阱": "Common Pitfalls",
        "系列导航": "Series Navigation",
    }

    for series in sorted(series_files):
        if "zh" not in series_files[series] or "en" not in series_files[series]:
            continue

        # Get standard sections used across the series for each lang
        zh_sections_all = set()
        en_sections_all = set()

        for fp in series_files[series]["zh"]:
            headings = file_headings.get(fp, [])
            for level, text, lineno in headings:
                if level != 2:
                    continue
                stripped = strip_number_prefix(text)
                if stripped in ZH_STANDARD:
                    zh_sections_all.add(stripped)
                elif stripped in ZH_VARIANTS:
                    zh_sections_all.add(ZH_VARIANTS[stripped])

        for fp in series_files[series]["en"]:
            headings = file_headings.get(fp, [])
            for level, text, lineno in headings:
                if level != 2:
                    continue
                stripped = strip_number_prefix(text)
                if stripped in EN_STANDARD:
                    en_sections_all.add(stripped)
                elif stripped in EN_VARIANTS:
                    en_sections_all.add(EN_VARIANTS[stripped])

        # Check if ZH has a section but EN doesn't have the equivalent
        for zh_sec, en_sec in zh_en_map.items():
            if zh_sec in zh_sections_all and en_sec not in en_sections_all:
                issues_found_5 = True
                out(f"  [{series}] ZH has '{zh_sec}' but EN missing '{en_sec}'")
            elif en_sec in en_sections_all and zh_sec not in zh_sections_all:
                issues_found_5 = True
                out(f"  [{series}] EN has '{en_sec}' but ZH missing '{zh_sec}'")

    if not issues_found_5:
        out("  No cross-language inconsistencies found.")
    out()

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    out("=" * 80)
    out("SUMMARY OF ALL UNIQUE ## HEADINGS BY SERIES")
    out("=" * 80)
    out()

    for series in sorted(series_files):
        for lang in sorted(series_files[series]):
            files = series_files[series][lang]
            all_h2 = set()
            for fp in files:
                headings = file_headings.get(fp, [])
                for level, text, lineno in headings:
                    if level == 2:
                        all_h2.add(strip_number_prefix(text))
            if all_h2:
                out(f"  [{lang}/{series}] ({len(files)} articles)")
                for h in sorted(all_h2):
                    # Mark if it's a standard section
                    standards = ZH_STANDARD if lang == "zh" else EN_STANDARD
                    variants = ZH_VARIANTS if lang == "zh" else EN_VARIANTS
                    marker = ""
                    if h in standards:
                        marker = " [STANDARD]"
                    elif h in variants:
                        marker = f" [VARIANT -> {variants[h]}]"
                    out(f"    - {h}{marker}")
                out()

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    out()
    out(f"Report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
