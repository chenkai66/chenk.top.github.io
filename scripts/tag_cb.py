#!/usr/bin/env python3
"""Auto-tag bare ``` code fences with inferred language."""
import glob, re

def infer_lang(content):
    """Return language tag based on content. Returns 'text' if uncertain."""
    s = content.strip()
    if not s:
        return "text"
    first_lines = "\n".join(s.split("\n")[:5])

    # Shell prompts and common shell commands
    if re.search(r'^[\$#]\s', first_lines, re.MULTILINE):
        return "bash"
    if re.search(r'^(\$|sudo |apt |apt-get |yum |dnf |brew |npm |yarn |pnpm |pip |pip3 |docker |kubectl |git |cd |ls |mkdir |export |source |curl |wget |ssh |systemctl |service |grep |find |awk |sed |chmod |chown |tar |zip)', first_lines, re.MULTILINE):
        return "bash"

    # Python REPL
    if ">>>" in first_lines or first_lines.strip().startswith("Python "):
        return "python"
    # Python code
    if re.search(r'^(import |from \w+ import|def |class |if __name__|@\w+\n(?:def |class ))', s, re.MULTILINE):
        return "python"

    # JavaScript / TypeScript
    if re.search(r'^(import .+ from |const |let |var |function |export |async function|interface |type \w+ = )', first_lines, re.MULTILINE):
        if "interface " in s or ": string" in s or ": number" in s or ": boolean" in s or "<T>" in s:
            return "typescript"
        return "javascript"

    # Go
    if re.search(r'^(package \w+|func \w|import \(\n)', first_lines, re.MULTILINE):
        return "go"

    # Rust
    if re.search(r'^(fn \w|use \w+::|struct \w+ \{|impl \w|pub fn )', first_lines, re.MULTILINE):
        return "rust"

    # Java
    if re.search(r'^(public class |private |@Override|public static void main)', first_lines, re.MULTILINE):
        return "java"

    # C / C++
    if re.search(r'^(#include |int main\(|void main\()', first_lines, re.MULTILINE):
        return "c"

    # SQL
    if re.search(r'^\s*(SELECT |INSERT |UPDATE |DELETE |CREATE TABLE|ALTER TABLE|DROP TABLE|WITH \w+ AS)', s, re.MULTILINE | re.IGNORECASE):
        return "sql"

    # YAML
    if re.search(r'^\w[\w-]*:\s*$', first_lines, re.MULTILINE) and not re.search(r'\{|\}', first_lines):
        if re.search(r'^(version|kind|apiVersion|metadata|spec|name):', first_lines, re.MULTILINE):
            return "yaml"
        return "yaml"

    # JSON
    if s.startswith("{") and s.rstrip().endswith("}"):
        if re.search(r'"\w+":\s*[\{\[\d"]', s):
            return "json"
    if s.startswith("[") and s.rstrip().endswith("]"):
        if '"' in s or "{" in s:
            return "json"

    # HCL / Terraform
    if re.search(r'^(resource|provider|module|variable|output|terraform|locals)\s+["{]', first_lines, re.MULTILINE):
        return "hcl"

    # Dockerfile
    if re.search(r'^(FROM |RUN |COPY |WORKDIR |CMD |EXPOSE |ENV |ARG |LABEL )', first_lines):
        return "dockerfile"

    # HTML
    if re.search(r'<(html|div|span|p|a|head|body|script|style)[\s>]', first_lines):
        return "html"

    # CSS
    if re.search(r'^[.\#][\w-]+\s*\{|^@media', first_lines, re.MULTILINE):
        return "css"

    # TOML
    if re.search(r'^\[[\w\.]+\]', first_lines, re.MULTILINE):
        return "toml"

    # Markdown (rare in code blocks)
    if re.search(r'^#{1,6} \w', first_lines, re.MULTILINE):
        return "markdown"

    # Math / latex
    if re.search(r'\\(frac|sum|int|alpha|beta|gamma|theta|nabla|mathbb)', s):
        return "latex"

    return "text"


total_fixed = 0
total_files = 0
lang_counts = {}
for path in glob.glob("/root/chenk-hugo/content/**/*.md", recursive=True):
    if "_index" in path: continue
    with open(path) as f: c = f.read()
    parts = c.split("---", 2)
    if len(parts) < 3: continue
    fm_with_delim = "---" + parts[1] + "---"
    body = parts[2]

    lines = body.split("\n")
    new_lines = []
    in_code = False
    fixed_in_file = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```(.*)$", line)
        if m:
            lang = m.group(1).strip()
            if not in_code:
                if not lang:
                    j = i + 1
                    block_lines = []
                    while j < len(lines):
                        if lines[j].rstrip() == "```":
                            break
                        block_lines.append(lines[j])
                        j += 1
                    content = "\n".join(block_lines)
                    inferred = infer_lang(content)
                    new_lines.append(f"```{inferred}")
                    fixed_in_file += 1
                    lang_counts[inferred] = lang_counts.get(inferred, 0) + 1
                else:
                    new_lines.append(line)
                in_code = True
            else:
                new_lines.append(line)
                in_code = False
        else:
            new_lines.append(line)
        i += 1

    new_body = "\n".join(new_lines)
    if fixed_in_file > 0:
        with open(path, "w") as f:
            f.write(fm_with_delim + new_body)
        total_files += 1
        total_fixed += fixed_in_file

print(f"\nTagged {total_fixed} bare code blocks in {total_files} files")
print("\nLanguage distribution:")
for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang}: {n}")
