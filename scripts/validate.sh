#!/bin/bash
# scripts/validate.sh — gate before deploy
# Usage: bash scripts/validate.sh <series-slug>
# Exit 0 if all checks pass, non-zero with diagnosis otherwise.
set -u
SERIES="${1:?usage: validate.sh <series-slug>}"
HUGO_DIR="${HUGO_DIR:-/root/chenk-hugo}"
EN="$HUGO_DIR/content/en/$SERIES"
ZH="$HUGO_DIR/content/zh/$SERIES"
ERR=0

fail() { echo "✗ $*" >&2; ERR=$((ERR+1)); }
ok()   { echo "✓ $*"; }

if [ ! -d "$EN" ] && [ ! -d "$ZH" ]; then
    echo "Series dir not found: $EN / $ZH" >&2; exit 2
fi

# A. 篇数对齐
en=$(ls $EN/*.md 2>/dev/null | grep -v _index | wc -l)
zh=$(ls $ZH/*.md 2>/dev/null | grep -v _index | wc -l)
[ "$en" = "$zh" ] && ok "篇数对齐 ($en EN + $zh ZH)" || fail "EN=$en ZH=$zh"

# B. front matter 必填字段
for f in $EN/*.md $ZH/*.md; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = "_index.md" ] && continue
    for field in "title:" "date:" "lang:" "series:" "series_order:" "translationKey:"; do
        grep -q "^$field" "$f" || fail "$(basename $f) 缺 $field"
    done
done
[ "$ERR" = "0" ] && ok "所有 front matter 必填字段就位"

# C. H2 末尾合规
# 两种合法 archetype:
#   A. Tutorial: 末尾内容 H2 必须是 "What's next" / "Where to go from here" / "接下来" / "下一步"
#   B. Reference: 末尾内容 H2 是 "Summary" / "Conclusion" / "总结" / "小结" 之类的总结性 H2
# trailers (FAQ / References / Acknowledgements / 参考文献 等) 可以混在其后/前都行
# 判定: 找到 "What's next" 或 "Summary" 作为 anchor，确认它后面只剩 trailer
TRAILER_EN='^## (References|Summary|FAQ|Frequently Asked|Acknowled|Conclusion|Bibliography|Appendix)'
TRAILER_ZH='^## (参考文献|参考资料|致谢|常见问题|FAQ|总结|小结|附录|结语|结论)'
ANCHOR_EN="What'?s next|Where to go from here|Summary|Conclusion"
ANCHOR_ZH="接下来|下一步|总结|小结|结语|结论"

check_archetype() {
    local f=$1 anchor_re=$2 trailer_re=$3
    mapfile -t h2s < <(grep -nE '^## ' "$f" | sed 's/^[0-9]*://')
    [ "${#h2s[@]}" = "0" ] && return 0
    local anchor_idx=-1
    for i in "${!h2s[@]}"; do
        if echo "${h2s[$i]}" | grep -qiE "$anchor_re"; then anchor_idx=$i; fi
    done
    [ "$anchor_idx" = "-1" ] && return 0
    for ((i=anchor_idx+1; i<${#h2s[@]}; i++)); do
        if ! echo "${h2s[$i]}" | grep -qiE "$trailer_re"; then
            fail "$(basename $f) anchor 之后非 trailer: ${h2s[$i]}"
            return
        fi
    done
}

for f in $EN/*.md; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = "_index.md" ] && continue
    check_archetype "$f" "$ANCHOR_EN" "$TRAILER_EN"
done
for f in $ZH/*.md; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = "_index.md" ] && continue
    check_archetype "$f" "$ANCHOR_ZH" "$TRAILER_ZH"
done

# D. 表格里没有 \|
for f in $EN/*.md $ZH/*.md; do
    [ -f "$f" ] || continue
    if grep -nE '^\|.*\\\|.*\|' "$f" > /dev/null 2>&1; then
        fail "$(basename $f) 表格 cell 含 \\|（改用 \\mid）"
    fi
done

# E. 未来日期
today=$(date +%Y-%m-%d)
for f in $EN/*.md $ZH/*.md; do
    [ -f "$f" ] || continue
    d=$(grep -m1 '^date:' "$f" | awk '{print $2}')
    [[ "$d" > "$today" ]] && fail "$(basename $f) date=$d 在未来"
done

# F. 图片 URL HEAD 检查 (only if curl available)
if command -v curl >/dev/null; then
    imgs=$(grep -hoE 'https://blog-pic-ck[^)]+' $EN/*.md $ZH/*.md 2>/dev/null | sort -u)
    [ -n "$imgs" ] && {
        broken=0
        for u in $imgs; do
            code=$(curl -sI -o /dev/null -w '%{http_code}' "$u" --max-time 10)
            [ "$code" != "200" ] && fail "img $code: $u" && broken=$((broken+1))
        done
        n_total=$(echo "$imgs" | wc -l)
        [ "$broken" = "0" ] && ok "$n_total 张图全部 200"
    }
fi

# G. Hugo 构建无 warning
cd "$HUGO_DIR"
build_log=$(hugo --minify 2>&1)
if echo "$build_log" | grep -iE 'WARN|ERROR|broken' > /dev/null; then
    fail "Hugo 构建有 warning/error"
    echo "$build_log" | grep -iE 'WARN|ERROR|broken' | head -5
else
    ok "Hugo 构建无 warning"
fi

# H. 跨章引用超链接化
for f in $EN/*.md; do
    [ -f "$f" ] || continue
    unlinked=$(grep -nE '(Chapter|Part|Section) [0-9]+' "$f" 2>/dev/null | grep -vE '\]\(|http|^[^:]*:\s*#' | head -3)
    [ -n "$unlinked" ] && {
        fail "$(basename $f) 有未链接的 Part/Section 引用:"
        echo "$unlinked" | sed 's/^/    /'
    }
done
for f in $ZH/*.md; do
    [ -f "$f" ] || continue
    unlinked=$(grep -nE '第 ?[0-9]+ ?[章节部]' "$f" 2>/dev/null | grep -vE '\]\(|http|^[^:]*:\s*#' | head -3)
    [ -n "$unlinked" ] && {
        fail "$(basename $f) 有未链接的 第N章 引用:"
        echo "$unlinked" | sed 's/^/    /'
    }
done

echo "==="
[ "$ERR" = "0" ] && { echo "ALL PASSED — safe to deploy"; exit 0; } \
                 || { echo "FAILED $ERR check(s) — fix before deploy"; exit 1; }
