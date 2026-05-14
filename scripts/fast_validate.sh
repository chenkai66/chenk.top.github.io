#!/bin/bash
# fast-validate: structural checks only (no curl, no hugo build)
set -u
SERIES="${1:?usage: fast-validate.sh <series-slug>}"
HUGO_DIR="${HUGO_DIR:-/root/chenk-hugo}"
EN="$HUGO_DIR/content/en/$SERIES"
ZH="$HUGO_DIR/content/zh/$SERIES"
ERR=0
fail() { echo "✗ $*"; ERR=$((ERR+1)); }

TRAILER_EN='^## (References|Summary|FAQ|Frequently Asked|Acknowled|Conclusion|Bibliography|Appendix|Exercises|Further [Rr]eading|Practice|Resources|Notes|Glossary)'
TRAILER_ZH='^## (参考文献|参考资料|致谢|常见问题|FAQ|总结|小结|附录|结语|结论|练习题|延伸阅读|深入阅读|进一步阅读|资源|备注|术语表)'
ANCHOR_EN="What'?s next|Where to go from here|Summary|Conclusion"
ANCHOR_ZH="接下来|下一步|总结|小结|结语|结论"

check_archetype() {
    local f=$1 anchor_re=$2 trailer_re=$3
    mapfile -t h2s < <(grep -nE '^## ' "$f" | sed 's/^[0-9]*://')
    [ "${#h2s[@]}" = "0" ] && return 0
    # Only flag REAL tail-bloat: anchor exists AND content sections come AFTER it.
    # Articles without explicit anchor are accepted (many use creative endings).
    local anchor_idx=-1
    for i in "${!h2s[@]}"; do
        if echo "${h2s[$i]}" | grep -qiE "$anchor_re"; then anchor_idx=$i; fi
    done
    [ "$anchor_idx" = "-1" ] && return 0  # no anchor → accept
    # anchor exists — every H2 after must be a trailer (References/Summary/etc)
    for ((i=anchor_idx+1; i<${#h2s[@]}; i++)); do
        if ! echo "${h2s[$i]}" | grep -qiE "$trailer_re"; then
            fail "$(basename $f) anchor 之后非 trailer: ${h2s[$i]}"
            return
        fi
    done
}

en_count=$(ls $EN/*.md 2>/dev/null | grep -v _index | wc -l)
zh_count=$(ls $ZH/*.md 2>/dev/null | grep -v _index | wc -l)
[ "$en_count" = "$zh_count" ] || fail "篇数: EN=$en_count ZH=$zh_count"

for f in $EN/*.md; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = "_index.md" ] && continue
    check_archetype "$f" "$ANCHOR_EN" "$TRAILER_EN"
    grep -nE '^\|.*\\\|.*\|' "$f" >/dev/null 2>&1 && fail "$(basename $f) 表格 \\|"
    unlinked=$(grep -nE '(Chapter|Part|Section) [0-9]+' "$f" 2>/dev/null | grep -vE '\]\(|http|^[^:]*:\s*#' | head -2)
    [ -n "$unlinked" ] && fail "$(basename $f) 未链接 Part/Section ref"
done
for f in $ZH/*.md; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = "_index.md" ] && continue
    check_archetype "$f" "$ANCHOR_ZH" "$TRAILER_ZH"
    grep -nE '^\|.*\\\|.*\|' "$f" >/dev/null 2>&1 && fail "$(basename $f) 表格 \\|"
    unlinked=$(grep -nE '第 ?[0-9]+ ?[章节部]' "$f" 2>/dev/null | grep -vE '\]\(|http|^[^:]*:\s*#' | head -2)
    [ -n "$unlinked" ] && fail "$(basename $f) 未链接 第N章 ref"
done

echo "$SERIES → $ERR issues"
