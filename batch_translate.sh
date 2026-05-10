#!/bin/bash
# Batch translate a series from EN to ZH
# Usage: batch_translate.sh <series_name>
# Example: batch_translate.sh llm-engineering

set -e
SERIES=$1
EN_DIR="/root/chenk-hugo/content/en/${SERIES}"
ZH_DIR="/root/chenk-hugo/content/zh/${SERIES}"

export DASHSCOPE_API_KEY="sk-e15119caf6aa4e50bfe74fb4a9cb22ae"

if [ -z "$SERIES" ]; then
    echo "Usage: $0 <series_name>"
    exit 1
fi

echo "=== Batch translating: ${SERIES} ==="

for f in ${EN_DIR}/[0-9]*.md; do
    fname=$(basename "$f")
    zh_file="${ZH_DIR}/${fname}"

    if [ ! -f "$f" ]; then
        continue
    fi

    echo ""
    echo ">>> ${fname}"
    python3 /root/chenk-hugo/translate_article.py "$f" "$zh_file"
    echo "<<< ${fname} done"
    sleep 2
done

echo ""
echo "=== All done: ${SERIES} ==="
