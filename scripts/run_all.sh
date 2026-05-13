#!/bin/bash
# Master runner: rerun failed llm-engineering articles, then process all other 26 series.
# Auto-commit + deploy after each series. Logs per-series.

set -u
cd /root/chenk-hugo

LOG_DIR=/tmp/retrans_logs
mkdir -p $LOG_DIR

# All 27 series in order (excluding llm-engineering failed-only first)
SERIES=(
  "aliyun-bailian"
  "aliyun-fullstack"
  "aliyun-pai"
  "claude-code-learn"
  "cloud-computing"
  "computer-fundamentals"
  "databases"
  "docker-containers"
  "leetcode"
  "linear-algebra"
  "linux"
  "ml-math-derivations"
  "nlp"
  "ode"
  "openclaw-quickstart"
  "optimization-theory"
  "pde-ml"
  "probability-statistics"
  "python-engineering"
  "recommendation-systems"
  "reinforcement-learning"
  "standalone"
  "system-design"
  "terraform-agents"
  "time-series"
  "transfer-learning"
)

# First: retry the 4 failed llm-engineering articles inline
echo "=== Retry failed llm-engineering articles ==="
python3 -u -c "
import sys; sys.path.insert(0, 'scripts')
from full_retranslate import build_series_context, find_pair, process_article
ctx = build_series_context('llm-engineering')
for fn in ['01-architectures.md', '03-pretraining.md', '05-inference.md', '06-long-context.md']:
    zh = '/root/chenk-hugo/content/zh/llm-engineering/' + fn
    en = find_pair(zh)
    if not en:
        print(f'  ✗ {fn}: no EN pair'); continue
    ok, msg = process_article(zh, en, ctx)
    print(f'  {(\"✓\" if ok else \"✗\")} {fn}: {msg}', flush=True)
" > $LOG_DIR/_llm_retry.log 2>&1

# Commit any retry results
if ! git diff --quiet content/zh/llm-engineering/; then
  git add -A content/zh/llm-engineering/
  git commit -m "rewrite: retry-fix remaining ZH llm-engineering articles"
  bash deploy.sh > $LOG_DIR/_llm_retry_deploy.log 2>&1
fi

# Then process each series
for s in "${SERIES[@]}"; do
  echo "=== $s ==="
  date
  python3 -u scripts/full_retranslate.py "$s" > $LOG_DIR/${s}.log 2>&1
  if ! git diff --quiet content/zh/${s}/; then
    git add -A content/zh/${s}/
    git commit -m "rewrite: ZH ${s} via qwen3-max with EN context"
    bash deploy.sh > $LOG_DIR/${s}_deploy.log 2>&1
    echo "  committed + deployed"
  else
    echo "  no changes"
  fi
done

echo "=== ALL SERIES DONE ==="
date
