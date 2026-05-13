#!/bin/bash
set -e
cd /root/chenk-hugo
LOG=/tmp/run_remaining.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Key pre-test ==="
python3 scripts/full_retranslate.py --pretest

echo ""
echo "=== Retry previously failed articles ==="
date
python3 scripts/full_retranslate.py --files \
  content/zh/llm-engineering/03-pretraining.md \
  content/zh/aliyun-bailian/02-qwen-llm-api.md \
  content/zh/aliyun-bailian/03-qwen-omni-multimodal.md \
  content/zh/aliyun-fullstack/03-vpc-networking.md \
  content/zh/aliyun-fullstack/06-ram-security.md \
  content/zh/aliyun-fullstack/11-pai-ml-platform.md \
  content/zh/aliyun-fullstack/12-terraform-e2e.md \
  content/zh/aliyun-pai/05-pai-designer-vs-quickstart.md \
  content/zh/cloud-computing/storage-systems.md \
  content/zh/computer-fundamentals/04-motherboard-gpu.md \
  2>&1 | tee /tmp/retrans_logs/_retry2.log

git add -A content/zh/ && git diff --cached --quiet || {
  git commit -m "rewrite: retry previously failed ZH articles via qwen3-max (JSON v2)"
  bash deploy.sh 2>&1 | tail -5
  echo "  retries committed + deployed"
}

SERIES=(
  databases
  docker-containers
  leetcode
  linear-algebra
  linux
  ml-math-derivations
  nlp
  ode
  openclaw-quickstart
  optimization-theory
  pde-ml
  probability-statistics
  python-engineering
  recommendation-systems
  reinforcement-learning
  standalone
  system-design
  terraform-agents
  time-series
  transfer-learning
)

for s in "${SERIES[@]}"; do
  echo ""
  echo "=== $s ==="
  date
  python3 scripts/full_retranslate.py "$s" 2>&1 | tee /tmp/retrans_logs/${s}.log

  git add -A content/zh/ && git diff --cached --quiet || {
    git commit -m "rewrite: ZH $s via qwen3-max with EN context"
    bash deploy.sh 2>&1 | tee /tmp/retrans_logs/${s}_deploy.log | tail -5
    echo "  committed + deployed"
  }
done

echo ""
echo "=== ALL DONE ==="
date
