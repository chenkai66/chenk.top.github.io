#!/bin/bash
# chenk.top deploy: sync ZH dates from EN, build, push source + deploy
set -e
HUGO_DIR=/root/chenk-hugo
DEPLOY_DIR=/tmp/chenk-deploy

echo '=== [1/4] Sync ZH dates from EN ==='
cd "$HUGO_DIR"
python3 "$HUGO_DIR/scripts/sync-zh-dates-from-en.py"

echo '=== [2/4] Commit source changes (branch: source) ==='
cd "$HUGO_DIR"
git add -A
if ! git diff --cached --quiet; then
  MSG="${1:-Update source}"
  git -c user.name=chenkai66 -c user.email=chenkai.nb.666@gmail.com commit -m "$MSG" | tail -3
  git push origin source 2>&1 | tail -3
else
  echo '  (no source changes)'
fi

echo '=== [3/4] Hugo build ==='
cd "$HUGO_DIR"
hugo --gc --minify

echo '=== [4/4] Push to chenk.top.github.io master (deploy) ==='
cd "$DEPLOY_DIR"
git pull --rebase
rm -rf en zh covers css js *.html *.xml
cp -r "$HUGO_DIR/public/"* .
echo www.chenk.top > CNAME
git add -A
if git diff --cached --quiet; then
  echo '  (no deploy changes)'
  exit 0
fi
MSG="${1:-Update site}"
git -c user.name=chenkai66 -c user.email=chenkai.nb.666@gmail.com commit -m "$MSG"
git push origin master
echo "Deployed: $(git log -1 --oneline)"
