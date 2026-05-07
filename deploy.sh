#!/bin/bash
set -e
HUGO_DIR=/root/chenk-hugo
DEPLOY_DIR=/tmp/chenk-deploy

echo '=== [1/3] Sync ZH dates from EN ==='
cd "$HUGO_DIR"
python3 "$HUGO_DIR/scripts/sync-zh-dates-from-en.py"

echo '=== [2/3] Hugo build ==='
cd "$HUGO_DIR"
hugo --gc --minify

echo '=== [3/3] Push to chenk.top.github.io ==='
cd "$DEPLOY_DIR"
git pull --rebase
rm -rf en zh covers css js *.html *.xml
cp -r "$HUGO_DIR/public/"* .
echo www.chenk.top > CNAME
git add -A
if git diff --cached --quiet; then
  echo 'No changes to deploy.'
  exit 0
fi
MSG="${1:-Update site}"
git -c user.name=chenkai66 -c user.email=chenkai.nb.666@gmail.com commit -m "$MSG"
git push origin master
echo "Deployed: $(git log -1 --oneline)"
