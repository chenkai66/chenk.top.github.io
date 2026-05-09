#!/bin/bash
# Step 1: Extract all unique OSS illustration.jpg paths
cd /root/chenk-hugo
grep -roh "https://blog-pic-ck[^)]*illustration[^)]*\.jpg" content/ | sort -u > /tmp/jpg_paths.txt
TOTAL=$(wc -l < /tmp/jpg_paths.txt)
echo "Found $TOTAL unique .jpg paths to fix"

# Step 2: Copy each .jpg to .png on OSS
echo ""
echo "=== Copying .jpg → .png on OSS ==="
COUNT=0
FAIL=0
while IFS= read -r url; do
    COUNT=$((COUNT + 1))
    # Convert URL to OSS path
    oss_path=$(echo "$url" | sed "s|https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/||")
    oss_png_path=$(echo "$oss_path" | sed "s/\.jpg$/.png/")
    
    ossutil cp "oss://blog-pic-ck/$oss_path" "oss://blog-pic-ck/$oss_png_path" -f 2>/dev/null
    if [ $? -eq 0 ]; then
        printf "\r  [%d/%d] OK: %s" "$COUNT" "$TOTAL" "$(basename $oss_png_path)"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $oss_path"
    fi
done < /tmp/jpg_paths.txt
echo ""
echo "OSS copy complete: $COUNT total, $FAIL failed"

# Step 3: Update all markdown references
echo ""
echo "=== Updating markdown references ==="
cd /root/chenk-hugo
FILES=$(grep -rl "blog-pic-ck.*illustration.*\.jpg" content/)
FCOUNT=0
for f in $FILES; do
    sed -i "s|illustration_\([0-9]*\)\.jpg|illustration_\1.png|g" "$f"
    sed -i "s|illustration\([0-9]*\)\.jpg|illustration\1.png|g" "$f"
    FCOUNT=$((FCOUNT + 1))
done
echo "Updated $FCOUNT markdown files"
echo ""
echo "Done!"
