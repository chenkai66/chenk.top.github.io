#!/usr/bin/env python3
"""DRY-RUN: list OSS objects under blog-pic-ck that are NOT referenced by any
content/*.md or theme template. Does NOT delete anything.

Set OSS_AK and OSS_SK environment variables before running."""
import os, glob, re, sys
import oss2

AK = os.environ.get("OSS_AK")
SK = os.environ.get("OSS_SK")
if not AK or not SK:
    print("ERROR: set OSS_AK and OSS_SK environment variables", file=sys.stderr)
    sys.exit(1)

auth = oss2.Auth(AK, SK)
bucket = oss2.Bucket(auth, "https://oss-cn-beijing.aliyuncs.com", "blog-pic-ck")

referenced = set()
for pattern in [
    "/root/chenk-hugo/content/**/*.md",
    "/root/chenk-hugo/themes/**/*.html",
    "/root/chenk-hugo/themes/**/*.toml",
]:
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                c = f.read()
        except Exception:
            continue
        for m in re.finditer(r"https://blog-pic-ck\.oss-cn-beijing\.aliyuncs\.com/(\S+?)(?=[\s)\"\047>])", c):
            referenced.add(m.group(1))

all_keys = set()
sizes = {}
total_size = 0
marker = ""
while True:
    result = bucket.list_objects(prefix="posts/", marker=marker, max_keys=1000)
    for obj in result.object_list:
        all_keys.add(obj.key)
        sizes[obj.key] = obj.size
        total_size += obj.size
    if not result.is_truncated:
        break
    marker = result.next_marker

orphans = all_keys - referenced
orphan_size = sum(sizes.get(k, 0) for k in orphans)
print(f"Referenced: {len(referenced)}")
print(f"OSS total: {len(all_keys)}, {total_size/1024/1024:.1f} MB")
print(f"Orphans: {len(orphans)}, {orphan_size/1024/1024:.1f} MB")
