---
title: "Python Engineering (5): I/O, Serialization, and Data Formats"
date: 2022-04-19 09:00:00
tags:
  - Python
  - Serialization
  - Data Formats
categories:
  - Python Engineering
series: python-engineering
lang: en
description: "Handle files, paths, encodings, and data formats in Python. Compare JSON, YAML, TOML, CSV, pickle, and Parquet with practical examples."
disableNunjucks: true
series_order: 5
translationKey: "python-engineering-5"
---

Most programs are just plumbing between data formats. Read a CSV, transform it, write JSON. Load a config file, validate it, pass settings to the application. Every Python developer writes this code, and most of them get encoding, path handling, or serialization subtleties wrong at least once.

This article covers every common I/O pattern in Python, from basic file reading to columnar data formats, with a focus on the pitfalls that waste your time.

## File I/O: The Basics


![I/O pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/05-io-pipeline.png)

### Opening Files

```python
# The correct way: always use context managers
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# What happens without 'with':
f = open("data.txt", "r")
content = f.read()
f.close()  # Easy to forget, especially if an exception is raised above
```

The `with` statement guarantees `f.close()` runs even if an exception is raised. There is no reason to ever open a file without `with`.

### File Modes

| Mode | Name | Creates? | Truncates? | Position |
|------|------|----------|------------|----------|
| `"r"` | Read | No (error if missing) | No | Start |
| `"w"` | Write | Yes | Yes | Start |
| `"a"` | Append | Yes | No | End |
| `"x"` | Exclusive create | Yes (error if exists) | N/A | Start |
| `"r+"` | Read+write | No | No | Start |
| `"w+"` | Write+read | Yes | Yes | Start |
| `"rb"` | Read binary | No | No | Start |
| `"wb"` | Write binary | Yes | Yes | Start |

### Reading Patterns

```python
# Read entire file as string
with open("data.txt", encoding="utf-8") as f:
    content = f.read()

# Read as list of lines
with open("data.txt", encoding="utf-8") as f:
    lines = f.readlines()
# Each line includes the trailing '\n'

# Iterate line by line (memory efficient for large files)
with open("data.txt", encoding="utf-8") as f:
    for line in f:
        process(line.rstrip("\n"))

# Read specific number of bytes
with open("data.bin", "rb") as f:
    header = f.read(4)  # first 4 bytes
    rest = f.read()     # remaining bytes
```

### Writing Patterns

```python
# Write string
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, world\n")

# Write multiple lines
lines = ["first", "second", "third"]
with open("output.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
# Or:
with open("output.txt", "w", encoding="utf-8") as f:
    f.writelines(line + "\n" for line in lines)

# Append to existing file
with open("log.txt", "a", encoding="utf-8") as f:
    f.write(f"[{timestamp}] Event occurred\n")

# Write binary
with open("output.bin", "wb") as f:
    f.write(b"\x00\x01\x02\x03")
```

## pathlib.Path: Modern File Path Handling

The `pathlib` module replaces `os.path` with an object-oriented API. Use it everywhere.

![pathlib vs os.path](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/05-pathlib-vs-os.png)


```python
from pathlib import Path

# Create paths
project = Path("/home/user/project")
config = project / "config" / "settings.toml"
# Result: PosixPath('/home/user/project/config/settings.toml')

# Current directory and home
cwd = Path.cwd()
home = Path.home()

# Path components
p = Path("/home/user/project/data/file.csv")
p.name       # 'file.csv'
p.stem       # 'file'
p.suffix     # '.csv'
p.parent     # PosixPath('/home/user/project/data')
p.parents[1] # PosixPath('/home/user/project')
p.parts      # ('/', 'home', 'user', 'project', 'data', 'file.csv')
```

### Common Operations

```python
from pathlib import Path

p = Path("data")

# Check existence
p.exists()       # True/False
p.is_file()      # True if it's a file
p.is_dir()       # True if it's a directory

# Create directories
p.mkdir(parents=True, exist_ok=True)

# List directory contents
for child in p.iterdir():
    print(child)

# Glob patterns
for csv_file in p.glob("*.csv"):
    print(csv_file)

# Recursive glob
for py_file in p.rglob("*.py"):
    print(py_file)

# Read and write (convenience methods)
text = p.joinpath("config.txt").read_text(encoding="utf-8")
p.joinpath("output.txt").write_text("hello\n", encoding="utf-8")
data = p.joinpath("image.png").read_bytes()
p.joinpath("copy.png").write_bytes(data)

# File metadata
stat = p.stat()
stat.st_size     # File size in bytes
stat.st_mtime    # Modification time (Unix timestamp)

# Rename and delete
p.rename("new_name")
p.unlink()          # Delete file
p.rmdir()           # Delete empty directory
```

### os.path vs pathlib

| Operation | os.path | pathlib |
|-----------|---------|---------|
| Join paths | `os.path.join(a, b)` | `Path(a) / b` |
| Get filename | `os.path.basename(p)` | `p.name` |
| Get extension | `os.path.splitext(p)[1]` | `p.suffix` |
| Get parent | `os.path.dirname(p)` | `p.parent` |
| Check exists | `os.path.exists(p)` | `p.exists()` |
| Read file | `open(p).read()` | `p.read_text()` |
| Glob | `glob.glob("*.txt")` | `Path(".").glob("*.txt")` |
| Absolute path | `os.path.abspath(p)` | `p.resolve()` |

pathlib is cleaner in every case. The `/` operator for joining paths is reason enough to switch.

## Encoding: UTF-8 Everywhere


![Encoding flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/05-encoding-flow.png)


![Data serialization formats json yaml toml as different conta](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/05-data-serialization-formats-json-yaml-toml-as-different-conta.jpg)

### The Problem

```python
# This works on your Mac but fails on a Windows server:
with open("data.txt") as f:
    content = f.read()
# UnicodeDecodeError: 'cp1252' codec can't decode byte 0xe9
```

When you do not specify `encoding`, Python uses the platform default. On macOS and Linux, this is usually UTF-8. On Windows, it is often cp1252 (Windows-1252). This means code that works on your machine breaks in production.

### The Solution

Always specify encoding:

```python
# Always do this
with open("data.txt", encoding="utf-8") as f:
    content = f.read()
```

Starting with Python 3.15 (PEP 686), UTF-8 will be the default. Until then, be explicit.

### Handling Encoding Errors

```python
# Skip invalid bytes
with open("messy.txt", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Replace invalid bytes with ?
with open("messy.txt", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Detect encoding (when you don't know)
import chardet

with open("mystery.txt", "rb") as f:
    raw = f.read()
    detected = chardet.detect(raw)
    # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

content = raw.decode(detected["encoding"])
```

### BOM (Byte Order Mark)

Some Windows tools prepend a BOM (`﻿`) to UTF-8 files. Use `utf-8-sig` to handle it:

```python
# Reading: strips BOM if present
with open("windows_file.csv", encoding="utf-8-sig") as f:
    content = f.read()

# Writing: adds BOM (for Windows compatibility)
with open("output.csv", "w", encoding="utf-8-sig") as f:
    f.write("data\n")
```

## JSON

JSON is the most common data interchange format. Python's `json` module handles it natively.

### Reading and Writing

```python
import json

# Parse JSON string
data = json.loads('{"name": "Alice", "age": 30}')
# data = {'name': 'Alice', 'age': 30}

# Serialize to JSON string
text = json.dumps(data)
# '{"name": "Alice", "age": 30}'

# Pretty print
text = json.dumps(data, indent=2, sort_keys=True)
# {
#   "age": 30,
#   "name": "Alice"
# }

# Read from file
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

# Write to file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

The `ensure_ascii=False` parameter is important for non-ASCII text. Without it, characters like Chinese or emoji are escaped as `\uXXXX`.

### Custom Serializers

JSON does not support `datetime`, `Path`, `set`, `bytes`, or custom objects. Handle them with `default`:

![Serialization formats](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/05-serialization-formats.png)


```python
import json
from datetime import datetime
from pathlib import Path


def json_serializer(obj):
    """Handle types that json.dumps cannot serialize."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


data = {
    "timestamp": datetime.now(),
    "path": Path("/home/user/data"),
    "tags": {"python", "coding"},
}

text = json.dumps(data, default=json_serializer, indent=2)
```

### Command-Line JSON Tool

Python includes a JSON formatter:

```bash
# Pretty print a JSON file
$ python -m json.tool data.json

# From a pipe
$ curl -s https://api.example.com/data | python -m json.tool
```

## YAML

YAML is popular for configuration files because it is human-readable and supports comments.

```bash
(.venv) $ pip install pyyaml
```

```python
import yaml

# Read YAML
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Write YAML
with open("output.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
```

### safe_load vs load

**Always use `safe_load`**. The `load` function can execute arbitrary Python code embedded in YAML:

```yaml
# This YAML executes Python code with yaml.load():
!!python/object/apply:os.system
  args: ["rm -rf /"]
```

`safe_load` rejects these tags. There is no reason to use `load` unless you fully trust the source.

### YAML Gotchas

```yaml
# YAML has surprising type coercion:
norway: NO       # Parsed as boolean False!
version: 3.10    # Parsed as float 3.1!
port: 8080       # Parsed as integer (usually what you want)
zip: 01onal      # Parsed as string

# Always quote ambiguous values:
norway: "NO"
version: "3.10"
```

This is a real source of bugs. Use `safe_load` and quote anything that looks like a boolean or number but is not.

## TOML

TOML is the modern alternative to YAML for configuration. It has no type coercion surprises and is the standard for Python packaging (`pyproject.toml`).

### Reading TOML

Python 3.11+ includes `tomllib`:

```python
# Python 3.11+
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Or from string
config = tomllib.loads("""
[server]
host = "0.0.0.0"
port = 8080
debug = false

[database]
url = "postgresql://localhost/mydb"
pool_size = 5
""")
```

Note: `tomllib` opens in binary mode (`"rb"`), not text mode.

For Python 3.10 and earlier:

```bash
(.venv) $ pip install tomli
```

```python
import tomli

with open("config.toml", "rb") as f:
    config = tomli.load(f)
```

### Writing TOML

The standard library does not include a TOML writer. Use `tomli-w`:

```bash
(.venv) $ pip install tomli-w
```

```python
import tomli_w

config = {
    "server": {"host": "0.0.0.0", "port": 8080},
    "database": {"url": "postgresql://localhost/mydb"},
}

with open("config.toml", "wb") as f:
    tomli_w.dump(config, f)
```

## CSV

CSV is everywhere in data work. Python's `csv` module handles it correctly (quoting, escaping, different delimiters).

### Reading CSV

```python
import csv

# As lists
with open("data.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        print(row)  # ['Alice', '30', 'alice@example.com']

# As dictionaries (usually better)
with open("data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])
```

### Writing CSV

```python
import csv

# Write with DictWriter
rows = [
    {"name": "Alice", "age": 30, "email": "alice@example.com"},
    {"name": "Bob", "age": 25, "email": "bob@example.com"},
]

with open("output.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age", "email"])
    writer.writeheader()
    writer.writerows(rows)
```

The `newline=""` parameter is important on Windows. Without it, you get double line breaks.

### CSV Edge Cases

```python
# Tab-separated values
with open("data.tsv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")

# Semicolons (common in European locales)
with open("data.csv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")

# Handle BOM in CSV from Excel
with open("excel_export.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
```

## Binary Formats


![Format size comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/05-format-sizes.png)


![File io pipeline data flowing from disk through buffers to a](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/05-file-io-pipeline-data-flowing-from-disk-through-buffers-to-a.jpg)

### pickle: Python Object Serialization

pickle serializes any Python object to bytes and back. It is fast and convenient.

```python
import pickle

data = {"key": [1, 2, 3], "nested": {"a": "b"}}

# Serialize
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Deserialize
with open("data.pkl", "rb") as f:
    loaded = pickle.load(f)
```

**pickle is dangerous.** Loading a pickle file executes arbitrary code. Never unpickle data from untrusted sources. Pickle files are also not portable between Python versions or between different machines. Use pickle only for temporary caching within your own system.

| Format | Human-readable | Cross-language | Safe to load from untrusted | Python-specific |
|--------|---------------|----------------|---------------------------|-----------------|
| JSON | Yes | Yes | Yes | No |
| YAML | Yes | Yes | Yes (safe_load) | No |
| TOML | Yes | Yes | Yes | No |
| pickle | No | No | **No (dangerous)** | Yes |
| msgpack | No | Yes | Yes | No |

### struct: Binary Data Packing

For working with binary protocols or file formats:

```python
import struct

# Pack data into bytes
packed = struct.pack(">IHB", 1024, 256, 42)
# > = big-endian, I = uint32, H = uint16, B = uint8
# Result: b'\x00\x00\x04\x00\x01\x00\x2a'

# Unpack bytes into values
values = struct.unpack(">IHB", packed)
# (1024, 256, 42)
```

### msgpack: Fast Binary Serialization

msgpack is like JSON but binary and faster:

```bash
(.venv) $ pip install msgpack
```

```python
import msgpack

data = {"name": "Alice", "scores": [95, 87, 91]}

# Serialize
packed = msgpack.packb(data)
# b'\x82\xa4name\xa5Alice\xa6scores\x93_W['

# Deserialize
unpacked = msgpack.unpackb(packed)
```

## Parquet and Arrow: Columnar Data

For large datasets, row-oriented formats (CSV, JSON) are slow and wasteful. Parquet stores data in columns, which enables compression and fast analytical queries.

```bash
(.venv) $ pip install pyarrow pandas
```

```python
import pandas as pd

# Read CSV, write Parquet
df = pd.read_csv("large_data.csv")
df.to_parquet("large_data.parquet", engine="pyarrow")

# Read Parquet
df = pd.read_parquet("large_data.parquet")

# Read specific columns (Parquet can skip unused columns)
df = pd.read_parquet("large_data.parquet", columns=["name", "age"])
```

Size and speed comparison for a 1 million row dataset:

| Format | File Size | Write Time | Read Time | Read 2 Columns |
|--------|-----------|------------|-----------|-----------------|
| CSV | 120 MB | 8.2s | 5.1s | 5.1s (reads all) |
| JSON | 200 MB | 12.5s | 9.8s | 9.8s (reads all) |
| Parquet | 15 MB | 1.8s | 0.4s | 0.1s |

Parquet is 8x smaller and 12x faster to read than CSV for this example.

## Configuration Patterns

### .env Files with python-dotenv

```bash
(.venv) $ pip install python-dotenv
```

```
# .env
DATABASE_URL=postgresql://localhost/mydb
API_KEY=sk-abc123
DEBUG=true
SECRET_KEY=super-secret-key-do-not-commit
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env into os.environ

database_url = os.environ["DATABASE_URL"]
api_key = os.environ["API_KEY"]
debug = os.environ.get("DEBUG", "false").lower() == "true"
```

Always add `.env` to `.gitignore`. Commit a `.env.example` with placeholder values:

```
# .env.example
DATABASE_URL=postgresql://localhost/mydb
API_KEY=your-api-key-here
DEBUG=false
SECRET_KEY=generate-a-random-key
```

## Comparison Table: Config File Formats

| Feature | JSON | YAML | TOML | .env |
|---------|------|------|------|------|
| Comments | No | Yes | Yes | Yes |
| Nested structures | Yes | Yes | Yes | No (flat only) |
| Type safety | Good | Poor (coercion) | Good | None (all strings) |
| Human-readable | Good | Good | Good | Good |
| Standard Python support | stdlib | PyYAML | stdlib (3.11+) | python-dotenv |
| Multi-line strings | Escaped | Yes | Yes | Limited |
| Common use case | APIs, data | Kubernetes, Docker Compose | pyproject.toml, Cargo | Secrets, env vars |
| Footgun risk | Low | Medium (type coercion) | Low | Low |

**Recommendations:**
- **Application config:** TOML (clear, typed, no surprises)
- **Secrets and env vars:** `.env` files (never committed)
- **Data interchange:** JSON (universal, every language supports it)
- **Avoid YAML** unless you are working with tools that require it (Kubernetes, GitHub Actions)

## What's Next

Files and data formats are the I/O layer. But what happens when your program needs to do many I/O operations at once, like downloading 100 files or querying 50 APIs? Sequential execution wastes most of its time waiting. In the next article, we will tackle concurrency with threads, processes, and asyncio, and learn which tool to use for which problem.
