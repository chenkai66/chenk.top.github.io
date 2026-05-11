---
title: "Python Engineering (2): Project Structure — From Script to Package"
date: 2022-04-12 09:00:00
tags:
  - Python
  - Packaging
  - CLI
categories:
  - Python Engineering
series: python-engineering
lang: en
description: "Learn how to organize Python code into proper packages with imports, entry points, and CLI tools. Build a real command-line application from scratch."
disableNunjucks: true
series_order: 2
translationKey: "python-engineering-2"
---

Every project starts as a single file. You write `main.py`, it works, you add features, and one day you realize you have 1,500 lines in one file with functions that call other functions that depend on globals defined 800 lines above. The code works, but nobody (including future you) can understand it.

The jump from script to package is the first real engineering decision in a Python project. Get it right early and everything else (testing, packaging, deployment) becomes easier. Get it wrong and you spend weeks untangling circular imports.

## When a Single File Is Not Enough

A single-file script is fine when:
- The code is under 300 lines
- There is one clear flow from top to bottom
- You are the only person who will ever read it
- It is a throwaway script, not a maintained tool

You need a package when:
- Multiple people work on the code
- You want to test individual components
- You need to reuse functions across scripts
- The code has distinct logical sections (config, data, logic, CLI)
- You plan to distribute it (pip install)

## Flat Layout vs src Layout

There are two dominant project structures in the Python ecosystem.

![Flat vs src layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-flat-vs-src.png)


### Flat Layout

```
my_tool/
  my_tool/
    __init__.py
    core.py
    cli.py
    utils.py
  tests/
    test_core.py
    test_cli.py
  pyproject.toml
  README.md
```

The package directory sits at the project root. This is simpler and used by many projects including Flask and Requests.

### src Layout

```
my_tool/
  src/
    my_tool/
      __init__.py
      core.py
      cli.py
      utils.py
  tests/
    test_core.py
    test_cli.py
  pyproject.toml
  README.md
```

The package directory is inside `src/`. This layout is recommended by the Python Packaging Authority (PyPA) and has one critical advantage: **it forces you to install your package before testing it.** This catches packaging errors (missing files, broken imports) before you ship.

With the flat layout, `import my_tool` resolves to the local directory even if the package is not properly installable. With src layout, Python cannot find `my_tool` unless you run `pip install -e .` first. This is a feature, not a bug.

### Which to Choose

| Criterion | Flat Layout | src Layout |
|-----------|-------------|------------|
| Simplicity | Simpler | Slightly more nesting |
| Testing accuracy | May hide packaging bugs | Catches them early |
| Popular examples | Flask, Requests, FastAPI | pytest, pip, setuptools |
| PyPA recommendation | Acceptable | Recommended |
| Import safety | Accidental imports possible | Must install first |

**Use src layout for libraries** you plan to publish. **Use flat layout for applications** where you control the deployment environment. When in doubt, use src layout.

## `__init__.py`: The Package Marker


![Python package import resolution detective following sys pat](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/02-python-package-import-resolution-detective-following-sys-pat.jpg)

A directory becomes a Python package when it contains `__init__.py`. This file can be empty or contain initialization code.

![Package structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-package-structure.png)


```python
# src/my_tool/__init__.py

"""My Tool — a file downloader CLI."""

__version__ = "0.1.0"
```

### What `__init__.py` Does

1. **Marks a directory as a package** so Python can import from it
2. **Runs on import** — code in `__init__.py` executes when someone does `import my_tool`
3. **Controls the public API** via `__all__`

![__init__.py patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-init-purpose.png)


```python
# src/my_tool/__init__.py

from my_tool.core import download_file, validate_url
from my_tool.utils import format_size

__all__ = ["download_file", "validate_url", "format_size"]
```

Now users can write `from my_tool import download_file` instead of `from my_tool.core import download_file`.

### When `__init__.py` Should Be Empty

Keep it empty when:
- The package has submodules with distinct purposes
- You want users to import from specific submodules
- There are circular dependency risks between submodules

Example: `import numpy` has a large `__init__.py` that wires everything together. `import sqlalchemy` keeps `__init__.py` minimal and expects `from sqlalchemy.orm import Session`.

### Namespace Packages (No `__init__.py`)

Since Python 3.3, directories without `__init__.py` are namespace packages. These allow a package to span multiple directories on disk. Unless you are building a plugin system, always include `__init__.py`.

## Relative vs Absolute Imports

```python

![Import resolution order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-import-resolution.png)

# Absolute import — always works, always clear
from my_tool.core import download_file
from my_tool.utils import format_size

# Relative import — works inside the package
from .core import download_file
from .utils import format_size
from ..other_module import something  # parent package
```

### Rules of Thumb

| Situation | Use |
|-----------|-----|
| Importing from within the same package | Relative (`.module`) |
| Importing stdlib or third-party | Absolute (`import os`, `import requests`) |
| In `__init__.py` | Either, but be consistent |
| In scripts run directly (`python script.py`) | Absolute only |
| In tests | Absolute |

Relative imports fail when you run a module directly as a script (`python src/my_tool/core.py`) because Python does not know the package context. Use `python -m my_tool.core` instead.

### Circular Imports

Circular imports happen when module A imports from module B, and module B imports from module A.

```python
# core.py
from my_tool.utils import format_size  # utils imports from core!

# utils.py
from my_tool.core import DEFAULT_TIMEOUT  # core imports from utils!
```

Solutions:
1. **Move shared constants to a separate module** (`constants.py` or `config.py`)
2. **Import inside functions** instead of at module level (delays the import)
3. **Restructure** — if two modules are tightly coupled, maybe they should be one module

## `pyproject.toml` for Package Metadata

The full project metadata in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my-tool"
version = "0.1.0"
description = "A CLI file downloader"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
keywords = ["download", "cli", "tool"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "requests>=2.28",
    "click>=8.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
]

[project.scripts]
my-tool = "my_tool.cli:main"

[project.urls]
Homepage = "https://github.com/you/my-tool"
Repository = "https://github.com/you/my-tool"
Issues = "https://github.com/you/my-tool/issues"

[tool.setuptools.packages.find]
where = ["src"]
```

## Entry Points and Console Scripts

The `[project.scripts]` section in `pyproject.toml` creates executable commands when the package is installed:

```toml
[project.scripts]
my-tool = "my_tool.cli:main"
```

After `pip install .`, you can run `my-tool` from anywhere. It calls the `main()` function in `my_tool/cli.py`.

This is how CLI tools like `black`, `ruff`, `pytest`, and `flask` work. You `pip install flask` and the `flask` command appears in your PATH.

### How It Works Internally

`pip install` creates a small wrapper script in the venv's `bin/` directory:

```bash
$ cat .venv/bin/my-tool
#!/home/user/project/.venv/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from my_tool.cli import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

## `__main__.py` for Runnable Packages

`__main__.py` lets you run a package with `python -m`:

```bash
$ python -m my_tool
```

Python looks for `my_tool/__main__.py` and executes it.

```python
# src/my_tool/__main__.py

"""Allow running as: python -m my_tool"""

from my_tool.cli import main

if __name__ == "__main__":
    main()
```

This is useful during development (before installing the package) and for modules that need to be both importable and runnable.

## CLI with argparse


![Python project structure as a well organized filing cabinet](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/02-python-project-structure-as-a-well-organized-filing-cabinet-.jpg)

The standard library includes `argparse` for command-line interfaces:

![CLI entry point architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-cli-architecture.png)


```python
# src/my_tool/cli.py

import argparse
import sys

from my_tool.core import download_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="my-tool",
        description="Download files from URLs",
    )
    parser.add_argument(
        "url",
        help="URL to download",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: derive from URL)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = download_file(
            url=args.url,
            output=args.output,
            quiet=args.quiet,
            timeout=args.timeout,
        )
        if not args.quiet:
            print(f"Downloaded: {path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
```

Usage:

```bash
$ my-tool https://example.com/data.csv -o data.csv --timeout 60
Downloading: data.csv [============================] 100% 2.4MB
Downloaded: data.csv

$ my-tool --help
usage: my-tool [-h] [-o OUTPUT] [-q] [--timeout TIMEOUT] url

Download files from URLs

positional arguments:
  url                   URL to download

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: derive from URL)
  -q, --quiet           Suppress progress output
  --timeout TIMEOUT     Request timeout in seconds (default: 30)
```

The `argv` parameter in `parse_args` and `main` makes testing easy:

```python
def test_parse_args():
    args = parse_args(["https://example.com/file.txt", "-o", "out.txt"])
    assert args.url == "https://example.com/file.txt"
    assert args.output == "out.txt"
```

## CLI with click

For more complex CLIs, `click` is the de facto standard. It uses decorators instead of imperative parser setup:

```python
# src/my_tool/cli.py

import click

from my_tool.core import download_file


@click.command()
@click.argument("url")
@click.option("-o", "--output", default=None, help="Output file path")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option("--timeout", default=30, type=int, help="Timeout in seconds")
def main(url: str, output: str | None, quiet: bool, timeout: int) -> None:
    """Download files from URLs."""
    try:
        path = download_file(url=url, output=output, quiet=quiet, timeout=timeout)
        if not quiet:
            click.echo(f"Downloaded: {path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
```

click advantages over argparse:

| Feature | argparse | click |
|---------|----------|-------|
| Subcommands | Possible but verbose | `@click.group()` |
| Type validation | Basic | Extensible `click.Path`, `click.Choice` |
| Testing | Parse argv manually | `CliRunner` built in |
| Colored output | Manual | `click.style()`, `click.echo()` |
| Prompts | Manual | `click.prompt()`, `click.confirm()` |
| Progress bars | Not included | `click.progressbar()` |

### click with Subcommands

```python
@click.group()
@click.version_option()
def cli():
    """My Tool — file downloader and converter."""
    pass


@cli.command()
@click.argument("url")
@click.option("-o", "--output", default=None)
def download(url: str, output: str | None) -> None:
    """Download a file from a URL."""
    path = download_file(url=url, output=output)
    click.echo(f"Downloaded: {path}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_format", type=click.Choice(["csv", "json", "parquet"]))
def convert(input_file: str, output_format: str) -> None:
    """Convert a file to another format."""
    result = convert_file(input_file, output_format)
    click.echo(f"Converted: {result}")
```

Usage:

```bash
$ my-tool download https://example.com/data.csv
$ my-tool convert data.csv json
$ my-tool --help
```

## Real Example: Building a File Downloader

Let us build the complete project structure for the downloader tool.

### Project Layout

```
my-downloader/
  src/
    my_downloader/
      __init__.py         # Package version, public API
      __main__.py          # python -m my_downloader
      cli.py               # Click CLI interface
      core.py              # Download logic
      utils.py             # Helper functions
      config.py            # Constants, defaults
  tests/
    __init__.py
    conftest.py            # Shared fixtures
    test_core.py
    test_cli.py
    test_utils.py
  pyproject.toml
  requirements.txt
  .python-version
  .gitignore
  README.md
```

### `config.py` — Constants

```python
# src/my_downloader/config.py

"""Application constants and defaults."""

DEFAULT_TIMEOUT = 30
DEFAULT_CHUNK_SIZE = 8192
MAX_RETRIES = 3
USER_AGENT = "my-downloader/0.1.0"
```

### `utils.py` — Helpers

```python
# src/my_downloader/utils.py

"""Utility functions for file operations and formatting."""

from pathlib import Path
from urllib.parse import urlparse


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        size_bytes: Number of bytes.

    Returns:
        Formatted string like '2.4 MB'.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} PB"


def filename_from_url(url: str) -> str:
    """Extract filename from a URL.

    Args:
        url: The URL to parse.

    Returns:
        The filename portion of the URL path,
        or 'download' if none can be determined.
    """
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name if name else "download"
```

### `core.py` — Business Logic

```python
# src/my_downloader/core.py

"""Core download logic."""

from pathlib import Path

import requests

from my_downloader.config import DEFAULT_CHUNK_SIZE, DEFAULT_TIMEOUT, USER_AGENT
from my_downloader.utils import filename_from_url, format_size


def download_file(
    url: str,
    output: str | None = None,
    quiet: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """Download a file from a URL.

    Args:
        url: The URL to download from.
        output: Output file path. Derived from URL if None.
        quiet: If True, suppress progress output.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If the request fails.
    """
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()

    dest = Path(output) if output else Path(filename_from_url(url))
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if not quiet and total > 0:
                pct = downloaded / total * 100
                print(
                    f"\rDownloading: {dest.name} "
                    f"[{pct:5.1f}%] {format_size(downloaded)}",
                    end="",
                    flush=True,
                )

    if not quiet:
        print()  # newline after progress

    return dest
```

### Install in Development Mode

```bash
$ cd my-downloader
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -e ".[dev]"
```

The `-e` flag installs in "editable" mode. Code changes take effect immediately without reinstalling.

After installation, the `my-downloader` command is available:

```bash
(.venv) $ my-downloader https://example.com/data.csv
Downloading: data.csv [100.0%] 1.2 KB
Downloaded: data.csv
```

And `python -m my_downloader` also works because of `__main__.py`.

## Common Import Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'my_tool'` | Package not installed | `pip install -e .` |
| `ImportError: attempted relative import with no known parent package` | Running file directly | Use `python -m my_tool.module` |
| `ImportError: cannot import name 'X' from 'my_tool'` | X not in `__init__.py` or circular import | Check `__init__.py`, break circular deps |
| `ModuleNotFoundError: No module named 'my_tool.core'` | Missing `__init__.py` or wrong package structure | Verify `__init__.py` exists, check `find` config in pyproject.toml |

## What's Next

With a proper project structure in place, the next step is making sure it actually works. Testing is not about writing tests for the sake of coverage numbers. It is about building confidence that your code does what you think it does. In the next article, we will set up pytest, write meaningful tests with fixtures and parametrize, and learn to debug efficiently when tests reveal problems.
