---
title: "Python 工程实践（二）：项目结构 —— 从脚本到包"
date: 2022-04-12 09:00:00
tags:
  - Python
  - Packaging
  - CLI
categories:
  - Python Engineering
series: python-engineering
lang: zh
description: "学习如何将 Python 代码组织为规范的包，涵盖导入机制、入口点（entry points）和命令行工具开发。从零构建一个真实的命令行应用。"
disableNunjucks: true
series_order: 2
translationKey: "python-engineering-2"
---

每个项目都始于单个文件。你写下 `main.py`，它能运行；你添加功能，某天突然发现这个文件已膨胀至 1500 行——函数调用其他函数，而这些函数又依赖于 800 行之上定义的全局变量。代码虽能工作，但没人（包括未来的你自己）能理解它。

**从脚本跃迁至包，是 Python 项目中第一个真正的工程决策**。早期做对，后续所有环节（测试、打包、部署）都将变得轻松；若做错，则可能耗费数周时间去解开循环导入（circular imports）的死结。

## 单文件何时不再够用？

单文件脚本适用场景：
- 代码量少于 300 行  
- 逻辑清晰、自上而下线性执行  
- 仅你一人阅读和维护  
- 是一次性脚本，非长期维护的工具  

你需要转向包（package）结构的信号：
- 多人协作开发  
- 需要对独立组件进行单元测试  
- 需在多个脚本间复用函数  
- 代码存在明确的逻辑分层（如 config / data / logic / CLI）  
- 计划分发该工具（例如支持 `pip install`）

## 平铺布局（Flat Layout） vs `src` 布局

Python 生态中存在两种主流项目结构。

![Flat vs src layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-flat-vs-src.png)


### 平铺布局（Flat Layout）

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

包目录直接位于项目根目录。结构更简单，Flask、Requests 等知名项目均采用此方式。

### `src` 布局

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

包目录置于 `src/` 子目录内。该布局被 Python 打包权威机构（PyPA）推荐，其关键优势在于：**强制你在测试前先安装包**。这能在发布前就暴露打包错误（如遗漏文件、导入失败）。

在平铺布局中，`import my_tool` 会直接解析到本地目录，即使该包根本无法正确安装；而在 `src` 布局中，Python 根本找不到 `my_tool`，除非你先执行 `pip install -e .`。这是设计特性，而非缺陷。

### 如何选择？

| 判定维度 | 平铺布局 | `src` 布局 |
|----------|-----------|-------------|
| 简洁性 | 更简洁 | 目录嵌套略深 |
| 测试准确性 | 可能掩盖打包缺陷 | 提前捕获缺陷 |
| 典型案例 | Flask、Requests、FastAPI | pytest、pip、setuptools |
| PyPA 推荐程度 | 可接受 | **推荐** |
| 导入安全性 | 可能意外导入本地未安装版本 | 必须先安装才能导入 |

✅ **发布库（library）请使用 `src` 布局**  
✅ **部署环境可控的应用（application）可选用平铺布局**  
❓ 不确定时，请优先选择 `src` 布局。

## `__init__.py`：包的标识符

当一个目录包含 `__init__.py` 文件时，它即成为一个 Python 包。该文件可以为空，也可包含初始化逻辑。

![__init__.py patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-init-purpose.png)


```python
# src/my_tool/__init__.py

"""My Tool — a file downloader CLI."""

__version__ = "0.1.0"
```

### `__init__.py` 的作用

1. **标记目录为包**：使 Python 能从中导入模块  
2. **导入时自动执行**：当用户执行 `import my_tool` 时，其中代码即运行  
3. **控制公共 API**：通过 `__all__` 显式声明导出内容  

```python
# src/my_tool/__init__.py

from my_tool.core import download_file, validate_url
from my_tool.utils import format_size

__all__ = ["download_file", "validate_url", "format_size"]
```

此后用户可直接写 `from my_tool import download_file`，无需 `from my_tool.core import download_file`。

### 何时让 `__init__.py` 保持为空？

保持为空的典型场景：
- 包内含多个职责分明的子模块  
- 期望用户显式从具体子模块导入（如 `from my_tool.core import X`）  
- 子模块间存在循环依赖风险  

示例对比：  
- `import numpy` 的 `__init__.py` 极大，负责整合全部功能；  
- `import sqlalchemy` 的 `__init__.py` 极小，用户需显式 `from sqlalchemy.orm import Session`。

### 命名空间包（Namespace Packages，无 `__init__.py`）

自 Python 3.3 起，不含 `__init__.py` 的目录可作为命名空间包（namespace package），允许多个物理目录共同构成一个逻辑包。**除非你在构建插件系统，否则务必包含 `__init__.py`。**

![Package structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-package-structure.png)


## 相对导入 vs 绝对导入

```python
# 绝对导入 —— 永远有效，语义清晰
from my_tool.core import download_file
from my_tool.utils import format_size

# 相对导入 —— 仅在包内部有效
from .core import download_file
from .utils import format_size
from ..other_module import something  # 父包
```

### 实践建议

| 场景 | 推荐方式 |
|------|----------|
| 同一包内模块互相导入 | 相对导入（`.module`） |
| 导入标准库或第三方包 | 绝对导入（`import os`, `import requests`） |
| 在 `__init__.py` 中 | 二者皆可，但需全包统一风格 |
| 在直接运行的脚本中（`python script.py`） | **仅限绝对导入** |
| 在测试文件中 | 绝对导入 |

⚠️ 注意：相对导入在直接运行模块时会失败（如 `python src/my_tool/core.py`），因为 Python 无法推断包上下文。此时应改用 `python -m my_tool.core`。

### 循环导入（Circular Imports）

当模块 A 导入模块 B，而模块 B 又导入模块 A 时，即发生循环导入：

![Import resolution order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/02-import-resolution.png)


```python
# core.py
from my_tool.utils import format_size  # utils imports from core!

# utils.py
from my_tool.core import DEFAULT_TIMEOUT  # core imports from utils!
```

解决方案：
1. **将共享常量提取至独立模块**（如 `constants.py` 或 `config.py`）  
2. **延迟导入**：在函数体内而非模块顶层导入（推迟实际导入时机）  
3. **重构模块**：若两模块高度耦合，或许它们本该属于同一模块  

## `pyproject.toml`：包元数据配置

完整 `pyproject.toml` 示例：

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

## 入口点（Entry Points）与控制台脚本（Console Scripts）

`pyproject.toml` 中的 `[project.scripts]` 定义了包安装后生成的可执行命令：

```toml
[project.scripts]
my-tool = "my_tool.cli:main"
```

执行 `pip install .` 后，你即可在任意位置运行 `my-tool`，它将调用 `my_tool/cli.py` 中的 `main()` 函数。

这就是 `black`、`ruff`、`pytest`、`flask` 等 CLI 工具的工作原理：`pip install flask` 后，`flask` 命令便自动出现在你的 `PATH` 中。

### 内部实现原理

`pip install` 会在虚拟环境的 `bin/` 目录下创建一个轻量级包装脚本：

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

## `__main__.py`：让包可直接运行

`__main__.py` 使你能以 `python -m` 方式运行整个包：

```bash
$ python -m my_tool
```

Python 将查找 `my_tool/__main__.py` 并执行其中代码。

```python
# src/my_tool/__main__.py

"""Allow running as: python -m my_tool"""

from my_tool.cli import main

if __name__ == "__main__":
    main()
```

该机制在开发阶段（尚未安装包时）非常有用，也适用于那些既需被导入、又需直接运行的模块。

## 使用 `argparse` 构建 CLI

标准库 `argparse` 是构建命令行接口的基础方案：

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

使用示例：

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

`parse_args` 和 `main` 中的 `argv` 参数极大简化了测试：

```python
def test_parse_args():
    args = parse_args(["https://example.com/file.txt", "-o", "out.txt"])
    assert args.url == "https://example.com/file.txt"
    assert args.output == "out.txt"
```

## 使用 `click` 构建 CLI

对于更复杂的 CLI，`click` 是事实标准。它采用装饰器（decorator）而非命令式解析器构建：

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

`click` 相比 `argparse` 的核心优势：

| 特性 | `argparse` | `click` |
|------|------------|---------|
| 子命令（Subcommands） | 支持但冗长 | `@click.group()` 简洁优雅 |
| 类型校验 | 基础支持 | 可扩展的 `click.Path`, `click.Choice` |
| 测试 | 需手动构造 `argv` | 内置 `CliRunner` |
| 彩色输出 | 需手动实现 | `click.style()`, `click.echo()` |
| 交互式提示 | 需手动实现 | `click.prompt()`, `click.confirm()` |
| 进度条 | 不内置 | `click.progressbar()` |

### `click` 子命令实战

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

使用方式：

```bash
$ my-tool download https://example.com/data.csv
$ my-tool convert data.csv json
$ my-tool --help
```

## 实战：构建一个文件下载器

我们来搭建一个完整的下载器项目结构。

### 项目布局

```
my-downloader/
  src/
    my_downloader/
      __init__.py         # 包版本号、公共 API
      __main__.py          # 支持 python -m my_downloader
      cli.py               # Click CLI 接口
      core.py              # 下载核心逻辑
      utils.py             # 工具函数
      config.py            # 常量与默认值
  tests/
    __init__.py
    conftest.py            # 共享 fixture
    test_core.py
    test_cli.py
    test_utils.py
  pyproject.toml
  requirements.txt
  .python-version
  .gitignore
  README.md
```

### `config.py` —— 常量定义

```python
# src/my_downloader/config.py

"""Application constants and defaults."""

DEFAULT_TIMEOUT = 30
DEFAULT_CHUNK_SIZE = 8192
MAX_RETRIES = 3
USER_AGENT = "my-downloader/0.1.0"
```

### `utils.py` —— 工具函数

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

### `core.py` —— 业务逻辑

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

### 开发模式安装

```bash
$ cd my-downloader
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -e ".[dev]"
```

`-e` 标志启用“可编辑安装”（editable mode），代码修改后立即生效，无需重复安装。

安装完成后，`my-downloader` 命令即可全局使用：

```bash
(.venv) $ my-downloader https://example.com/data.csv
Downloading: data.csv [100.0%] 1.2 KB
Downloaded: data.csv
```

同时 `python -m my_downloader` 也能运行，这得益于 `__main__.py`。

## 常见导入错误及修复方案

| 错误信息 | 原因 | 解决方法 |
|----------|------|-----------|
| `ModuleNotFoundError: No module named 'my_tool'` | 包未安装 | `pip install -e .` |
| `ImportError: attempted relative import with no known parent package` | 直接运行 `.py` 文件 | 改用 `python -m my_tool.module` |
| `ImportError: cannot import name 'X' from 'my_tool'` | `X` 未在 `__init__.py` 中导出，或存在循环导入 | 检查 `__init__.py`，拆解循环依赖 |
| `ModuleNotFoundError: No module named 'my_tool.core'` | 缺少 `__init__.py` 或 `pyproject.toml` 中 `find` 配置错误 | 确认 `__init__.py` 存在，检查 `pyproject.toml` 的 `find` 配置 |

## 下一步

项目结构已就绪，下一步是确保它真正可靠地工作。**测试不是为了追求覆盖率数字，而是为了建立信心：确信你的代码正如你所设想的那样运行。** 在下一篇文章中，我们将配置 `pytest`，编写有意义的测试（含 fixtures 和 `parametrize`），并学习如何在测试暴露问题时高效调试。