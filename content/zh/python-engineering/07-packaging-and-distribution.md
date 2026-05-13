---
title: "Python 工程实践（七）：打包分发 —— 从 pip install 到 PyPI"
date: 2022-04-24 09:00:00
tags:
  - Python
  - Packaging
  - Docker
  - PyPI
categories: Python Engineering
series: python-engineering
lang: zh
description: "将 Python 代码打包为可分发形式，通过 pip 安装、发布至 PyPI、构建 Docker 镜像，并管理版本号。本指南完整覆盖从本地项目到可安装包的全流程。"
disableNunjucks: true
series_order: 7
translationKey: "python-engineering-7"
---

你写了一个实用的小工具。一位同事向你索要。你把整个文件夹压缩成 ZIP，通过邮件发给他。他解压后运行 `python main.py`，却得到 `ModuleNotFoundError` —— 因为他没有安装依赖。接着他手动安装了依赖，但版本不匹配；随后又发现他使用的是 Python 3.8，而你的代码中使用了 f-string 内的海象运算符（walrus operator），该语法在 Python 3.8 中尚不可用，导致解析失败。

**规范的打包能彻底解决所有这些问题。** 只需执行 `pip install your-tool`，一切就绪：依赖自动安装、版本精确匹配、 CLI 命令开箱即用。

## 包、模块与库

这些术语常被混用，但在 Python 生态中有明确的定义。

![打包流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/07-packaging-pipeline.png)


| 术语 | 定义 | 示例 |
|------|------|------|
| **Module （模块）** | 单个 `.py` 文件 | `utils.py` |
| **Package （包）** | 包含 `__init__.py` 的目录，内含多个模块 | `my_tool/` 目录 |
| **Library （库）** | 为复用而发布的包（或包集合） | `requests`, `flask` |
| **Distribution （分发包）** | 可安装的归档文件（wheel 或 sdist），托管于 PyPI | `requests-2.31.0-py3-none-any.whl` |
| **Script （脚本）** | 可直接运行的独立 `.py` 文件 | `download.py` |

当有人说“安装 requests 库”，其真实含义是：从 PyPI 下载 `requests` 分发包，该包中包含 `requests` 包及其子包。

## 构建分发包

Python 包支持两种标准分发格式。

![语义化版本控制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/07-versioning.png)


### sdist （源码分发包）

`.tar.gz` 格式的源码归档。接收方需具备构建工具链（如 C 扩展所需的编译器等）才能安装。

### wheel （预编译分发包）

`.whl` 文件（本质是 zip 归档）。已预先构建完成，无需编译，安装更快。`pip` 默认使用 wheel。

![Wheel 与 sdist 比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/07-wheel-vs-sdist.png)


```bash
# 安装构建工具
(.venv) $ pip install build

# 同时构建 sdist 和 wheel
(.venv) $ python -m build
* Creating venv isolated environment...
* Installing packages in isolated environment... (setuptools>=68.0, wheel)
* Getting build dependencies for sdist...
* Building sdist...
* Building wheel from sdist
Successfully built my_tool-0.1.0.tar.gz and my_tool-0.1.0-py3-none-any.whl
```

输出位于 `dist/` 目录下：

```
dist/
  my_tool-0.1.0.tar.gz                    # sdist
  my_tool-0.1.0-py3-none-any.whl          # wheel
```

wheel 文件名编码了元数据：`{name}-{version}-{python}-{abi}-{platform}.whl`。其中 `py3-none-any` 表示“仅支持 Python 3、无 ABI 依赖、跨平台”（即纯 Python 包）。

### 查看 wheel 内容

```bash
# .whl 本质就是 zip 文件
$ unzip -l dist/my_tool-0.1.0-py3-none-any.whl
Archive:  dist/my_tool-0.1.0-py3-none-any.whl
  Length      Date    Time    Name
---------  ---------- -----   ----
      142  2024-01-15 10:00   my_tool/__init__.py
     1845  2024-01-15 10:00   my_tool/core.py
      923  2024-01-15 10:00   my_tool/cli.py
      456  2024-01-15 10:00   my_tool/utils.py
      178  2024-01-15 10:00   my_tool/config.py
      631  2024-01-15 10:00   my_tool-0.1.0.dist-info/METADATA
       92  2024-01-15 10:00   my_tool-0.1.0.dist-info/WHEEL
       50  2024-01-15 10:00   my_tool-0.1.0.dist-info/entry_points.txt
      654  2024-01-15 10:00   my_tool-0.1.0.dist-info/RECORD
```

## 使用 pyproject.toml 进行打包配置


![从脚本到发布到 PyPI 的 Python 打包过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/07-python-packaging-journey-from-script-to-pypi-published-packa.jpg)

完整的打包配置如下：

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my-tool"
version = "0.1.0"
description = "A CLI tool for downloading and converting files"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
keywords = ["download", "convert", "cli"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Utilities",
]
dependencies = [
    "requests>=2.28,<3",
    "click>=8.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff>=0.3",
    "mypy",
    "build",
    "twine",
]

[project.scripts]
my-tool = "my_tool.cli:main"

[project.urls]
Homepage = "https://github.com/you/my-tool"
Documentation = "https://my-tool.readthedocs.io"
Repository = "https://github.com/you/my-tool"
Issues = "https://github.com/you/my-tool/issues"
Changelog = "https://github.com/you/my-tool/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
my_tool = ["py.typed"]
```

### 关键配置项说明

**`[build-system]`**：告知 `pip` 使用哪个构建工具。`setuptools` 是默认选项，也可选用 `flit`、`hatchling` 或 `poetry-core`。

**`[project]`**：遵循 PEP 621 的标准元数据定义，包括名称、版本、依赖等。

**`[project.scripts]`**：声明 CLI 入口点。`pip install` 后，`my-tool` 命令将自动加入 `PATH`。

**`[project.optional-dependencies]`**：按用途分组的可选依赖。可通过 `pip install my-tool[dev]` 安装。

**`[tool.setuptools.packages.find]`**：指示 `setuptools` 在 `src/` 目录下查找包。

### 包含非 Python 数据文件

如果包需要加载模板、配置文件或数据文件，必须显式声明。

```toml
[tool.setuptools.package-data]
my_tool = [
    "templates/*.html",
    "data/*.json",
    "py.typed",  # PEP 561 类型检查标记
]
```

或创建 `MANIFEST.in` 专用于 sdist：

```
include LICENSE
include README.md
recursive-include src/my_tool/templates *.html
recursive-include src/my_tool/data *.json
```

## 发布到 PyPI

### 先在 TestPyPI 上测试

**务必先在 TestPyPI 测试，再上传至正式 PyPI。**

```bash
# 安装 twine
(.venv) $ pip install twine

# 构建
(.venv) $ python -m build

# 上传至 TestPyPI
(.venv) $ twine upload --repository testpypi dist/*
Uploading distributions to https://test.pypi.org/legacy/
Uploading my_tool-0.1.0-py3-none-any.whl [========================================] 100%
Uploading my_tool-0.1.0.tar.gz [========================================] 100%

# 从 TestPyPI 测试安装
(.venv) $ pip install --index-url https://test.pypi.org/simple/ my-tool
```

### 创建 PyPI 账户与 API Token

1. 访问 https://pypi.org/account/register/ 注册账户。
2. 进入 Account Settings > API Tokens。
3. 创建一个作用域限定于你项目的 token（首次上传可选“all projects”）。

配置 `~/.pypirc`：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 正式上传至 PyPI

```bash
(.venv) $ twine upload dist/*
Uploading distributions to https://upload.pypi.org/legacy/
Uploading my_tool-0.1.0-py3-none-any.whl [========================================] 100%
Uploading my_tool-0.1.0.tar.gz [========================================] 100%

View at:
https://pypi.org/project/my-tool/0.1.0/
```

现在任何人都可一键安装。

```bash
$ pip install my-tool
```

### 发布前检查清单

```bash
# 检查分发包常见错误
(.venv) $ twine check dist/*
Checking dist/my_tool-0.1.0-py3-none-any.whl: PASSED
Checking dist/my_tool-0.1.0.tar.gz: PASSED
```

`twine check` 可捕获的典型问题：
- 缺少 `README`
- `long_description` 格式非法
- 必填元数据缺失

## 私有包索引（Private Package Indexes）

并非所有包都适合公开发布，内部工具应使用私有索引。

![私有包索引](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/07-private-index.png)


### devpi

```bash
# 安装 devpi 服务端与客户端
$ pip install devpi-server devpi-client

# 启动服务端
$ devpi-server --start --port 3141

# 配置客户端
$ devpi use http://localhost:3141
$ devpi login root --password ""
$ devpi index -c dev

# 上传包
$ devpi upload dist/*

# 从私有索引安装
$ pip install my-tool --index-url http://localhost:3141/root/dev/+simple/
```

### pip 配置私有索引

```ini
# ~/.pip/pip.conf（Linux/macOS）或 %APPDATA%\pip\pip.ini（Windows）

[global]
extra-index-url = http://internal-pypi.company.com/simple/
trusted-host = internal-pypi.company.com
```

或在 `pyproject.toml` 中按项目指定（非标准，但 `pip-tools` 支持）：

```toml
# This is not standard but supported by pip-tools
# For pip itself, use pip.conf or command-line flags
```

```bash
# 按命令指定
$ pip install my-internal-tool --extra-index-url http://internal-pypi.company.com/simple/
```

## 使用 Docker 构建 Python 镜像


![Docker 中的 Python 打包](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/07-docker-python.png)

### 基础镜像选择

| 基础镜像 | 大小 | 适用场景 |
|----------|------|----------|
| `python:3.11` | ~900MB | 开发环境，含构建工具 |
| `python:3.11-slim` | ~150MB | 生产环境，精简版 |
| `python:3.11-alpine` | ~50MB | 极致精简，但 musl libc 可能引发兼容性问题 |
| `python:3.11-bookworm` | ~900MB | Debian Bookworm，兼容性佳 |
| `python:3.11-slim-bookworm` | ~150MB | Debian Bookworm 生产环境 |

**推荐：生产环境使用 `python:3.11-slim`。除非明确需要极小体积且能处理 musl 兼容性问题（如 numpy、 pandas 等含 C 扩展的包可能失败），否则避免 Alpine。**

### 基础 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（如需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 优先复制依赖文件（提升缓存命中率）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ src/
COPY pyproject.toml .

# 安装当前包
RUN pip install --no-cache-dir .

# 以非 root 用户运行
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["my-tool", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### 多阶段构建（Multi-Stage Build）

通过分离构建与运行阶段来减小最终镜像体积。

```dockerfile
# 第一阶段：构建
FROM python:3.11 AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# 第二阶段：运行时
FROM python:3.11-slim

WORKDIR /app

# 仅从 builder 复制已安装的包
COPY --from=builder /install /usr/local

# 创建非 root 用户
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["my-tool", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

构建阶段包含 `gcc` 和构建工具；运行阶段仅保留已安装的包。

### .dockerignore

```
.venv/
.git/
.mypy_cache/
__pycache__/
*.pyc
dist/
build/
*.egg-info/
.env
.env.*
tests/
docs/
*.md
```

### 构建与运行

```bash
# 构建
$ docker build -t my-tool:0.1.0 .

# 运行
$ docker run -p 8000:8000 my-tool:0.1.0

# 运行并传入环境变量
$ docker run -p 8000:8000 \
    -e DATABASE_URL=postgresql://db:5432/mydb \
    -e API_KEY=sk-abc123 \
    my-tool:0.1.0
```

## 使用 Poetry 构建与发布


![Wheel 与 sdist 比较：预组装家具 vs 宜家平板包装](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/07-wheel-vs-sdist-comparison-prebuilt-furniture-vs-ikea-flatpac.jpg)

若项目采用 Poetry 而非 setuptools：

```bash
# 安装 Poetry
$ curl -sSL https://install.python-poetry.org | python3 -

# 初始化新项目
$ poetry new my-tool
$ cd my-tool

# 或转换现有项目
$ poetry init
```

Poetry 使用 `pyproject.toml` 中的 `[tool.poetry]` 区块：

```toml
[tool.poetry]
name = "my-tool"
version = "0.1.0"
description = "A CLI tool"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "my_tool", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28"
click = "^8.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
ruff = "^0.3"

[tool.poetry.scripts]
my-tool = "my_tool.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

```bash
# 构建
$ poetry build
Building my-tool (0.1.0)
  - Building sdist
  - Built my_tool-0.1.0.tar.gz
  - Building wheel
  - Built my_tool-0.1.0-py3-none-any.whl

# 发布至 TestPyPI
$ poetry config repositories.testpypi https://test.pypi.org/legacy/
$ poetry publish --repository testpypi

# 发布至正式 PyPI
$ poetry publish
```

## 版本管理

### 语义化版本（SemVer）

标准格式：`MAJOR.MINOR.PATCH`

| 组件 | 何时递增 | 示例 |
|--------|-----------|--------|
| MAJOR | API 不兼容变更 | `1.0.0` → `2.0.0` |
| MINOR | 新功能（向后兼容） | `1.0.0` → `1.1.0` |
| PATCH | Bug 修复（向后兼容） | `1.0.0` → `1.0.1` |

预发布版本：`1.0.0a1`（alpha）、`1.0.0b1`（beta）、`1.0.0rc1`（release candidate）。

### 版本号设置方式

**方式一：单一信源 —— `pyproject.toml` 中定义**

```toml
[project]
version = "0.1.0"
```

运行时读取：

```python
from importlib.metadata import version

__version__ = version("my-tool")
```

**方式二：在 `__init__.py` 中定义 `__version__`**

```python
# src/my_tool/__init__.py
__version__ = "0.1.0"
```

并在 `pyproject.toml` 中引用：

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "my_tool.__version__"}
```

**方式三：基于 Git Tag 的动态版本（推荐）**

```bash
(.venv) $ pip install setuptools-scm
```

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]

[project]
dynamic = ["version"]

[tool.setuptools_scm]
```

版本号将自动从 Git tag 推导：

```bash
$ git tag v0.1.0
$ python -m build
# 包版本即为 0.1.0
```

## 实战示例：完整打包与发布流程

以下是以本系列前文中的下载器工具为例的完整发布流程：

```bash
# 1. 确保测试全部通过
(.venv) $ pytest -v --cov=my_tool
========================= 14 passed in 0.45s ==========================

# 2. 确保代码风格合规
(.venv) $ ruff check src/ tests/
All checks passed!

# 3. 确保类型检查无误
(.venv) $ mypy src/
Success: no issues found

# 4. 构建分发包
(.venv) $ python -m build

# 5. 检查构建结果
(.venv) $ twine check dist/*
Checking dist/my_tool-0.1.0-py3-none-any.whl: PASSED
Checking dist/my_tool-0.1.0.tar.gz: PASSED

# 6. 在全新虚拟环境中测试安装
$ python -m venv /tmp/test-install
$ source /tmp/test-install/bin/activate
(test-install) $ pip install dist/my_tool-0.1.0-py3-none-any.whl
(test-install) $ my-tool --help
Usage: my-tool [OPTIONS] URL
...
(test-install) $ deactivate

# 7. 先上传至 TestPyPI
(.venv) $ twine upload --repository testpypi dist/*

# 8. 从 TestPyPI 测试安装
$ pip install --index-url https://test.pypi.org/simple/ my-tool
$ my-tool --version
my-tool 0.1.0

# 9. 最终上传至正式 PyPI
(.venv) $ twine upload dist/*

# 10. 打上 Git Release Tag
$ git tag -a v0.1.0 -m "Release 0.1.0"
$ git push origin v0.1.0
```

### 使用 GitHub Actions 自动化发布

```yaml
# .github/workflows/publish.yml

name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build

      - name: Build
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # Uses trusted publishing (no API token needed)
```

启用 Trusted Publishing 后，PyPI 将信任你的 GitHub Actions 工作流，无需手动管理 API Token。

## 下一步

你的包已成功发布并可供安装。但它够快吗？在下一篇文章中，我们将学习如何对 Python 代码进行性能剖析、定位瓶颈、应用缓存与向量化优化，并掌握一项关键技能：**判断何时该优化、何时不该优化**。