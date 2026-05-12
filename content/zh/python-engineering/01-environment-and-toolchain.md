---
title: "Python 工程实践（一）：环境搭建——pyenv、venv 与依赖地狱"
date: 2022-04-10 09:00:00
tags:
  - Python
  - Environment
  - Toolchain
categories:
  - Python Engineering
series: python-engineering
lang: zh
description: "掌握使用 pyenv、虚拟环境及现代依赖管理工具进行 Python 环境管理。彻底告别依赖地狱。"
disableNunjucks: true
series_order: 1
translationKey: "python-engineering-1"
---

每位 Python 开发者都经历过这样的时刻：你在同事的机器上运行一段脚本，结果崩溃了——因为对方用的是 Python 3.8，而你是在 3.11 上编写的。更糟的是，你执行了 `pip install` 全局安装，却意外破坏了一个完全无关的项目。Python 的环境管理体系本身非常强大，但开箱即用的默认体验却像一片布满地雷的雷区。

本文将从零开始，完整梳理整套工具链。读完后，你将拥有一套可复现、隔离良好、版本锁定的开发环境，且在每一台机器上行为完全一致。

## Python 版本问题

大多数操作系统都自带一个系统级 Python。macOS 曾长期预装 Python 2.7（Monterey 中已移除）；Ubuntu 22.04 预装的是 Python 3.10。这个系统 Python 被 OS 层工具所依赖。向其中安装包或升级它，都可能导致操作系统异常。

![Dependency resolution flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-dep-resolution.png)


核心问题如下：

| 问题 | 示例 |
|---------|---------|
| 系统 Python 过于陈旧 | Ubuntu 20.04 自带 3.8，而你需要 3.11 的新特性 |
| 多个项目需要不同 Python 版本 | 项目 A 需要 3.9，项目 B 需要 3.12 |
| 全局 pip 安装引发冲突 | 包 X 要求 `requests>=2.28`，包 Y 却锁定 `requests==2.25` |
| 可复现性失效 | “在我机器上能跑”——只因各处版本不一致 |
| 系统工具依赖系统 Python | Ubuntu 的 `apt` 在内部使用系统 Python |

解决方案是一个三层架构：

1. **pyenv**：管理多个 Python 版本（并行安装 3.9、3.10、3.11 等）
2. **venv**：为每个项目隔离依赖
3. **pip-tools** 或 **Poetry**：精确锁定版本，保障可复现性

## pyenv：无痛管理多版本 Python

pyenv 通过拦截 `python` 命令，并将其重定向至你配置的任意 Python 版本，来实现版本切换。它通过在你的 `$PATH` 中插入 shim 脚本来完成这一操作。

![pyenv shim mechanism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-pyenv-shim.png)


### 安装

macOS 上：

```bash
brew install pyenv
```

Linux 上：

```bash
curl https://pyenv.run | bash
```

安装完成后，在你的 shell 配置文件（`~/.bashrc`、`~/.zshrc` 或 `~/.bash_profile`）中添加以下内容：

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

重新加载 shell：

```bash
source ~/.zshrc  # 或 ~/.bashrc
```

验证安装：

```bash
$ pyenv --version
pyenv 2.3.36
```

### 安装 Python 版本

列出可用版本：

```bash
$ pyenv install --list | grep "^  3\." | tail -10
  3.11.5
  3.11.6
  3.11.7
  3.12.0
  3.12.1
  3.12.2
  3.12.3
  3.12.4
  3.13.0a3
  3.13.0a4
```

安装指定版本：

```bash
$ pyenv install 3.11.7
Downloading Python-3.11.7.tar.xz...
Installing Python-3.11.7...
Installed Python-3.11.7 to /home/user/.pyenv/versions/3.11.7
```

在 macOS 上若构建失败，你很可能需要先安装：

```bash
brew install openssl readline sqlite3 xz zlib tcl-tk
```

在 Ubuntu/Debian 上：

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

### 设置当前激活版本

pyenv 支持三级版本选择机制，按优先级从高到低依次为：

```bash
# 全局默认（最低优先级）
$ pyenv global 3.11.7

# 按目录设置（会创建 .python-version 文件）
$ cd ~/projects/my-api
$ pyenv local 3.11.7
$ cat .python-version
3.11.7

# 仅当前 shell 会话生效（最高优先级）
$ pyenv shell 3.12.2
```

解析顺序为：`shell` > `local`（`.python-version`）> `global`（`~/.pyenv/version`）。

查看当前激活版本及其来源：

```bash
$ pyenv version
3.11.7 (set by /home/user/projects/my-api/.python-version)

$ pyenv versions
  system
  3.9.18
* 3.11.7 (set by /home/user/projects/my-api/.python-version)
  3.12.2
```

**请将 `.python-version` 提交至代码仓库。** 这能确保所有开发者使用完全一致的 Python 版本。零成本，却能杜绝大量因版本错配导致的 bug。

## 虚拟环境：依赖隔离


![Dependency hell tangled wires vs clean resolved dependencies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/01-dependency-hell-tangled-wires-vs-clean-resolved-dependencies.jpg)

即使有了正确的 Python 版本，你仍需依赖隔离。否则，`pip install` 会把包安装到共享位置，而两个需要同一包不同版本的项目就会发生冲突。

![Version management stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-version-stack.png)


### 创建虚拟环境

```bash
$ cd ~/projects/my-api
$ python -m venv .venv
```

该命令将创建一个 `.venv` 目录，其结构如下：

```
.venv/
  bin/          # python、pip、activate 等脚本
  include/      # C 头文件（用于编译扩展）
  lib/          # 已安装的包存放于此
  pyvirst.cfg   # 指向基础 Python 的路径
```

### 激活虚拟环境

```bash
# macOS / Linux
$ source .venv/bin/activate
(.venv) $

# Windows（PowerShell）
> .venv\Scripts\Activate.ps1

# Windows（cmd）
> .venv\Scripts\activate.bat
```

激活后，`python` 和 `pip` 将指向虚拟环境内的副本：

```bash
(.venv) $ which python
/home/user/projects/my-api/.venv/bin/python

(.venv) $ which pip
/home/user/projects/my-api/.venv/bin/pip
```

完成工作后退出：

```bash
(.venv) $ deactivate
$
```

### 为什么叫 `.venv`？

前缀 `.` 使其在文件列表中隐藏。大多数工具（VS Code、PyCharm、pytest）都能自动识别 `.venv`。请立即将其加入 `.gitignore`：

![Virtual environment isolation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-venv-isolation.png)


```bash
echo ".venv/" >> .gitignore
```

**切勿提交虚拟环境。** 它包含平台特定的二进制文件，不具备可移植性。

### venv vs virtualenv vs conda

| 特性 | venv | virtualenv | conda |
|---------|------|------------|-------|
| 是否内置标准库 | 是（Python 3.3+） | 否（需 `pip install`） | 否（需独立安装） |
| 速度 | 中等 | 快 | 慢 |
| Python 版本管理 | 否 | 否 | 是 |
| 非 Python 依赖支持 | 否 | 否 | 是（如 C 库等） |
| 跨平台支持 | 是 | 是 | 是 |
| 环境体积 | 小 | 小 | 大（200MB+） |
| 最适用场景 | 通用 Python 项目 | 遗留项目 / 追求速度 | 数据科学（含 CUDA、MKL 等难编译的 C 依赖） |

**建议：** 绝大多数项目使用 `venv`；仅当你需要预编译的科学计算库（如 CUDA、MKL）且源码编译极其困难时，才选用 `conda`。

## pip：Python 包安装器


![Python virtual environment isolated bubbles each with differ](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/01-python-virtual-environment-isolated-bubbles-each-with-differ.jpg)

在虚拟环境激活状态下，`pip` 将把包安装到该隔离环境中。

### 基础命令

```bash
# 安装包
(.venv) $ pip install requests

# 安装指定版本
(.venv) $ pip install requests==2.31.0

# 安装满足约束的版本
(.venv) $ pip install "requests>=2.28,<3.0"

# 升级包
(.venv) $ pip install --upgrade requests

# 卸载包
(.venv) $ pip uninstall requests

# 查看包信息
(.venv) $ pip show requests
Name: requests
Version: 2.31.0
Location: /home/user/projects/my-api/.venv/lib/python3.11/site-packages
Requires: certifi, charset-normalizer, idna, urllib3
```

### requirements.txt

传统依赖记录方式：

```bash
# 从当前环境生成
(.venv) $ pip freeze > requirements.txt

# 从文件安装
(.venv) $ pip install -r requirements.txt
```

典型的 `pip freeze` 输出 `requirements.txt`：

```
certifi==2023.11.17
charset-normalizer==3.3.2
idna==3.6
requests==2.31.0
urllib3==2.1.0
```

`pip freeze` 的问题在于：它会导出所有已安装包（包括传递依赖）。你无法区分哪些是你直接声明的依赖，哪些是间接引入的。卸载某个包后，其依赖项仍会残留。

## pip-tools：实现可复现安装

pip-tools 通过分离「你想要什么」和「实际安装什么」，解决了 `pip freeze` 的缺陷。

![Toolchain comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-toolchain-comparison.png)


### 安装

```bash
(.venv) $ pip install pip-tools
```

### 工作流

创建 `requirements.in`，仅列出你的直接依赖：

```
# requirements.in
requests>=2.28
flask>=3.0
pydantic>=2.0
```

编译生成完全锁定的 `requirements.txt`：

```bash
(.venv) $ pip-compile requirements.in
#
# This file is autogenerated by pip-compile with Python 3.11
# by the following command:
#
#    pip-compile requirements.in
#
blinker==1.7.0
    # via flask
certifi==2023.11.17
    # via requests
charset-normalizer==3.3.2
    # via requests
click==8.1.7
    # via flask
flask==3.0.0
    # via -r requirements.in
idna==3.6
    # via requests
itsdangerous==2.1.2
    # via flask
jinja2==3.1.2
    # via flask
markupsafe==2.1.3
    # via
    #   jinja2
    #   werkzeug
pydantic==2.5.3
    # via -r requirements.in
pydantic-core==2.14.6
    # via pydantic
requests==2.31.0
    # via -r requirements.in
urllib3==2.1.0
    # via requests
werkzeug==3.0.1
    # via flask
```

每行都注明了依赖来源。使用以下命令精确同步环境：

```bash
(.venv) $ pip-sync requirements.txt
```

`pip-sync` 会**移除**不在 `requirements.txt` 中的包，而 `pip install -r` 仅做新增。

### 升级依赖

```bash
# 升级全部包
(.venv) $ pip-compile --upgrade requirements.in

# 升级单个包
(.venv) $ pip-compile --upgrade-package requests requirements.in
```

### 开发依赖

为开发工具单独建一个文件：

```
# requirements-dev.in
-c requirements.txt
pytest>=7.0
pytest-cov
mypy
ruff
```

其中 `-c requirements.txt` 表示开发依赖必须与生产依赖兼容。

```bash
(.venv) $ pip-compile requirements-dev.in
(.venv) $ pip-sync requirements.txt requirements-dev.txt
```

## Poetry vs pip-tools vs PDM

| 特性 | pip-tools | Poetry | PDM |
|---------|-----------|--------|-----|
| 配置文件 | requirements.in | pyproject.toml | pyproject.toml |
| 锁定文件 | requirements.txt | poetry.lock | pdm.lock |
| 虚拟环境管理 | 否（需手动） | 是（自动） | 是（自动） |
| 构建与发布 | 否 | 是 | 是 |
| 速度 | 快 | 中等 | 快 |
| 是否符合 PEP 621 | N/A | 否（自定义格式） | 是 |
| 学习曲线 | 低 | 中等 | 中等 |
| 稳定性 | 极其稳定 | 稳定 | 稳定 |
| 依赖解析器 | pip 内置解析器 | 自研 | 自研 |

**pip-tools** 是最轻量的选择：紧贴 pip 生态，仅增加必要功能。**Poetry** 在需要构建+发布的库或应用中广受欢迎。**PDM** 则最严格遵循 PEP 标准。

## pyproject.toml：现代标准配置文件

`pyproject.toml` 取代了 `setup.py`、`setup.cfg`、`MANIFEST.in` 以及绝大多数工具专属配置文件。其规范由 [PEP 518](https://peps.python.org/pep-0518/) 和 [PEP 621](https://peps.python.org/pep-0621/) 定义。

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my-api"
version = "0.1.0"
description = "A sample API project"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
dependencies = [
    "requests>=2.28",
    "flask>=3.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "mypy",
    "ruff",
]

[project.scripts]
my-api = "my_api.cli:main"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

所有工具配置集中于一个文件，不再需要散落的 `.flake8`、`mypy.ini`、`pytest.ini`。

## 实际工作流：从克隆到运行

以下是零起点的正确初始化流程：

```bash
# 1. 克隆仓库
$ git clone git@github.com:team/my-api.git
$ cd my-api

# 2. pyenv 自动读取 .python-version
$ python --version
Python 3.11.7

# 3. 创建并激活虚拟环境
$ python -m venv .venv
$ source .venv/bin/activate

# 4. 安装依赖
(.venv) $ pip install -r requirements.txt
# 或使用 pip-tools：
(.venv) $ pip-sync requirements.txt requirements-dev.txt

# 5. 验证测试
(.venv) $ python -m pytest
========================= test session starts ==========================
collected 42 items
...
========================= 42 passed in 3.21s ===========================

# 6. 运行应用
(.venv) $ python -m my_api
 * Running on http://127.0.0.1:5000
```

用 Makefile 自动化步骤 2–4：

```makefile
.PHONY: setup test run

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip-sync requirements.txt requirements-dev.txt

test:
	.venv/bin/python -m pytest

run:
	.venv/bin/python -m my_api
```

## 常见陷阱与解决方案

| 陷阱 | 表象 | 解决方案 |
|---------|---------|----------|
| 使用系统 pip | `pip install X` 修改了系统包 | 总是先激活虚拟环境 |
| 忘记版本锁定 | `pip install requests` 安装最新版 | 使用 `pip-compile` 锁定版本 |
| 提交了 `.venv` | 仓库体积暴增，含平台特定二进制 | 将 `.venv/` 加入 `.gitignore` |
| pyenv 未加入 PATH | `pyenv: command not found` | 在 shell 配置中添加 init 行 |
| Linux 缺少构建依赖 | `pyenv install` 期间报 `ModuleNotFoundError` | 安装 `build-essential`、`libssl-dev` 等 |
| 全局/本地 Python 冲突 | 激活了错误版本 | 执行 `pyenv version` 查看哪个配置生效 |
| pip 缓存过期 | 升级后仍安装旧版本 | 使用 `pip install --no-cache-dir` 或 `pip cache purge` |
| 混用 conda 与 pip | 环境状态损坏 | 二选一：仅用 conda，或仅用 venv+pip |
| `requirements.txt` 无哈希校验 | 存在供应链攻击风险 | `pip-compile --generate-hashes` |
| 忘记更新锁文件 | 新开发者获得不同版本 | CI 应校验锁文件是否最新 |

## 初始化后的目录结构

```
my-api/
  .python-version        # pyenv 读取（已提交）
  .venv/                  # 虚拟环境（已加入 .gitignore）
  pyproject.toml          # 项目元数据与工具配置
  requirements.in         # 直接依赖声明
  requirements.txt        # 锁定的完整依赖树
  requirements-dev.in     # 开发专用直接依赖
  requirements-dev.txt    # 锁定的开发依赖树
  .gitignore              # 包含 .venv/
  src/
    my_api/
      __init__.py
      ...
  tests/
    ...
```

## 下一步

当你的开发环境已完全锁定，下一个问题是：如何组织代码？单个 `main.py` 适合脚本，但超过 200 行的项目就需要合理结构。下一篇文章，我们将从零构建一个规范的 Python 项目，涵盖包目录布局、导入机制、入口点（entry points）以及 CLI 工具开发。