---
title: "Python Engineering (1): Environment Setup — pyenv, venv, and Dependency Hell"
date: 2022-04-10 09:00:00
tags:
  - Python
  - Environment
  - Toolchain
categories:
  - Python Engineering
series: python-engineering
lang: en
description: "Master Python environment management with pyenv, virtual environments, and modern dependency tools. Escape dependency hell for good."
disableNunjucks: true
series_order: 1
translationKey: "python-engineering-1"
---

Every Python developer has lived through this moment: you run a script on your colleague's machine and it crashes because they have Python 3.8 while you wrote it on 3.11. Or worse, you `pip install` something globally and break a completely unrelated project. Python's environment story is powerful once you understand it, but the default experience is a minefield.

This article walks through the entire toolchain from scratch. By the end, you will have a reproducible, isolated, version-pinned setup that works the same way on every machine.

## The Python Version Problem

Most operating systems ship with a system Python. On macOS, it used to be Python 2.7 (removed in Monterey). On Ubuntu 22.04, it is Python 3.10. This system Python is used by OS-level tools. Installing packages into it or upgrading it can break your operating system.

![Version management stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-version-stack.png)


The core problems:

| Problem | Example |
|---------|---------|
| System Python is outdated | Ubuntu 20.04 ships 3.8, you need 3.11 features |
| Multiple projects need different versions | Project A needs 3.9, Project B needs 3.12 |
| Global pip installs cause conflicts | Package X needs `requests>=2.28`, Package Y pins `requests==2.25` |
| Reproducibility fails | "Works on my machine" because versions differ |
| OS tools depend on system Python | `apt` on Ubuntu uses system Python internally |

The solution is a three-layer stack:

1. **pyenv** manages Python versions (install 3.9, 3.10, 3.11 side by side)
2. **venv** isolates per-project dependencies
3. **pip-tools** or **Poetry** pins exact versions for reproducibility

## pyenv: Multiple Python Versions Without Pain

pyenv intercepts the `python` command and redirects it to whichever version you have configured. It does this by inserting shims into your `$PATH`.

![pyenv shim mechanism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-pyenv-shim.png)


### Installation

On macOS:

```bash
brew install pyenv
```

On Linux:

```bash
curl https://pyenv.run | bash
```

After installation, add these lines to your shell config (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell:

```bash
source ~/.zshrc  # or ~/.bashrc
```

Verify the installation:

```bash
$ pyenv --version
pyenv 2.3.36
```

### Installing Python Versions

List available versions:

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

Install a specific version:

```bash
$ pyenv install 3.11.7
Downloading Python-3.11.7.tar.xz...
Installing Python-3.11.7...
Installed Python-3.11.7 to /home/user/.pyenv/versions/3.11.7
```

On macOS, if the build fails, you likely need:

```bash
brew install openssl readline sqlite3 xz zlib tcl-tk
```

On Ubuntu/Debian:

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

### Setting the Active Version

pyenv has three levels of version selection, from most specific to least:

```bash
# Global default (lowest priority)
$ pyenv global 3.11.7

# Per-directory (creates .python-version file)
$ cd ~/projects/my-api
$ pyenv local 3.11.7
$ cat .python-version
3.11.7

# Current shell session only (highest priority)
$ pyenv shell 3.12.2
```

The resolution order is: `shell` > `local` (.python-version) > `global` (~/.pyenv/version).

Check which version is active and why:

```bash
$ pyenv version
3.11.7 (set by /home/user/projects/my-api/.python-version)

$ pyenv versions
  system
  3.9.18
* 3.11.7 (set by /home/user/projects/my-api/.python-version)
  3.12.2
```

**Commit `.python-version` to your repo.** This ensures every developer uses the same Python version. It costs nothing and prevents version-mismatch bugs.

## Virtual Environments: Isolating Dependencies


![Dependency hell tangled wires vs clean resolved dependencies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/01-dependency-hell-tangled-wires-vs-clean-resolved-dependencies.jpg)

Even with the right Python version, you still need dependency isolation. Without it, `pip install` puts packages into a shared location, and two projects needing different versions of the same package will conflict.

![Dependency resolution flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-dep-resolution.png)


### Creating a Virtual Environment

```bash
$ cd ~/projects/my-api
$ python -m venv .venv
```

This creates a `.venv` directory containing:

```
.venv/
  bin/          # python, pip, activate scripts
  include/      # C headers for building extensions
  lib/          # installed packages go here
  pyvirst.cfg   # points back to the base Python
```

### Activation

```bash
# macOS / Linux
$ source .venv/bin/activate
(.venv) $

# Windows (PowerShell)
> .venv\Scripts\Activate.ps1

# Windows (cmd)
> .venv\Scripts\activate.bat
```

When activated, `python` and `pip` point to the venv copies:

```bash
(.venv) $ which python
/home/user/projects/my-api/.venv/bin/python

(.venv) $ which pip
/home/user/projects/my-api/.venv/bin/pip
```

Deactivate when done:

```bash
(.venv) $ deactivate
$
```

### Why .venv?

The `.` prefix hides it in file listings. Most tools (VS Code, PyCharm, pytest) auto-detect `.venv`. Add it to `.gitignore` immediately:

![Virtual environment isolation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-venv-isolation.png)


```bash
echo ".venv/" >> .gitignore
```

Never commit the virtual environment. It contains platform-specific binaries and is not portable.

### venv vs virtualenv vs conda

| Feature | venv | virtualenv | conda |
|---------|------|------------|-------|
| Included in stdlib | Yes (3.3+) | No (pip install) | No (separate installer) |
| Speed | Moderate | Fast | Slow |
| Python version management | No | No | Yes |
| Non-Python dependencies | No | No | Yes (C libs, etc.) |
| Cross-platform | Yes | Yes | Yes |
| Environment size | Small | Small | Large (200MB+) |
| Best for | General Python | Legacy/speed | Data science with C deps |

**Recommendation:** Use `venv` for most projects. Use conda only if you need compiled scientific libraries (CUDA, MKL) that are painful to build from source.

## pip: The Package Installer


![Python virtual environment isolated bubbles each with differ](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/01-python-virtual-environment-isolated-bubbles-each-with-differ.jpg)

With your venv activated, `pip` installs packages into the isolated environment.

### Basic Commands

```bash
# Install a package
(.venv) $ pip install requests

# Install a specific version
(.venv) $ pip install requests==2.31.0

# Install with version constraints
(.venv) $ pip install "requests>=2.28,<3.0"

# Upgrade a package
(.venv) $ pip install --upgrade requests

# Uninstall
(.venv) $ pip uninstall requests

# Show package info
(.venv) $ pip show requests
Name: requests
Version: 2.31.0
Location: /home/user/projects/my-api/.venv/lib/python3.11/site-packages
Requires: certifi, charset-normalizer, idna, urllib3
```

### requirements.txt

The traditional way to record dependencies:

```bash
# Generate from current environment
(.venv) $ pip freeze > requirements.txt

# Install from file
(.venv) $ pip install -r requirements.txt
```

A typical `requirements.txt` from `pip freeze`:

```
certifi==2023.11.17
charset-normalizer==3.3.2
idna==3.6
requests==2.31.0
urllib3==2.1.0
```

The problem with `pip freeze`: it dumps every installed package, including transitive dependencies. You cannot tell which packages you actually need versus which are dependencies of dependencies. Removing a package leaves its dependencies behind.

## pip-tools: Reproducible Installs

pip-tools solves the `pip freeze` problem by separating what you want from what gets installed.

![Toolchain comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/01-toolchain-comparison.png)


### Installation

```bash
(.venv) $ pip install pip-tools
```

### Workflow

Create `requirements.in` with your direct dependencies:

```
# requirements.in
requests>=2.28
flask>=3.0
pydantic>=2.0
```

Compile it to a fully pinned `requirements.txt`:

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

Every line shows where each dependency comes from. Sync your environment exactly:

```bash
(.venv) $ pip-sync requirements.txt
```

`pip-sync` removes packages not in the file, unlike `pip install -r` which only adds.

### Upgrading

```bash
# Upgrade all packages
(.venv) $ pip-compile --upgrade requirements.in

# Upgrade one package
(.venv) $ pip-compile --upgrade-package requests requirements.in
```

### Dev Dependencies

Create a separate file for development tools:

```
# requirements-dev.in
-c requirements.txt
pytest>=7.0
pytest-cov
mypy
ruff
```

The `-c requirements.txt` constrains dev deps to be compatible with production deps.

```bash
(.venv) $ pip-compile requirements-dev.in
(.venv) $ pip-sync requirements.txt requirements-dev.txt
```

## Poetry vs pip-tools vs PDM

| Feature | pip-tools | Poetry | PDM |
|---------|-----------|--------|-----|
| Config file | requirements.in | pyproject.toml | pyproject.toml |
| Lock file | requirements.txt | poetry.lock | pdm.lock |
| Venv management | No (manual) | Yes (auto) | Yes (auto) |
| Build & publish | No | Yes | Yes |
| Speed | Fast | Moderate | Fast |
| PEP 621 compliant | N/A | No (custom format) | Yes |
| Learning curve | Low | Medium | Medium |
| Stability | Very stable | Stable | Stable |
| Resolver | pip's resolver | Custom | Custom |

**pip-tools** is the simplest choice: it stays close to pip and adds only what is needed. **Poetry** is popular for libraries and applications that need build+publish. **PDM** follows PEP standards most closely.

## pyproject.toml: The Modern Standard

`pyproject.toml` replaces `setup.py`, `setup.cfg`, `MANIFEST.in`, and most tool-specific config files. It is defined by PEP 518 and PEP 621.

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

All tool configuration in one file. No more scattered `.flake8`, `mypy.ini`, `pytest.ini`.

## Real Workflow: Clone to Running

Here is what a correct setup looks like from zero:

```bash
# 1. Clone the repo
$ git clone git@github.com:team/my-api.git
$ cd my-api

# 2. pyenv reads .python-version automatically
$ python --version
Python 3.11.7

# 3. Create and activate venv
$ python -m venv .venv
$ source .venv/bin/activate

# 4. Install dependencies
(.venv) $ pip install -r requirements.txt
# Or with pip-tools:
(.venv) $ pip-sync requirements.txt requirements-dev.txt

# 5. Verify
(.venv) $ python -m pytest
========================= test session starts ==========================
collected 42 items
...
========================= 42 passed in 3.21s ===========================

# 6. Run the application
(.venv) $ python -m my_api
 * Running on http://127.0.0.1:5000
```

Automate steps 2-4 with a Makefile:

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

## Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Using system pip | `pip install X` modifies system packages | Always activate venv first |
| Forgetting to pin versions | `pip install requests` installs latest | Use `pip-compile` to pin |
| Committing .venv | Huge repo, platform-specific binaries | Add `.venv/` to `.gitignore` |
| pyenv not in PATH | `pyenv: command not found` | Add init lines to shell config |
| Build deps missing on Linux | `ModuleNotFoundError` during `pyenv install` | Install build-essential, libssl-dev, etc. |
| Conflicting global/local Python | Wrong version active | Check `pyenv version` to see which config wins |
| pip cache stale | Old package version installed despite upgrade | `pip install --no-cache-dir` or `pip cache purge` |
| Mixing conda and pip | Broken environment state | Pick one: conda OR venv+pip |
| requirements.txt has no hashes | Supply chain attack risk | `pip-compile --generate-hashes` |
| Forgot to update lock file | New dev gets different versions | CI should verify lock file is up to date |

## Directory Structure After Setup

```
my-api/
  .python-version        # pyenv reads this (committed)
  .venv/                  # virtual environment (gitignored)
  pyproject.toml          # project metadata and tool config
  requirements.in         # direct dependencies
  requirements.txt        # pinned full dependency tree
  requirements-dev.in     # dev-only direct dependencies
  requirements-dev.txt    # pinned dev dependency tree
  .gitignore              # includes .venv/
  src/
    my_api/
      __init__.py
      ...
  tests/
    ...
```

## What's Next

With your environment locked down, the next question is how to organize your code. A single `main.py` works for scripts, but anything beyond 200 lines needs structure. In the next article, we will build a proper Python project from scratch, covering package layouts, imports, entry points, and CLI tools.
