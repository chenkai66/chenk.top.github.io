---
title: "Python Engineering (4): Type Hints, Linting, and Code Quality"
date: 2022-04-17 09:00:00
tags:
  - Python
  - Type Hints
  - Linting
  - Code Quality
categories: Python Engineering
series: python-engineering
lang: en
description: "Add type safety with mypy, enforce style with ruff and black, and automate checks with pre-commit hooks. Make code reviews about logic, not formatting."
disableNunjucks: true
series_order: 4
translationKey: "python-engineering-4"
---

Code reviews should be about logic and design, not about whether someone used single quotes or double quotes. Formatting debates are a waste of engineering time. The solution is to let machines handle style and let humans focus on correctness.

This article covers three layers of automated code quality: type hints catch logical errors before runtime, linters catch style violations and common bugs, and pre-commit hooks enforce everything automatically on every commit.

## Type Hints: Basic Annotations

Python is dynamically typed, but since 3.5 it supports optional type annotations. They do not affect runtime behavior. They are metadata that tools like mypy can check.

![Type system hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/04-type-system.png)


### Primitive Types

```python
# Basic types
name: str = "Alice"
age: int = 30
height: float = 1.75
active: bool = True

# Function annotations
def greet(name: str, excited: bool = False) -> str:
    if excited:
        return f"Hello, {name}!"
    return f"Hello, {name}."
```

### Collection Types

Since Python 3.9, you can use built-in types directly for generics:

```python
# Python 3.9+
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95, "Bob": 87}
coordinates: tuple[float, float] = (40.7128, -74.0060)
unique_ids: set[int] = {1, 2, 3}

# Nested types
matrix: list[list[int]] = [[1, 2], [3, 4]]
config: dict[str, list[str]] = {"hosts": ["a.com", "b.com"]}
```

For Python 3.7-3.8, import from `typing`:

```python
from typing import Dict, List, Set, Tuple

names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 95}
```

Or use `from __future__ import annotations` at the top of the file to enable the 3.9+ syntax in older versions.

### Optional and Union

```python
from typing import Optional, Union

# Optional means "this type or None"
def find_user(user_id: int) -> Optional[dict]:
    """Returns user dict or None if not found."""
    ...

# Union means "one of these types"
def process(value: Union[str, int]) -> str:
    return str(value)

# Python 3.10+ shorthand
def find_user(user_id: int) -> dict | None:
    ...

def process(value: str | int) -> str:
    return str(value)
```

### Any, Callable, and Iterator

```python
from typing import Any, Callable, Iterator

# Any disables type checking for this value
def log(message: Any) -> None:
    print(message)

# Callable[[arg_types], return_type]
def retry(func: Callable[[str], bool], attempts: int = 3) -> bool:
    for _ in range(attempts):
        if func("test"):
            return True
    return False

# Iterator and Generator
def count_up(start: int, end: int) -> Iterator[int]:
    current = start
    while current < end:
        yield current
        current += 1
```

### Type Aliases

```python
# Simple alias
UserId = int
UserRecord = dict[str, Any]

# Complex types benefit from aliases
Headers = dict[str, str]
Callback = Callable[[str, int], bool]
Matrix = list[list[float]]

def fetch(url: str, headers: Headers, on_progress: Callback) -> bytes:
    ...
```

## Generic Types: TypeVar and Protocol

### TypeVar

When you need a function that works with any type but preserves the relationship:

```python
from typing import TypeVar, Sequence

T = TypeVar("T")

def first(items: Sequence[T]) -> T:
    """Return the first item. Type of return matches type of items."""
    return items[0]

# mypy knows these types:
x: int = first([1, 2, 3])        # T = int
y: str = first(["a", "b", "c"])  # T = str
```

### Bounded TypeVar

```python
from typing import TypeVar

# T must be a subclass of int or float
Numeric = TypeVar("Numeric", int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

add(1, 2)       # OK: int
add(1.0, 2.0)   # OK: float
add("a", "b")   # Error: str is not int or float
```

### Protocol (Structural Subtyping)

Protocol defines an interface by structure, not inheritance. If it has the right methods, it matches:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Readable(Protocol):
    def read(self, size: int = -1) -> bytes:
        ...


def process_stream(source: Readable) -> bytes:
    """Accepts anything with a .read() method."""
    return source.read()


# This works with any object that has .read(), without inheriting Readable
import io
data = process_stream(io.BytesIO(b"hello"))  # OK
```

### TypedDict

For dictionaries with a known structure:

```python
from typing import TypedDict


class UserRecord(TypedDict):
    name: str
    age: int
    email: str


class UserRecordPartial(TypedDict, total=False):
    name: str
    age: int
    email: str  # all fields are optional


def create_user(data: UserRecord) -> int:
    # mypy knows data["name"] is str, data["age"] is int
    ...

# OK
create_user({"name": "Alice", "age": 30, "email": "a@b.com"})

# Error: missing "email"
create_user({"name": "Alice", "age": 30})
```

## Type Checking with mypy

mypy reads your type annotations and reports errors without running the code.

![mypy type checking flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/04-mypy-flow.png)


### Installation and Basic Usage

```bash
(.venv) $ pip install mypy
(.venv) $ mypy src/
```

### Strictness Levels

```bash
# Default: only checks annotated code
(.venv) $ mypy src/

# Strict: requires annotations everywhere, catches more errors
(.venv) $ mypy --strict src/

# Check a single file
(.venv) $ mypy src/my_tool/core.py
```

### Common mypy Errors and Fixes

```python
# Error: Incompatible return value type (got "Optional[str]", expected "str")
def get_name(user_id: int) -> str:
    result = lookup(user_id)  # returns Optional[str]
    return result  # Error!

# Fix: handle the None case
def get_name(user_id: int) -> str:
    result = lookup(user_id)
    if result is None:
        raise ValueError(f"User {user_id} not found")
    return result  # Now mypy knows result is str


# Error: Item "None" of "Optional[dict]" has no attribute "get"
def get_email(user: dict | None) -> str:
    return user.get("email", "")  # Error: user might be None

# Fix: narrow the type
def get_email(user: dict | None) -> str:
    if user is None:
        return ""
    return user.get("email", "")


# Error: Need type annotation for "items"
items = []  # mypy doesn't know the element type

# Fix: annotate
items: list[str] = []


# Error: Argument 1 to "open" has incompatible type "Optional[str]"
def read_file(path: str | None) -> str:
    with open(path) as f:  # Error: path might be None
        return f.read()

# Fix: check first or change the type
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()
```

### mypy Configuration

```toml
# pyproject.toml

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

# Per-module overrides
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true
```

### Gradual Adoption

You do not need to annotate everything at once. Start with:
1. New code: always add type hints
2. Public API functions: annotate return types and parameters
3. Core modules: add full annotations
4. Tests: annotate fixtures and helpers, but test functions can be loose

Use `# type: ignore[error-code]` to suppress specific errors temporarily:

```python
result = some_untyped_function()  # type: ignore[no-untyped-call]
```

## Linting with ruff


![Code quality pipeline raw code passing through linter format](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/04-code-quality-pipeline-raw-code-passing-through-linter-format.jpg)

ruff is a Python linter written in Rust. It is 10-100x faster than flake8 and replaces flake8, isort, pyflakes, pycodestyle, pydocstyle, and many flake8 plugins in a single tool.

![Linting pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/04-linting-pipeline.png)


### Installation and Usage

```bash
(.venv) $ pip install ruff

# Lint
(.venv) $ ruff check src/
src/my_tool/core.py:3:1: F401 [*] `os` imported but unused
src/my_tool/utils.py:15:80: E501 Line too long (92 > 88)
Found 2 errors.
[*] 1 fixable with `--fix`.

# Auto-fix
(.venv) $ ruff check --fix src/
Found 2 errors (1 fixed, 1 remaining).

# Format (replaces black)
(.venv) $ ruff format src/
2 files reformatted.
```

### ruff Configuration

```toml

![Ruff vs other linters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/04-ruff-vs-others.png)

# pyproject.toml

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # pyflakes
    "I",     # isort
    "N",     # pep8-naming
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "SIM",   # flake8-simplify
    "C4",    # flake8-comprehensions
    "DTZ",   # flake8-datetimez
    "T20",   # flake8-print (no print in prod code)
    "RET",   # flake8-return
    "PTH",   # flake8-use-pathlib
    "ERA",   # eradicate (commented-out code)
    "RUF",   # ruff-specific rules
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["T20", "S101"]  # allow print and assert in tests

[tool.ruff.lint.isort]
known-first-party = ["my_tool"]
```

### What ruff Catches

```python
# F401: imported but unused
import os  # <-- ruff removes this

# F841: local variable assigned but never used
def process():
    result = compute()  # <-- ruff flags this
    return None

# B006: mutable default argument
def append_to(item, target=[]):  # <-- Bug! Shared mutable default
    target.append(item)
    return target

# SIM108: use ternary instead of if-else
if condition:
    x = 1
else:
    x = 2
# ruff suggests: x = 1 if condition else 2

# UP035: use PEP 604 union syntax
from typing import Optional  # <-- ruff suggests: str | None
def f(x: Optional[str]): ...

# C4: use dict/list comprehension
dict([(k, v) for k, v in items])  # <-- ruff suggests: {k: v for k, v in items}
```

## Formatting with black

black is an opinionated code formatter. It makes style decisions for you so you never argue about formatting again.

```bash
(.venv) $ pip install black

# Check without modifying
(.venv) $ black --check src/
would reformat src/my_tool/core.py
Oh no!
1 file would be reformatted.

# Show what would change
(.venv) $ black --diff src/my_tool/core.py

# Format in place
(.venv) $ black src/
reformatted src/my_tool/core.py
All done!
1 file reformatted.
```

Note: `ruff format` is now a drop-in replacement for black, so you can skip installing black separately and just use `ruff format`.

### black Configuration

```toml
# pyproject.toml

[tool.black]
line-length = 88
target-version = ["py311"]
```

black has very few options by design. The point is to stop debating. Use the defaults.

## Pre-commit Hooks

Pre-commit runs checks automatically before every `git commit`. If any check fails, the commit is blocked until you fix the issue.

![Pre-commit hooks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/04-precommit.png)


### Installation

```bash
(.venv) $ pip install pre-commit
```

### Configuration

Create `.pre-commit-config.yaml` in the project root:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - pydantic

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: debug-statements
```

### Install the Hooks

```bash
(.venv) $ pre-commit install
pre-commit installed at .git/hooks/pre-commit

# Run against all files (first time)
(.venv) $ pre-commit run --all-files
ruff.....................................................................Passed
ruff-format..............................................................Passed
mypy.....................................................................Passed
trailing whitespace......................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
check for added large files..............................................Passed
debug statements.........................................................Passed
```

Now every `git commit` runs these checks. If ruff or black reformats a file, the commit fails and you need to `git add` the reformatted file and commit again.

### Skipping Hooks (Emergency Only)

```bash
# Skip all hooks
$ git commit --no-verify -m "hotfix: emergency patch"

# Skip specific hooks
$ SKIP=mypy git commit -m "WIP: types incomplete"
```

Use `--no-verify` sparingly. If you skip hooks regularly, your CI will catch the errors anyway, and you will waste more time fixing them after the fact.

## Comparison: ruff vs flake8 vs pylint

| Feature | ruff | flake8 | pylint |
|---------|------|--------|--------|
| Language | Rust | Python | Python |
| Speed (10k files) | ~0.1s | ~30s | ~120s |
| Auto-fix | Yes | No (plugins) | No |
| Import sorting | Built-in (isort) | Plugin | Built-in |
| Formatting | Built-in (black-compatible) | No | No |
| Type checking | No (use mypy) | No | Basic |
| Plugin ecosystem | Growing | Huge | Built-in |
| Configuration | pyproject.toml | .flake8 or setup.cfg | .pylintrc |
| Rules count | 800+ | ~200 (core) | 400+ |
| Active development | Very active | Maintenance | Active |

**Recommendation:** Use ruff. It is faster, fixes issues automatically, and consolidates multiple tools. Add mypy separately for type checking.

## Complete pyproject.toml Configuration


![Type hints as blueprint annotations on python code architect](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/04-type-hints-as-blueprint-annotations-on-python-code-architect.jpg)

Here is a production-ready tool configuration section:

```toml
# pyproject.toml

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "C4", "DTZ", "RET", "PTH", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["T20", "S101"]

[tool.ruff.lint.isort]
known-first-party = ["my_tool"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=my_tool --cov-report=term-missing"

[tool.coverage.run]
source = ["my_tool"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
```

## CI Integration: GitHub Actions

Run all checks in CI so nothing slips through:

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Check formatting
        run: ruff format --check src/ tests/

      - name: Type check with mypy
        run: mypy src/

      - name: Run tests
        run: pytest --cov=my_tool --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
```

## What's Next

Your code is now type-safe, consistently formatted, and automatically checked on every commit. But Python programs do more than compute; they read files, parse configs, and serialize data in a dozen formats. In the next article, we will master I/O, tackle encoding headaches, and compare every serialization format from JSON to Parquet.
