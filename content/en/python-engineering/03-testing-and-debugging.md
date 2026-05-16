---
title: "Python Engineering (3): Testing — pytest, Fixtures, and the Confidence Loop"
date: 2022-04-14 09:00:00
tags:
  - Python
  - Testing
  - pytest
categories: Python Engineering
series: python-engineering
lang: en
description: "Build confidence in your code with pytest fixtures, parametrize, mocking, and coverage. Learn debugging techniques that save hours."
disableNunjucks: true
series_order: 3
series_total: 8
translationKey: "python-engineering-3"
---

You change one line and three unrelated features break. You refactor a function and spend two hours manually clicking through the app to check if everything still works. You deploy on Friday and get paged at midnight. All of these are symptoms of the same disease: no tests.

Tests are not bureaucracy. They are the fastest way to know that your code does what you think it does. A good test suite runs in seconds and catches the bugs that would take hours to find manually.


---

## Why Test

Writing tests costs time up front. Not writing tests costs more time later. Here is the math:

![Testing pyramid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-test-pyramid.png)


| Activity | Without Tests | With Tests |
|----------|--------------|------------|
| Initial development | Faster (no tests to write) | Slower (tests add 20-40% time) |
| Refactoring | Terrifying (did I break something?) | Confident (tests catch regressions) |
| Debugging | Read the whole codebase | Run tests, see exactly what broke |
| Onboarding new developer | "Ask Sarah, she knows how it works" | Tests document expected behavior |
| Deploying to production | Manual QA, hope for the best | Automated gate, deploy with confidence |
| Bug reported by customer | Reproduce manually, fix, manually verify | Write test that reproduces bug, fix, test verifies |

The real payoff comes on the second change to any piece of code. The first time, you know the code works because you just wrote it. Every subsequent change, you do not know unless you test.

## pytest Basics

pytest is the de facto testing framework for Python. It uses plain `assert` statements and automatic test discovery.

### Installation

```bash
(.venv) $ pip install pytest
```

### Your First Test

```python
# tests/test_math.py

def test_addition():
    assert 1 + 1 == 2


def test_string_upper():
    assert "hello".upper() == "HELLO"


def test_list_append():
    items = [1, 2, 3]
    items.append(4)
    assert items == [1, 2, 3, 4]
    assert len(items) == 4
```

Run it:

```bash
(.venv) $ pytest
========================= test session starts ==========================
platform linux -- Python 3.11.7, pytest-7.4.4
rootdir: /home/user/project
collected 3 items

tests/test_math.py ...                                            [100%]

========================= 3 passed in 0.02s ============================
```

### Test Discovery Rules

pytest finds tests automatically by looking for:

- Files named `test_*.py` or `*_test.py`
- Functions named `test_*` inside those files
- Classes named `Test*` with methods named `test_*`

### Useful Command-Line Flags

```bash
# Verbose output (show each test name)
(.venv) $ pytest -v

# Stop on first failure
(.venv) $ pytest -x

# Run tests matching a keyword expression
(.venv) $ pytest -k "download"

# Run tests in a specific file
(.venv) $ pytest tests/test_core.py

# Run a specific test function
(.venv) $ pytest tests/test_core.py::test_download_file

# Show print statements (not captured)
(.venv) $ pytest -s

# Show local variables in tracebacks
(.venv) $ pytest -l

# Re-run only failed tests from last run
(.venv) $ pytest --lf

# Run failed tests first, then the rest
(.venv) $ pytest --ff
```

### Testing Exceptions

Use `pytest.raises` to verify that code raises expected exceptions:

```python
import pytest

from my_tool.utils import validate_url


def test_invalid_url_raises():
    with pytest.raises(ValueError, match="Invalid URL"):
        validate_url("not-a-url")


def test_empty_url_raises():
    with pytest.raises(ValueError):
        validate_url("")
```

The `match` parameter checks the exception message with a regex.

### Testing Approximate Values

```python
def test_float_division():
    # Float comparison with tolerance
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 3.14 == pytest.approx(3.14159, abs=0.01)
```

## Fixtures: Reusable Test Setup


![Testing pyramid ancient egyptian pyramid with unit integrati](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/03-testing-pyramid-ancient-egyptian-pyramid-with-unit-integrati.jpg)

Fixtures replace the `setUp`/`tearDown` pattern from unittest. They provide test dependencies through function arguments.

![Fixture scopes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-fixture-scope.png)


### Basic Fixture

```python
import pytest

from my_tool.core import FileDownloader


@pytest.fixture
def downloader():
    """Create a configured downloader instance."""
    return FileDownloader(timeout=5, retries=1)


def test_download_sets_timeout(downloader):
    assert downloader.timeout == 5


def test_download_sets_retries(downloader):
    assert downloader.retries == 1
```

pytest sees `downloader` in the test function's parameter list, finds the fixture with that name, calls it, and passes the result. Each test gets a fresh instance.

### Fixture with Teardown

Use `yield` to run cleanup code after the test:

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory, clean up after test."""
    path = Path(tempfile.mkdtemp())
    yield path
    # Cleanup runs after the test, even if the test fails
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def test_file_creation(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    assert test_file.read_text() == "hello"
```

### Fixture Scope

By default, fixtures run once per test function. Change the scope for expensive setup:

```python
@pytest.fixture(scope="session")
def database_connection():
    """Create a database connection, shared across all tests."""
    conn = create_connection("test.db")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def sample_data():
    """Load sample data, shared within one test file."""
    return load_test_data("fixtures/sample.json")


@pytest.fixture(scope="class")
def api_client():
    """Create API client, shared within one test class."""
    return APIClient(base_url="http://localhost:8000")


@pytest.fixture  # scope="function" is the default
def clean_state():
    """Fresh state for each test."""
    return {}
```

| Scope | Created | Destroyed | Use When |
|-------|---------|-----------|----------|
| `function` | Before each test | After each test | Default, for most fixtures |
| `class` | Before first test in class | After last test in class | Shared state within a test class |
| `module` | Before first test in file | After last test in file | Expensive file-level setup |
| `session` | Before first test overall | After last test overall | Database connections, server startup |

### conftest.py: Shared Fixtures

Fixtures in `conftest.py` are available to all tests in the same directory and subdirectories without importing:

```text
tests/
  conftest.py          # fixtures available to all tests
  test_core.py
  test_cli.py
  integration/
    conftest.py        # additional fixtures for integration tests
    test_api.py
```

```python
# tests/conftest.py

import pytest


@pytest.fixture
def sample_url():
    return "https://httpbin.org/get"


@pytest.fixture
def sample_headers():
    return {"User-Agent": "test-agent/1.0"}
```

Any test file in `tests/` can use `sample_url` and `sample_headers` without importing them.

### Built-in Fixtures

pytest provides several useful built-in fixtures:

```python
def test_capture_output(capsys):
    """capsys captures stdout and stderr."""
    print("hello world")
    captured = capsys.readouterr()
    assert captured.out == "hello world\n"


def test_temp_path(tmp_path):
    """tmp_path provides a unique temporary directory."""
    file = tmp_path / "data.txt"
    file.write_text("content")
    assert file.read_text() == "content"


def test_monkeypatch_env(monkeypatch):
    """monkeypatch modifies environment for the test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    import os
    assert os.environ["API_KEY"] == "test-key-123"
```

## Parametrize: Testing Multiple Cases


![Pytest fixture mechanism factory producing test data assembl](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/03-pytest-fixture-mechanism-factory-producing-test-data-assembl.jpg)

Instead of writing separate test functions for each case, use `@pytest.mark.parametrize`:

![Parametrized tests](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-parametrize.png)


```python
import pytest

from my_tool.utils import format_size


@pytest.mark.parametrize("size_bytes,expected", [
    (0, "0.0 B"),
    (1023, "1023.0 B"),
    (1024, "1.0 KB"),
    (1536, "1.5 KB"),
    (1048576, "1.0 MB"),
    (1073741824, "1.0 GB"),
    (5368709120, "5.0 GB"),
])
def test_format_size(size_bytes, expected):
    assert format_size(size_bytes) == expected
```

Output:

```bash
(.venv) $ pytest -v tests/test_utils.py::test_format_size
tests/test_utils.py::test_format_size[0-0.0 B] PASSED
tests/test_utils.py::test_format_size[1023-1023.0 B] PASSED
tests/test_utils.py::test_format_size[1024-1.0 KB] PASSED
tests/test_utils.py::test_format_size[1536-1.5 KB] PASSED
tests/test_utils.py::test_format_size[1048576-1.0 MB] PASSED
tests/test_utils.py::test_format_size[1073741824-1.0 GB] PASSED
tests/test_utils.py::test_format_size[5368709120-5.0 GB] PASSED
```

Seven test cases, one function. Each case appears as a separate test in the output with its own pass/fail status.

### Parametrize with IDs

Make output clearer with explicit IDs:

```python
@pytest.mark.parametrize("url,expected_filename", [
    ("https://example.com/data.csv", "data.csv"),
    ("https://example.com/path/to/file.txt", "file.txt"),
    ("https://example.com/", "download"),
    ("https://example.com", "download"),
], ids=["simple", "nested-path", "trailing-slash", "no-path"])
def test_filename_from_url(url, expected_filename):
    assert filename_from_url(url) == expected_filename
```

## Mocking: Isolating Your Code

When testing a function that calls an external service, you do not want your tests to make real HTTP requests. Mocking replaces parts of your code with controlled fakes.

![Mock architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-mock-architecture.png)


### unittest.mock.patch

```python
from unittest.mock import patch, MagicMock

from my_tool.core import download_file


@patch("my_tool.core.requests.get")
def test_download_file_success(mock_get, tmp_path):
    # Configure the mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "11"}
    mock_response.iter_content.return_value = [b"hello world"]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    output = tmp_path / "test.txt"
    result = download_file("https://example.com/test.txt", str(output), quiet=True)

    assert result == output
    assert output.read_bytes() == b"hello world"
    mock_get.assert_called_once()


@patch("my_tool.core.requests.get")
def test_download_file_http_error(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_get.return_value = mock_response

    with pytest.raises(Exception, match="404"):
        download_file("https://example.com/missing.txt", quiet=True)
```

### When to Mock

| Mock | Do Not Mock |
|------|-------------|
| HTTP requests | Pure functions |
| Database queries | Data transformations |
| File system (sometimes) | String manipulation |
| Time-dependent code | Math |
| External service APIs | Your own internal logic |

Over-mocking is a common mistake. If you mock everything, your tests verify the mocks, not your code. Mock at the boundary (network, disk, clock) and test real logic with real code.

### Monkeypatch as a Simpler Alternative

For simple cases, `monkeypatch` is cleaner than `patch`:

```python
def test_download_with_env_config(monkeypatch):
    monkeypatch.setenv("DOWNLOAD_TIMEOUT", "60")
    monkeypatch.setattr("my_tool.config.DEFAULT_TIMEOUT", 60)
    # test code that reads from config
```

## Coverage: Measuring What Is Tested

Install pytest-cov:

![Coverage report](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-coverage-report.png)


```bash
(.venv) $ pip install pytest-cov
```

Run with coverage:

```bash
(.venv) $ pytest --cov=my_tool --cov-report=term-missing
========================= test session starts ==========================
...

---------- coverage: platform linux, python 3.11.7 ----------
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
src/my_tool/__init__.py         3      0   100%
src/my_tool/cli.py             28      5    82%   41-45
src/my_tool/config.py           4      0   100%
src/my_tool/core.py            31      3    90%   52-54
src/my_tool/utils.py           15      0   100%
---------------------------------------------------------
TOTAL                          81      8    90%

========================= 12 passed in 0.45s ===========================
```

The `Missing` column tells you exactly which lines lack coverage. `41-45` means lines 41 through 45 are not exercised by any test.

### Coverage Configuration

```toml
# pyproject.toml

[tool.coverage.run]
source = ["my_tool"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "if TYPE_CHECKING:",
]
```

`branch = true` enables branch coverage, which checks that both sides of `if/else` are tested.

### Coverage Targets

| Coverage Level | What It Means | Appropriate For |
|---------------|---------------|-----------------|
| < 50% | Barely tested | Legacy code you are starting to test |
| 50-70% | Basic coverage | Internal tools, scripts |
| 70-85% | Good coverage | Most applications |
| 85-95% | Strong coverage | Libraries, critical services |
| 95-100% | Near-complete | Payment processing, security code |

Do not chase 100% coverage. Some code (error handlers for impossible states, `__repr__` methods) is not worth testing. Focus on business logic and edge cases.

## Test Organization

### Unit Tests

Test individual functions in isolation. Fast, focused, many of them.

```python
# tests/test_utils.py

def test_format_size_zero():
    assert format_size(0) == "0.0 B"

def test_format_size_kilobytes():
    assert format_size(1024) == "1.0 KB"
```

### Integration Tests

Test how components work together. Slower, fewer of them.

```python
# tests/integration/test_download.py

def test_download_real_file(tmp_path):
    """Integration test: actually downloads from httpbin."""
    result = download_file(
        "https://httpbin.org/bytes/1024",
        str(tmp_path / "data.bin"),
        quiet=True,
    )
    assert result.stat().st_size == 1024
```

Mark integration tests so you can skip them in fast runs:

```python
import pytest

pytestmark = pytest.mark.integration

def test_download_real_file(tmp_path):
    ...
```

```bash
# Run only unit tests (skip integration)
(.venv) $ pytest -m "not integration"

# Run only integration tests
(.venv) $ pytest -m integration
```

Register custom marks in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### End-to-End Tests

Test the full application from the user's perspective:

```python
# tests/e2e/test_cli.py

from click.testing import CliRunner
from my_tool.cli import main


def test_cli_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["https://httpbin.org/bytes/100", "-o", "test.bin", "-q"])
    assert result.exit_code == 0
    assert (tmp_path / "test.bin").exists()


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Download files from URLs" in result.output
```

## Debugging: When Tests Fail

### breakpoint() and pdb

Python 3.7 added the `breakpoint()` built-in:

```python
def download_file(url, output=None, quiet=False, timeout=30):
    response = requests.get(url, stream=True, timeout=timeout)
    breakpoint()  # Execution stops here, drops into debugger
    response.raise_for_status()
    ...
```

Run the test normally (without `-s` pytest will capture stdin, but `breakpoint()` forces it open):

```bash
(.venv) $ pytest -s tests/test_core.py::test_download_file_success
> /home/user/src/my_tool/core.py(15)download_file()
-> response.raise_for_status()
(Pdb)
```

### Essential pdb Commands

| Command | Short | What It Does |
|---------|-------|-------------|
| `next` | `n` | Execute current line, step over function calls |
| `step` | `s` | Execute current line, step into function calls |
| `continue` | `c` | Continue until next breakpoint |
| `print expr` | `p expr` | Print the value of an expression |
| `pp expr` | | Pretty-print the value |
| `list` | `l` | Show source code around current line |
| `where` | `w` | Show the call stack |
| `up` | `u` | Move up one frame in the call stack |
| `down` | `d` | Move down one frame in the call stack |
| `quit` | `q` | Exit the debugger |
| `help` | `h` | Show help |

### ipdb: A Better Debugger

`ipdb` adds syntax highlighting and tab completion:

```bash
(.venv) $ pip install ipdb
```

```python
import ipdb; ipdb.set_trace()  # or set PYTHONBREAKPOINT=ipdb.set_trace
```

Or set it globally:

```bash
$ export PYTHONBREAKPOINT=ipdb.set_trace
$ pytest -s tests/test_core.py  # breakpoint() now uses ipdb
```

### Post-Mortem Debugging

Debug a test that just failed, at the point of failure:

```bash
(.venv) $ pytest --pdb tests/test_core.py
# Drops into pdb at the point of failure
```

This is extremely powerful. You do not need to add `breakpoint()` anywhere. Just run with `--pdb` and pytest will open the debugger at the exact line where the assertion failed, with all local variables intact.

## Property-Based Testing with Hypothesis

Traditional tests check specific examples. Property-based testing checks *invariants* that should hold for any valid input. [Hypothesis](https://hypothesis.readthedocs.io/) generates hundreds of random inputs to find edge cases you would never think to test manually.

### Installation

```bash
(.venv) $ pip install hypothesis
```

### Basic Properties

```python
from hypothesis import given, assume, settings
from hypothesis import strategies as st

@given(st.lists(st.integers()))
def test_sort_is_idempotent(xs):
    """Sorting twice gives the same result as sorting once."""
    assert sorted(sorted(xs)) == sorted(xs)

@given(st.lists(st.integers()))
def test_sort_preserves_length(xs):
    """Sorting does not add or remove elements."""
    assert len(sorted(xs)) == len(xs)

@given(st.lists(st.integers(), min_size=1))
def test_sort_result_is_ordered(xs):
    """Every element is <= the next."""
    result = sorted(xs)
    for a, b in zip(result, result[1:]):
        assert a <= b
```

Instead of testing `sorted([3, 1, 2]) == [1, 2, 3]`, we test properties that *any* correct sort must satisfy. Hypothesis will try empty lists, single elements, duplicates, negative numbers, huge values, and more.

### Custom Strategies

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
    email: str

# Build a strategy for generating Users
users = st.builds(
    User,
    name=st.text(min_size=1, max_size=50),
    age=st.integers(min_value=0, max_value=150),
    email=st.emails(),
)

@given(users)
def test_user_serialization_roundtrip(user):
    """Serialize → deserialize should return the original object."""
    data = serialize(user)
    restored = deserialize(data)
    assert restored == user
```

### Stateful Testing

For systems with state (databases, APIs), Hypothesis can generate sequences of operations:

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition

class SetMachine(RuleBasedStateMachine):
    """Test that our CustomSet behaves like Python's built-in set."""

    def __init__(self):
        super().__init__()
        self.model = set()         # reference
        self.actual = CustomSet()  # system under test

    @rule(value=st.integers())
    def add(self, value):
        self.model.add(value)
        self.actual.add(value)
        assert value in self.actual

    @rule(value=st.integers())
    def remove(self, value):
        if value in self.model:
            self.model.remove(value)
            self.actual.remove(value)
        assert value not in self.actual

    @rule()
    def check_length(self):
        assert len(self.actual) == len(self.model)

TestSet = SetMachine.TestCase
```

### When Hypothesis Finds a Bug

Hypothesis shrinks the failing input to the smallest example that still fails:

```
Falsifying example: test_parse_date(
    s='0000-00-00',  # Shrunk from '9812-23-71'
)
```

It then stores this example in `.hypothesis/` so future runs re-test it. Commit `.hypothesis/examples/` to your repo.

### Settings and Profiles

```python
from hypothesis import settings, Phase, Verbosity

# Slow but thorough (CI)
@settings(max_examples=1000, deadline=None)
@given(st.text())
def test_thorough(s):
    ...

# Fast iteration (dev)
@settings(max_examples=50)
@given(st.text())
def test_quick(s):
    ...

# Configure profiles globally in conftest.py
settings.register_profile("ci", max_examples=1000)
settings.register_profile("dev", max_examples=50)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
```

## Snapshot Testing

Snapshot testing captures the output of a function and compares it against a saved "golden" file. Useful for:
- CLI output formatting
- Serialization formats (JSON, YAML responses)
- Template rendering
- Code generation output

### Using pytest-snapshot (syrupy)

```bash
(.venv) $ pip install syrupy
```

```python
def test_user_json_format(snapshot):
    user = User(name="Alice", age=30, email="alice@example.com")
    result = user.to_json(indent=2)
    assert result == snapshot

def test_error_message(snapshot):
    with pytest.raises(ValidationError) as exc_info:
        validate_config({"port": "not_a_number"})
    assert str(exc_info.value) == snapshot

def test_cli_help_output(snapshot):
    result = runner.invoke(app, ["--help"])
    assert result.output == snapshot
```

First run creates `__snapshots__/` files. Subsequent runs compare against them.

```bash
# Update snapshots when output intentionally changes
(.venv) $ pytest --snapshot-update
```

### When to Use Snapshots vs Assertions

| Approach | Best for | Drawback |
|----------|----------|----------|
| Explicit assertions | Logic, calculations, state transitions | Verbose for complex outputs |
| Snapshots | Formatting, rendering, serialization | Easy to over-approve changes |

Rule of thumb: use snapshots when you care about the *exact shape* of output and would need >5 assertions to verify it manually.

## Detecting Slow Tests

As test suites grow, individual slow tests accumulate and kill feedback loops. Catch them early.

### pytest-duration

```bash
# Show 10 slowest tests
(.venv) $ pytest --durations=10

============================= slowest 10 durations =============================
1.23s call     tests/test_api.py::test_database_migration
0.89s call     tests/test_export.py::test_large_csv_export
0.67s setup    tests/test_integration.py::test_full_pipeline
...
```

### Enforcing Time Limits

```python
import pytest

@pytest.mark.timeout(5)
def test_api_response():
    """Must complete within 5 seconds."""
    response = client.get("/api/heavy-endpoint")
    assert response.status_code == 200
```

Or globally in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
timeout = 30  # seconds — any single test exceeding this fails
```

### Parallelizing Tests

```bash
(.venv) $ pip install pytest-xdist

# Run tests across all CPU cores
(.venv) $ pytest -n auto

# Run tests across 4 workers
(.venv) $ pytest -n 4
```

Tests must be independent (no shared state, no fixed ports) to run in parallel. Use fixtures with unique temp directories and random ports.

### CI Optimization Pattern

```yaml
# .github/workflows/test.yml
jobs:
  test:
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - run: pytest --splits 4 --group ${{ matrix.shard }}
```

Split tests across multiple CI runners for faster feedback on large suites.

## Real Example: Testing a Data Processor

Here is a function that processes log entries:

```python
# src/my_tool/processor.py

from datetime import datetime


def parse_log_entry(line: str) -> dict:
    """Parse a log line into structured data.

    Expected format: YYYY-MM-DD HH:MM:SS LEVEL message

    Args:
        line: Raw log line string.

    Returns:
        Dict with keys: timestamp, level, message.

    Raises:
        ValueError: If the line format is invalid.
    """
    parts = line.strip().split(" ", 3)
    if len(parts) < 4:
        raise ValueError(f"Invalid log format: {line!r}")

    date_str, time_str, level, message = parts
    timestamp = datetime.fromisoformat(f"{date_str} {time_str}")
    level = level.upper()

    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Unknown log level: {level!r}")

    return {
        "timestamp": timestamp,
        "level": level,
        "message": message,
    }
```

And thorough tests:

```python
# tests/test_processor.py

from datetime import datetime

import pytest

from my_tool.processor import parse_log_entry


class TestParseLogEntry:
    """Tests for parse_log_entry function."""

    def test_valid_info_line(self):
        result = parse_log_entry("2024-01-15 10:30:00 INFO Server started")
        assert result == {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
            "level": "INFO",
            "message": "Server started",
        }

    def test_valid_error_line(self):
        result = parse_log_entry("2024-01-15 10:30:00 ERROR Connection refused")
        assert result["level"] == "ERROR"
        assert result["message"] == "Connection refused"

    def test_message_with_spaces(self):
        result = parse_log_entry(
            "2024-01-15 10:30:00 WARNING Disk usage at 90% on /dev/sda1"
        )
        assert result["message"] == "Disk usage at 90% on /dev/sda1"

    def test_level_case_insensitive(self):
        result = parse_log_entry("2024-01-15 10:30:00 info lowercase level")
        assert result["level"] == "INFO"

    @pytest.mark.parametrize("level", [
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    ])
    def test_all_valid_levels(self, level):
        line = f"2024-01-15 10:30:00 {level} test message"
        result = parse_log_entry(line)
        assert result["level"] == level

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid log format"):
            parse_log_entry("")

    def test_incomplete_line_raises(self):
        with pytest.raises(ValueError, match="Invalid log format"):
            parse_log_entry("2024-01-15 10:30:00")

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Unknown log level"):
            parse_log_entry("2024-01-15 10:30:00 TRACE message")

    def test_invalid_timestamp_raises(self):
        with pytest.raises(ValueError):
            parse_log_entry("not-a-date 10:30:00 INFO message")

    def test_strips_whitespace(self):
        result = parse_log_entry("  2024-01-15 10:30:00 INFO padded  \n")
        assert result["level"] == "INFO"
```

Run with verbose output and coverage:

```bash
(.venv) $ pytest -v --cov=my_tool tests/test_processor.py
tests/test_processor.py::TestParseLogEntry::test_valid_info_line PASSED
tests/test_processor.py::TestParseLogEntry::test_valid_error_line PASSED
tests/test_processor.py::TestParseLogEntry::test_message_with_spaces PASSED
tests/test_processor.py::TestParseLogEntry::test_level_case_insensitive PASSED
tests/test_processor.py::TestParseLogEntry::test_all_valid_levels[DEBUG] PASSED
tests/test_processor.py::TestParseLogEntry::test_all_valid_levels[INFO] PASSED
tests/test_processor.py::TestParseLogEntry::test_all_valid_levels[WARNING] PASSED
tests/test_processor.py::TestParseLogEntry::test_all_valid_levels[ERROR] PASSED
tests/test_processor.py::TestParseLogEntry::test_all_valid_levels[CRITICAL] PASSED
tests/test_processor.py::TestParseLogEntry::test_empty_string_raises PASSED
tests/test_processor.py::TestParseLogEntry::test_incomplete_line_raises PASSED
tests/test_processor.py::TestParseLogEntry::test_invalid_level_raises PASSED
tests/test_processor.py::TestParseLogEntry::test_invalid_timestamp_raises PASSED
tests/test_processor.py::TestParseLogEntry::test_strips_whitespace PASSED
14 passed
```

## What's Next

Tests tell you that your code works. Type hints and linting tell you that your code is correct before you even run it. In the next article, we will add type annotations to our codebase, set up mypy for static type checking, and configure ruff and black so that style arguments become automated, not debated.
