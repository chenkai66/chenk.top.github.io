---
title: "Python 工程实践（三）：测试——pytest、Fixture 与信心循环"
date: 2022-04-14 09:00:00
tags:
  - Python
  - Testing
  - pytest
categories: Python Engineering
series: python-engineering
lang: zh
description: "借助 pytest fixture、parametrize、mocking 和覆盖率分析，建立对代码的信心。掌握可节省数小时的调试技巧。"
disableNunjucks: true
series_order: 3
translationKey: "python-engineering-3"
---

你只改了一行代码，却导致三个完全无关的功能崩溃；重构一个函数后，不得不花两小时手动检查整个应用以确认一切是否依然正常；周五上线部署后，午夜又被报警电话叫醒——所有这些现象，都指向同一个病根：**没有测试**。

测试不是官僚主义流程，而是你**最快验证代码行为是否符合预期**的手段。一套优秀的测试套件能在几秒内完成运行，捕获那些手动排查需要耗费数小时才能发现的缺陷。

## 为何要写测试

编写测试在前期会消耗时间，但不写测试，后期将付出更多时间。以下是对比：

![Testing pyramid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-test-pyramid.png)


| 活动 | 无测试 | 有测试 |
|----------|--------------|------------|
| 初始开发 | 更快（无需编写测试） | 更慢（测试增加 20–40% 时间） |
| 重构 | 令人恐惧（我是否破坏了什么？） | 充满信心（测试可捕获回归问题） |
| 调试 | 需通读整个代码库 | 运行测试，立即定位具体失败点 |
| 新成员上手 | “去问 Sarah，她知道怎么运作” | 测试即文档，明确描述预期行为 |
| 生产环境部署 | 手动 QA，听天由命 | 自动化守门员，自信部署 |
| 客户报告 Bug | 手动复现 → 修复 → 手动验证 | 编写复现 Bug 的测试 → 修复 → 测试自动验证 |

真正的价值体现在对任意一段代码进行**第二次修改时**：第一次你刚写完，自然知道它能工作；但此后每次变更，如果没有测试保障，就无法确信它仍然正确。

## pytest 基础

pytest 是 Python 事实上的标准测试框架。它使用原生 `assert` 语句，并支持自动发现测试用例。

### 安装

```bash
(.venv) $ pip install pytest
```

### 你的第一个测试

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

运行它：

```bash
(.venv) $ pytest
========================= test session starts ==========================
platform linux -- Python 3.11.7, pytest-7.4.4
rootdir: /home/user/project
collected 3 items

tests/test_math.py ...                                            [100%]

========================= 3 passed in 0.02s ============================
```

### 测试发现规则

pytest 通过以下方式自动查找测试：

- 文件名匹配 `test_*.py` 或 `*_test.py`
- 这些文件中，函数名以 `test_*` 开头
- 类名以 `Test*` 开头，且其方法名以 `test_*` 开头

### 实用命令行参数

```bash
# 详细输出（显示每个测试名称）
(.venv) $ pytest -v

# 首次失败即停止
(.venv) $ pytest -x

# 运行匹配关键词表达式的测试
(.venv) $ pytest -k "download"

# 运行指定文件中的测试
(.venv) $ pytest tests/test_core.py

# 运行指定测试函数
(.venv) $ pytest tests/test_core.py::test_download_file

# 显示 print 输出（不捕获 stdout/stderr）
(.venv) $ pytest -s

# 在 traceback 中显示局部变量
(.venv) $ pytest -l

# 仅重跑上一次运行中失败的测试
(.venv) $ pytest --lf

# 先运行失败的测试，再运行其余测试
(.venv) $ pytest --ff
```

### 测试异常

使用 `pytest.raises` 验证代码是否抛出预期异常：

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

`match` 参数使用正则表达式校验异常消息内容。

### 测试近似值

```python
def test_float_division():
    # 浮点数比较需容忍误差
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 3.14 == pytest.approx(3.14159, abs=0.01)
```

## Fixture：可复用的测试准备逻辑


![Testing pyramid ancient egyptian pyramid with unit integrati](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/03-testing-pyramid-ancient-egyptian-pyramid-with-unit-integrati.jpg)

Fixture 替代了 unittest 中的 `setUp`/`tearDown` 模式，通过函数参数向测试注入依赖。

![Fixture scopes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-fixture-scope.png)


### 基础 Fixture

```python
import pytest

from my_tool.core import FileDownloader


@pytest.fixture
def downloader():
    """创建一个已配置的下载器实例。"""
    return FileDownloader(timeout=5, retries=1)


def test_download_sets_timeout(downloader):
    assert downloader.timeout == 5


def test_download_sets_retries(downloader):
    assert downloader.retries == 1
```

pytest 发现测试函数参数列表中的 `downloader`，找到同名 fixture，调用它并将返回值传入测试函数。每个测试均获得一个全新实例。

### 含清理逻辑的 Fixture

使用 `yield` 在测试结束后执行清理代码（即使测试失败也会执行）：

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """创建临时目录，测试后自动清理。"""
    path = Path(tempfile.mkdtemp())
    yield path
    # 清理逻辑在测试结束后执行，无论成功或失败
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def test_file_creation(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    assert test_file.read_text() == "hello"
```

### Fixture 作用域（Scope）

默认情况下，fixture 在每个测试函数前运行一次。对开销较大的初始化操作，可调整作用域：

```python
@pytest.fixture(scope="session")
def database_connection():
    """创建数据库连接，被全部测试共享。"""
    conn = create_connection("test.db")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def sample_data():
    """加载样本数据，在单个测试文件内共享。"""
    return load_test_data("fixtures/sample.json")


@pytest.fixture(scope="class")
def api_client():
    """创建 API 客户端，在单个测试类内共享。"""
    return APIClient(base_url="http://localhost:8000")


@pytest.fixture  # scope="function" 是默认值
def clean_state():
    """为每个测试提供干净状态。"""
    return {}
```

| 作用域 | 创建时机 | 销毁时机 | 适用场景 |
|-------|---------|-----------|----------|
| `function` | 每个测试前 | 每个测试后 | 默认，适用于大多数 fixture |
| `class` | 类中首个测试前 | 类中最后一个测试后 | 测试类内共享状态 |
| `module` | 文件中首个测试前 | 文件中最后一个测试后 | 开销大的文件级初始化 |
| `session` | 全局首个测试前 | 全局最后一个测试后 | 数据库连接、服务启动等 |

### conftest.py：共享 Fixture

定义在 `conftest.py` 中的 fixture 对同目录及所有子目录下的测试自动可见，无需显式导入：

```
tests/
  conftest.py          # fixture 对所有测试可用
  test_core.py
  test_cli.py
  integration/
    conftest.py        # 为集成测试额外定义的 fixture
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

`tests/` 下任意测试文件均可直接使用 `sample_url` 和 `sample_headers`，无需导入。

### 内置 Fixture

pytest 提供若干实用内置 fixture：

```python
def test_capture_output(capsys):
    """capsys 捕获 stdout 和 stderr。"""
    print("hello world")
    captured = capsys.readouterr()
    assert captured.out == "hello world\n"


def test_temp_path(tmp_path):
    """tmp_path 提供唯一临时目录。"""
    file = tmp_path / "data.txt"
    file.write_text("content")
    assert file.read_text() == "content"


def test_monkeypatch_env(monkeypatch):
    """monkeypatch 为测试临时修改环境变量。"""
    monkeypatch.setenv("API_KEY", "test-key-123")
    import os
    assert os.environ["API_KEY"] == "test-key-123"
```

## Parametrize：批量测试多组用例


![Pytest fixture mechanism factory producing test data assembl](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/03-pytest-fixture-mechanism-factory-producing-test-data-assembl.jpg)

无需为每种输入单独编写测试函数，使用 `@pytest.mark.parametrize` 即可：

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

输出：

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

7 组用例， 1 个函数。每组用例在输出中作为独立测试项呈现，各自拥有独立的通过/失败状态。

### 使用 ID 命名 Parametrize 用例

为提升输出可读性，可以显式指定用例 ID。

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

## Mocking：隔离你的代码

当测试调用外部服务的函数时，你不希望测试真正发起 HTTP 请求。Mocking 将代码中部分组件替换为可控的模拟对象。

![Mock architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-mock-architecture.png)


### unittest.mock.patch

```python
from unittest.mock import patch, MagicMock

from my_tool.core import download_file


@patch("my_tool.core.requests.get")
def test_download_file_success(mock_get, tmp_path):
    # 配置 mock 响应
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

### 何时该 Mock？

| 应 Mock | 不应 Mock |
|------|-------------|
| HTTP 请求 | 纯函数 |
| 数据库查询 | 业务数据转换 |
| 文件系统（有时） | 字符串处理 |
| 时间相关逻辑 | 数学计算 |
| 外部服务 API | 自己的内部逻辑 |

过度 Mock 是一种常见误区：如果 Mock 了所有依赖，测试实际上只验证了 Mock 的行为，而非你自己的代码逻辑。**只在边界处 Mock（网络、磁盘、时钟），并在真实环境中测试核心逻辑。**

### Monkeypatch：更简洁的替代方案

对于简单场景，`monkeypatch` 比 `patch` 更轻量：

```python
def test_download_with_env_config(monkeypatch):
    monkeypatch.setenv("DOWNLOAD_TIMEOUT", "60")
    monkeypatch.setattr("my_tool.config.DEFAULT_TIMEOUT", 60)
    # 测试读取配置的代码
```

## Coverage：衡量测试覆盖度

安装 pytest-cov：

![Coverage report](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/03-coverage-report.png)


```bash
(.venv) $ pip install pytest-cov
```

运行并生成覆盖率报告：

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

`Missing` 列精确指出哪些行未被任何测试覆盖。`41-45` 表示第 41 至 45 行未被执行。

### Coverage 配置

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

`branch = true` 启用分支覆盖率，确保 `if/else` 的两个分支均被测试。

### 覆盖率目标建议

| 覆盖率水平 | 含义 | 适用场景 |
|---------------|---------------|-----------------|
| < 50% | 几乎未测试 | 正在开始补测的遗留代码 |
| 50–70% | 基础覆盖 | 内部工具、脚本 |
| 70–85% | 良好覆盖 | 大多数应用程序 |
| 85–95% | 强覆盖 | 库、关键服务 |
| 95–100% | 接近完整 | 支付处理、安全敏感代码 |

**切勿盲目追求 100% 覆盖率**。某些代码（如不可能触发的错误处理器、`__repr__` 方法）不值得测试。应聚焦于业务逻辑和边界条件。

## 测试组织策略

### 单元测试（Unit Tests）

隔离测试单个函数，快速、专注、数量众多。

```python
# tests/test_utils.py

def test_format_size_zero():
    assert format_size(0) == "0.0 B"

def test_format_size_kilobytes():
    assert format_size(1024) == "1.0 KB"
```

### 集成测试（Integration Tests）

测试多个组件如何协同工作，较慢、数量较少。

```python
# tests/integration/test_download.py

def test_download_real_file(tmp_path):
    """集成测试：实际从 httpbin 下载文件。"""
    result = download_file(
        "https://httpbin.org/bytes/1024",
        str(tmp_path / "data.bin"),
        quiet=True,
    )
    assert result.stat().st_size == 1024
```

为集成测试打标记，便于在快速运行中跳过：

```python
import pytest

pytestmark = pytest.mark.integration

def test_download_real_file(tmp_path):
    ...
```

```bash
# 仅运行单元测试（跳过集成测试）
(.venv) $ pytest -m "not integration"

# 仅运行集成测试
(.venv) $ pytest -m integration
```

在 `pyproject.toml` 中注册自定义标记：

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### 端到端测试（End-to-End Tests）

从用户视角测试整个应用：

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

## 调试：当测试失败时

### `breakpoint()` 与 `pdb`

Python 3.7 引入了内置 `breakpoint()` 函数：

```python
def download_file(url, output=None, quiet=False, timeout=30):
    response = requests.get(url, stream=True, timeout=timeout)
    breakpoint()  # 执行在此暂停，进入调试器
    response.raise_for_status()
    ...
```

正常运行测试（`-s` 可避免 pytest 捕获 stdin，但 `breakpoint()` 会强制打开）：

```bash
(.venv) $ pytest -s tests/test_core.py::test_download_file_success
> /home/user/src/my_tool/core.py(15)download_file()
-> response.raise_for_status()
(Pdb)
```

### 核心 pdb 命令

| 命令 | 缩写 | 功能说明 |
|---------|-------|-------------|
| `next` | `n` | 执行当前行，跳过函数调用（step over） |
| `step` | `s` | 执行当前行，进入函数调用（step into） |
| `continue` | `c` | 继续执行至下一个断点 |
| `print expr` | `p expr` | 打印表达式值 |
| `pp expr` | | 美观打印表达式值 |
| `list` | `l` | 显示当前行附近的源码 |
| `where` | `w` | 显示调用栈 |
| `up` | `u` | 向上调用栈移动一帧 |
| `down` | `d` | 向下调用栈移动一帧 |
| `quit` | `q` | 退出调试器 |
| `help` | `h` | 显示帮助 |

### ipdb：更强大的调试器

`ipdb` 提供语法高亮与 Tab 补全：

```bash
(.venv) $ pip install ipdb
```

```python
import ipdb; ipdb.set_trace()  # 或设置 PYTHONBREAKPOINT=ipdb.set_trace
```

或全局启用：

```bash
$ export PYTHONBREAKPOINT=ipdb.set_trace
$ pytest -s tests/test_core.py  # 此时 breakpoint() 将使用 ipdb
```

### 失败后调试（Post-Mortem Debugging）

对刚刚失败的测试，在失败点直接进入调试器：

```bash
(.venv) $ pytest --pdb tests/test_core.py
# 在失败点自动进入 pdb
```

这极为强大：你无需手动添加 `breakpoint()`。只需加上 `--pdb` 参数， pytest 就会在断言失败的**精确行号**处启动调试器，并保留所有局部变量。

## 真实案例：测试日志处理器

以下是一个解析日志条目的函数：

```python
# src/my_tool/processor.py

from datetime import datetime


def parse_log_entry(line: str) -> dict:
    """将日志行解析为结构化数据。

    预期格式：YYYY-MM-DD HH:MM:SS LEVEL message

    Args:
        line: 原始日志行字符串。

    Returns:
        包含键值：timestamp、level、message 的字典。

    Raises:
        ValueError: 若日志格式无效。
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

配套的全面测试：

```python
# tests/test_processor.py

from datetime import datetime

import pytest

from my_tool.processor import parse_log_entry


class TestParseLogEntry:
    """parse_log_entry 函数的测试集。"""

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

带详细输出与覆盖率运行：

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

## 下一步

测试告诉你**代码是否能工作**；类型提示与静态检查则在你运行代码之前就告诉你**代码是否正确**。在下一篇文章中，我们将为代码库添加类型注解，配置 mypy 进行静态类型检查，并设置 ruff 与 black，让风格争议自动化，而非靠争论解决。