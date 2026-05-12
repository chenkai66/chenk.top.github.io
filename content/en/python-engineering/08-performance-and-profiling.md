---
title: "Python Engineering (8): Performance — Profiling, Caching, and Knowing When to Stop"
date: 2022-04-27 09:00:00
tags:
  - Python
  - Performance
  - Profiling
  - Optimization
categories: Python Engineering
series: python-engineering
lang: en
description: "Profile Python code to find real bottlenecks, apply caching and vectorization where they matter, and avoid the trap of premature optimization."
disableNunjucks: true
series_order: 8
translationKey: "python-engineering-8"
---

Donald Knuth's famous quote gets half-remembered. The full version is: "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%." The missing second sentence is the important one. Performance work is not about making everything fast. It is about finding the 3% that matters and making that fast.

This article is about finding that 3%. You will learn to profile first, optimize second, and measure the impact of every change.

## Manual Benchmarking

### time.perf_counter()

The simplest profiling tool. Use it to time specific sections:

```python
import time

start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"Took {elapsed:.4f}s")
```

`perf_counter()` uses the highest-resolution timer available. `time.time()` is lower resolution on some platforms. Always use `perf_counter()` for benchmarking.

### A Reusable Timer

```python
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str = "Block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")


# Usage
with timer("Data loading"):
    data = load_large_file("data.csv")

with timer("Processing"):
    result = process(data)

with timer("Writing output"):
    write_results(result)
```

Output:

```
Data loading: 2.3451s
Processing: 0.0123s
Writing output: 0.8901s
```

Now you know where the time goes. Data loading is the bottleneck, not processing.

### timeit Module

For micro-benchmarks of small code snippets:

```bash
# Command line
$ python -m timeit -n 1000000 '"hello" + " " + "world"'
1000000 loops, best of 5: 0.0523 usec per loop

$ python -m timeit -n 1000000 'f"hello world"'
1000000 loops, best of 5: 0.0168 usec per loop

$ python -m timeit -n 1000000 '" ".join(["hello", "world"])'
1000000 loops, best of 5: 0.0891 usec per loop
```

```python
# In code
import timeit

# Time a function
time_taken = timeit.timeit(
    stmt='sorted(data)',
    setup='import random; data = random.sample(range(10000), 1000)',
    number=1000,
)
print(f"1000 iterations: {time_taken:.4f}s")
print(f"Per iteration: {time_taken/1000*1000:.4f}ms")
```

### Comparing Alternatives

```python
import timeit


def approach_a():
    """List comprehension."""
    return [x**2 for x in range(10000)]


def approach_b():
    """Map function."""
    return list(map(lambda x: x**2, range(10000)))


def approach_c():
    """For loop with append."""
    result = []
    for x in range(10000):
        result.append(x**2)
    return result


for func in [approach_a, approach_b, approach_c]:
    time_taken = timeit.timeit(func, number=1000)
    print(f"{func.__doc__.strip():25s}: {time_taken:.4f}s")
```

Output:

```
List comprehension.      : 1.2345s
Map function.            : 1.5678s
For loop with append.    : 1.8901s
```

List comprehension wins. But the difference is often small enough that readability should win over micro-optimization.

## cProfile: Function-Level Profiling

cProfile tracks every function call, how many times it was called, and how long it took.

![Profiling workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/08-profiling-flow.png)


### Basic Usage

```bash
$ python -m cProfile -s cumtime my_script.py
         12456 function calls (11789 primitive calls) in 3.456 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    3.456    3.456 my_script.py:1(<module>)
        1    0.002    0.002    2.345    2.345 my_script.py:15(load_data)
     1000    1.234    0.001    1.234    0.001 my_script.py:30(parse_row)
        1    0.890    0.890    0.890    0.890 my_script.py:50(write_output)
        1    0.012    0.012    0.210    0.210 my_script.py:45(transform)
     1000    0.198    0.000    0.198    0.000 my_script.py:35(validate)
      ...
```

### Understanding the Output

| Column | Meaning |
|--------|---------|
| `ncalls` | Number of calls to this function |
| `tottime` | Total time spent in this function (excluding subfunctions) |
| `percall` | `tottime / ncalls` |
| `cumtime` | Cumulative time (including subfunctions) |
| `percall` | `cumtime / ncalls` |

Sort options:

```bash
$ python -m cProfile -s tottime my_script.py    # By total time in function
$ python -m cProfile -s cumtime my_script.py    # By cumulative time (default, most useful)
$ python -m cProfile -s calls my_script.py      # By number of calls
```

### Profiling Specific Code

```python
import cProfile
import pstats

def main():
    data = load_data("input.csv")
    result = process(data)
    write_output(result)

# Profile and save results
cProfile.run("main()", "profile_output.prof")

# Analyze saved results
stats = pstats.Stats("profile_output.prof")
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions
```

### Visualizing Profiles with snakeviz

```bash
(.venv) $ pip install snakeviz

# Generate profile data
$ python -m cProfile -o profile.prof my_script.py

# Visualize in browser
$ snakeviz profile.prof
```

snakeviz opens an interactive visualization showing a sunburst chart of function call times. Functions that take the most time are immediately visible.

## line_profiler: Per-Line Timing


![Optimization journey slow python code transforming to fast o](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/08-optimization-journey-slow-python-code-transforming-to-fast-o.jpg)

cProfile tells you which functions are slow. `line_profiler` tells you which lines within those functions are slow.

```bash
(.venv) $ pip install line_profiler
```

Decorate the functions you want to profile:

```python
# my_script.py

@profile  # This decorator is recognized by kernprof
def process_data(records):
    results = []
    for record in records:
        # Validate
        if not record.get("id"):
            continue

        # Transform
        name = record["name"].strip().lower()
        score = float(record["score"])

        # Normalize
        normalized = score / 100.0

        # Store
        results.append({
            "id": record["id"],
            "name": name,
            "score": normalized,
        })
    return results
```

Run with `kernprof`:

```bash
$ kernprof -l -v my_script.py
Wrote profile results to my_script.py.lprof

Timer unit: 1e-06 s

Total time: 2.34567 s
File: my_script.py
Function: process_data at line 3

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     3                                           @profile
     4                                           def process_data(records):
     5         1          2.0      2.0      0.0      results = []
     6    100001      45123.0      0.5      1.9      for record in records:
     7    100000      89234.0      0.9      3.8          if not record.get("id"):
     8       100         45.0      0.5      0.0              continue
     9                                           
    10     99900     456789.0      4.6     19.5          name = record["name"].strip().lower()
    11     99900     123456.0      1.2      5.3          score = float(record["score"])
    12                                           
    13     99900      98765.0      1.0      4.2          normalized = score / 100.0
    14                                           
    15     99900    1532146.0     15.3     65.3          results.append({
    16                                                       "id": record["id"],
    17                                                       "name": name,
    18                                                       "score": normalized,
    19                                                   })
    20         1          1.0      1.0      0.0      return results
```

Line 15 (the `append` with dict creation) takes 65% of the time. That is the optimization target.

## memory_profiler: Tracking Memory Usage

```bash
(.venv) $ pip install memory_profiler
```

![Python object memory layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/08-memory-layout.png)


```python
from memory_profiler import profile


@profile
def load_large_data():
    # Each step shows memory change
    data = [i for i in range(1_000_000)]        # +8 MB
    strings = [str(i) for i in range(1_000_000)] # +60 MB
    combined = list(zip(data, strings))          # +40 MB
    del data                                     # -8 MB
    del strings                                  # -60 MB
    return combined
```

```bash
$ python -m memory_profiler my_script.py
Filename: my_script.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     4     45.2 MiB     45.2 MiB           1   @profile
     5                                         def load_large_data():
     6     53.4 MiB      8.2 MiB           1       data = [i for i in range(1_000_000)]
     7    113.6 MiB     60.2 MiB           1       strings = [str(i) for i in range(1_000_000)]
     8    153.8 MiB     40.2 MiB           1       combined = list(zip(data, strings))
     9    145.6 MiB     -8.2 MiB           1       del data
    10     85.4 MiB    -60.2 MiB           1       del strings
    11     85.4 MiB      0.0 MiB           1       return combined
```

## functools.lru_cache: Memoization

Memoization stores the results of expensive function calls and returns the cached result when the same inputs occur again.

![lru_cache benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/08-lru-cache.png)


```python
from functools import lru_cache
import time


def fibonacci_slow(n: int) -> int:
    """Naive recursive Fibonacci. Exponential time."""
    if n < 2:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)


@lru_cache(maxsize=128)
def fibonacci_fast(n: int) -> int:
    """Cached Fibonacci. Linear time."""
    if n < 2:
        return n
    return fibonacci_fast(n - 1) + fibonacci_fast(n - 2)


# fibonacci_slow(35) takes ~3 seconds
# fibonacci_fast(35) takes ~0.00001 seconds
```

### Python 3.9+ @cache

For unlimited cache:

```python
from functools import cache

@cache
def expensive_computation(x: int, y: int) -> float:
    """Result is cached forever (until process exits)."""
    time.sleep(2)
    return x ** y / (x + y)

# First call: 2 seconds
result1 = expensive_computation(10, 20)

# Second call with same args: instant
result2 = expensive_computation(10, 20)
```

### Cache Statistics

```python
@lru_cache(maxsize=256)
def fetch_user(user_id: int) -> dict:
    return database.query(f"SELECT * FROM users WHERE id = {user_id}")

# After some usage:
print(fetch_user.cache_info())
# CacheInfo(hits=847, misses=52, maxsize=256, currsize=52)

# Clear the cache
fetch_user.cache_clear()
```

### When to Use Caching

| Use Cache | Do Not Cache |
|-----------|-------------|
| Pure functions (same input -> same output) | Functions with side effects |
| Expensive computations (API calls, DB queries) | Functions that return mutable objects (lists, dicts) |
| Frequently repeated calls with same args | Functions where args are not hashable |
| Read-heavy, write-rarely data | Real-time data that changes frequently |

**Warning:** `lru_cache` stores results in memory. For functions that return large objects or are called with many different arguments, the cache can consume significant memory. Set `maxsize` to limit it.

### Caching Mutable Returns

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_default_config() -> dict:
    return {"timeout": 30, "retries": 3}

# Danger: callers can mutate the cached dict!
config = get_default_config()
config["timeout"] = 60  # This modifies the cached object!

# Next call returns the mutated version:
config2 = get_default_config()
print(config2["timeout"])  # 60, not 30!

# Fix: return a copy or use frozen types
import copy

def get_default_config_safe() -> dict:
    return copy.deepcopy(_get_default_config_cached())

@lru_cache(maxsize=32)
def _get_default_config_cached() -> dict:
    return {"timeout": 30, "retries": 3}
```

## NumPy Vectorization

Python loops are slow because each iteration involves type checking, reference counting, and bytecode interpretation. NumPy pushes the loop into optimized C code.

![NumPy vectorization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/08-vectorization.png)


```python
import numpy as np
import time


def python_distance(points_a, points_b):
    """Pure Python Euclidean distance calculation."""
    distances = []
    for a, b in zip(points_a, points_b):
        dist = sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5
        distances.append(dist)
    return distances


def numpy_distance(points_a, points_b):
    """NumPy vectorized Euclidean distance calculation."""
    a = np.array(points_a)
    b = np.array(points_b)
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


# Generate test data
n = 100_000
points_a = [[i, i+1, i+2] for i in range(n)]
points_b = [[i+3, i+4, i+5] for i in range(n)]

# Benchmark
start = time.perf_counter()
result_py = python_distance(points_a, points_b)
py_time = time.perf_counter() - start

start = time.perf_counter()
result_np = numpy_distance(points_a, points_b)
np_time = time.perf_counter() - start

print(f"Python: {py_time:.4f}s")
print(f"NumPy:  {np_time:.4f}s")
print(f"Speedup: {py_time/np_time:.1f}x")
```

Output:

```
Python: 3.4521s
NumPy:  0.0234s
Speedup: 147.5x
```

### Common Vectorization Patterns

```python
import numpy as np

data = np.random.randn(1_000_000)

# Instead of a loop to filter:
# Bad
filtered = [x for x in data if x > 0]

# Good (100x faster)
filtered = data[data > 0]


# Instead of a loop to transform:
# Bad
result = [x ** 2 + 2 * x + 1 for x in data]

# Good
result = data ** 2 + 2 * data + 1


# Instead of a loop to aggregate:
# Bad
total = sum(x for x in data if x > 0)

# Good
total = data[data > 0].sum()
```

## Generators and Lazy Evaluation


![Python profiling xray scanning code for performance bottlene](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/08-python-profiling-xray-scanning-code-for-performance-bottlene.jpg)

Generators produce values one at a time instead of creating the entire collection in memory.

```python
# This creates a list of 10 million items in memory (~80MB)
squares_list = [x**2 for x in range(10_000_000)]

# This creates nothing until iterated (~0MB)
squares_gen = (x**2 for x in range(10_000_000))

# Both work the same way in a for loop:
for sq in squares_gen:
    if sq > 1000:
        break
```

### Generator Functions with yield

```python
def read_large_file(path: str):
    """Read a file line by line without loading it all into memory."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def filter_records(lines):
    """Filter records lazily."""
    for line in lines:
        if line.startswith("ERROR"):
            yield line


def parse_records(lines):
    """Parse records lazily."""
    for line in lines:
        parts = line.split("\t")
        yield {"level": parts[0], "message": parts[1]}


# Pipeline: each stage processes one record at a time
# Memory usage is constant regardless of file size
lines = read_large_file("huge_log.txt")       # Lazy
errors = filter_records(lines)                 # Lazy
records = parse_records(errors)                # Lazy

for record in records:  # Only here does processing actually happen
    print(record["message"])
```

### When to Use Generators

| Use Generator | Use List |
|---------------|----------|
| Data is larger than memory | Data fits comfortably in memory |
| You only iterate once | You need random access (indexing) |
| You need the first N items | You need `len()` |
| Pipeline processing | You need to iterate multiple times |
| Reading from files/network | Small datasets (< 10K items) |

## `__slots__` for Memory-Efficient Classes

By default, Python objects store attributes in a `__dict__` dictionary. `__slots__` replaces this with a fixed-size array.

```python
import sys


class PointDict:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class PointSlots:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


p_dict = PointDict(1.0, 2.0, 3.0)
p_slots = PointSlots(1.0, 2.0, 3.0)

print(f"Dict object:  {sys.getsizeof(p_dict)} bytes + {sys.getsizeof(p_dict.__dict__)} dict")
print(f"Slots object: {sys.getsizeof(p_slots)} bytes (no dict)")
```

Output:

```
Dict object:  48 bytes + 104 dict
Slots object: 56 bytes (no dict)
```

For one million objects:

```python
dict_objects = [PointDict(i, i, i) for i in range(1_000_000)]
# Memory: ~152 MB

slots_objects = [PointSlots(i, i, i) for i in range(1_000_000)]
# Memory: ~56 MB
```

Roughly 3x memory savings. Use `__slots__` when you create millions of instances of the same class (data points, graph nodes, ORM rows).

**Trade-off:** `__slots__` objects cannot have arbitrary attributes added dynamically, and they do not support multiple inheritance with other `__slots__` classes easily.

## Cython: When Python Is Not Fast Enough

Cython compiles Python-like code to C. It can speed up CPU-bound code by 10-100x.

```bash
(.venv) $ pip install cython
```

```python
# distance.pyx (Cython source file)

def euclidean_distance_cy(double[:] a, double[:] b):
    cdef int n = a.shape[0]
    cdef double total = 0.0
    cdef int i

    for i in range(n):
        total += (a[i] - b[i]) ** 2

    return total ** 0.5
```

Compile with a `setup.py`:

```python
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("distance.pyx"))
```

```bash
$ python setup.py build_ext --inplace
```

Cython is a big topic. For most applications, the optimization techniques earlier in this article (vectorization, caching, generators) are sufficient. Reach for Cython when you have verified with profiling that a specific hot loop is the bottleneck, and NumPy cannot help because the computation is not easily vectorizable.

## Performance Optimization Checklist

Before optimizing, ask these questions in order:

![Optimization checklist](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/08-optimization-checklist.png)


| Step | Question | Action |
|------|----------|--------|
| 1 | Is it actually slow? | Measure with `time.perf_counter()` |
| 2 | Where is it slow? | Profile with cProfile |
| 3 | Which lines are slow? | Profile with line_profiler |
| 4 | Is it I/O-bound or CPU-bound? | Check if CPU usage is high or low during the slow part |
| 5 | Can I use a better algorithm? | O(n) vs O(n^2) matters more than any micro-optimization |
| 6 | Can I cache results? | `@lru_cache` for repeated computations |
| 7 | Can I vectorize? | NumPy for numerical array operations |
| 8 | Can I use generators? | Lazy evaluation for memory-bound problems |
| 9 | Can I use concurrency? | Threading for I/O, multiprocessing for CPU |
| 10 | Can I use C extensions? | Cython, pybind11 (last resort) |

### Optimization Approaches by Impact

| Technique | Typical Speedup | Effort | Best For |
|-----------|----------------|--------|----------|
| Better algorithm | 10-1000x | Medium | O(n^2) -> O(n log n) |
| NumPy vectorization | 10-200x | Low | Numerical array operations |
| Caching | 2-1000x | Low | Repeated expensive calls |
| asyncio/threading | 2-50x | Medium | I/O-bound tasks |
| Multiprocessing | 2-Nx (N=cores) | Medium | CPU-bound tasks |
| Generators | 1x (saves memory) | Low | Large data pipelines |
| `__slots__` | 1x (saves memory) | Low | Millions of small objects |
| Cython | 10-100x | High | Hot loops after all else fails |
| PyPy | 2-10x | None (just use PyPy) | Pure Python code |

### The Golden Rule

Profile before you optimize. Measure the impact after you optimize. If the optimization makes the code harder to read and the speedup is less than 2x, revert it. Readable code that takes 0.2 seconds is better than clever code that takes 0.15 seconds, because the person who maintains it (including future you) will spend more time understanding the clever version than the 0.05 seconds it saves per execution.

## Series Wrap-Up

Over eight articles, we have built a complete Python engineering toolkit:

1. **Environment** — pyenv, venv, pip-tools for reproducible setups
2. **Structure** — packages, imports, and CLI tools
3. **Testing** — pytest, fixtures, parametrize, and debugging
4. **Quality** — type hints, ruff, black, and pre-commit
5. **I/O** — files, encodings, and serialization formats
6. **Concurrency** — threads, processes, and asyncio
7. **Packaging** — building, publishing, and Docker
8. **Performance** — profiling, caching, and vectorization

These are not theoretical topics. They are the daily tools of professional Python development. The difference between a script that works and a project that scales is not cleverness — it is engineering discipline applied consistently.
