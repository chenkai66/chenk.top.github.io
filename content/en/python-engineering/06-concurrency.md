---
title: "Python Engineering (6): Concurrency — Threads, Processes, and asyncio"
date: 2022-04-21 09:00:00
tags:
  - Python
  - Concurrency
  - asyncio
categories: Python Engineering
series: python-engineering
lang: en
description: "Understand the GIL, master threading, multiprocessing, and asyncio. Learn which concurrency model to use for I/O-bound vs CPU-bound workloads."
disableNunjucks: true
series_order: 6
series_total: 8
translationKey: "python-engineering-6"
---

Your script downloads 100 files one at a time. Each download takes 2 seconds, mostly waiting for the server to respond. Total time: 200 seconds. Your CPU is idle for 99% of that time, wasting compute and money on network latency. Concurrency can fix this.

Python has three concurrency models, each designed for different problems. Choosing the wrong one can make your code slow or full of race conditions. This article explains when to use each.

---

## The GIL: What It Is and Why It Matters

The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects. Only one thread can execute Python bytecode at a time, even on a multi-core machine.

![GIL impact on parallelism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/06-gil-impact.png)


### What the GIL Prevents

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(1_000_000):
        counter += 1  # This is NOT atomic

threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)
# Without GIL: race condition, counter < 4_000_000
# With GIL: still a race condition! counter < 4_000_000
```

Wait, the GIL does not prevent this? Correct. `counter += 1` compiles to multiple bytecodes (LOAD, ADD, STORE), and the GIL can release between them. The GIL protects interpreter internals, not your application logic.

### What the GIL Does

- Prevents multiple threads from corrupting Python's internal data structures (reference counts, object allocation)
- Makes single-threaded code faster (no locking overhead)
- Releases during I/O operations (file reads, network calls, sleep)

### What the GIL Means for You

| Workload Type | Threading | Multiprocessing |
|---------------|-----------|-----------------|
| I/O-bound (network, disk) | Works well (GIL releases during I/O) | Works but overkill |
| CPU-bound (math, parsing) | No speedup (GIL blocks parallel execution) | Works well (separate processes, separate GILs) |
| Mixed | Depends on ratio | Usually the safe choice |

## Threading: I/O-Bound Concurrency

Threads share the same memory space and are lightweight. The GIL releases during I/O operations, making them effective for network calls, file operations, and database queries.

![Concurrency models comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/06-concurrency-models.png)


### Basic Thread Usage

```python
import threading
import time

import requests


def download(url: str) -> int:
    """Download a URL, return response size."""
    response = requests.get(url, timeout=10)
    return len(response.content)


urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
]

# Sequential: ~4 seconds
start = time.perf_counter()
results = [download(url) for url in urls]
sequential_time = time.perf_counter() - start
print(f"Sequential: {sequential_time:.2f}s")

# Threaded: ~1 second
start = time.perf_counter()
threads = []
for url in urls:
    t = threading.Thread(target=download, args=(url,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
threaded_time = time.perf_counter() - start
print(f"Threaded: {threaded_time:.2f}s")
```

Output:

```text
Sequential: 4.12s
Threaded: 1.08s
```

### ThreadPoolExecutor

`concurrent.futures` provides a higher-level API with result collection:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def download(url: str) -> tuple[str, int]:
    response = requests.get(url, timeout=10)
    return url, len(response.content)


urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/3",
]

with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all tasks
    futures = {executor.submit(download, url): url for url in urls}

    # Process results as they complete
    for future in as_completed(futures):
        url = futures[future]
        try:
            result_url, size = future.result()
            print(f"Downloaded {result_url}: {size} bytes")
        except Exception as e:
            print(f"Failed {url}: {e}")
```

### Thread-Safe Data Structures

When threads share data, use locks or thread-safe collections:

```python
import threading
from collections import deque

# Lock for protecting shared state
lock = threading.Lock()
results = []

def worker(item):
    processed = expensive_computation(item)
    with lock:  # Only one thread can execute this block at a time
        results.append(processed)

# Queue for producer-consumer patterns
from queue import Queue

work_queue: Queue[str] = Queue()
for url in urls:
    work_queue.put(url)

def consumer():
    while True:
        url = work_queue.get()
        if url is None:  # Poison pill
            break
        download(url)
        work_queue.task_done()

# Start consumers
threads = [threading.Thread(target=consumer) for _ in range(4)]
for t in threads:
    t.start()

# Wait for all work to complete
work_queue.join()

# Send poison pills
for _ in threads:
    work_queue.put(None)
for t in threads:
    t.join()
```

## Multiprocessing: CPU-Bound Concurrency

Each process has its own Python interpreter and GIL, enabling true parallelism on multiple CPU cores.

![Concurrency decision tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/06-decision-tree.png)


### Basic Usage

```python
import multiprocessing
import time


def cpu_intensive(n: int) -> int:
    """Simulate CPU-bound work."""
    total = 0
    for i in range(n):
        total += i * i
    return total


numbers = [10_000_000] * 4

# Sequential
start = time.perf_counter()
results = [cpu_intensive(n) for n in numbers]
print(f"Sequential: {time.perf_counter() - start:.2f}s")

# Parallel
start = time.perf_counter()
with multiprocessing.Pool(processes=4) as pool:
    results = pool.map(cpu_intensive, numbers)
print(f"Parallel:   {time.perf_counter() - start:.2f}s")
```

Output on a 4-core machine:

```text
Sequential: 8.45s
Parallel:   2.31s
```

### ProcessPoolExecutor

Same API as ThreadPoolExecutor, making it easy to switch between the two:

```python
from concurrent.futures import ProcessPoolExecutor

def factorize(n: int) -> list[int]:
    """Find all factors of n."""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


numbers = [112272535095293, 112582705942171, 115280095190773, 115797848077099]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(factorize, numbers))
    for n, factors in zip(numbers, results):
        print(f"{n}: {factors}")
```

### Multiprocessing Overhead

Processes are heavier than threads:

| Aspect | Threads | Processes |
|--------|---------|-----------|
| Memory | Shared (lightweight) | Separate (each copies data) |
| Startup cost | ~1ms | ~50-100ms |
| Communication | Direct (shared memory, but need locks) | Serialization (pickle via pipes) |
| GIL limitation | Yes (CPU-bound limited) | No (separate interpreters) |
| Debugging | Harder (shared state bugs) | Easier (isolated state) |

Don't use multiprocessing for small tasks. The serialization and process startup overhead can make it slower than sequential execution.

### Sharing Data Between Processes

```python
import multiprocessing

def worker(shared_array, index, value):
    shared_array[index] = value

if __name__ == "__main__":
    # Shared array
    arr = multiprocessing.Array("i", 4)  # 4 integers

    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=worker, args=(arr, i, i * 10))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(list(arr))  # [0, 10, 20, 30]
```

Note the `if __name__ == "__main__":` guard. This is required on macOS and Windows because multiprocessing uses `spawn` to create new processes, which re-imports the module.

## concurrent.futures: Unified API


![Asyncio event loop as a spinning wheel processing coroutines](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/06-asyncio-event-loop-as-a-spinning-wheel-processing-coroutines.jpg)

The beauty of `concurrent.futures` is that switching between threads and processes requires changing one line:

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def process_item(item):
    # ... some work ...
    return result

items = range(100)

# For I/O-bound work:
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_item, items))

# For CPU-bound work (change only this line):
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_item, items))
```

### Timeout and Exception Handling

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def slow_download(url):
    import time
    time.sleep(10)
    return f"Done: {url}"

with ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(slow_download, "https://example.com")

    try:
        result = future.result(timeout=5)  # Wait max 5 seconds
    except TimeoutError:
        print("Task timed out!")
        future.cancel()
    except Exception as e:
        print(f"Task failed: {e}")
```

## asyncio: Cooperative Concurrency

asyncio uses a single thread with an event loop. Functions voluntarily give up control at `await` points, allowing other tasks to run. No threads, no locks, no GIL worries.

![asyncio event loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/06-asyncio-loop.png)


### Basic async/await

```python
import asyncio


async def say_after(delay: float, message: str) -> str:
    """Wait, then return a message."""
    await asyncio.sleep(delay)
    return message


async def main():
    # Sequential: takes 3 seconds
    result1 = await say_after(1, "hello")
    result2 = await say_after(2, "world")
    print(result1, result2)

    # Concurrent: takes 2 seconds (max of delays)
    results = await asyncio.gather(
        say_after(1, "hello"),
        say_after(2, "world"),
    )
    print(results)  # ['hello', 'world']


asyncio.run(main())
```

### Creating Tasks

```python
import asyncio


async def download(url: str) -> str:
    print(f"Start: {url}")
    await asyncio.sleep(1)  # Simulate network I/O
    print(f"Done: {url}")
    return f"Content of {url}"


async def main():
    # Create tasks (start running immediately)
    tasks = [
        asyncio.create_task(download(f"https://example.com/{i}"))
        for i in range(5)
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    print(f"Downloaded {len(results)} pages")


asyncio.run(main())
```

Output:

```text
Start: https://example.com/0
Start: https://example.com/1
Start: https://example.com/2
Start: https://example.com/3
Start: https://example.com/4
Done: https://example.com/0
Done: https://example.com/1
Done: https://example.com/2
Done: https://example.com/3
Done: https://example.com/4
Downloaded 5 pages
```

All five downloads start immediately and complete together after ~1 second.

### aiohttp for Async HTTP

The `requests` library is synchronous. For async HTTP, use `aiohttp`:

```bash
(.venv) $ pip install aiohttp
```

```python
import asyncio
import time

import aiohttp


async def download(session: aiohttp.ClientSession, url: str) -> int:
    async with session.get(url) as response:
        content = await response.read()
        return len(content)


async def main():
    urls = [f"https://httpbin.org/delay/1" for _ in range(10)]

    async with aiohttp.ClientSession() as session:
        tasks = [download(session, url) for url in urls]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    print(f"Downloaded {len(results)} URLs in {elapsed:.2f}s")
    print(f"Total bytes: {sum(results)}")


asyncio.run(main())
```

Output:

```text
Downloaded 10 URLs in 1.15s
Total bytes: 4230
```

Ten URLs with 1-second delay each, completed in just over 1 second.

### Semaphore: Controlling Concurrency

Unlimited concurrency can overwhelm servers or hit rate limits:

```python
import asyncio
import aiohttp


async def download(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> int:
    async with semaphore:  # At most N concurrent downloads
        async with session.get(url) as response:
            content = await response.read()
            return len(content)


async def main():
    urls = [f"https://httpbin.org/delay/1" for _ in range(100)]
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

    async with aiohttp.ClientSession() as session:
        tasks = [download(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)
    print(f"Downloaded {len(results)} URLs")


asyncio.run(main())
```

### Timeout with asyncio

```python
import asyncio


async def slow_operation():
    await asyncio.sleep(10)
    return "done"


async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=3.0)
    except asyncio.TimeoutError:
        print("Operation timed out after 3 seconds")


asyncio.run(main())
```

## When to Use Which

The decision depends on your workload type:

**I/O-bound work (network, disk, database):**
- Small number of concurrent tasks (< 50): `ThreadPoolExecutor`
- Large number of concurrent tasks (50+): `asyncio`
- Simple scripts, one-off tasks: `ThreadPoolExecutor`
- Web servers, long-running services: `asyncio`

**CPU-bound work (math, image processing, parsing):**
- Always: `ProcessPoolExecutor` or `multiprocessing.Pool`

**Mixed workloads:**
- Use asyncio for I/O, offload CPU work to `ProcessPoolExecutor`:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

import aiohttp


def cpu_work(data: bytes) -> dict:
    """CPU-intensive processing (runs in separate process)."""
    # Parse, transform, compute...
    return {"result": len(data)}


async def fetch_and_process(session, url, process_pool):
    """Fetch data (async I/O) then process it (CPU in process pool)."""
    async with session.get(url) as response:
        data = await response.read()

    # Offload CPU work to process pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(process_pool, cpu_work, data)
    return result


async def main():
    urls = [f"https://example.com/{i}" for i in range(20)]

    with ProcessPoolExecutor(max_workers=4) as process_pool:
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_and_process(session, url, process_pool)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)
```

## Real Benchmark: Sequential vs Threaded vs Async

![Concurrency benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/python-engineering/06-benchmark.png)

```python
"""Benchmark: download 20 URLs with different concurrency models."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import requests

URL = "https://httpbin.org/delay/1"
COUNT = 20


def sequential():
    for _ in range(COUNT):
        requests.get(URL, timeout=10)


def threaded():
    def download(url):
        requests.get(url, timeout=10)

    with ThreadPoolExecutor(max_workers=COUNT) as executor:
        list(executor.map(download, [URL] * COUNT))


async def async_download():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(COUNT):
            tasks.append(session.get(URL))
        responses = await asyncio.gather(*tasks)
        for r in responses:
            await r.read()
            r.close()


def benchmark(name, func):
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    print(f"{name:12s}: {elapsed:.2f}s")


benchmark("Sequential", sequential)
benchmark("Threaded", threaded)
benchmark("Async", lambda: asyncio.run(async_download()))
```

Results:

```text
Sequential  : 21.34s
Threaded    : 1.18s
Async       : 1.09s
```

Both threaded and async complete in about 1 second (the server delay). Async uses fewer system resources because it doesn't have thread stacks or context switches.

## Comparison Table


![Python gil bottleneck single lane bridge with threads waitin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/python-engineering/06-python-gil-bottleneck-single-lane-bridge-with-threads-waitin.jpg)

| Feature | threading | multiprocessing | asyncio |
|---------|-----------|-----------------|---------|
| Parallelism type | Concurrent (GIL-limited) | True parallel | Concurrent (cooperative) |
| Best for | I/O-bound | CPU-bound | I/O-bound (many tasks) |
| Memory overhead | Low (~8KB per thread) | High (~30MB per process) | Very low (~1KB per coroutine) |
| Max practical tasks | ~100-1000 | CPU core count | ~10,000+ |
| Shared state | Yes (need locks) | No (serialized) | Yes (no locks needed, single thread) |
| Debugging difficulty | Hard (race conditions) | Medium (isolation helps) | Medium (stack traces less clear) |
| Library ecosystem | All libraries work | All libraries work | Needs async-compatible libraries |
| Startup cost | ~1ms | ~50-100ms | ~0.01ms |
| GIL affected | Yes | No | N/A (single thread) |

## Common Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Using threads for CPU work | No speedup, 100% one core | Switch to ProcessPoolExecutor |
| Too many threads | Memory exhaustion, slow context switching | Limit thread pool size (10-50 typically) |
| Forgetting `if __name__ == "__main__":` | RuntimeError on macOS/Windows | Always guard multiprocessing code |
| Mixing sync and async | `RuntimeError: This event loop is already running` | Use `asyncio.run()` at top level only |
| Not awaiting coroutines | `RuntimeWarning: coroutine was never awaited` | Always `await` async function calls |
| Blocking in async code | Event loop freezes, other tasks starve | Offload blocking work to thread/process pool |
| Race conditions with shared state | Inconsistent data, intermittent bugs | Use locks, queues, or immutable data |
| Deadlocks | Program hangs forever | Acquire locks in consistent order, use timeouts |

## What's Next

Your code is now concurrent and fast. But before you share it with the world, you need to package it properly. In the next article, we will build distributable Python packages, publish to PyPI, create Docker images, and set up a complete distribution pipeline.
