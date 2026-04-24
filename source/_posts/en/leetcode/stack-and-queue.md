---
title: "LeetCode Patterns: Stack and Queue"
date: 2024-01-25 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 9
series_total: 10
lang: en
mathjax: false
description: "Master stack and queue applications: valid parentheses, monotonic stack for next greater element, BFS with queues, priority queues for top-K problems, and deque for sliding window maximum."
---

Stacks and queues look unassuming next to graphs or DP, but they sit underneath an astonishing fraction of interview problems. The reason is simple: most algorithmic questions are really questions about *order of access*. Stacks give you LIFO (last in, first out); queues give you FIFO (first in, first out); and once you add the variants — monotonic stack, deque, priority queue — you have efficient answers for bracket matching, next-greater-element, sliding-window extrema, top-K, BFS, and a long tail of "implement X using Y" puzzles.

This part of the series walks the whole landscape end-to-end. We start from the bare data structures, then work through six representative LeetCode problems with full traces, and finish with a comparison table and a Q&A that targets the mistakes I see candidates make most often.

![Stack vs Queue: same input, opposite output order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/stack-and-queue/fig1_lifo_vs_fifo.png)

# Series Navigation

**LeetCode Algorithm Masterclass** (10 parts):

1. Hash Tables — Two Sum, Longest Consecutive, Group Anagrams
2. Two Pointers — collision pointers, fast/slow, sliding window
3. Linked List Operations — reversal, cycle detection, merge
4. Binary Tree Traversal & Recursion — in/pre/post-order, LCA
5. Dynamic Programming Intro — 1D / 2D DP, state transition
6. Backtracking — permutations, combinations, pruning
7. Binary Search Advanced — integer / real / answer binary search
8. Sliding Window Patterns — fixed and variable windows
9. **Stack & Queue** — monotonic stack, priority queue, deque ← *you are here*
10. Greedy & Bit Manipulation — greedy strategies, bitwise tricks

# 1. Stack and Queue Fundamentals

## 1.1 Stack — LIFO

A **stack** is a linear container in which the most recently inserted element is the first to leave. Think of a pile of plates: you add to the top and remove from the top. The interface is tiny:

- `push(x)` — append `x` on top
- `pop()` — remove and return the top
- `peek()` / `top()` — read the top without removing it
- `empty()`, `size()` — housekeeping

All five operations run in **O(1)** worst case. The trick is the matching access pattern: a stack only ever talks about its *current* top, so any algorithm that naturally asks "what was the most recent X?" can be expressed with one.

In Python you almost always reuse `list`, because both `append` and `pop` from the end are amortised O(1):

```python
stack = []
stack.append(1)        # push
stack.append(2)
top = stack[-1]        # peek -> 2
val = stack.pop()      # pop  -> 2
empty = not stack
```

In Java you can use `ArrayDeque` (preferred over the legacy `Stack` class):

```java
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);
int top = stack.peek();
int val = stack.pop();
```

In C++:

```cpp
#include <stack>
std::stack<int> st;
st.push(1);
int top = st.top();
st.pop();
```

## 1.2 Queue — FIFO

A **queue** is the mirror image: the *oldest* element leaves first. Picture a line at a coffee shop. The standard interface is:

- `enqueue(x)` / `offer(x)` — append at the rear
- `dequeue()` / `poll()` — remove and return the front
- `peek()` / `front()` — read the front
- `empty()`, `size()`

A naive implementation backed by a Python `list` looks fine until you measure it: `list.pop(0)` is O(n) because every other element shifts down. The right primitive is `collections.deque`, which gives O(1) on both ends:

```python
from collections import deque
q = deque()
q.append(1)            # enqueue
q.append(2)
front = q[0]           # peek -> 1
val = q.popleft()      # dequeue -> 1
```

In Java the analogue is `ArrayDeque` again (used as a queue via `offer` / `poll`); in C++ you have `std::queue<int>`.

A practical mental model: **stack = function call history, queue = work to-do list**. That single sentence already tells you which one BFS, DFS, undo/redo, task schedulers, and bracket matchers want.

# 2. Stack Classics

## 2.1 LeetCode 20 — Valid Parentheses

> Given a string `s` containing only the characters `()[]{}`, decide whether every opening bracket is closed by the same type, in the correct nested order.

The key observation is the *nesting* rule: when a closing bracket arrives, it must match the **most recently opened** one that is still unclosed. "Most recently opened" is exactly what a stack tracks.

**Algorithm.** Walk the string. Push opening brackets. On a closing bracket, the stack must be non-empty *and* the top must be the matching opener — otherwise the string is invalid. After the loop, the string is valid iff the stack is empty.

```python
def isValid(s: str) -> bool:
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    for ch in s:
        if ch in pairs:                       # closing bracket
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:                                  # opening bracket
            stack.append(ch)
    return not stack
```

Figure 2 traces the algorithm on `({[]})`. Each column is one step: the highlighted character is the one being read, the bottom column shows the stack *after* the action. Notice how the stack grows to depth 3, then unwinds symmetrically.

![Valid Parentheses: stack trace on "({[]})"](./stack-and-queue/fig2_valid_parentheses.png)

**Complexity.** Each character is pushed and popped at most once, so time is **O(n)** and space is **O(n)** in the worst case (`((((` would push everything).

**Easy traps.**

- Forgetting to check `not stack` *before* the comparison — popping an empty stack throws.
- Returning early on the first match without checking the rest of the string.
- Returning `True` when the loop ends without checking that the stack is empty (`"((("` would pass).

## 2.2 LeetCode 155 — Min Stack

> Design a stack that, in addition to `push`, `pop`, `top`, supports `getMin()` in **O(1)**.

Scanning the stack to find the minimum is O(n); we need to remember the minimum at every depth. The cleanest formulation stores pairs `(value, min_so_far)`:

```python
class MinStack:
    def __init__(self) -> None:
        self.stack: list[tuple[int, int]] = []

    def push(self, val: int) -> None:
        cur_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, cur_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

Every operation is O(1) and the space overhead is one extra integer per element. The "two parallel stacks" variant works too but is fiddlier when duplicates of the minimum appear (you must compare `<= ` instead of `<` on push, and pop the auxiliary stack only when the popped value equals the current min).

## 2.3 LeetCode 150 — Evaluate Reverse Polish Notation

Postfix notation eliminates parentheses by writing operators after their operands: `((2 + 1) * 3)` becomes `2 1 + 3 *`. The evaluation rule is a one-line stack invariant: **the stack always holds the operands waiting for their next operator**.

```python
def evalRPN(tokens: list[str]) -> int:
    stack: list[int] = []
    ops = {"+", "-", "*", "/"}
    for tok in tokens:
        if tok in ops:
            b = stack.pop()                # right operand pops first!
            a = stack.pop()
            if   tok == "+": stack.append(a + b)
            elif tok == "-": stack.append(a - b)
            elif tok == "*": stack.append(a * b)
            else:            stack.append(int(a / b))   # truncate toward zero
        else:
            stack.append(int(tok))
    return stack[0]
```

Two subtleties bite people:

- **Operand order.** The first element popped is the *right* operand. `a - b` and `b - a` are different.
- **Division.** LeetCode wants truncation toward zero (`-7 / 2 == -3` in C, `-3` here too), but Python's `//` rounds toward negative infinity (`-7 // 2 == -4`). Use `int(a / b)`.

# 3. Monotonic Stack

A **monotonic stack** is just a stack with the discipline that its contents are kept in strictly increasing or decreasing order. Whenever a new element would break the invariant, we pop until it doesn't. That single rule turns a quadratic-looking "for each i, find the next j such that..." into an amortised O(n) sweep, because every index is pushed and popped at most once.

The pattern matches four families of questions:

- next greater element to the right → decreasing stack
- next smaller element to the right → increasing stack
- previous greater / previous smaller → same logic, scan from the right (or pre-process)

## 3.1 LeetCode 739 — Daily Temperatures

> For each day `i`, how many days until a strictly warmer day? Output `0` if none.

Brute force is the textbook O(n²) double loop. The monotonic-stack rewrite keeps a stack of **indices** whose temperatures are decreasing from bottom to top. When today's temperature is higher than the top index's, today is the answer for that index — pop it, fill in `answer[idx] = i - idx`, repeat.

```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    answer = [0] * n
    stack: list[int] = []                     # indices, temps decreasing
    for i, t in enumerate(temperatures):
        while stack and t > temperatures[stack[-1]]:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer
```

Figure 3 makes the resolution visual. Each green arc is "this colder day was finally resolved by that warmer day". The bottom row shows the stack of indices after every step — note how it always stays decreasing in temperature.

![Daily Temperatures: monotonic decreasing stack of indices](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/stack-and-queue/fig3_monotonic_stack.png)

**Complexity.** Each index enters and leaves the stack at most once, so total work across the inner `while` is O(n). Space is O(n) for the stack.

**Trap:** the stack stores **indices**, not temperatures, because we need both the value (to compare) and the position (to compute the gap).

# 4. Queues and BFS

Queues' best-known job is implementing **breadth-first search**: explore all distance-1 neighbours, then all distance-2, and so on. The skeleton fits in eight lines:

```python
from collections import deque

def bfs(start, neighbours_of):
    seen = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        # ... process node ...
        for nb in neighbours_of(node):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
```

Two implementation rules carry over to almost every BFS problem:

- **Mark on enqueue, not on dequeue.** Otherwise the same node can be queued many times before it is processed, blowing up time and memory.
- **Use `deque`, not `list`.** `list.pop(0)` is O(n). For BFS that turns the algorithm from O(V+E) into O(V·(V+E)).

Level-order traversal is BFS with one extra trick: snapshot `len(q)` at the start of each level so you know when to start a new sublist (LeetCode 102, code below for reference):

```python
def levelOrder(root):
    if not root: return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):                # important: capture *now*
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

# 5. Priority Queue (Heap)

A **priority queue** breaks the FIFO rule deliberately: the next element out is always the one with the smallest (or largest) key. The standard implementation is a **binary heap** — a complete binary tree stored in an array, where every parent compares ≤ (min-heap) or ≥ (max-heap) to its children.

![Min-heap as a binary tree and the underlying array](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/stack-and-queue/fig4_heap_priority_queue.png)

The array layout means parent-child arithmetic is index-only:

- parent of `i` is `(i - 1) // 2`
- children of `i` are `2*i + 1` and `2*i + 2`

That is what makes `heappush` and `heappop` cost **O(log n)**, while `peek` (root) is **O(1)** and `heapify` of a fresh array is **O(n)**.

In Python, `heapq` is always a min-heap. To get max-heap behaviour, push the negation of your key.

## 5.1 LeetCode 347 — Top K Frequent Elements

> Return the `k` most frequent elements of `nums`.

Three solutions, in order of cleverness:

```python
import heapq
from collections import Counter

# (1) Min-heap of size k.  Time: O(n log k), space: O(n + k)
def topKFrequent(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)
    heap: list[tuple[int, int]] = []          # (frequency, value)
    for val, f in freq.items():
        heapq.heappush(heap, (f, val))
        if len(heap) > k:
            heapq.heappop(heap)               # drop the least frequent
    return [v for _, v in heap]
```

```python
# (2) Bucket sort.  Time: O(n), space: O(n)
def topKFrequent(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for v, f in freq.items():
        buckets[f].append(v)
    out: list[int] = []
    for f in range(len(buckets) - 1, 0, -1):
        out.extend(buckets[f])
        if len(out) >= k:
            return out[:k]
    return out
```

The min-heap idea generalises beyond top-K: it is the same trick you use for **Merge K Sorted Lists** (one heap entry per list head), **K-th Largest in a Stream**, and Dijkstra's shortest paths.

# 6. Deque — Sliding Window Maximum

A **deque** (double-ended queue) supports O(1) push/pop on **both** ends. It is exactly what you need when a sliding window asks for an extremum of its current contents.

## 6.1 LeetCode 239 — Sliding Window Maximum

> Slide a window of size `k` across `nums`; return the max of each window.

The trick is a **monotonic deque** of indices, kept decreasing by value:

- when an index falls outside the window (`dq[0] <= i - k`), pop it from the front;
- before pushing `i`, pop indices from the back whose values are `< nums[i]` — they can never be the max while `i` is in the window;
- the maximum of the current window is always `nums[dq[0]]`.

```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    dq: deque[int] = deque()
    out: list[int] = []
    for i, x in enumerate(nums):
        # 1. drop indices that have left the window
        if dq and dq[0] <= i - k:
            dq.popleft()
        # 2. maintain decreasing order from front to back
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        # 3. record once we have a full window
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
```

Each index enters and leaves the deque at most once, so the total cost is O(n). A naive solution that recomputes the max per window would be O(n·k). Why a deque and not a heap? Because a heap can't cheaply *evict* an element that has just slid out of the window — you would have to lazy-delete and the bound becomes O(n log n) instead of O(n).

# 7. Bridges — Building One on Top of the Other

These two problems show up surprisingly often as warm-up questions for system-design interviews because they test whether you really *understand* the access patterns.

## 7.1 LeetCode 232 — Implement Queue using Stacks

A queue needs FIFO; a stack gives LIFO. Reversing twice gets you back to the original order, so we keep two stacks: an **input stack** for pushes and an **output stack** for pops. A pop that finds the output stack empty drains the input stack into it; otherwise it just pops the output top.

![Queue using two stacks: push to input, drain on pop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/stack-and-queue/fig5_queue_via_two_stacks.png)

```python
class MyQueue:
    def __init__(self) -> None:
        self.in_stk: list[int] = []
        self.out_stk: list[int] = []

    def push(self, x: int) -> None:
        self.in_stk.append(x)

    def pop(self) -> int:
        self._shift()
        return self.out_stk.pop()

    def peek(self) -> int:
        self._shift()
        return self.out_stk[-1]

    def empty(self) -> bool:
        return not self.in_stk and not self.out_stk

    def _shift(self) -> None:
        if not self.out_stk:
            while self.in_stk:
                self.out_stk.append(self.in_stk.pop())
```

`push` is O(1). `pop` and `peek` are O(1) **amortised**: every element is moved across at most once during its lifetime, so n operations cost O(n) total even though one individual `pop` may do a big shift.

## 7.2 LeetCode 225 — Implement Stack using Queues

The reverse direction is uglier because rotation is expensive. The single-queue approach is the simplest: on every push, append `x`, then rotate the previous `n - 1` elements to the back so `x` sits at the front.

```python
from collections import deque

class MyStack:
    def __init__(self) -> None:
        self.q: deque[int] = deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q
```

`push` is O(n), `pop` and `top` are O(1). The asymmetry tells you something deep: queues are *not* stacks-with-extra-capabilities; the LIFO discipline genuinely needs different machinery.

# 8. Complexity Cheat Sheet

| Container | push / enqueue | pop / dequeue | peek | search |
|---|---|---|---|---|
| Stack | O(1) | O(1) | O(1) | O(n) |
| Queue (deque-backed) | O(1) | O(1) | O(1) | O(n) |
| Deque | O(1) both ends | O(1) both ends | O(1) | O(n) |
| Min/Max heap | O(log n) | O(log n) | O(1) | O(n) |
| Heap (`heapify` from array) | O(n) build | — | — | — |

Algorithm-level summary:

- bracket / RPN matching — O(n) time, O(n) space
- monotonic stack (next greater / smaller) — O(n) time, O(n) space
- BFS over a graph with `V` vertices and `E` edges — O(V + E) time, O(V) space
- sliding-window extremum via monotonic deque — O(n) time, O(k) space
- top-K frequent via min-heap of size K — O(n log k) time, O(n) space

# 9. Q&A

**Q1. Why use `deque` instead of `list` for queues in Python?**

`list.pop(0)` is O(n) because the remaining elements have to shift left. `deque.popleft()` is O(1). For BFS or any queue-heavy workload this changes the asymptotics, not just a constant.

**Q2. When do I reach for a monotonic stack instead of a regular one?**

When the question is shaped like "for each index, find the closest index to the left/right whose value is greater/smaller". The monotonic discipline turns the obvious O(n²) sweep into O(n).

**Q3. How do I get a max-heap out of Python's `heapq`?**

Push `-x` and negate again on pop. Or, for tuples, push `(-priority, value)` so that the highest priority becomes the smallest negative.

**Q4. Why does sliding-window maximum use a deque instead of a heap?**

A heap can give you the current max in O(log n), but it can't cheaply remove an element that has just left the window — you would have to lazy-delete and pay extra logs. The monotonic deque drops out-of-window indices in O(1) at the front, and out-of-order indices in O(1) at the back, giving a true O(n) algorithm.

**Q5. Min-heap or max-heap for "top K largest"?**

A **min-heap of size K**. Once the heap has K elements, every new element bigger than the root replaces the root; the root is always the smallest of the K best so far. Symmetrically, "top K smallest" wants a max-heap of size K.

**Q6. Why is "queue via two stacks" called amortised O(1)?**

Each element is moved from input to output exactly once before it leaves. n operations therefore do O(n) total work, so the *per-operation* average is O(1) even though a single `pop` may do a big transfer.

**Q7. Stack vs queue for tree / graph traversal?**

Stack (or recursion) gives you DFS — depth first, follow one path until it terminates, then backtrack. Queue gives you BFS — visit by distance from the start. They explore the same graph but in different orders, so the right choice depends on what you are looking for (any path, shortest path, etc.).

# 10. Summary

The mental model that makes all of this stick is short:

- **Stack** — when you need the *most recent* unresolved thing.
- **Queue** — when you need things *in arrival order*.
- **Monotonic stack** — when "most recent" should also satisfy a comparison invariant; turns O(n²) into O(n).
- **Deque** — both ends are interesting; sliding-window extrema, work-stealing, undo-redo with bounded history.
- **Priority queue** — order does not matter, *priority* does; top-K, scheduling, Dijkstra.

Once a problem statement maps cleanly onto one of those five sentences, the implementation is mechanical. Most of the work in interviews is recognising the mapping; the data structure does the rest.
