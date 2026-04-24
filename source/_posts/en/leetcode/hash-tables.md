---
title: "LeetCode Patterns: Hash Tables"
date: 2024-03-01 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 1
series_total: 10
lang: en
mathjax: false
description: "Master hash table patterns through four classic LeetCode problems — Two Sum, Group Anagrams, Longest Substring Without Repeating Characters, and Top K Frequent Elements. Learn complement search, canonical-form grouping, sliding window with counts, and bucket sort by frequency."
disableNunjucks: true
---

A hash table is the cheapest superpower in your toolbox. You spend a constant amount of memory per stored item, and in return every "is *x* in here?" question costs roughly one CPU instruction. Whole families of `O(n²)` brute-force solutions collapse into a single `O(n)` pass once you reach for one.

This article is the first installment of the **LeetCode Patterns** series. We will build hash table intuition from scratch, then work through four template problems — **Two Sum**, **Group Anagrams**, **Longest Substring Without Repeating Characters**, and **Top K Frequent Elements** — each illustrating a reusable pattern you will see again and again on harder problems.

# Series navigation

**LeetCode Patterns** (10 articles):

1. **Hash Tables** — Two Sum, Group Anagrams, Longest Substring Without Repeating Characters, Top K Frequent ← *you are here*
2. Two Pointers — collision pointers, fast / slow, partitioning
3. Linked Lists — reversal, cycle detection, merging
4. Sliding Window — fixed and variable windows
5. Binary Search — on indices, on answers, on real numbers
6. Binary Tree Traversal — recursion, BFS, LCA
7. Dynamic Programming — 1D, 2D, knapsack family
8. Backtracking — permutations, combinations, pruning
9. Greedy Algorithms — exchange argument, scheduling
10. Stack & Queue — monotonic stack, deque tricks

# 1. What a hash table really is

A hash table stores `(key, value)` pairs in an internal array. The trick is the **hash function**, which turns any key into an integer index that says *which array slot the pair belongs in*.

![Hash function: keys flow through hash(key) % N to bucket indices](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/hash-tables/fig1_hash_function.png)

Three guarantees come out of this design:

- **Determinism.** The same key always hashes to the same slot. That is what makes lookups possible.
- **Average-case `O(1)`.** With a reasonable hash function, the work per insert / lookup / delete does not grow with `n`.
- **Worst-case `O(n)`.** If too many keys collide into the same slot, you degrade to a linear scan. In practice this only happens with adversarial input or a broken hash function.

## Collisions are inevitable

Two different keys can hash to the same slot — that is a **collision**. Hash tables handle them with one of two strategies:

![Separate chaining (Java HashMap) vs open addressing (Python dict)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/hash-tables/fig2_collision_resolution.png)

- **Separate chaining** keeps a small linked list (or balanced tree) inside each slot. Collisions just append to the chain. Used by Java's `HashMap` and C++'s `std::unordered_map`.
- **Open addressing** keeps one entry per slot. On a collision it probes forward (`slot+1`, `slot+2`, …) until it finds an empty one. Used by Python's `dict` and Go's `map`.

You almost never implement collision handling yourself. But knowing the mechanism explains why hash tables can occasionally be slower than `O(1)` — and why a good hash function matters.

## Hash table vs alternatives

Why reach for a hash table instead of a sorted array or a plain array?

![Lookup cost: hash table O(1) vs binary search O(log n) vs linear scan O(n)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/hash-tables/fig4_complexity_compare.png)

The gap is enormous as `n` grows. At `n = 1000`, a linear scan does ~1000 comparisons per lookup, binary search does ~10, a hash lookup does ~1. Memory is the price you pay: a Python `dict` carries roughly 8–16× the byte cost of a raw array, because it stores the hash, the key, the value, and keeps the table sparse to avoid collisions.

## Pick the right structure

Before you write code, decide what you actually need:

![Decision tree: array vs hash set vs hash map](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/hash-tables/fig5_set_vs_map_decision.png)

The rule of thumb is **array > set > map** in memory cost. Use the lightest structure that solves your problem. Concretely:

| Need | Use |
| --- | --- |
| Index by integer in `[0, N)` | plain `list` / array |
| "Have I seen this before?" | `set` |
| "What value goes with this key?" | `dict` |
| Counting occurrences | `collections.Counter` |
| Auto-creating empty buckets | `collections.defaultdict(list)` |

In Python, `set` and `dict` share the same hash machinery, so use whichever expresses your intent — that is also the easiest code to read and debug.

# 2. Two Sum — the complement pattern

> **LeetCode 1.** Given an array `nums` and an integer `target`, return the indices of the two numbers that add up to `target`. You may assume exactly one solution exists, and you may not use the same element twice.

```
Input:  nums = [2, 7, 11, 15], target = 9
Output: [0, 1]      # because 2 + 7 == 9
```

## Brute force first

The naive solution checks every pair, which is `O(n²)` time and `O(1)` space:

```python
def two_sum_brute(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

For `n = 10⁴` that is roughly 50 million comparisons. We can do better.

## The hash table insight

While scanning left to right, every time we look at `num`, the question is not "is `num` in the array?" but "**have I already seen `target − num`?**" If yes, we have our pair. So we keep a map `value → index` of everything seen so far.

![Two Sum trace on nums = [2, 7, 11, 15], target = 9](./hash-tables/fig3_two_sum_flow.png)

```python
from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """One-pass hash table solution. O(n) time, O(n) space."""
    seen: dict[int, int] = {}          # value -> earliest index

    for i, num in enumerate(nums):
        complement = target - num
        # Check BEFORE inserting: that prevents pairing num with itself.
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []   # unreachable per the problem statement
```

Two details that matter:

- **Check before insert.** If you store `seen[num] = i` first, an input like `nums = [3, 0, 0]` with `target = 6` would happily pair `nums[0]` with itself.
- **Return order.** `seen[complement]` is the *earlier* index, so it must come first: `[seen[complement], i]`.

## Why it is `O(n)`

One pass through `nums` does `n` iterations. Each iteration is one dictionary lookup and possibly one insert — both `O(1)` average. Total: `O(n)` time, `O(n)` space (the dictionary holds at most `n` entries).

You cannot do better in time *and* space for the unsorted version of this problem. If the array were already sorted, the two-pointer technique (covered in part 2) achieves `O(n)` time and `O(1)` space — but that requires a sort, which is `O(n log n)`.

## Pattern: complement search

Whenever you are looking for **pairs that satisfy a relation involving the other element**, ask whether you can rephrase the problem as "for each `x`, is there a previously-seen `y` such that `f(x, y)` holds?" If yes, a hash table will usually win:

| Problem | What you store | What you ask |
| --- | --- | --- |
| Two Sum | seen values | `target − num` in `seen`? |
| Pairs with given difference (LeetCode 532) | seen values | `num + k` and `num − k` in `seen`? |
| Subarray sum equals `k` (LeetCode 560) | prefix sums | `prefix − k` in `seen`? |
| Continuous subarray sum (LeetCode 523) | prefix sum % k | same residue seen before? |

# 3. Group Anagrams — the canonical-form pattern

> **LeetCode 49.** Given an array of strings, group the anagrams together.

```
Input:  ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

Two strings are anagrams if they have the same multiset of characters but in different orders. The whole game is to design a **key** that maps every anagram to the same bucket.

## Approach 1 — sorted string as the key

The most direct canonical form: sort the characters.

```python
from collections import defaultdict
from typing import List

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """O(n · k log k) where n = len(strs), k = max word length."""
    groups: dict[str, list[str]] = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))   # "eat", "tea", "ate" -> "aet"
        groups[key].append(s)
    return list(groups.values())
```

Simple, generic, works for any character set. The cost is the sort: `O(k log k)` per string.

## Approach 2 — character count as the key

If the alphabet is small (say lowercase English, 26 letters), counting characters is `O(k)` and we skip the sort:

```python
from collections import defaultdict
from typing import List

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """O(n · k) using a 26-slot count array as the key."""
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        # Lists are unhashable; convert to a tuple to use as a dict key.
        groups[tuple(count)].append(s)
    return list(groups.values())
```

The tuple matters. Dictionary keys must be **immutable and hashable** — a list cannot be a key, but its tuple version can.

## Pattern: canonical form

Whenever a problem says "group / find / count things that are equal up to some symmetry", ask **"what is the canonical representation that is the same for every member of an equivalence class?"** Then use that representation as a hash key.

| Problem | Equivalence | Canonical key |
| --- | --- | --- |
| Group Anagrams | same character multiset | sorted string or 26-tuple of counts |
| Group Shifted Strings (LeetCode 249) | same gap pattern between letters | tuple of consecutive differences |
| Isomorphic Strings (LeetCode 205) | same letter pattern | normalized "first-occurrence" string |
| Group strings by length | same length | the length itself |

# 4. Longest Substring Without Repeating Characters — sliding window with a hash map

> **LeetCode 3.** Given a string `s`, find the length of the longest substring with no repeated characters.

```
Input:  s = "abcabcbb"
Output: 3            # "abc"
```

This is where hash tables team up with the sliding window technique. We maintain a window `[left, right]` and slide `right` forward one character at a time. As soon as the new character would duplicate one already in the window, we shrink from the left until the window is valid again.

The hash map's job: for each character, remember **the most recent index where we saw it**. That lets us jump `left` straight past the duplicate in `O(1)` instead of walking it forward one step at a time.

```python
def length_of_longest_substring(s: str) -> int:
    """Sliding window with last-seen-index map. O(n) time, O(min(n, |Σ|)) space."""
    last_seen: dict[str, int] = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        # If ch was seen *inside* the current window, jump left past it.
        # max(...) guards against stale entries left of the window.
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1

        last_seen[ch] = right
        best = max(best, right - left + 1)

    return best
```

The trace for `"abcabcbb"`:

| `right` | `ch` | `last_seen` after update | `left` | window | `best` |
| --- | --- | --- | --- | --- | --- |
| 0 | `a` | `{a:0}` | 0 | `a` | 1 |
| 1 | `b` | `{a:0, b:1}` | 0 | `ab` | 2 |
| 2 | `c` | `{a:0, b:1, c:2}` | 0 | `abc` | 3 |
| 3 | `a` | `{a:3, b:1, c:2}` | 1 | `bca` | 3 |
| 4 | `b` | `{a:3, b:4, c:2}` | 2 | `cab` | 3 |
| 5 | `c` | `{a:3, b:4, c:5}` | 3 | `abc` | 3 |
| 6 | `b` | `{a:3, b:6, c:5}` | 5 | `cb` | 3 |
| 7 | `b` | `{a:3, b:7, c:5}` | 7 | `b` | 3 |

## Pattern: variable-size window with a hash map

Use this template whenever you need the **longest / shortest** substring or subarray satisfying a constraint that you can update in `O(1)` as you add or drop a single element:

```python
left = 0
state = {}                       # whatever you need to track the window

for right, x in enumerate(arr):
    add(x, state)                # extend the window

    while not valid(state):      # shrink until the window is valid again
        remove(arr[left], state)
        left += 1

    update_answer(left, right)
```

Variants you will meet later: minimum window substring (LeetCode 76), longest substring with at most K distinct characters (LeetCode 340), permutation in string (LeetCode 567).

# 5. Top K Frequent Elements — frequency map + bucket sort

> **LeetCode 347.** Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

```
Input:  nums = [1, 1, 1, 2, 2, 3], k = 2
Output: [1, 2]
```

Step one is obvious — count frequencies with a hash map (or `Counter`). Step two is the interesting design choice: how do you pick the top `k`?

## Approach 1 — heap (`O(n log k)`)

Push each `(count, value)` pair through a min-heap of size `k`:

```python
import heapq
from collections import Counter
from typing import List

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """O(n log k) time, O(n) space."""
    freq = Counter(nums)
    # nlargest does the heap dance internally.
    return [val for val, _ in freq.most_common(k)]
```

`Counter.most_common` already uses a heap underneath. Clean, idiomatic, and good enough most of the time.

## Approach 2 — bucket sort (`O(n)`)

Frequencies are bounded by `n`. So we can put each value into a bucket indexed by its count, then read the buckets back from highest to lowest:

```python
from collections import Counter
from typing import List

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """Bucket sort by frequency. O(n) time, O(n) space."""
    freq = Counter(nums)

    # buckets[f] = list of values that appear exactly f times.
    # A value can appear at most n times, so we need n+1 slots (0..n).
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for value, count in freq.items():
        buckets[count].append(value)

    # Walk from the highest-frequency bucket downward.
    result: list[int] = []
    for f in range(len(buckets) - 1, 0, -1):
        for value in buckets[f]:
            result.append(value)
            if len(result) == k:
                return result
    return result
```

This is the classic "**counts are bounded, so use the count as an index**" trick. It is genuinely `O(n)` — no logarithmic factor — and shows up again in problems like Sort Characters by Frequency (LeetCode 451).

## Pattern: frequency counting

Three Python idioms you will use constantly:

```python
from collections import Counter, defaultdict

# 1. Count everything in one line.
freq = Counter(nums)                       # {1: 3, 2: 2, 3: 1}

# 2. Auto-initialise buckets when grouping.
groups = defaultdict(list)
for x in items:
    groups[key_of(x)].append(x)

# 3. Compare two multisets in one line.
def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
```

Frequency maps power: anagram checks, majority element, first unique character, sort by frequency, top-K, and any "how often does X happen?" problem.

# 6. Common pitfalls

A short but high-yield list — most hash table bugs in interviews fall into one of these:

- **Inserting before checking.** In Two Sum, store *after* you check. Otherwise you may pair an element with itself.
- **Unhashable keys.** Lists, sets, and dicts cannot be dict keys. Convert to `tuple(...)` (or `frozenset(...)`) first.
- **Mutating a key after insertion.** If you change the contents of an object that is being used as a key, the hash code becomes stale and you may never find the entry again. Treat keys as immutable.
- **Using `dict` when you only need `set`.** Storing `seen[x] = True` wastes ~30–40% memory compared to `seen.add(x)`.
- **Iterating while mutating.** Adding or deleting keys mid-iteration can raise `RuntimeError`. Iterate over a snapshot — `list(d.items())` — if you must mutate.
- **Forgetting that hash tables are unordered (in spirit).** Python 3.7+ preserves insertion order, but do not rely on key order for algorithm correctness.

# 7. Interview checklist

When you suspect a hash table fits, run through this in your head before writing code:

- **What is the key?** A value, a sorted form, a count tuple, a prefix sum?
- **What is the value?** An index, a count, a list of indices, a boolean?
- **Set or map?** If you only need membership, a set is lighter.
- **Where does the answer come from?** A hit during the scan, or a final pass over the table?
- **Edge cases.** Empty input, single element, all duplicates, all unique, negatives.

Out loud, the script in interviews is roughly:

> "Brute force is `O(n²)` because for each element I compare against every other. I can avoid that by storing a hash map of `<key>` → `<value>` as I go, so each lookup becomes `O(1)`. Total time `O(n)`, space `O(n)`."

# 8. What to practise next

Once these four patterns feel automatic, here is a graded set of follow-ups:

- **Easy.** Contains Duplicate (217), Valid Anagram (242), Intersection of Two Arrays (349).
- **Medium.** Subarray Sum Equals K (560), Longest Consecutive Sequence (128), 4Sum II (454), Insert Delete GetRandom O(1) (380).
- **Hard.** First Missing Positive (41 — uses the input array as a hash table), Substring with Concatenation of All Words (30), Longest Substring with At Most K Distinct Characters (340).

In **part 2 (Two Pointers)** we will look at the other side of the trade-off: when the input is sorted (or can afford to be sorted), two pointers can match a hash table's `O(n)` time while using only `O(1)` extra space. The interesting question is when to pick which — and we will answer it with concrete problems.
