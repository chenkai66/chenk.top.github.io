---
title: "LeetCode Patterns: Binary Search"
date: 2022-06-30 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 5
series_total: 10
lang: en
mathjax: false
description: "A working coder's guide to binary search: three templates, rotated arrays, peak finding, and binary search on the answer with worked traces and figures."
disableNunjucks: true
---

Binary search is the algorithm everyone thinks they understand until they have to write it under interview pressure. The idea is one sentence — *halve the search space at every step* — but the implementation is a minefield of off-by-one errors, infinite loops, and subtly wrong return values. The goal of this article is not to give you yet another recitation of the standard template; it is to give you a **mental model** that explains why each template looks the way it does, and a small toolkit (three templates plus the answer-space pattern) that covers the vast majority of LeetCode problems.

We will build everything from a single invariant: at every iteration the answer, if it exists, lives inside our current search interval. Once that invariant is internalised, the choice between `<` and `<=`, between `right = mid` and `right = mid - 1`, becomes mechanical instead of mysterious.

## 1. The Mental Model

![Binary search halves the window each iteration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-search/fig1_search_space.png)

Binary search is a **shrinking interval** problem. We maintain two pointers `l` and `r` that delimit the slice of the array still under consideration. Each iteration we pick the midpoint `m`, look at `nums[m]`, and use that single observation to throw away half of the remaining interval. The figure above shows the typical behaviour: a sorted array of 16 elements, target `23`, four iterations and the window collapses from 16 to 1.

The whole algorithm rests on one invariant:

> **Invariant.** If the target exists, it is in `[l, r]` (or `[l, r)`, depending on convention) at every iteration.

If you preserve the invariant, the algorithm is correct. If you violate it — for example by setting `r = mid - 1` after observing `nums[mid] == target` while still searching for the leftmost occurrence — you will lose the answer and the rest of the run is garbage.

Why is binary search logarithmic? After $k$ iterations the window contains at most $\lceil n / 2^k \rceil$ elements, so we need $k = \lceil \log_2 n \rceil$ iterations to shrink to one. A billion elements take about 30 comparisons, which is the entire reason databases, file systems and version-control bisect tools rely on it.

**When binary search applies.** The classical condition is "the array is sorted", but the deeper condition is **monotonicity of a predicate**. If you can write a function `feasible(x)` whose answer flips from `False` to `True` (or vice versa) exactly once as `x` increases, you can binary-search for the flip point. The "answer-space" problems later in this article exploit this directly.

## 2. The Standard Template

The standard template finds *any* index where `nums[i] == target` in a sorted array, returning `-1` if none exists. We use the **closed interval** `[l, r]` and the loop condition `l <= r`:

```python
def binary_search(nums, target):
    """Return any index i with nums[i] == target, else -1."""
    l, r = 0, len(nums) - 1
    while l <= r:
        m = l + (r - l) // 2     # avoids overflow in fixed-width languages
        if nums[m] == target:
            return m
        if nums[m] < target:
            l = m + 1            # target lives strictly to the right
        else:
            r = m - 1            # target lives strictly to the left
    return -1
```

There are exactly three things to get right:

1. **Initial bounds.** `r = len(nums) - 1` because `r` is an *inclusive* index.
2. **Loop condition.** `l <= r` because the interval `[l, r]` is non-empty whenever `l <= r`.
3. **Updates.** When `nums[m] != target`, the value at `m` is decisively *not* the answer, so we exclude it via `m + 1` or `m - 1`.

The midpoint formula `l + (r - l) // 2` is mathematically the same as `(l + r) // 2` but it does not overflow when `l + r` exceeds `INT_MAX`. Python's bigints make this academic, but the habit transfers to C++ and Java, where it is not academic at all.

### LeetCode 704 — Binary Search

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            if nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return -1
```

This is literally the template. Edge cases handle themselves: an empty array gives `r = -1`, the loop body never executes, we return `-1`. A single-element array works because `l == r == 0` and `l <= r` is true exactly once.

## 3. Boundary Templates: First / Last Occurrence

Many problems do not just ask whether the target exists; they ask for the **first** or **last** position where some condition holds, or for the position where a new element should be inserted. The standard template is not enough because when `nums[m] == target` we cannot return immediately — there might be earlier (or later) matches.

![Left-bound vs right-bound template trace on duplicates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-search/fig2_left_right_bound.png)

The figure above traces both templates on `nums = [1, 2, 5, 5, 5, 5, 8, 9]` with `target = 5`. They share almost identical code; the entire difference is whether the `<` is strict.

### Left bound: first index `i` with `nums[i] >= target`

```python
def left_bound(nums, target):
    """Half-open [l, r). Returns the leftmost insertion point."""
    l, r = 0, len(nums)
    while l < r:
        m = l + (r - l) // 2
        if nums[m] < target:
            l = m + 1            # m is too small, drop it
        else:
            r = m                # m might be the answer, keep it in [l, r)
    return l                     # l == r at exit
```

Three details are non-negotiable:

- **`r = len(nums)`**, not `len(nums) - 1`. We use the half-open interval `[l, r)` so that the answer "should be inserted at the very end" is representable as `l = len(nums)`.
- **`l < r`** matches the half-open convention: `[l, r)` is empty exactly when `l == r`.
- **`r = m`**, not `m - 1`. When `nums[m] >= target`, position `m` itself is a valid candidate for the answer, so we cannot exclude it.

After the loop, `l` is the leftmost index where `nums[i] >= target`. To turn this into "first occurrence of `target`", check `l < len(nums) and nums[l] == target`.

### Right bound: last index `i` with `nums[i] <= target`

```python
def right_bound(nums, target):
    """Half-open [l, r). Returns the position one past the last match."""
    l, r = 0, len(nums)
    while l < r:
        m = l + (r - l) // 2
        if nums[m] <= target:
            l = m + 1            # keep moving right
        else:
            r = m
    return l - 1                 # last match (or -1 if no match exists)
```

Same skeleton, the only change is `<=` instead of `<`. After the loop, `l` is the leftmost index where `nums[i] > target`, so `l - 1` is the rightmost match. Validate with `l - 1 >= 0 and nums[l - 1] == target`.

### LeetCode 35 — Search Insert Position

This problem is the left-bound template, no modifications:

```python
class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums)
        while l < r:
            m = l + (r - l) // 2
            if nums[m] < target:
                l = m + 1
            else:
                r = m
        return l
```

For `nums = [1, 3, 5, 6]`, `target = 2`: `m = 2, nums[2] = 5 >= 2`, so `r = 2`. Then `m = 1, nums[1] = 3 >= 2`, `r = 1`. Then `m = 0, nums[0] = 1 < 2`, `l = 1`. Now `l == r`, return `1`. Inserting `2` at index `1` gives `[1, 2, 3, 5, 6]`, sorted.

### LeetCode 34 — First and Last Position

Apply both templates and validate:

```python
class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        def left_bound():
            l, r = 0, len(nums)
            while l < r:
                m = l + (r - l) // 2
                if nums[m] < target:
                    l = m + 1
                else:
                    r = m
            return l

        def right_bound():
            l, r = 0, len(nums)
            while l < r:
                m = l + (r - l) // 2
                if nums[m] <= target:
                    l = m + 1
                else:
                    r = m
            return l - 1

        lo = left_bound()
        if lo == len(nums) or nums[lo] != target:
            return [-1, -1]
        return [lo, right_bound()]
```

### LeetCode 278 — First Bad Version

The same left-bound template, but the array is replaced by a black-box predicate `isBadVersion(v)` that flips from `False` to `True` exactly once. We are searching the *predicate*, not an array:

```python
class Solution:
    def firstBadVersion(self, n: int) -> int:
        l, r = 1, n
        while l < r:
            m = l + (r - l) // 2
            if isBadVersion(m):
                r = m            # m might be the first bad
            else:
                l = m + 1        # m is good, first bad is later
        return l
```

This is the cleanest possible illustration that binary search is about monotonic predicates, not sorted arrays.

## 4. Two Interval Conventions

You may have noticed that Sections 2 and 3 use different conventions: the standard search uses closed `[l, r]`, the boundary templates use half-open `[l, r)`. Both are correct; mixing them within one function is what causes infinite loops.

![Closed vs half-open interval semantics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-search/fig5_template_comparison.png)

The figure shows the two conventions running side by side on the same input. The interval bracket beneath each step makes the difference visible: closed `[l, r]` includes the right endpoint, half-open `[l, r)` does not. The pairing rule is mechanical:

| Convention      | Init `r`       | Loop      | Right update      |
| :-------------- | :------------- | :-------- | :---------------- |
| Closed `[l, r]` | `len(nums)-1`  | `l <= r`  | `r = m - 1`       |
| Half-open `[l, r)` | `len(nums)`  | `l < r`   | `r = m`           |

Pick a convention per function and stick to it. The most common bug — the one that hangs your code in the interview — is using `l < r` with `r = m - 1`: the interval can shrink to `[l, l]` and `m = l`, after which `r = l - 1 < l` exits, but only after potentially skipping the answer. The reverse mismatch (`l <= r` with `r = m`) is worse: when `l == r == m` you set `r = m = l` again and loop forever.

## 5. Search in Rotated Sorted Array

So far the array has been sorted. Rotated arrays are not, but they retain a useful structural property: **after splitting at any index, at least one of the two halves is sorted**. That is enough to keep using binary search.

![Rotated array trace, target = 0](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-search/fig3_rotated_array.png)

The figure traces `nums = [4, 5, 6, 7, 0, 1, 2]` searching for `0`. At each step we identify which half is sorted (compare `nums[l]` with `nums[m]`), check whether the target falls inside that sorted half's range, and recurse into the sorted half if it does, otherwise into the other half.

### LeetCode 33 — Search in Rotated Sorted Array

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            if nums[l] <= nums[m]:
                # Left half [l..m] is sorted.
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                # Right half [m..r] is sorted.
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1
```

The single subtlety is `nums[l] <= nums[m]` (note the `<=`). When `l == m` the left "half" is the single element `nums[l]`, which is trivially sorted; the equality covers that case.

## 6. Find Peak Element

A peak is an element strictly greater than both neighbours. The array is *not* sorted, yet binary search still works because we have a monotonic property of a different flavour: the slope.

### LeetCode 162 — Find Peak Element

```python
class Solution:
    def findPeakElement(self, nums: list[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r - l) // 2
            if nums[m] < nums[m + 1]:
                l = m + 1        # ascending at m, peak is to the right
            else:
                r = m            # descending or peak at m, search left
        return l
```

If `nums[m] < nums[m + 1]` we are climbing; since `nums[n] = -inf` by convention, we must eventually descend, which means there is a peak strictly to the right of `m`. If `nums[m] >= nums[m + 1]` we are descending or at the top; since `nums[-1] = -inf`, there is a peak at `m` or to the left. Either way we halve the interval and the invariant — *the interval contains at least one peak* — is preserved.

This problem is the prototype for "binary search on a monotonic property of an unsorted array". Once you spot the local-monotonicity argument, the template is mechanical.

## 7. Binary Search on the Answer

The most powerful generalisation of binary search drops the array entirely. Instead, we search over **the answer space** — the range of possible answer values — using a feasibility check.

![Answer-space binary search: capacity to ship packages](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-search/fig4_answer_space.png)

The pattern works whenever:

1. The answer is a single integer (or real) in a known range $[lo, hi]$.
2. There is a function `feasible(x)` that returns `True` iff `x` is a valid answer.
3. `feasible` is **monotonic**: there is a single threshold $x^*$ such that `feasible(x)` is `False` below it and `True` above it (or vice versa).

The binary search then locates $x^*$ in $\log(hi - lo)$ iterations, each costing one `feasible` call. The figure above shows this for *Capacity To Ship Packages*: the top panel plots days-needed against capacity (a step function that monotonically decreases), the bottom panel shows the bracket `[l, r]` shrinking to the answer.

### LeetCode 1011 — Capacity To Ship Packages Within D Days

We must ship a list of `weights` in order within `D` days; the daily ship capacity is constant and we want the minimum capacity that still meets the deadline.

```python
class Solution:
    def shipWithinDays(self, weights: list[int], D: int) -> int:
        def feasible(cap: int) -> bool:
            days, cur = 1, 0
            for w in weights:
                if cur + w > cap:
                    days += 1
                    cur = w
                    if days > D:
                        return False
                else:
                    cur += w
            return True

        l, r = max(weights), sum(weights)
        while l < r:
            m = l + (r - l) // 2
            if feasible(m):
                r = m
            else:
                l = m + 1
        return l
```

Three design decisions:

- **Lower bound `max(weights)`.** A capacity below the heaviest single package can never ship that package.
- **Upper bound `sum(weights)`.** A capacity equal to the total ships everything in one day, so it is trivially feasible.
- **Monotonicity.** If capacity `c` is feasible, any `c' > c` is also feasible, because more capacity per day can only finish sooner. So the feasible set is an upward-closed interval `[x*, +inf)` and we use the left-bound template to find `x*`.

### LeetCode 875 — Koko Eating Bananas

Same pattern, almost identical code; the feasibility check is the only thing that changes:

```python
class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        def feasible(k: int) -> bool:
            hours = sum((p + k - 1) // k for p in piles)
            return hours <= h

        l, r = 1, max(piles)
        while l < r:
            m = l + (r - l) // 2
            if feasible(m):
                r = m
            else:
                l = m + 1
        return l
```

Once you have written one of these, you have written all of them. *Split Array Largest Sum* (410), *Minimum Number of Days to Make m Bouquets* (1482), *Find K-th Smallest Pair Distance* (719) — they all fit this skeleton, and the entire creative work is the feasibility predicate.

## 8. The Bug Catalogue

These are the only four bugs you will write, in roughly decreasing order of frequency.

**Mismatched interval and update.** Using `l < r` with `r = m - 1`, or `l <= r` with `r = m`. Lock the pairing in your head: closed goes with `<=` and `m ± 1`, half-open goes with `<` and `m`. Anything else is wrong.

**Wrong return value.** Standard template returns `m` (or `-1`). Left-bound returns `l`. Right-bound returns `l - 1`. Forgetting the `- 1` is the classic right-bound mistake. Always validate with a bounds check (`0 <= idx < len(nums) and nums[idx] == target`) before claiming the target exists.

**Forgetting the `+1` trick for `mid` rounding.** Patterns like `l = m` (rather than `l = m + 1`) need *upper* mid: `m = l + (r - l + 1) // 2`. Otherwise when `r = l + 1`, `m = l`, and `l = m` makes no progress — infinite loop. *LeetCode 69 (sqrt)* is the canonical example.

**Integer overflow on `(l + r) // 2`.** Harmless in Python, fatal in C++/Java when `l, r` approach $2^{31}$. The habit of writing `l + (r - l) // 2` costs nothing and saves you the day you switch languages.

A simple debugging recipe: add `print(f"l={l} r={r} m={m} nums[m]={nums[m]}")` inside the loop and verify three things: (a) the interval shrinks every iteration, (b) the invariant "answer is inside `[l, r]`" holds, (c) the value at the returned index is what you expect.

## 9. Picking a Template

A short decision tree covers most problems:

- **Find any matching index, no duplicates relevant.** Standard template, closed interval.
- **First match, insertion point, smallest valid value.** Left-bound template.
- **Last match, largest valid value.** Right-bound template.
- **Predicate over a black box (versions, capacities, speeds).** Left-bound on the answer space; ignore the array entirely.
- **Unsorted but locally monotonic (peaks, rotated arrays).** Standard template, but the comparison logic checks the *structural* property (which half is sorted, which direction climbs) instead of comparing to the target.

When in doubt, prefer the half-open `[l, r)` convention. It generalises better — it accommodates "insert at the end" naturally, it works unchanged when you switch from indices to floats, and the loop condition `l < r` makes the termination criterion explicit.

## 10. Where Binary Search Lives in Practice

Binary search is not just an interview trick. It is one of the half-dozen algorithms that genuinely run the world.

Database indexes — B-trees, B+-trees, LSM-tree sorted runs — are layered binary searches with branching factor higher than two. A `WHERE id = 12345` lookup on a billion-row table is essentially $\log_{B} n \approx 4$ disk reads where $B$ is the page fanout. Without binary search, OLTP would not exist.

`git bisect` is binary search on commit history. Mark a commit as good and a later one as bad, then check out the midpoint, test, and recurse. A thousand-commit regression hunt becomes ten test runs.

Standard library functions — `bisect_left` / `bisect_right` in Python, `lower_bound` / `upper_bound` in C++, `Arrays.binarySearch` in Java — are exactly the templates from sections 2 and 3. The naming is a giveaway: "lower bound" is left-bound, "upper bound" is right-bound minus one.

Learning rate finders, hyperparameter scheduling, and rate limiters all do binary search on the answer when the predicate ("does training diverge?", "does p99 latency exceed SLA?") is expensive but monotonic. Production systems answer "what is the largest QPS we can serve at 99% success?" with the same template that solves *Koko Eating Bananas*.

## 11. Quick Reference

A compact cheat sheet for revision the night before the interview.

```python
# Standard: find any index of target, else -1.
def std(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = l + (r - l) // 2
        if nums[m] == target: return m
        if nums[m] < target:  l = m + 1
        else:                  r = m - 1
    return -1

# Left bound: smallest i with nums[i] >= target.
def lb(nums, target):
    l, r = 0, len(nums)
    while l < r:
        m = l + (r - l) // 2
        if nums[m] < target: l = m + 1
        else:                 r = m
    return l

# Right bound: largest i with nums[i] <= target (or -1 if none).
def rb(nums, target):
    l, r = 0, len(nums)
    while l < r:
        m = l + (r - l) // 2
        if nums[m] <= target: l = m + 1
        else:                  r = m
    return l - 1

# Answer space: smallest x in [lo, hi] with feasible(x) == True.
def search_answer(lo, hi, feasible):
    l, r = lo, hi
    while l < r:
        m = l + (r - l) // 2
        if feasible(m): r = m
        else:            l = m + 1
    return l
```

Five problems to drill until the templates are muscle memory: **704** (standard), **35** (left bound), **34** (both bounds), **278** (predicate), **33** (rotated), **162** (peak), **1011** (answer space), **875** (answer space). If you can write each from scratch in under five minutes, binary search will never trip you up again.
