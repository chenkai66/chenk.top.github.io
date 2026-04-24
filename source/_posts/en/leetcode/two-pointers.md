---
title: "LeetCode Patterns: Two Pointers"
date: 2024-01-04 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 2
series_total: 10
lang: en
mathjax: true
description: "A deep tour of the two-pointer family: collision, fast/slow, sliding window, and partition pointers. Five problems (Two Sum II, 3Sum, Container With Most Water, Linked List Cycle, Move Zeroes) explained with invariants, complexity proofs, and a decision tree."
---

Hash tables buy you speed by spending memory. Two pointers is the opposite trade: spend a little structural assumption — the array is sorted, the list might have a cycle, the answer lives in a contiguous window — and you get $O(n)$ time with $O(1)$ extra space. The pattern looks trivial in code (two indices and a `while` loop) but it has more failure modes than any other beginner technique: off-by-one indices, infinite loops, missed duplicates, wrong pointer moved on tie. The cure is to think in **invariants** rather than in moves.

This article walks through four flavors of two-pointer thinking — **collision**, **fast/slow**, **sliding window**, and **partition** — and pins each one to a problem that you will see on a real interview: Two Sum II, 3Sum, Container With Most Water, Linked List Cycle, and Move Zeroes.

# Series Navigation

**LeetCode Algorithm Masterclass** (10 parts):

1. Hash Tables — Two Sum, Longest Consecutive, Group Anagrams
2. **→ Two Pointers** — Collision, Fast/Slow, Sliding Window, Partition  ← *you are here*
3. Linked List Operations — Reverse, cycle entry, merge
4. Binary Tree Traversal & Construction — Inorder/Preorder/Postorder, LCA
5. Dynamic Programming Basics — 1D / 2D DP, state transition
6. Backtracking — Permutations, combinations, pruning
7. Binary Search — On indices, on answers, on real numbers
8. Stack & Queue — Monotonic stack, priority queue, deque
9. Graph Algorithms — BFS / DFS, topological sort, union-find
10. Greedy & Bit Manipulation

# Why Two Pointers Works

The brute-force pair search on an array of length $n$ is $\Theta(n^2)$ — every pair gets visited once. Two pointers reduces this to $\Theta(n)$ in the cases where one fact lets you **discard half the remaining work after every comparison**:

| Structural fact you exploit                         | Pattern             | Discarded work after each comparison        |
| --------------------------------------------------- | ------------------- | ------------------------------------------- |
| The array is sorted                                 | Collision           | All pairs through the rejected pointer      |
| One pointer can lap another in a finite cycle       | Fast / slow         | All non-cyclic positions                    |
| Validity is monotone in window length               | Sliding window      | Smaller windows ending at the same index    |
| Each value belongs to one of $k$ regions            | Partition (k-way)   | Already-classified prefix and suffix        |

Each pattern is a different way of saying *"a comparison eliminates a slab of the search space, not just a single candidate."* When that property is **not** present (unsorted array, no cycle, non-monotone validity, no clean partition), two pointers degenerates back to $O(n^2)$ or stops being correct.

![Decision tree: when to reach for which two-pointer pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/two-pointers/fig5_decision_tree.png)

# Pattern 1 — Collision Pointers (Sorted Array)

## The invariant

Two pointers `left` and `right` start at the ends of a sorted array and walk toward each other. The invariant is:

> The answer, if it exists, lies in the closed interval `[left, right]`.

A comparison either confirms the current pair or rules out one of the boundary elements. The pointer pointing at the ruled-out element moves; the other stays.

![Collision pointers walking through Two Sum II](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/two-pointers/fig1_collision_pointers.png)

## LeetCode 167 — Two Sum II

**Problem.** Given a sorted array `numbers` (1-indexed) and a target, return the 1-indexed positions of two numbers whose sum equals the target. Exactly one solution exists.

```python
def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """Return 1-indexed positions of the unique pair summing to target."""
    left, right = 0, len(numbers) - 1
    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]
        if s < target:
            left += 1          # need a larger sum -> drop the smallest value
        else:
            right -= 1         # need a smaller sum -> drop the largest value
    return []                  # unreachable given the problem guarantees
```

**Why moving the *smaller* side on `s < target` is safe.** Suppose `s < target`. Then `numbers[left]` paired with **anything** in `[left+1, right]` is `<= numbers[left] + numbers[right] = s < target`. So no pair starting at `left` can reach the target — we can discard the entire row `left * [left+1 .. right]` of the pair matrix in one move. The symmetric argument handles `s > target`.

**Complexity.** $O(n)$ time, $O(1)$ space. Each iteration moves at least one pointer toward the other; the loop ends when they meet.

## LeetCode 11 — Container With Most Water

**Problem.** Given heights $h_0, h_1, \dots, h_{n-1}$, pick two indices $i < j$ that maximize $\text{area}(i,j) = (j - i) \cdot \min(h_i, h_j)$.

```python
def max_area(height: list[int]) -> int:
    """Largest rectangle bounded by two vertical lines and the x-axis."""
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        h = min(height[left], height[right])
        best = max(best, (right - left) * h)
        # Move the *shorter* side: it is the only way to possibly grow the area
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
```

**Greedy correctness — moving the shorter side is the only useful move.** Fix the current pair $(i, j)$ with $h_i \le h_j$. Any pair $(i, k)$ with $k < j$ has width $(k - i) < (j - i)$ and height $\min(h_i, h_k) \le h_i$, so its area is at most $(j - i) \cdot h_i$ — already the area we just recorded. Therefore no pair anchored at $i$ can beat the current candidate; index $i$ can be retired. The argument is symmetric on the other side.

The point worth memorizing: it is **not** true that the next pair will be larger; it is only true that no pair anchored at the discarded index can be. That is enough to make the greedy correct.

## LeetCode 15 — 3Sum

**Problem.** Return every unique triplet `[a, b, c]` with `a + b + c == 0`.

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """All unique triplets summing to zero."""
    nums.sort()                                # O(n log n), enables collision pointers
    n = len(nums)
    out: list[list[int]] = []

    for i in range(n - 2):
        # Pruning 1: smallest of the remaining values is positive -> sum > 0 always
        if nums[i] > 0:
            break
        # Dedup the first slot
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1
        target = -nums[i]
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                out.append([nums[i], nums[left], nums[right]])
                # Dedup slots 2 and 3 *after* recording the answer
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
    return out
```

**Three deduplication slots, not one.** Sorting groups duplicates; the algorithm has to walk *past* each duplicate group at every level it picks from:

- **First slot.** `if i > 0 and nums[i] == nums[i - 1]: continue` — never pick the same outer value twice.
- **Second slot.** After a hit, advance `left` while the new value equals the just-used value.
- **Third slot.** Same idea on the right.

Skipping any of these slots produces duplicate triplets on inputs like `[-2, 0, 0, 2, 2]`. Skipping all of them and deduplicating with a set works but pays an extra $O(\log n)$ per insert and obscures the algorithm.

**Complexity.** Sorting $O(n \log n)$, outer loop $O(n)$, inner two-pointer scan $O(n)$, total $O(n^2)$ time and $O(\log n)$ stack for the sort.

# Pattern 2 — Fast / Slow Pointers (Floyd's Tortoise and Hare)

## The invariant

Two pointers move along the same sequence at different speeds (canonically slow $= 1$ step, fast $= 2$ steps). The invariant is:

> If a cycle exists, fast catches slow inside the cycle; if no cycle exists, fast reaches the end first.

The mechanism is purely arithmetic: inside a cycle the gap between fast and slow shrinks by exactly $1$ each iteration, so the pointers must meet within $\lambda$ iterations (where $\lambda$ is the cycle length).

![Fast and slow pointers detecting a cycle in a linked list](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/two-pointers/fig2_fast_slow_cycle.png)

## LeetCode 141 — Linked List Cycle

```python
class ListNode:
    def __init__(self, val: int = 0, next: "ListNode | None" = None) -> None:
        self.val = val
        self.next = next


def has_cycle(head: ListNode | None) -> bool:
    """True if the linked list contains a cycle. O(1) extra space."""
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

**Why the meeting is guaranteed.** Let the tail (the path from `head` to the cycle entry) have length $\mu$ and the cycle have length $\lambda$. After both pointers are inside the cycle, the position of `fast` minus the position of `slow` (mod $\lambda$) decreases by exactly $1$ per iteration. Starting from any non-zero offset $d \in \{1, 2, \dots, \lambda - 1\}$, that offset reaches $0$ after $d$ iterations. So total iterations $\le \mu + \lambda$, and the algorithm is $O(n)$ time, $O(1)$ space.

**Why it has to be 1 vs 2, not 1 vs 3 or 1 vs 5.** Any speed pair $(a, b)$ with $a < b$ and $\gcd(b - a, \lambda) = 1$ works. The pair $(1, 2)$ is special because $b - a = 1$ always divides $\lambda$, so correctness holds *regardless* of the cycle length — no number-theoretic gotchas.

## Cycle entry point (LeetCode 142)

Once `slow` and `fast` meet at some node `m` inside the cycle, reset one pointer to `head` and step both by 1. They meet again exactly at the cycle entry.

```python
def detect_cycle(head: ListNode | None) -> ListNode | None:
    """Return the node where the cycle begins, or None."""
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # Phase 2: walk a fresh pointer from head at the same speed as slow
            entry = head
            while entry is not slow:
                entry = entry.next
                slow = slow.next
            return entry
    return None
```

**One-line proof of the entry trick.** When `slow` and `fast` meet, slow has walked $\mu + k$ steps for some $0 \le k < \lambda$, and fast has walked exactly twice that: $2(\mu + k)$. Both end up at the same node, so the difference $\mu + k$ is a multiple of $\lambda$. Therefore walking $\mu$ more steps from `head`, and $\mu$ more from the meeting point, both land on the cycle entry — they meet there.

# Pattern 3 — Sliding Window

## The invariant

A window $[l, r]$ over an array or string is **valid** iff some constraint holds (e.g., no duplicate characters, sum $\ge k$, contains every required character). Sliding window applies when validity is **monotone in window contents**:

> If $[l, r]$ is invalid, every window $[l, r']$ with $r' \ge r$ that still contains $[l, r]$ is also invalid (or vice versa).

That monotonicity is what lets `left` advance without ever going back. Each index is touched at most twice (once by `right`, once by `left`), so the total work is $O(n)$.

![Sliding window expanding right and shrinking left when a duplicate appears](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/two-pointers/fig3_sliding_window.png)

## LeetCode 3 — Longest Substring Without Repeating Characters

```python
def length_of_longest_substring(s: str) -> int:
    """Length of the longest substring of s with all-distinct characters."""
    last_seen: dict[str, int] = {}     # char -> most recent index in s
    left = 0
    best = 0
    for right, ch in enumerate(s):
        # If we've seen this char *inside* the current window, jump left past it.
        # max(...) is essential: a previous duplicate may have already pushed
        # left further right than this character's last index.
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

**The `>= left` guard, explained on `"abba"`.** At `right = 2` (second `b`), we set `left = 2`. At `right = 3` (second `a`), `last_seen['a'] = 0` is *outside* the current window `[2, 3]`. Without the guard, `left` would jump back to `1`, breaking the never-go-back invariant and reporting the wrong answer.

**Complexity.** $O(n)$ time (each character is added once and removed at most once), $O(\min(n, |\Sigma|))$ space for the map.

# Pattern 4 — Partition Pointers (k-way)

## The invariant

Maintain $k$ regions in the same array. With $k = 3$ (Dutch National Flag), three pointers `lo`, `mid`, `hi` carve the array into:

> `[0, lo)` are settled "low" values; `[lo, mid)` are settled "middle" values; `[mid, hi]` are unprocessed; `(hi, n)` are settled "high" values.

Each step processes `arr[mid]` and either swaps it into the low region, leaves it, or swaps it into the high region — preserving the invariant.

![Dutch National Flag: three pointers carve the array into 0/1/2 regions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/two-pointers/fig4_partition_pointers.png)

## LeetCode 75 — Sort Colors (Dutch National Flag)

```python
def sort_colors(nums: list[int]) -> None:
    """Sort an array of 0s, 1s, and 2s in-place in O(n) / O(1)."""
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        v = nums[mid]
        if v == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1                  # the swapped-in value came from [lo, mid), already a 1
        elif v == 2:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1                   # do NOT advance mid: swapped-in value is unprocessed
        else:                          # v == 1
            mid += 1
```

**The asymmetry between the 0 and 2 branches** is the single most-missed detail. When you swap into the **low** region, the displaced value used to live in `[lo, mid)`, which by invariant means it was already a `1` — safe to step past. When you swap into the **high** region, the displaced value came from `[mid+1, hi]`, which is **unclassified** — you must look at it next iteration, so `mid` does not advance.

## LeetCode 283 — Move Zeroes (k = 2 partition)

The two-region degenerate case: `[0, write)` are the non-zero values in original order, `[write, n)` is everything else.

```python
def move_zeroes(nums: list[int]) -> None:
    """Move every 0 to the end, preserving the relative order of non-zero values."""
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
```

The swap (instead of plain assignment) is what keeps the second region full of zeros: every value pulled forward leaves a zero behind because positions `[write, read)` are guaranteed to be zero by induction.

# Hash Table vs Two Pointers — When to Sort?

The table everyone wants but rarely sees written down:

| Question                                 | Hash table                | Two pointers                                |
| ---------------------------------------- | ------------------------- | ------------------------------------------- |
| Find a pair summing to $T$ (any indices) | $O(n)$ time, $O(n)$ space | $O(n \log n)$ + $O(n)$, $O(1)$ extra        |
| Find a pair, must return original indices | Hash table wins           | Sort with companion index array, $O(n)$ extra |
| Find all pairs / triplets / quadruplets   | Awkward dedup             | $O(n^{k-1})$, clean dedup by skipping runs |
| Detect a cycle in a linked list          | $O(n)$ extra              | $O(1)$ — Floyd                              |
| Longest valid contiguous range           | Often unnecessary         | Sliding window is the canonical answer     |

**Decision rule.** Use a hash table when you need original indices, or when the array is unsorted and sorting is forbidden. Use two pointers when the structure is already sorted (or sortable), when the answer is contiguous, or when $O(1)$ extra space matters.

# Five Failure Modes (and How to Avoid Them)

These are the bugs that crash interviews — pattern recognition is easy, getting the loop right is the hard part.

**1. Wrong pointer moves on tie.** On Container With Most Water, moving the *taller* side keeps the height capped at the same value but shrinks the width. Move the shorter side, or either on a tie.

**2. Forgetting to advance after a hit.** In 3Sum, after recording a triplet you must move both `left` and `right`; otherwise the same pair gets re-tested forever. Pair this with the dedup `while` loops.

**3. `left` regressing in a sliding window.** `left = last_seen[ch] + 1` without a `max(left, ...)` guard breaks correctness on inputs like `"abba"`. Either guard with `max`, or explicitly delete characters as `left` advances.

**4. Off-by-one on `fast.next`.** `fast.next.next` requires `fast and fast.next` to be non-`None`. Checking only `fast` lets you crash on a 2-node list. Always test both.

**5. Mid advancing on the high-swap branch in Dutch National Flag.** The most common partition-pointer bug. Memorize: *low-swap advances mid, high-swap does not.*

# Interview Communication

A short script that gets you 80% of the credit on any two-pointer question:

> "I notice the array is sorted / we need a contiguous window / this is a linked-list cycle problem, so I'd reach for **\<pattern\>**. The brute force is $O(n^2)$; two pointers gives $O(n)$ because every comparison rules out a whole row/column/prefix. The invariant I'll maintain is **\<state the invariant\>**. Edge cases I'll handle up front: empty input, single element, all-equal values."

Stating the **invariant** before coding is the single best signal that you understand the technique rather than having memorized templates. Interviewers reliably reward it.

# Practice Set (Ordered by Difficulty)

| LeetCode # | Problem                                       | Pattern             | Difficulty |
| ---------- | --------------------------------------------- | ------------------- | ---------- |
| 167        | Two Sum II – Input Array Is Sorted            | Collision           | Easy       |
| 283        | Move Zeroes                                   | Partition (k=2)     | Easy       |
| 141        | Linked List Cycle                             | Fast / slow         | Easy       |
| 11         | Container With Most Water                     | Collision (greedy)  | Medium     |
| 15         | 3Sum                                          | Sort + collision    | Medium     |
| 75         | Sort Colors                                   | Partition (k=3)     | Medium     |
| 142        | Linked List Cycle II                          | Fast / slow + math  | Medium     |
| 3          | Longest Substring Without Repeating           | Sliding window      | Medium     |
| 209        | Minimum Size Subarray Sum                     | Sliding window      | Medium     |
| 76         | Minimum Window Substring                      | Sliding window      | Hard       |

# Further Reading

- Robert Floyd, *Nondeterministic Algorithms*, J. ACM 14(4), 1967 — the original cycle-detection paper.
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*, Chapter 8 — the partitioning material that underpins quicksort and Dutch National Flag.
- Donald Knuth, *TAOCP* Vol. 2, §3.1 — the rho algorithm for cycle finding (the original "tortoise and hare" application).

# What's Next

In **Part 3 — Linked List Operations**, the same fast/slow pointer machinery shows up in disguise: finding the middle, splitting for merge sort, and reversing in groups of $k$. The collision-pointer mindset transfers directly to palindrome detection on a list. The trick that makes everything click is the **dummy head node** — wait for it.
