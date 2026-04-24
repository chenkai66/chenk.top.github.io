---
title: "LeetCode Patterns: Greedy Algorithms"
date: 2024-01-28 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 10
series_total: 10
lang: en
mathjax: false
description: "When does picking the locally best option give the global optimum? A working tour of greedy algorithms with proofs (exchange argument), classic problems (Jump Game, Gas Station, Interval Scheduling, Task Scheduler, Partition Labels), and a clear rule for when greedy fails."
---

Greedy is the algorithm paradigm that feels too good to be true: at every step, take the choice that looks best right now, never look back, and somehow end up at the global optimum. When it works, the code is almost embarrassingly short. When it doesn't, it produces confidently wrong answers — which is why the real skill is not writing greedy code, but recognising **when greedy is allowed**.

This article walks through the structural reason greedy is correct on some problems and broken on others, then applies that lens to seven LeetCode classics: **Jump Game**, **Jump Game II**, **Gas Station**, **Best Time to Buy and Sell Stock II**, **Non-overlapping Intervals**, **Task Scheduler**, and **Partition Labels**.

## Series Navigation

**LeetCode Patterns** (10 articles total):

1. Hash Tables — Two Sum, Longest Consecutive Sequence, Group Anagrams
2. Two Pointers — opposite, fast/slow, sliding window
3. Linked List Operations — reversal, cycle detection, merging
4. Sliding Window
5. Binary Search
6. Binary Tree Traversal & Construction
7. Dynamic Programming Basics — 1D / 2D DP, state transitions
8. Backtracking — permutations, combinations, pruning
9. **→ Greedy Algorithms** ← *current article*
10. Stack & Queue — bracket matching, monotonic stack

## What Makes a Problem "Greedy"?

A **greedy algorithm** builds a solution incrementally by committing, at each step, to the choice that is locally best according to some criterion — and **never undoing it**. That last clause is what separates greedy from backtracking and from DP: there is no `undo`, no memo table of alternative branches.

For this strategy to be correct, the problem must have two structural properties:

- **Greedy choice property.** There exists a locally optimal choice that is part of *some* globally optimal solution. (Equivalently: making this choice never destroys the best possible future.)
- **Optimal substructure.** Once that choice is committed, the remainder of the problem is itself a smaller instance of the same problem, and an optimal solution for it combines with the greedy choice into an optimal solution for the original.

Most problems satisfy optimal substructure. The greedy choice property is the rarer one, and it is the part that needs a proof. **If you cannot prove the greedy choice property, you do not have a greedy algorithm — you have a guess.**

### Greedy vs DP vs Backtracking

| Property | Greedy | Dynamic Programming | Backtracking |
| --- | --- | --- | --- |
| Decision rule | One locally optimal pick | Try all, keep optimum | Try all, prune infeasible |
| Revocability | Never revoked | N/A (computes all states) | Revoked on backtrack |
| Typical time | O(n) or O(n log n) | O(n²)–O(n·k) | Exponential |
| Required structure | Greedy choice + optimal substructure | Optimal substructure + overlapping subproblems | Verifiable solution check |
| Proof burden | Heavy (must prove correctness) | Mechanical (transition + base case) | None (it tries everything) |

The takeaway: **greedy trades correctness work for runtime**. You pay in proof effort and are rewarded with linear or near-linear time.

### When greedy fails: a concrete warning

![Greedy works vs greedy fails on coin change](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/09-greedy-algorithms/fig5_greedy_vs_dp.png)

Coin change is the canonical cautionary tale. With the canonical US coin set `{1, 5, 10, 25}`, "always take the largest coin that fits" produces an optimal answer for every amount. With a slightly different set `{1, 3, 4}` and amount `6`, greedy picks `4 + 1 + 1` (three coins) while the optimal is `3 + 3` (two coins). Same algorithm, same intuition — but the second instance lacks the structural property (technically: the coin set is not a *matroid*) that makes greedy correct.

The lesson: **greedy correctness is a property of the problem, not the algorithm**. Always either prove it or recognise it from a known pattern.

## A Vocabulary of Greedy Strategies

Across LeetCode, greedy solutions tend to fall into a small number of recurring shapes:

- **Sort by end** — interval scheduling, non-overlapping intervals, minimum-arrows.
- **Sort by start** — meeting-room allocation, merging intervals.
- **Sort by ratio / value density** — fractional knapsack, profit-weighted scheduling.
- **Maintain a running extremum** — Jump Game (`max_reach`), Best Time to Buy/Sell Stock (`min_price` so far), Maximum Subarray (`max_ending_here`).
- **Prefix-sum reset** — Gas Station, Maximum Subarray (Kadane), Partition Labels (last-occurrence sweep).
- **Frequency / heap-based scheduling** — Task Scheduler, Reorganize String.
- **Local adjustment in two passes** — Candy, Wiggle Subsequence.

Most of the problems below are instances of one of these patterns. Recognising the pattern is 90% of the solving work.

## How to Prove Greedy Correctness: Exchange Argument

The most common proof technique for greedy is the **exchange argument**:

1. Take any optimal solution `OPT`.
2. Show that you can replace the first non-greedy choice in `OPT` with the greedy choice, producing a new solution `OPT'` that is **no worse**.
3. By induction, `OPT'` can be transformed step by step into the greedy solution without losing optimality.
4. Therefore the greedy solution is itself optimal.

![Exchange argument: swap non-greedy choice for greedy choice](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/09-greedy-algorithms/fig4_exchange_argument.png)

Worked example for interval scheduling: `OPT` schedules some interval `X` first, but the greedy choice `G` is the interval with the earliest finish time. Because `G.end ≤ X.end`, every interval that was compatible with `X` (and came after it) is still compatible with `G`. Swap `X` for `G`; size is unchanged, feasibility is preserved. Repeat for the second pick, the third, and so on. The greedy schedule has the same size as `OPT`, so it is optimal.

The same template proves correctness for Jump Game II, Task Scheduler's lower-bound argument, and many others.

---

## LeetCode 55: Jump Game

> Given a non-negative integer array `nums` where `nums[i]` is the maximum jump length at position `i`, starting from index 0, return `true` if you can reach the last index.

**Examples:** `[2,3,1,1,4] → true`, `[3,2,1,0,4] → false`.

### The greedy insight

You do not need to track *which* jumps you make — only **how far you could possibly have got by now**. Maintain `max_reach`, the farthest index reachable using any sequence of jumps from positions `0..i`. Sweep left to right; if at any point `i > max_reach`, you are stranded.

![Jump Game: maintain max_reach frontier](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/09-greedy-algorithms/fig2_jump_game.png)

The figure compares the two cases. On the left, `max_reach` jumps to 4 at `i=1` and we are done. On the right, the zero at index 3 freezes `max_reach` at 3, and on the next iteration `i=4 > max_reach=3` triggers the failure.

### Why greedy is correct

The proof is almost a tautology: if position `i` is reachable, then every position in `[0, i + nums[i]]` is reachable through it. `max_reach` is the union of those intervals up to step `i`, which is exactly the set of all reachable positions. Local update = exact global state.

### Implementation

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """O(n) time, O(1) space."""
        max_reach = 0
        n = len(nums)
        for i, step in enumerate(nums):
            if i > max_reach:          # current index is unreachable
                return False
            max_reach = max(max_reach, i + step)
            if max_reach >= n - 1:     # early exit: goal already in range
                return True
        return True
```

**Common pitfalls:**

- Forgetting the `i > max_reach` check and updating `max_reach` from an unreachable index. The update would be meaningless (we never had the right to be at `i`), but the algorithm would return wrong answers like `True` for `[0, 1]`.
- Comparing `max_reach == n - 1` instead of `>=`. The reachable frontier can overshoot the goal in a single jump.

---

## LeetCode 45: Jump Game II

> Same setup as LC 55, but you are guaranteed to be able to reach the end. Return the **minimum** number of jumps.

The trick is to think of jumps as **levels of a BFS**: jump 1 reaches a contiguous prefix `[1 .. nums[0]]`; from anywhere in that prefix, jump 2 can reach a wider prefix, etc. We do not need an explicit BFS queue — we just need to know the right boundary of the current level.

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        """
        Greedy BFS in O(n) / O(1).

        - current_end is the right boundary of the jump we are committed to.
        - farthest is the best right boundary reachable from any index inside
          the current level.
        - When i hits current_end we must spend a jump; the next level reaches
          up to farthest.
        """
        jumps = 0
        current_end = 0
        farthest = 0
        for i in range(len(nums) - 1):       # last index needs no jump from
            farthest = max(farthest, i + nums[i])
            if i == current_end:             # exhausted current level
                jumps += 1
                current_end = farthest
                if current_end >= len(nums) - 1:
                    break
        return jumps
```

**Why this is optimal.** Among all positions reachable in `k` jumps, picking the one whose `i + nums[i]` is largest never hurts — by an exchange argument, any optimal sequence using a "shorter" intermediate position can be modified to use the farther-reaching one without increasing the jump count. So always extending `farthest` is safe.

---

## LeetCode 134: Gas Station

> A circular route has `n` gas stations. Station `i` has `gas[i]` litres; driving from `i` to `i+1` costs `cost[i]` litres. Starting with an empty tank, return the index from which you can complete the loop, or `-1` if impossible. The answer is unique.

### Two facts that drive the algorithm

1. **Feasibility test.** A solution exists iff `sum(gas) >= sum(cost)`. If the total deficit is negative, no starting point can compensate.
2. **Localising the start.** Define `Δ[i] = gas[i] - cost[i]`. Run the prefix sum of `Δ` starting from station 0. The optimal start is **one position past the global minimum of this prefix sum**.

![Gas Station: cumulative tank diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/09-greedy-algorithms/fig3_gas_station.png)

The figure plots the running tank for `gas = [1,2,3,4,5]`, `cost = [3,4,5,1,2]`. The prefix sum dips to its minimum after station 2, so starting at station 3 makes the tank non-negative everywhere going forward; the surplus from `Δ[3]+Δ[4] = +6` covers the earlier deficit on the wrap-around.

### Why the greedy reset works

Suppose we attempted to start at index `s` and got stuck at index `j` (`current_tank < 0`). Then **no index in `[s, j]` can be a valid start**: each prefix from `s` was non-negative until `j`, so any later starting index `s' ∈ (s, j]` had even less fuel available at `j`, and would also fail. The next candidate to try is `j + 1`. Each station is visited at most twice across the whole sweep, giving O(n) overall.

### Implementation

```python
from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        """O(n) time, O(1) space."""
        total_tank = 0     # global feasibility check
        current_tank = 0   # tank since the candidate start
        start = 0
        for i in range(len(gas)):
            diff = gas[i] - cost[i]
            total_tank += diff
            current_tank += diff
            if current_tank < 0:
                start = i + 1   # everything in [start, i] is disqualified
                current_tank = 0
        return start if total_tank >= 0 else -1
```

---

## LeetCode 122: Best Time to Buy and Sell Stock II

> Given daily stock prices, you can buy and sell as many times as you like (but never hold more than one share). Maximise total profit.

### The greedy reframe

The optimal profit is `Σ max(0, prices[i] - prices[i-1])`. That is, capture **every** upward day-over-day move and skip every downward one. There is no need to identify peaks and troughs explicitly.

**Why it works.** Decompose any buy-low/sell-high pair `(p_lo, p_hi)` into the telescoping sum of consecutive day differences: `p_hi - p_lo = Σ_{i=lo+1}^{hi} (p_i - p_{i-1})`. Including the negative differences strictly hurts, so the optimal is to take only the positive ones. Each positive difference is independently realisable because you can always sell at the end of an up day and rebuy at the start of the next.

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """O(n) / O(1). Sum every positive day-over-day delta."""
        return sum(
            max(0, prices[i] - prices[i - 1])
            for i in range(1, len(prices))
        )
```

**Pitfall.** This is *not* the algorithm for LC 121 (single transaction) or LC 123 / 188 (limited transactions). Those problems do not have the greedy choice property — multiple buys would interfere — and require DP.

---

## LeetCode 435: Non-overlapping Intervals

> Return the minimum number of intervals to remove so that the rest do not overlap.

This is the classical **activity selection** problem in disguise: maximise the number of mutually compatible intervals, then `answer = n − max_compatible`.

### Strategy

Sort by **end time** ascending. Greedily keep the next interval whose start is `≥` the last kept interval's end.

![Activity selection: pick the earliest-finishing compatible interval](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/09-greedy-algorithms/fig1_interval_scheduling.png)

The figure shows 11 intervals after sorting by end time. The greedy sweep keeps `[1,4)`, `[5,7)`, `[8,11)`, `[12,16)` — four intervals. Every other interval overlaps one of these four and is rejected.

### Why "earliest end" and not "earliest start" or "shortest"

- **Earliest start** can lock in a hugely long interval (e.g. `[1,10)`) and crowd out many short ones — see counterexample `[[1,10],[2,3],[4,5]]`.
- **Shortest** can fail when a short interval sits in the middle of two longer ones it forces you to drop — counterexample `[[1,5],[4,6],[5,10]]`: shortest is `[4,6]` (length 2), picking it kills both neighbours.
- **Earliest end** is provably optimal by the exchange argument from earlier: swapping any first pick for the earliest-ending one preserves feasibility and size.

### Implementation

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """O(n log n) for the sort, O(1) extra space."""
        if not intervals:
            return 0
        intervals.sort(key=lambda iv: iv[1])
        kept = 1
        last_end = intervals[0][1]
        for s, e in intervals[1:]:
            if s >= last_end:    # touching at endpoint counts as non-overlap
                kept += 1
                last_end = e
        return len(intervals) - kept
```

**Pitfall.** Use `>=` (not `>`) for the overlap test — the problem treats `[1,2]` and `[2,3]` as non-overlapping. For LC 452 (arrows / balloon-bursting) the convention flips and you want strict `>`.

---

## LeetCode 621: Task Scheduler

> Tasks are letters; identical tasks must be separated by at least `n` cooldown slots (any letter or `idle`). Return the minimum total length of the schedule.

### The greedy lower-bound formula

Let `f_max` be the frequency of the most common task, and let `k` be the number of tasks tied at that frequency. Then the answer is

```
max(len(tasks), (f_max - 1) * (n + 1) + k)
```

**Intuition.** Lay out the most frequent task in a grid of `n+1` columns. The first `f_max - 1` rows are full (length `(f_max-1)*(n+1)`), and the last row contains exactly `k` tasks (the ones tied for most frequent). Any other tasks can be slotted into the empty positions of these rows without violating cooldown. If everything fits, the formula above gives the schedule length; if there are more tasks than empty slots, no idles are needed at all and the answer is just `len(tasks)`.

### Implementation

```python
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        """O(N) time (N = len(tasks)), O(1) extra space (alphabet size = 26)."""
        counts = Counter(tasks).values()
        f_max = max(counts)
        k = sum(1 for c in counts if c == f_max)   # tasks tied at the max
        return max(len(tasks), (f_max - 1) * (n + 1) + k)
```

**Why no heap is needed.** A heap-based simulation also works (O(N log 26)) and is sometimes asked in interviews. The closed-form above is the elegant version once you internalise the row layout argument.

---

## LeetCode 763: Partition Labels

> Partition the string into as many parts as possible so each letter appears in at most one part. Return the lengths of the parts.

### The greedy insight

For each letter, record its **last occurrence**. Sweep the string maintaining `end = max(end, last[s[i]])`. When `i` hits `end`, every letter seen so far is fully contained in the current segment, so close it.

This is greedy because: at every position we extend the current segment to be the smallest one that can contain all letters seen so far — closing it any earlier would split a letter across segments; closing it any later wastes characters.

### Implementation

```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        """O(n) time, O(1) extra space (alphabet ≤ 26)."""
        last = {ch: i for i, ch in enumerate(s)}
        result = []
        start = 0
        end = 0
        for i, ch in enumerate(s):
            end = max(end, last[ch])
            if i == end:
                result.append(end - start + 1)
                start = i + 1
        return result
```

**Worked example** (`s = "ababcbacadefegdehijhklij"`):

```
last occurrences: a→8, b→5, c→7, d→14, e→15, f→11, g→13, h→19, i→22, j→23, k→20, l→21
i=0..8  end stays at 8 → segment "ababcbaca"   length 9
i=9..15 end = 15        → segment "defegde"     length 7
i=16..23 end = 23        → segment "hijhklij"   length 8
result: [9, 7, 8]
```

---

## Summary

### What I would internalise

1. **Greedy correctness is structural, not aesthetic.** The same `argmax` loop is right on Jump Game and wrong on `{1,3,4}` coin change. Always identify the structural reason — exchange argument, matroid, prefix-sum monotonicity — before trusting a greedy.
2. **Sort by the right key.** End time for scheduling, value/weight for fractional knapsack. The wrong key produces plausible-looking wrong answers.
3. **Maintain an invariant, not a path.** Jump Game's `max_reach`, Gas Station's `current_tank`, Buy/Sell Stock II's day-delta — these are running aggregates, not reconstructions. Greedy's space win comes from this.
4. **The exchange argument is the workhorse proof.** Memorise the template; you will reuse it for every new "earliest X / largest Y" pattern.

### Problem checklist

| Pattern | LeetCode | Greedy key | Proof technique |
| --- | --- | --- | --- |
| Interval scheduling | 435, 452, 56 | sort by end | exchange argument |
| Reachability frontier | 55, 45 | running `max_reach` | inductive invariant |
| Prefix-sum reset | 134, 53 | reset on negative tank | total-sum lemma |
| Day-delta capture | 122 | sum positive deltas | telescoping decomposition |
| Frequency layout | 621 | most-frequent-first formula | grid-fitting argument |
| Last-occurrence sweep | 763 | extend `end` to `last[ch]` | minimal-segment lemma |
| Counterexample (don't!) | 322 | greedy ≠ optimal | requires DP |

### Common mistakes

- Picking a sort key that *feels* right (start time, length) without proving it.
- Forgetting the reachability check (`i > max_reach`) before updating an aggregate.
- Treating LC 122 as a template for LC 121/123/188 — those need DP.
- Confusing "touching at endpoint" rules: `>=` for LC 435, `>` for LC 452.

When in doubt, **try a small adversarial input** (often `n ≤ 5` is enough) before believing your greedy. If you cannot find a counterexample and you can sketch an exchange argument, you are probably safe.

That completes the LeetCode Patterns series. From hash tables to greedy, the recurring meta-skill has been the same: identify the structural property of the problem first, then pick the algorithm whose preconditions match. Good luck.
