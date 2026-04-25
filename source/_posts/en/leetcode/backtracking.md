---
title: "LeetCode Patterns: Backtracking Algorithms"
date: 2022-08-14 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 8
series_total: 10
lang: en
mathjax: false
description: "Master the universal backtracking template through six classic problems: Permutations, Combinations, Subsets, Word Search, N-Queens, and Sudoku Solver. Learn the choose -> recurse -> un-choose pattern and the pruning techniques that make it fast."
disableNunjucks: true
---

Backtracking is the algorithm you reach for whenever a problem asks you to *enumerate* something — every permutation, every subset, every legal board, every path through a grid. It is brute force with a brain: you build a candidate solution one decision at a time, abandon it the moment a constraint says "this cannot work", and undo your last move so the next branch sees a clean slate. The whole technique fits in three lines:

> **choose** -> **recurse** -> **un-choose**

That single rhythm solves Permutations, Combinations, Subsets, Word Search, N-Queens, and Sudoku, and it is the same template you will use on roughly 90% of "find all..." problems on LeetCode. This article walks you through the template, then applies it to six canonical problems with full implementations, recursion-tree diagrams, complexity proofs, pruning tactics, and a debugging checklist for the bugs you will inevitably hit the first few times you write it.

# Series Navigation

**LeetCode Algorithm Masterclass Series** (10 Parts):

1. Hash Tables (Two Sum, Longest Consecutive, Group Anagrams)
2. Two Pointers (Collision pointers, fast/slow, sliding window)
3. Linked List Operations (Reverse, cycle detection, merge)
4. Binary Tree Traversal & Recursion (Inorder/Preorder/Postorder, LCA)
5. Dynamic Programming Intro (1D/2D DP, state transitions)
6. Binary Search Advanced (Integer/real binary search, answer search)
7. Dynamic Programming continued
8. **-> Backtracking Algorithms** (Permutations, combinations, pruning) <- *You are here*
9. Stack & Queue (Monotonic stack, priority queue, deque)
10. Greedy & Bit Manipulation (Greedy strategies, bitwise tricks)

# What backtracking actually is

Backtracking is depth-first search over an *implicit* tree of partial solutions. Each node is a partial answer (the `path` so far). Each edge is a single decision (add element `x`, place a queen on column `c`, write digit `d` in this cell). When you reach a leaf that satisfies the goal, you record it. When you reach a node that violates a constraint, you stop descending and back up. The "back" in backtracking is the explicit *undoing* of the last decision so that the algorithm can try a sibling branch with the state it had before.

That last point is what separates backtracking from a plain DFS: a graph traversal does not need to undo anything because each node is visited once, but a search over partial solutions reuses the same `path` and `used[]` data structures across millions of branches and would corrupt them without explicit restoration.

When to reach for backtracking:

- The problem says "return **all** ..." (permutations, combinations, subsets, partitions, parenthesizations).
- It is a constraint-satisfaction problem (N-Queens, Sudoku, graph coloring).
- The search space is exponential but most of it can be eliminated by checking partial constraints early.
- You need *one* solution and a witness, and a polynomial-time greedy / DP doesn't apply.

When **not** to reach for it:

- You only need a count or an optimal value, and the subproblems overlap -> use DP.
- The search space is small enough that you can enumerate states iteratively.
- The problem reduces to BFS shortest path or a known graph algorithm.

# The universal template

Every backtracking solution in this article is a thin specialization of one template. Memorize the shape; specialize the four slots.

![The backtracking template — choose, recurse, un-choose](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/backtracking/fig1_template.png)

```python
def backtrack(path, state):
    # 1. Goal: is this a complete solution? Save a *copy* and return.
    if is_goal(path, state):
        result.append(path[:])          # path[:] makes a snapshot
        return

    # 2..4. Iterate over the available choices at this node.
    for choice in choices(state):
        if not is_valid(choice, path, state):
            continue                    # prune

        path.append(choice)             # 2. choose
        update(state, choice)
        backtrack(path, state)          # 3. recurse
        path.pop()                      # 4. un-choose
        undo(state, choice)
```

Four things to get right and you are done:

1. **`is_goal`** — when is `path` a complete answer?
2. **`choices`** — what can we do from this node? (Often controlled by a `start` index, a `used[]` array, or both.)
3. **`is_valid`** — what makes a choice illegal in this branch? (Prune as early as you can.)
4. **`update` / `undo`** — every mutation made on the way down must be reversed on the way up. This is the bug that catches everyone the first time.

A subtle point about `path[:]`: `result.append(path)` would store a *reference* to the same list you are about to keep mutating. By the time recursion finishes, every entry in `result` would point at an empty list. Use `path[:]`, `path.copy()`, or `list(path)`.

# Backtracking vs DFS — the one-line distinction

DFS *traverses*: it visits each node once, never modifies what it walked over, and the recursion stack alone holds enough state. Backtracking *constructs*: the same `path` is mutated all the way down and restored all the way up, so the algorithm can reuse it across exponentially many branches without copying. Backtracking uses DFS as its motion, but adds the choose/un-choose discipline.

# LeetCode 46 — Permutations

> Given an array `nums` of distinct integers, return all possible permutations.

Example: `nums = [1,2,3]` -> `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`.

## How the template specializes

- **Goal**: `len(path) == len(nums)`.
- **Choices**: any number that hasn't been used yet.
- **Validity**: `not used[i]`.
- **State**: a `used[]` boolean array, flipped True on the way down and False on the way up.

The recursion tree for `[1,2,3]` is a perfect 3-2-1 fan-out — three first choices, two second choices, one final element forced — and produces exactly `3! = 6` leaves.

![Permutations recursion tree for [1,2,3]](./backtracking/fig2_permutations_tree.png)

## Implementation

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result: List[List[int]] = []
        path: List[int] = []
        used = [False] * n

        def backtrack() -> None:
            if len(path) == n:                  # goal
                result.append(path[:])          # snapshot, never the live list
                return
            for i in range(n):
                if used[i]:                     # prune: already in path
                    continue
                used[i] = True                  # choose
                path.append(nums[i])
                backtrack()                     # recurse
                path.pop()                      # un-choose
                used[i] = False

        backtrack()
        return result
```

## Complexity

There are `n!` permutations and each one costs `O(n)` to copy into the result, so the total time is `O(n * n!)`. The recursion stack and the `path` array each take `O(n)` auxiliary space (the output is not counted).

Why `n!`? At depth 0 there are `n` legal choices, at depth 1 there are `n-1`, and so on, so the leaf count is `n * (n-1) * ... * 1 = n!`.

# LeetCode 39 — Combinations (Combination Sum)

> Given an array of distinct positive integers `candidates` and a target `target`, return all unique combinations whose elements sum to `target`. The same number may be picked unlimited times.

Example: `candidates = [2,3,6,7], target = 7` -> `[[2,2,3], [7]]`.

## How the template specializes

- **Goal**: `remain == 0`.
- **Constraint pruning**: if `remain < 0`, abandon this branch immediately.
- **Avoiding duplicates**: this is the hard part. `[2,3]` and `[3,2]` are the *same* combination — order does not matter. The standard fix is a `start` index that says "from this branch on, you can only pick candidates at index `>= start`". That fixes a canonical ordering on every saved combination, so each one is generated exactly once.
- **Reuse**: when we recurse we pass `i` (not `i + 1`) so the same number can be picked again.

The decision tree for the example below shows pruning in action: every red `X` is a branch killed by `remain < 0`. Notice how aggressively the tree shrinks compared to enumerating all `4^k` candidate strings.

![Combination Sum decision tree with pruning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/backtracking/fig3_combinations_pruning.png)

## Implementation

```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remain: int) -> None:
            if remain == 0:                     # goal
                result.append(path[:])
                return
            if remain < 0:                      # prune: overshot
                return
            for i in range(start, len(candidates)):
                path.append(candidates[i])      # choose
                backtrack(i, remain - candidates[i])  # recurse, i (not i+1) -> reuse
                path.pop()                      # un-choose

        backtrack(0, target)
        return result
```

## Sort-then-break optimization

If you sort `candidates` first, the inner loop can `break` (not just `continue`) as soon as `candidates[i] > remain`, because every later candidate is even larger. On adversarial inputs this is a huge speedup.

```python
candidates.sort()
for i in range(start, len(candidates)):
    if candidates[i] > remain:
        break                                   # all later candidates are >= this one
    path.append(candidates[i])
    backtrack(i, remain - candidates[i])
    path.pop()
```

## Complexity

A clean upper bound is hard to write because reuse means paths can be longer than the input. With smallest candidate `m`, paths have length at most `target / m`, and at each step there are at most `n` choices, giving `O(n^(target/m))` worst case. Pruning shaves this down dramatically in practice. Space is `O(target / m)` for the recursion depth.

# LeetCode 78 — Subsets

> Given an array `nums` of unique integers, return all `2^n` subsets (the power set).

Example: `nums = [1,2,3]` -> `[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]`.

## How the template specializes

- **Goal**: there isn't a single goal — *every* node of the recursion tree is a valid subset, including the root (empty set). Save on entry to every call.
- **Choices**: include `nums[i]` for `i >= start`, then recurse with `start = i + 1`.

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            result.append(path[:])              # every node is a valid subset
            for i in range(start, len(nums)):
                path.append(nums[i])            # choose
                backtrack(i + 1)                # recurse: each element used at most once
                path.pop()                      # un-choose

        backtrack(0)
        return result
```

A second, perhaps more intuitive, formulation is "for each element, include or skip":

```python
def subsets_binary(nums):
    result, path = [], []
    n = len(nums)
    def backtrack(i: int) -> None:
        if i == n:
            result.append(path[:])
            return
        backtrack(i + 1)                        # skip nums[i]
        path.append(nums[i])                    # take nums[i]
        backtrack(i + 1)
        path.pop()
    backtrack(0)
    return result
```

Both run in `O(n * 2^n)` time and `O(n)` auxiliary space.

# LeetCode 79 — Word Search

> Given an `m x n` grid of characters and a string `word`, return True if `word` can be formed by a sequence of horizontally or vertically adjacent cells, with each cell used at most once.

This is backtracking on a grid. The choice at each step is "which neighbor do I move to next", the constraint is that the cell must match the next character of the word and not already be on my path, and the goal is matching the final character.

The cleanest way to track "is this cell on my current path" is to mutate the grid in place — write a sentinel like `'#'` when entering a cell, restore the original character when leaving. That is the choose/un-choose discipline applied to a 2D state.

```python
from typing import List

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])

        def dfs(r: int, c: int, k: int) -> bool:
            if k == len(word):                  # goal: matched the whole word
                return True
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[k]:
                return False                    # off-grid or mismatch -> prune

            tmp, board[r][c] = board[r][c], '#'  # choose: mark visited
            found = (dfs(r + 1, c, k + 1) or
                     dfs(r - 1, c, k + 1) or
                     dfs(r, c + 1, k + 1) or
                     dfs(r, c - 1, k + 1))
            board[r][c] = tmp                    # un-choose: restore
            return found

        for r in range(m):
            for c in range(n):
                if dfs(r, c, 0):
                    return True
        return False
```

Time is `O(m * n * 4^L)` where `L = len(word)` (each path can branch four ways and is at most `L` long, with `m * n` starting positions). The early character-mismatch check prunes almost all of that in practice. Space is `O(L)` for the recursion.

# LeetCode 51 — N-Queens

> Place `n` queens on an `n x n` board so that no two attack each other (no shared row, column, or diagonal). Return every solution.

The clever piece here is the diagonal trick. For any cell `(r, c)`:

- All cells on its `↘` (main) diagonal share the same value of `r - c`.
- All cells on its `↙` (anti) diagonal share the same value of `r + c`.

That means we can encode "is this diagonal taken?" as O(1) set membership instead of scanning the board. Combined with placing one queen per row (which automatically prevents row conflicts), we get three O(1) constraint sets that prune the search to a fraction of the naive `n^n` brute force.

![4-Queens: a valid solution and the three constraint sets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/backtracking/fig4_nqueens_board.png)

```python
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result: List[List[str]] = []
        board = [['.'] * n for _ in range(n)]
        cols: set[int] = set()
        diag1: set[int] = set()                 # r - c (main diagonals)
        diag2: set[int] = set()                 # r + c (anti diagonals)

        def backtrack(row: int) -> None:
            if row == n:
                result.append([''.join(r) for r in board])
                return
            for col in range(n):
                if col in cols or (row - col) in diag1 or (row + col) in diag2:
                    continue                    # prune: column or diagonal taken
                board[row][col] = 'Q'           # choose
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                backtrack(row + 1)              # recurse
                board[row][col] = '.'           # un-choose
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)

        backtrack(0)
        return result
```

The asymptotic complexity is famously hard to write in closed form. A loose upper bound is `O(n!)` because each row eliminates at least one column for the next row; a much tighter empirical bound is roughly `O(n!) / (n-ish factor)` thanks to diagonal pruning. The number of solutions itself grows roughly like `n! / (n^c)` and is non-trivial to compute — see OEIS A000170.

# LeetCode 37 — Sudoku Solver

> Solve a Sudoku puzzle in place. The rules: digits 1-9 must appear exactly once per row, column, and 3x3 box.

Sudoku is the cleanest pedagogical example of "backtracking with strong pruning". Most cells have only one or two legal candidates given the current board, so although the worst-case search tree is astronomical, real puzzles finish in milliseconds.

The pattern is "find the next empty cell, try every legal digit, recurse". On a dead end, blank the cell and try the next digit; if no digit works, return False so the caller can blank its cell and try the next digit, all the way up.

![One Sudoku step: candidates eliminated by row, column, and box](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/backtracking/fig5_sudoku_step.png)

```python
from typing import List

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        rows  = [set() for _ in range(9)]
        cols  = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        empty: List[tuple[int, int]] = []

        # Pre-compute occupied digits and the list of empty cells.
        for r in range(9):
            for c in range(9):
                v = board[r][c]
                if v == '.':
                    empty.append((r, c))
                else:
                    rows[r].add(v); cols[c].add(v); boxes[(r // 3) * 3 + c // 3].add(v)

        def backtrack(k: int) -> bool:
            if k == len(empty):                 # goal: all empties filled
                return True
            r, c = empty[k]
            b = (r // 3) * 3 + c // 3
            for d in '123456789':
                if d in rows[r] or d in cols[c] or d in boxes[b]:
                    continue                    # prune
                board[r][c] = d                 # choose
                rows[r].add(d); cols[c].add(d); boxes[b].add(d)
                if backtrack(k + 1):            # recurse
                    return True                 # found one; stop
                board[r][c] = '.'               # un-choose
                rows[r].remove(d); cols[c].remove(d); boxes[b].remove(d)
            return False                        # no digit worked here -> backtrack

        backtrack(0)
```

Two things worth pointing out:

1. We return `True` as soon as a solution is found, which short-circuits the whole tree. Backtracking does not have to enumerate everything when the question is "find one".
2. We pre-compute `empty[]` instead of scanning for the next blank inside `backtrack`. This turns each step from O(81) into O(1) cell lookup. A further speedup is "MRV" — pick the empty cell with the *fewest* legal candidates next, which dramatically prunes the branching factor near the leaves.

Worst-case complexity is exponential in the number of empty cells; pragmatically, a 9x9 Sudoku is solved in milliseconds because the constraints prune the tree so aggressively.

# Pruning techniques that matter

Constraint pruning (the `is_valid` check before recursing) is what separates a brute-force enumeration from a useful algorithm. Five techniques recur:

1. **Constraint pruning** — `if not is_valid(choice): continue`. This is the bread and butter; do it *before* recursing, never after.
2. **Bound pruning** — for sum-target problems, abort once `remain < 0` or once `remain` cannot be reached even if you take every remaining element.
3. **Sort + early break** — sort the input; once the current element exceeds the remaining budget, every subsequent one will too, so `break` instead of `continue`.
4. **Duplicate skipping** — sort, and inside the loop skip indices where `i > start and nums[i] == nums[i-1]`. This is the standard fix for "Combination Sum II" and "Permutations II".
5. **Symmetry breaking** — for problems with rotational symmetry (e.g. N-Queens with `n >= 4` solutions), restrict the first choice to half the board and mirror the rest.

# Complexity at a glance

| Problem            | Time                    | Space (aux) | Why                                   |
|--------------------|-------------------------|-------------|---------------------------------------|
| Permutations       | `O(n * n!)`             | `O(n)`      | `n!` leaves, `O(n)` to copy each      |
| Combination Sum    | `O(n^(target/m))` worst | `O(target)` | `m = min(candidates)`; pruned heavily |
| Subsets            | `O(n * 2^n)`            | `O(n)`      | `2^n` subsets, `O(n)` to copy         |
| Word Search        | `O(m*n * 4^L)`          | `O(L)`      | start anywhere, branch 4              |
| N-Queens           | ~ `O(n!)` worst         | `O(n)`      | row-by-row with diagonal pruning      |
| Sudoku             | exponential worst       | `O(81)`     | strong pruning makes it fast in practice |

# Bugs you will hit (and how to fix them)

**Bug 1 — saving a reference instead of a copy.**

```python
result.append(path)        # WRONG — every entry will end up empty
result.append(path[:])     # right
```

**Bug 2 — forgetting to un-choose.**

```python
path.append(c); backtrack(); # missing path.pop() -> the next iteration starts contaminated
```

**Bug 3 — un-choose only restores half the state.** If you mutated `used[i]`, the board, or three diagonal sets, you have to undo *all* of them. Make it a habit: every line below the recursive call mirrors a line above it.

**Bug 4 — checking the constraint after appending.** Always check `is_valid(choice)` *before* `path.append(choice)`. Otherwise pruning saves you nothing.

**Bug 5 — wrong loop start in combinations.** Combinations need a `start` parameter; permutations need `used[]`. Mixing them up either generates duplicates or misses solutions.

**Bug 6 — re-scanning instead of indexing in Sudoku.** The naive solver finds the next empty cell with a nested loop on every call. Pre-compute the list of empties once.

# 10 frequently asked questions

**Q1. Backtracking vs DFS — what is the actual difference?** DFS visits each node once and never modifies the structure; backtracking reuses the same `path`/state across exponentially many branches and *must* restore that state on the way up. Backtracking uses DFS as its motion.

**Q2. Backtracking vs DP — when do I pick which?** If you need *all* solutions, backtracking. If you need a count or an optimum and the subproblems overlap, DP. Sometimes both work and DP is faster (e.g. counting paths instead of listing them).

**Q3. Why `path[:]` and not `path`?** `path` is a reference to the list you keep mutating. Without the copy, `result` ends up as a list of references all pointing at the same eventually-empty list.

**Q4. How do I avoid duplicates when the input has duplicates?** Sort the input, then in the loop write `if i > start and nums[i] == nums[i-1]: continue`. This skips equal siblings at the same depth without skipping equal elements that appear at different depths.

**Q5. Combinations vs permutations — what changes in the code?** Permutations track `used[]` and let the loop start at 0 each call; combinations use a `start` index that fixes a canonical order on the chosen elements.

**Q6. My solution is correct but TLEs. Where to look?** First, are you pruning *before* recursing, not after? Second, are your validity checks `O(1)` (sets/arrays) instead of `O(n)` (`in path`)? Third, can you sort the input and `break` instead of `continue`?

**Q7. Can I make backtracking iterative?** Yes — replace the call stack with an explicit stack of `(path, state)` pairs. The recursive form is almost always cleaner; the iterative form helps only when you are hitting a recursion-depth limit.

**Q8. How do I find just *one* solution efficiently?** Have the recursive function return a `bool` (or the solution itself). Return `True` as soon as a goal is reached, and propagate that up the call chain so the caller stops iterating.

**Q9. Sudoku takes forever on hard puzzles — what should I add?** Use MRV (most-constrained variable): at each step, fill the empty cell with the *fewest* legal candidates. Combined with the `rows/cols/boxes` sets above, this cuts the branching factor near the bottom of the tree dramatically.

**Q10. What is the right order to learn these?** Permutations -> Subsets -> Combination Sum -> Word Search -> N-Queens -> Sudoku. Each one adds exactly one new wrinkle (used[] -> save-every-node -> start index + reuse -> grid traversal -> multi-set constraints -> early termination + heuristics).

# Summary

Backtracking is a single rhythm — **choose, recurse, un-choose** — wrapped around a constraint check. The whole skill is recognizing the four slots in the template (goal, choices, validity, state mutation) and filling them in for the problem in front of you. Permutations show off the `used[]` pattern. Subsets show that "every node is a solution". Combinations introduce the `start` index for canonical ordering. Word Search transplants the template onto a 2D grid by mutating cells in place. N-Queens demonstrates the diagonal-set encoding. Sudoku shows how aggressive constraint propagation turns a 10^81 search space into something that finishes in milliseconds.

If you internalize one thing from this article, make it this: every line of mutation on the way down the tree must be matched by an undoing line on the way up. Get that invariant right and the rest is just bookkeeping.
