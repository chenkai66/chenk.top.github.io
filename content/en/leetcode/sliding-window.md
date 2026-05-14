---
title: "LeetCode (4): Patterns: Sliding Window Technique"
date: 2022-06-15 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: leetcode
series_order: 4
series_total: 10
lang: en
mathjax: false
description: "Master fixed-size and variable-size sliding window patterns. Solve Maximum Sum Subarray, Longest Substring Without Repeating Characters, Minimum Window Substring, and Permutation in String."
disableNunjucks: true
translationKey: "leetcode-4"
---
![Chapter concept illustration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/04-sliding-window/illustration_1.png)

If you have ever caught yourself writing a double `for` loop to inspect every contiguous subarray, **sliding window** is probably the optimisation you are missing. It turns an $O(nk)$ or $O(n^2)$ scan into a single linear pass by *reusing the work* it has already done. This article walks through the technique from first principles, then drills four canonical LeetCode problems plus a monotonic-deque variant.

---

## The Idea in One Picture

A sliding window is a contiguous range `[left, right]` over an array or string. Instead of recomputing everything when the range moves, we **add the element entering on the right** and **remove the element leaving on the left**. Each element is touched at most twice, so the total cost is $O(n)$.

There are two flavours:

- **Fixed-size window** — width is a constant `k` and both pointers advance together.
- **Variable-size window** — `right` expands to grow the window, `left` contracts to repair it when an invariant breaks.

A picture is worth a thousand words. Below, the green cells are the live window and the right column shows the incremental sum update.

![Fixed-size window: max sum subarray of size k](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/sliding-window/fig1_fixed_window.png)

The orange cell is *entering* the window; the pink one is *leaving*. Notice the right column: each step is one subtraction and one addition, never a fresh `sum()` call. That is the entire trick.

## Fixed-Size Window — Maximum Sum Subarray of Size K

**Problem.** Given an array `arr` and an integer `k`, find the maximum sum of any contiguous subarray of length `k`.

**Naive approach.** Sum every length-`k` slice — $O(nk)$.

**Sliding window.** Compute the first window's sum, then slide one position at a time, subtracting the element that leaves and adding the element that enters.

```python
def max_sum_subarray(arr, k):
    """Return the maximum sum of any length-k contiguous subarray."""
    if len(arr) < k:
        return None

    window_sum = sum(arr[:k])     # first window
    best = window_sum

    for right in range(k, len(arr)):
        # entering: arr[right]    leaving: arr[right - k]
        window_sum += arr[right] - arr[right - k]
        best = max(best, window_sum)

    return best
```

**Java**

```java
public int maxSumSubarray(int[] arr, int k) {
    if (arr.length < k) return Integer.MIN_VALUE;
    int windowSum = 0;
    for (int i = 0; i < k; i++) windowSum += arr[i];
    int best = windowSum;
    for (int right = k; right < arr.length; right++) {
        windowSum += arr[right] - arr[right - k];
        best = Math.max(best, windowSum);
    }
    return best;
}
```

**Complexity.** Time $O(n)$ — one addition and one subtraction per slide. Space $O(1)$.

**Why it works.** The window's sum is a function of its multiset of elements. When the window shifts by one, the multiset changes by exactly one removal and one insertion, so we only need to apply that delta. This *incremental update* generalises to running counts, frequency maps, and even monotonic deques (Section 6).

## Variable-Size Window — Two Postures

When the window size is not given, we use two pointers and let the problem dictate the policy:

- **Find the longest valid window:** keep expanding `right`; whenever the window goes invalid, contract `left` until it is valid again, then record the length.
- **Find the shortest valid window:** keep expanding `right` until the window becomes valid; then contract `left` as far as possible while staying valid, recording the length on each shrink.

The picture below traces the longest-substring-without-repeating-characters algorithm on `"abcabcbb"`. Green = valid (expanding), pink = duplicate detected, purple = contracting to repair.

![Variable-size window: longest substring without repeating characters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/sliding-window/fig2_variable_window.png)

And here is what the window length looks like over time. Green bands are expansion phases, purple bands are contraction phases.

![Window length over time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/sliding-window/fig3_expand_contract.png)

The crucial invariant: **`left` only ever moves right**. So even though there is a `while` inside a `for`, the inner loop's *total* work across the whole run is bounded by `n`. That is the amortised-$O(n)$ argument.

### LeetCode 3 — Longest Substring Without Repeating Characters

![Longest substring without repeating characters: window expanding and contracting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/gifs/leetcode/sliding-window-substr.gif)


**Problem.** Given a string `s`, return the length of the longest substring with all distinct characters.

**Pattern.** Variable window, *find longest*. The invariant is "all characters in the window are unique". When `s[right]` is already in the window, slide `left` past its previous occurrence.

```python
def length_of_longest_substring(s):
    """LC 3 — longest substring with distinct characters."""
    last_seen = {}        # char -> most recent index
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            # jump left past the previous occurrence in O(1)
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)

    return best
```

**Java**

```java
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> lastSeen = new HashMap<>();
    int left = 0, best = 0;
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (lastSeen.containsKey(c) && lastSeen.get(c) >= left) {
            left = lastSeen.get(c) + 1;
        }
        lastSeen.put(c, right);
        best = Math.max(best, right - left + 1);
    }
    return best;
}
```

**Complexity.** Time $O(n)$ — each character is inserted into and removed from the map at most once. Space $O(\min(n, |\Sigma|))$ where $|\Sigma|$ is the alphabet size.

**Two implementation styles.** The version above uses *jump-left* — `left` leaps directly past the previous occurrence. The textbook alternative is *step-left*: hold a frequency map and `while count[s[right]] > 1: drop(s[left]); left += 1`. Both are $O(n)$; jump-left is one line shorter, step-left generalises better to "at most k" variants.

### LeetCode 76 — Minimum Window Substring

**Problem.** Given `s` and `t`, return the shortest substring of `s` that contains every character of `t` (with multiplicities). Return `""` if no such window exists.

**Pattern.** Variable window, *find shortest*. The invariant is "the window contains all required characters at the required counts". We expand until the invariant holds, then aggressively contract.

The cleanest way to track validity is a `valid` counter — the number of *distinct* required characters whose count in the window has reached the required threshold. The window is valid iff `valid == len(need)`. This avoids comparing two hash maps on every step.

```python
from collections import Counter

def min_window(s, t):
    """LC 76 — minimum window substring containing all chars of t."""
    if not s or not t or len(s) < len(t):
        return ""

    need = Counter(t)
    have = {}
    valid = 0                     # distinct chars whose count meets need
    left = 0
    best_len = float("inf")
    best_start = 0

    for right, ch in enumerate(s):
        if ch in need:
            have[ch] = have.get(ch, 0) + 1
            if have[ch] == need[ch]:
                valid += 1

        # shrink while still valid
        while valid == len(need):
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_start = left

            drop = s[left]
            left += 1
            if drop in need:
                if have[drop] == need[drop]:
                    valid -= 1            # about to fall below threshold
                have[drop] -= 1

    return "" if best_len == float("inf") else s[best_start:best_start + best_len]
```

**Java**

```java
public String minWindow(String s, String t) {
    if (s.length() < t.length()) return "";
    Map<Character, Integer> need = new HashMap<>();
    for (char c : t.toCharArray()) need.merge(c, 1, Integer::sum);

    Map<Character, Integer> have = new HashMap<>();
    int valid = 0, left = 0;
    int bestLen = Integer.MAX_VALUE, bestStart = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (need.containsKey(c)) {
            have.merge(c, 1, Integer::sum);
            if (have.get(c).intValue() == need.get(c).intValue()) valid++;
        }
        while (valid == need.size()) {
            if (right - left + 1 < bestLen) {
                bestLen = right - left + 1;
                bestStart = left;
            }
            char d = s.charAt(left++);
            if (need.containsKey(d)) {
                if (have.get(d).intValue() == need.get(d).intValue()) valid--;
                have.merge(d, -1, Integer::sum);
            }
        }
    }
    return bestLen == Integer.MAX_VALUE ? "" : s.substring(bestStart, bestStart + bestLen);
}
```

**Complexity.** Time $O(|s| + |t|)$ — both pointers advance monotonically. Space $O(|\Sigma|)$.

**The subtle line** is `if have[drop] == need[drop]: valid -= 1` — we decrement `valid` *before* the `have` count actually falls below the threshold, because we're about to make it fall. Get the order wrong and the counter desyncs.

### LeetCode 567 — Permutation in String

**Problem.** Return `True` iff some substring of `s2` is a permutation of `s1`.

**Pattern.** Fixed-size window of width `len(s1)`. Two strings are permutations of each other iff their character-frequency vectors are equal. So we maintain a length-26 frequency vector for the current window in `s2` and compare to the frequency vector of `s1`.

```python
def check_inclusion(s1, s2):
    """LC 567 — does s2 contain a permutation of s1?"""
    n, m = len(s1), len(s2)
    if n > m:
        return False

    need = [0] * 26
    have = [0] * 26
    for c in s1:
        need[ord(c) - ord('a')] += 1
    for c in s2[:n]:
        have[ord(c) - ord('a')] += 1

    # how many of the 26 buckets currently match
    matches = sum(1 for i in range(26) if need[i] == have[i])

    for right in range(n, m):
        if matches == 26:
            return True

        r = ord(s2[right]) - ord('a')
        l = ord(s2[right - n]) - ord('a')

        # element entering on the right
        have[r] += 1
        if have[r] == need[r]:
            matches += 1
        elif have[r] == need[r] + 1:
            matches -= 1

        # element leaving on the left
        have[l] -= 1
        if have[l] == need[l]:
            matches += 1
        elif have[l] == need[l] - 1:
            matches -= 1

    return matches == 26
```

**Java**

```java
public boolean checkInclusion(String s1, String s2) {
    int n = s1.length(), m = s2.length();
    if (n > m) return false;
    int[] need = new int[26], have = new int[26];
    for (int i = 0; i < n; i++) {
        need[s1.charAt(i) - 'a']++;
        have[s2.charAt(i) - 'a']++;
    }
    int matches = 0;
    for (int i = 0; i < 26; i++) if (need[i] == have[i]) matches++;
    for (int right = n; right < m; right++) {
        if (matches == 26) return true;
        int r = s2.charAt(right) - 'a';
        int l = s2.charAt(right - n) - 'a';
        have[r]++;
        if (have[r] == need[r]) matches++;
        else if (have[r] == need[r] + 1) matches--;
        have[l]--;
        if (have[l] == need[l]) matches++;
        else if (have[l] == need[l] - 1) matches--;
    }
    return matches == 26;
}
```

**Complexity.** Time $O(m)$ — every slide is a constant amount of work because we maintain `matches` incrementally instead of rescanning all 26 buckets. Space $O(1)$.

The same skeleton solves **LC 438 (Find All Anagrams)** verbatim — just collect the start indices instead of returning early.

### Variant: Sliding Window Maximum (Monotonic Deque)

When the window asks for the **min or max** of its elements, the simple "add / remove" trick is not enough — removing the current max means we need to know the next-best element instantly. The standard tool is a **monotonic deque** that stores values (or indices) in strictly decreasing order. The front of the deque is always the current window max.

![Sliding window maximum with monotonic deque](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/sliding-window/fig4_deque_max.png)

```python
from collections import deque

def max_sliding_window(nums, k):
    """LC 239 — max in each window of size k."""
    dq = deque()      # holds indices; values at these indices are decreasing
    out = []

    for i, x in enumerate(nums):
        # 1. evict indices whose value is dominated by x
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)

        # 2. evict the front if it has slid out of the window
        if dq[0] <= i - k:
            dq.popleft()

        # 3. once the window is full, the front is the max
        if i >= k - 1:
            out.append(nums[dq[0]])

    return out
```

Each index is pushed and popped at most once, so total work is $O(n)$ — the same linear bound as the basic patterns.

## When to Reach for Sliding Window

The decision tree below summarises which flavour fits which problem shape:

![When to use sliding window](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/sliding-window/fig5_decision_tree.png)

Reach for sliding window when **all** of the following are true:

1. The answer is over a **contiguous** range (subarray / substring).
2. The window's value can be **maintained incrementally** as the boundaries move (running sum, frequency map, monotonic deque, …).
3. The objective rewards either a **specific size** (fixed `k`) or a **boundary condition** (longest valid / shortest valid).

If any of those fails — non-contiguous picks, requiring a global view, or needing to undo arbitrary insertions — try prefix sums, two pointers on sorted data, dynamic programming, or a heap instead.

## Templates Worth Memorising

**Fixed-size window**

```python
def fixed_window(arr, k):
    state = init_state(arr[:k])           # e.g. sum, frequency map
    best = report(state)
    for right in range(k, len(arr)):
        state = add(state, arr[right])
        state = remove(state, arr[right - k])
        best = better(best, report(state))
    return best
```

**Variable window — longest valid**

```python
def longest_valid(arr):
    left = 0
    state = empty_state()
    best = 0
    for right, x in enumerate(arr):
        state = add(state, x)
        while not is_valid(state):
            state = remove(state, arr[left])
            left += 1
        best = max(best, right - left + 1)
    return best
```

**Variable window — shortest valid**

```python
def shortest_valid(arr):
    left = 0
    state = empty_state()
    best = float("inf")
    for right, x in enumerate(arr):
        state = add(state, x)
        while is_valid(state):
            best = min(best, right - left + 1)
            state = remove(state, arr[left])
            left += 1
    return best if best != float("inf") else 0
```

## Common Pitfalls

- **Off-by-one in length.** Window length is `right - left + 1`, not `right - left`.
- **Updating `valid` after the count change.** Increment `valid` exactly when `have[c]` first equals `need[c]`; decrement *before* it falls below. The order matters.
- **Using `list.pop(0)` for fixed windows.** That is $O(k)$. Use `collections.deque` (or simple index arithmetic) for $O(1)$.
- **Recomputing the window from scratch.** The whole point is incremental update — if your inner loop scans the window, you have lost the asymptotic improvement.
- **Negative numbers in "sum ≥ target".** Contracting may not reduce the sum, so the "shortest valid" template doesn't apply directly. Reach for prefix sums + monotonic deque, or a different framing.

## Practice Set

| Difficulty | Problem | Pattern |
|---|---|---|
| Easy | LC 643 Maximum Average Subarray I | fixed |
| Easy | LC 1456 Max Vowels in a Substring of Given Length | fixed |
| Medium | LC 3 Longest Substring Without Repeating Characters | longest |
| Medium | LC 159 / 340 Longest Substring with At Most K Distinct | longest |
| Medium | LC 209 Minimum Size Subarray Sum | shortest |
| Medium | LC 424 Longest Repeating Character Replacement | longest |
| Medium | LC 567 Permutation in String | fixed |
| Medium | LC 438 Find All Anagrams in a String | fixed |
| Medium | LC 713 Subarray Product Less Than K | longest |
| Medium | LC 1004 Max Consecutive Ones III | longest |
| Hard | LC 76 Minimum Window Substring | shortest |
| Hard | LC 239 Sliding Window Maximum | monotonic deque |
| Hard | LC 30 Substring with Concatenation of All Words | fixed + hashing |

## Where Sliding Window Shows Up Outside Interviews

The pattern lives well beyond LeetCode. Three places I've actually shipped it:

- **Rate limiting.** A token-bucket and a sliding-window counter are the two standard answers. The "1000 requests per minute" implementation is literally a deque of timestamps, evicting from the left. `redis-cell` and most API gateways do exactly this.
- **Streaming aggregations.** Any time series dashboard showing "errors per minute over the last hour" is a sliding window over an event stream. Kafka Streams' `TimeWindow` and Flink's tumbling/sliding windows are the same idea wrapped in a framework.
- **Network congestion control.** TCP's `cwnd` (congestion window) is a sliding window over in-flight bytes. The receiver's advertised window controls how far the sender's left edge can advance.

The interview problem is the toy version. The production version is the same algorithm with timestamps instead of array indices and TTL eviction instead of pointer increments.

## A Cleaner Python Idiom

C-style `while right < n` loops work, but a single `for right, x in enumerate(arr)` reads better and removes one source of off-by-one bugs. Compare:

```python
# C-style
right = 0
while right < len(arr):
    add(arr[right])
    while bad(): remove(arr[left]); left += 1
    best = max(best, right - left + 1)
    right += 1

# Pythonic
left = 0
for right, x in enumerate(arr):
    add(x)
    while bad():
        remove(arr[left]); left += 1
    best = max(best, right - left + 1)
```

The second version eliminates `right += 1` (the most-forgotten line in interview rooms) and avoids re-indexing `arr[right]` inside the loop. Same algorithm, fewer places to break it.

For frequency-map windows, prefer `collections.Counter` over a hand-rolled dict — the `+=` and `-=` semantics are well-tested:

```python
from collections import Counter

def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    cnt = Counter()
    left = best = 0
    for right, c in enumerate(s):
        cnt[c] += 1
        while len(cnt) > k:
            cnt[s[left]] -= 1
            if cnt[s[left]] == 0:
                del cnt[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
```

The `del` after decrement is what keeps `len(cnt)` honest — without it your "distinct count" silently includes zero-count keys.

## Edge-Case Test Battery

Before you submit, run your solution mentally against this list. Most "wrong answer" verdicts on sliding-window problems are one of these:

| Case | Why it breaks |
|---|---|
| Empty string / array | Off-by-one on the initial `best = 0` vs `best = float('inf')` |
| `k = 0` for "at most K distinct" | The valid window is empty, but loops that always increment `right` first will record length 1 |
| Single element | Whether you compare `>` or `>=` against `best` shows up here |
| All identical elements | The "longest" template usually shines; the "shortest" template can collapse to length 1 immediately and never grow |
| `target` larger than `sum(arr)` | "Min length subarray sum >= target" should return 0, not `float('inf')` |
| Negative numbers | Most sliding-window invariants assume monotonic growth on `add`; negatives break it. Switch to prefix sums + monotonic deque |
| Window larger than array | `len(arr) < k` for fixed-size templates needs an explicit guard |

Two minutes of mental dry-run on these saves the fifteen minutes of staring at "Wrong Answer on test case 87".

## Summary

Sliding window is fundamentally an **amortisation argument**: instead of recomputing each window's value from scratch, you maintain a tiny piece of state and update it as the window moves by one position. Two pointers, both monotonically increasing, give you an $O(n)$ algorithm where the brute force is $O(n^2)$.

The mental checklist when you see a contiguous-range problem:

1. Is the window size **fixed** or **variable**?
2. If variable, am I after the **longest** or the **shortest** valid window?
3. What **state** do I maintain — a sum, a frequency map, a monotonic deque?
4. What is the **invariant** that determines validity, and which pointer's move can break it?

Internalise those four questions and most sliding window problems collapse to filling in a template.
