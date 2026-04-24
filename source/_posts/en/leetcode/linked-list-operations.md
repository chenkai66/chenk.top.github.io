---
title: "LeetCode Patterns: Linked List Operations"
date: 2024-01-07 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 3
series_total: 10
lang: en
mathjax: true
description: "A complete linked list toolkit: reversal (iterative and recursive), merging sorted lists, Floyd's cycle detection, removing the nth node from end, and an LRU cache built on a doubly linked list plus hash map."
disableNunjucks: true
---

A linked list is the simplest data structure that forces you to **think in pointers**. Arrays let you index in $O(1)$ and forget about layout; linked lists hand you a head pointer and ask, *"now what?"* That single shift — from indices to references — is what makes linked-list problems so common in interviews. They are short to state, brutal to get right, and reward exactly the habits good engineers build: drawing pictures, naming pointers, and **never dereferencing without checking for `None`**.

This article walks through five problems that, taken together, cover every classical linked-list technique you will see in coding interviews:

- **Reverse a linked list** — the canonical pointer-rewiring exercise, both iterative and recursive.
- **Merge two sorted lists** — the dummy-node pattern and how it eliminates head-case clutter.
- **Linked list cycle detection (Floyd)** — fast/slow pointers and the algebra that proves they meet at the entrance.
- **Remove the nth node from end** — two pointers with a fixed gap, in a single pass.
- **LRU cache** — the doubly linked list + hash map combo that powers every real-world cache.

Every section keeps the same shape: problem, idea, code, complexity, and a worked example. The figures below show the pointer state before and after each rewiring, because that is how you should solve these in your head and on the whiteboard.

# Series Navigation

**LeetCode Patterns** (10 articles): Hash Tables, Two Pointers, **Linked List Operations** (this article), Binary Tree Traversal, Dynamic Programming, Backtracking, Binary Search, Stacks and Queues, Graphs, Greedy and Bit Manipulation.

# Linked List Fundamentals

A singly linked list node carries a value and one pointer:

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

That is the entire interface. Every algorithm in this article is just a sequence of `node.next = something` assignments performed in the right order, with the right safety checks.

## Linked list vs. array

| Feature | Array | Linked list |
|---|---|---|
| Random access | $O(1)$ | $O(n)$ |
| Insert / delete at known position | $O(n)$ (shift) | $O(1)$ (rewire) |
| Memory layout | Contiguous | Scattered |
| Cache locality | High | Low |
| Dynamic growth | Reallocate | Natural |

The trade-off is concrete: arrays win on access and cache behavior, linked lists win when you mutate the structure often and do not need to jump around. Once you internalize this, problem selection becomes obvious — *"can I reach the kth element?"* points to an array; *"do I need to splice out the middle without copying?"* points to a list.

## Insert and delete: rewire two pointers, never more

![Insert and delete pointer rewiring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/linked-list-operations/fig1_insert_delete.png)

Insertion of `X` between `A` and `B` is exactly two assignments, **in this order**:

```python
X.next = A.next   # X now points to B
A.next = X        # A now points to X
```

If you reverse the order, `A.next` is overwritten before you save it, and `B` is unreachable. Deletion is even simpler — one assignment redirects `A` past `B`, and the garbage collector reclaims `B`:

```python
A.next = B.next   # A skips B
```

This is the entire mental model: **operations on a linked list are pointer assignments performed in a careful order**. Everything else in this article is a variation on that theme.

# LeetCode 206: Reverse Linked List

> Given the head of a singly linked list, reverse the list and return its new head.
>
> Example: `1 → 2 → 3 → 4 → 5` becomes `5 → 4 → 3 → 2 → 1`.
>
> Follow-up: solve it iteratively and recursively.

This is the *"hello world"* of pointer manipulation. The challenge is that you have to flip every `next` pointer in place without losing access to the rest of the list.

![Iterative and recursive reversal](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/linked-list-operations/fig2_reverse.png)

## Iterative: three pointers, $O(1)$ space

The trick is to remember that flipping `curr.next = prev` immediately destroys our path forward, so we must save it first.

```python
def reverseList(head):
    """
    Reverse a singly linked list in place.

    Time:  O(n) — every node is visited exactly once.
    Space: O(1) — three pointers, no auxiliary structure.
    """
    prev, curr = None, head
    while curr:
        nxt = curr.next     # 1. save the path forward
        curr.next = prev    # 2. flip the current pointer
        prev = curr         # 3. advance prev
        curr = nxt          # 4. advance curr
    return prev             # prev is the new head (was the tail)
```

**Walkthrough on `1 → 2 → 3 → None`:**

| Step | `prev` | `curr` | After flip |
|---|---|---|---|
| 0 | `None` | `1` | — |
| 1 | `1` | `2` | `None ← 1`, remaining `2 → 3 → None` |
| 2 | `2` | `3` | `None ← 1 ← 2`, remaining `3 → None` |
| 3 | `3` | `None` | `None ← 1 ← 2 ← 3` |

Return `prev = 3`, which now heads the reversed list.

**Edge cases that the loop already handles correctly:**

- Empty list (`head is None`): the loop body never runs, return `prev = None`.
- Single node: one iteration, sets `1.next = None`, returns `1`.

## Recursive: rewire on the way back up

The recursive view says: *"reverse everything after `head`, then make the old `head.next` point back to `head`."*

```python
def reverseList_recursive(head):
    """
    Reverse a linked list recursively.

    Time:  O(n).
    Space: O(n) — recursion stack.
    """
    if not head or not head.next:
        return head

    new_head = reverseList_recursive(head.next)

    # head.next is the tail of the already-reversed sublist.
    # Make it point back to head, then sever head's outgoing link.
    head.next.next = head
    head.next = None
    return new_head
```

The single line that confuses everyone is `head.next.next = head`. Read it slowly:

- `head.next` is the node that comes right after `head` in the *original* list.
- After the recursive call, that node has become the **tail** of the reversed sublist.
- We want the tail to point back to `head`, so we assign to `head.next.next`.

Then `head.next = None` cuts `head`'s outgoing link so the new tail terminates cleanly.

## Iterative or recursive?

| Approach | Time | Space | When to prefer |
|---|---|---|---|
| Iterative | $O(n)$ | $O(1)$ | Production code, long lists, no stack overflow risk |
| Recursive | $O(n)$ | $O(n)$ | Whiteboard elegance; demonstrating you can think recursively |

In an interview, lead with iterative. Mention recursive as a follow-up to show range. In production, always pick iterative — Python's default recursion limit (1000) will bite you on long lists.

# LeetCode 21: Merge Two Sorted Lists

> Merge two sorted linked lists into one sorted list by splicing existing nodes.
>
> Example: `l1 = [1,2,4]`, `l2 = [1,3,4]` → `[1,1,2,3,4,4]`.

The interesting question here is not the merge — that is just standard two-pointer comparison — but how to handle the **first node** of the result. Without a helper, you need a special branch to decide which list contributes the head. With a *dummy node*, that branch disappears.

![Merging with a dummy node](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/linked-list-operations/fig4_merge.png)

## Iterative with a dummy node

```python
def mergeTwoLists(l1, l2):
    """
    Merge two sorted lists by splicing nodes (no allocation).

    Time:  O(m + n) — each input node is visited once.
    Space: O(1) — only the dummy node and a cursor.
    """
    dummy = ListNode()    # sentinel; never returned
    cur = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next

    # At most one list still has nodes; attach it whole.
    cur.next = l1 if l1 else l2

    return dummy.next
```

**Why the dummy?** It guarantees there is always a "previous node" to write to. Without it, the first iteration would need to *decide* whether the head is from `l1` or `l2`, which forks every subsequent branch. With the dummy, every iteration looks the same — the head case has been smuggled into the sentinel.

**Walkthrough on `l1 = [1,2,4]`, `l2 = [1,3,4]`:**

| Step | Compare | Picked | Result so far |
|---|---|---|---|
| 1 | `1 ≤ 1` | `l1` | `1` |
| 2 | `1 < 2` | `l2` | `1 → 1` |
| 3 | `2 < 3` | `l1` | `1 → 1 → 2` |
| 4 | `3 < 4` | `l2` | `1 → 1 → 2 → 3` |
| 5 | `4 ≤ 4` | `l1` | `1 → 1 → 2 → 3 → 4` |
| 6 | `l1` empty | attach rest of `l2` | `1 → 1 → 2 → 3 → 4 → 4` |

Return `dummy.next`.

## Recursive variant

```python
def mergeTwoLists_recursive(l1, l2):
    """
    Recursive merge. Elegant but uses O(m + n) stack.

    Time:  O(m + n).
    Space: O(m + n) — recursion depth equals total length.
    """
    if not l1: return l2
    if not l2: return l1
    if l1.val <= l2.val:
        l1.next = mergeTwoLists_recursive(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists_recursive(l1, l2.next)
        return l2
```

# LeetCode 141 / 142: Linked List Cycle (and where it begins)

> 141: Detect whether a linked list has a cycle.
>
> 142: Return the node where the cycle begins, or `None` if there is no cycle.
>
> Constraint: $O(1)$ extra space.

The brute-force solutions — hash every node, or count steps — are correct but use $O(n)$ space or two passes. **Floyd's tortoise-and-hare** algorithm does it in one pass with two pointers and a small piece of algebra.

![Floyd's cycle detection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/linked-list-operations/fig3_floyd_cycle.png)

## Phase 1: detect the cycle

```python
def hasCycle(head):
    """LeetCode 141. Detect a cycle using fast/slow pointers."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next         # 1 step
        fast = fast.next.next    # 2 steps
        if slow is fast:
            return True
    return False
```

Why does it work? If there is no cycle, `fast` reaches `None` and the loop exits. If there is a cycle, both pointers are eventually trapped inside it, and because `fast` gains one node per step on `slow`, the gap shrinks by 1 per iteration and they must collide within at most $L$ iterations (where $L$ is the cycle length).

## Phase 2: locate the cycle entrance

```python
def detectCycle(head):
    """
    LeetCode 142. Return the node where the cycle begins.

    Time:  O(n). Space: O(1).
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # Phase 2: reset slow, advance both by 1 step.
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

### The algebra (worth memorizing once)

Let $a$ = distance from head to the entrance, $b$ = distance from the entrance to the meeting point along the cycle, $c$ = the rest of the cycle (so cycle length $L = b + c$).

When `slow` and `fast` first meet:

$$
\text{slow walked: } a + b
$$

$$
\text{fast walked: } a + b + kL \quad \text{(for some integer } k \geq 1\text{)}
$$

Since `fast` moves at twice the speed of `slow`, $\text{fast} = 2 \cdot \text{slow}$, giving:

$$
2(a + b) = a + b + kL \implies a = kL - b = (k-1)L + c
$$

That last equality is the punchline. It says: *walking $a$ steps from the head lands on the entrance, and walking $a$ steps from the meeting point also lands on the entrance* (because $(k-1)L$ is just whole loops). So if we reset `slow` to `head` and step both pointers one node at a time, they meet at the entrance.

# LeetCode 19: Remove Nth Node From End

> Given the head of a linked list, remove the nth node from the end and return the head. Try to do it in one pass.
>
> Example: `head = [1,2,3,4,5]`, `n = 2` → `[1,2,3,5]`.

The naïve approach takes two passes: count the length, then walk to position `length - n`. The one-pass solution uses two pointers separated by a fixed gap of $n + 1$, plus a dummy node so deleting the head requires no special case.

```python
def removeNthFromEnd(head, n):
    """
    Remove the nth node from end in a single pass.

    Time:  O(L) where L is the list length.
    Space: O(1).
    """
    dummy = ListNode(0, head)    # handles deletion of the head uniformly
    slow = fast = dummy

    # Move fast n+1 steps ahead so that when fast hits None,
    # slow sits on the node BEFORE the one we want to remove.
    for _ in range(n + 1):
        fast = fast.next

    while fast:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next
```

**Why $n+1$, not $n$?** We need `slow` to land on the *predecessor* of the target so we can rewire `slow.next`. If `fast` only moves $n$ steps, the gap between them is $n$, so when `fast` is `None`, `slow` is *on* the target, not before it. Off by one is the single most common bug here.

**Why the dummy?** Consider `head = [1, 2]`, `n = 2` (delete the head). With the dummy:

- `fast` moves 3 steps, ending at `None`.
- `slow` is still at `dummy`.
- `dummy.next = dummy.next.next` sets `dummy.next = 2`.
- Return `dummy.next = 2`. Done.

Without the dummy, you would need a separate branch like `if n == length: return head.next`, which means computing the length first — and now you have a two-pass solution again.

# LeetCode 146: LRU Cache

> Design a data structure with `get(key)` and `put(key, value)` that both run in $O(1)$ amortized time. When capacity is exceeded, evict the least recently used key.

This problem comes up everywhere — operating system page caches, CDN edge caches, browser caches, your ORM's query cache. The standard solution is a small but beautiful composition: a **hash map** for $O(1)$ lookup, plus a **doubly linked list** for $O(1)$ reordering and eviction.

![LRU cache architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/linked-list-operations/fig5_lru.png)

## Why both data structures?

- A hash map alone gets us $O(1)$ lookup but no notion of recency.
- A linked list alone tracks recency but lookup is $O(n)$.
- Together: lookup the node in the map ($O(1)$), then unlink and re-link it at the head of the list ($O(1)$ because it is doubly linked).

The list maintains the ordering "most recently used at the head, least recently used at the tail". Two sentinel nodes (`head` and `tail`) eliminate every edge case for inserting at the front or evicting from the back.

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")
    def __init__(self, key=0, value=0):
        self.key, self.value = key, value
        self.prev = self.next = None


class LRUCache:
    """
    O(1) get and put using hash map + doubly linked list with sentinels.

    Invariant:
        head <-> most recent <-> ... <-> least recent <-> tail
    """

    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}                       # key -> Node
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    # ----- list primitives -----

    def _remove(self, node: "Node") -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: "Node") -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    # ----- public API -----

    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._add_to_front(node)            # mark as most recent
        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._add_to_front(node)
            return

        if len(self.map) >= self.cap:
            lru = self.tail.prev            # least recently used
            self._remove(lru)
            del self.map[lru.key]

        new_node = Node(key, value)
        self.map[key] = new_node
        self._add_to_front(new_node)
```

**Three things worth pausing on:**

1. **Sentinels are not optional.** They turn every insertion and removal into the same four-pointer dance, with no `if head is None` checks anywhere.
2. **The node carries the `key`.** When we evict the LRU node, we need to delete its key from the map — without storing the key on the node, we would have to walk the map to find it.
3. **We splice existing nodes; we never re-allocate on `get`.** That is what gives us true $O(1)$ amortized cost.

# Pattern Cheat Sheet

| Technique | Use it when | Examples in this article |
|---|---|---|
| **Three pointers** (`prev`, `curr`, `next`) | Reversing or rewiring in place | Reverse Linked List |
| **Dummy / sentinel node** | The head might be modified or returned | Merge Two Sorted Lists, Remove Nth From End, LRU |
| **Fast / slow pointers** | Find middle, detect cycle, kth from end | Cycle II, Remove Nth From End |
| **Hash map + linked list** | $O(1)$ lookup *and* ordering | LRU Cache |
| **Recursion** | Operation has natural divide structure | Recursive reverse / merge |

# Common Pitfalls

1. **Dereferencing without checking.** `curr.next.val` crashes the moment `curr.next` is `None`. Always guard: `while curr and curr.next:`.
2. **Losing the path forward.** Before writing `curr.next = something`, save the original `curr.next` if you still need it (this is the entire point of the `nxt` variable in iterative reverse).
3. **Forgetting to advance.** Every loop iteration must move at least one pointer toward termination, or you have an infinite loop.
4. **Off-by-one on gaps.** Two-pointer gap problems live and die on whether you advance `fast` by `n` or `n + 1` first.
5. **Skipping the dummy node.** Anytime the head can change, a dummy will collapse 5 lines of edge-case branching into 0.

# Practice Problems

Sorted roughly by difficulty within each group.

**Basics**
- LeetCode 206 — Reverse Linked List (covered)
- LeetCode 21 — Merge Two Sorted Lists (covered)
- LeetCode 19 — Remove Nth Node From End (covered)
- LeetCode 234 — Palindrome Linked List

**Cycles and intersection**
- LeetCode 141 — Linked List Cycle (covered)
- LeetCode 142 — Linked List Cycle II (covered)
- LeetCode 160 — Intersection of Two Linked Lists

**Advanced**
- LeetCode 25 — Reverse Nodes in k-Group
- LeetCode 23 — Merge K Sorted Lists
- LeetCode 138 — Copy List with Random Pointer
- LeetCode 146 — LRU Cache (covered)
- LeetCode 460 — LFU Cache

# Closing Thoughts

Linked-list problems are not really about linked lists. They are about being precise with mutation — naming the pointers, ordering the assignments, and never assuming the next field is non-null. The five problems above cover every technique you need: pointer rewiring, dummy nodes, fast/slow traversal, gap pointers, and the hash-map + list combination.

Once you can write `reverseList` from memory and explain why Floyd's algorithm works on a napkin, the rest of the linked-list canon falls out as variations on those themes. In the next article we move from one-dimensional pointer chains to **binary trees**, where recursion finally feels like the natural tool instead of a clever trick.

# Further Reading

- *Introduction to Algorithms* — Chapter 10, Elementary Data Structures.
- *Cracking the Coding Interview* — Chapter 2, Linked Lists.
- VisuAlgo — Linked List visualization: https://visualgo.net/en/list
- LeetCode Linked List tag — 100+ curated problems for practice.
