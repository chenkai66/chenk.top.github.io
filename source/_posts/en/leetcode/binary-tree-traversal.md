---
title: "LeetCode Patterns: Binary Tree Traversal and Construction"
date: 2024-01-16 09:00:00
tags:
  - LeetCode
  - Algorithms
  - Data Structures
categories: LeetCode
series: "LeetCode Patterns"
series_order: 6
series_total: 10
lang: en
mathjax: false
description: "A unified mental model for binary trees: how the four traversal orders are really one DFS recipe, why recursion and an explicit stack are the same algorithm, and how to reconstruct a tree from preorder + inorder. Includes Inorder Traversal, Level Order Traversal, Construct from Preorder + Inorder, Maximum Depth, and Validate BST."
disableNunjucks: true
---

A binary tree problem is rarely about the tree. It is about *the order in which you touch nodes* and *what you remember from the children before deciding what to do at the parent*. Once those two ideas click, the four traversal orders, the iterative rewrites, the construction problems, and even classics like Validate BST and Maximum Depth all collapse into a handful of variations on the same recipe. This article builds that recipe end to end.

We use a single example tree throughout so the figures and the code line up exactly:

```
        3
      /   \
     9    20
         /  \
        15   7
```

Its preorder is `[3, 9, 20, 15, 7]`, inorder is `[9, 3, 15, 20, 7]`, postorder is `[9, 15, 7, 20, 3]`, and level-order is `[[3], [9, 20], [15, 7]]`. Keep this picture nearby; almost every figure below refers back to it.

# Series Navigation

**LeetCode Algorithm Masterclass** (10 parts):

1. Hash Tables — Two Sum, Longest Consecutive, Group Anagrams
2. Two Pointers — collision pointers, fast/slow, sliding window
3. Linked List Operations — reverse, cycle detection, merge
4. Sliding Window — fixed and variable windows
5. Binary Search — boundaries, lower/upper bound, on the answer
6. **Binary Tree Traversal & Construction** — *you are here*
7. Dynamic Programming — 1D/2D DP, state transition
8. Backtracking — permutations, combinations, pruning
9. Greedy Algorithms — interval scheduling, jump game
10. Stacks & Queues — bracket matching, monotonic stack

# Binary tree fundamentals

## What a binary tree is — and the only vocabulary you need

A **binary tree** is a tree in which every node has at most two children, and crucially, the two children are *ordered* — left and right are not interchangeable. The empty tree counts as a binary tree, which is the base case that keeps every recursive definition clean.

Three labels appear over and over in problem statements:

- **Root** — the unique node with no parent.
- **Internal node** — any node with at least one child.
- **Leaf** — a node with no children.

And two numbers describe how "tall" a node sits inside the tree:

- The **depth** of a node is the number of edges from the root to it. The root has depth `0`.
- The **height** of a node is the number of edges on the longest downward path to any leaf below it. A leaf has height `0`. The height of the *tree* is the height of the root.

These five terms are enough vocabulary for almost every problem in this article. Figure 1 attaches them to the running example.

![Binary tree anatomy: root, internal nodes, leaves, depth, height](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-tree-traversal/fig1_tree_anatomy.png)

The minimal Python representation is a class with a value and two child pointers:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

That is the entire data model. Everything below is some way of walking it.

## The shapes that have names

A few special shapes show up as constraints in problem statements; recognising them often unlocks an asymptotically better solution.

- **Full binary tree** — every node has either `0` or `2` children, never exactly one. Common in expression trees.
- **Complete binary tree** — every level is full except possibly the last, which fills from left to right. Heaps are stored this way, which lets you implement them in a flat array.
- **Balanced binary tree** — for every node, the heights of the two subtrees differ by at most one. This is the property AVL and red-black trees maintain to keep operations $O(\log n)$.
- **Binary search tree (BST)** — for every node, *every* value in the left subtree is strictly less than the node and *every* value in the right subtree is strictly greater. We will exploit this directly when validating a BST.

# DFS: one recipe, three orders

Every depth-first traversal performs three actions at each node — visit the node, recurse left, recurse right — but in different orders. The order is what gives the traversal its name:

| Order | Action sequence | Resulting visit list on our tree |
| --- | --- | --- |
| **Preorder** | root → left → right | `[3, 9, 20, 15, 7]` |
| **Inorder** | left → root → right | `[9, 3, 15, 20, 7]` |
| **Postorder** | left → right → root | `[9, 15, 7, 20, 3]` |

Figure 2 shows all three on the same tree, with each node numbered by its visit step.

![Three DFS traversal orders on the same tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-tree-traversal/fig2_dfs_traversals.png)

The "where does the root sit" rule is the entire trick: the root is processed *first* in preorder, *between* the two children in inorder, and *last* in postorder. The recursive code makes this almost too obvious:

```python
def preorder(node, out):
    if not node: return
    out.append(node.val)         # root
    preorder(node.left, out)     # left
    preorder(node.right, out)    # right

def inorder(node, out):
    if not node: return
    inorder(node.left, out)      # left
    out.append(node.val)         # root
    inorder(node.right, out)     # right

def postorder(node, out):
    if not node: return
    postorder(node.left, out)    # left
    postorder(node.right, out)   # right
    out.append(node.val)         # root
```

The choice of order is dictated by *when you have the information you need*. Preorder is right when a parent decision must happen before its children — copying a tree, printing a directory tree, encoding for serialization. Inorder is right whenever the BST ordering matters — sorted output, validation, the $k$-th smallest. Postorder is right when a parent's answer is built from its children's answers — deleting a tree, computing subtree sums or sizes, and (as we'll see) computing height.

## LeetCode 94 — Inorder Traversal: the recursive and iterative views

Inorder is the most useful of the three to study iteratively, because the iterative version reveals exactly what the call stack was doing for you.

**Recursive.** The "left, root, right" pattern straight from the definition:

```python
from typing import List, Optional

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result: List[int] = []

        def visit(node: Optional[TreeNode]) -> None:
            if node is None:
                return
            visit(node.left)         # 1. dive left as far as possible
            result.append(node.val)  # 2. then record this node
            visit(node.right)        # 3. then handle the right subtree

        visit(root)
        return result
```

Time is $O(n)$ — every node is touched once. Space is $O(h)$ for the call stack, where $h$ is the tree height; that is $O(\log n)$ for a balanced tree and $O(n)$ for a degenerate "linked-list-shaped" tree.

**Iterative.** The recursive call stack is doing two things: it remembers the *path of ancestors* on the way down, and it *resumes* each ancestor after its left subtree is done. Both jobs can be performed by an explicit stack:

```python
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    result: List[int] = []
    stack: List[TreeNode] = []
    cur = root

    while cur is not None or stack:
        # Phase 1: walk left as far as possible, remembering ancestors.
        while cur is not None:
            stack.append(cur)
            cur = cur.left

        # Phase 2: pop the deepest unvisited ancestor, record it,
        # then jump to its right subtree to continue the traversal.
        cur = stack.pop()
        result.append(cur.val)
        cur = cur.right

    return result
```

The two phases mirror the two halves of the recursive call: pushing onto the stack stands in for "going down a recursive call", and popping stands in for "returning from one". Figure 4 puts the two side by side, frame by frame, on the running tree — they really are the same algorithm under different syntax.

![Recursive vs. iterative inorder: implicit call stack vs. explicit stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-tree-traversal/fig4_recursive_vs_stack.png)

Two pitfalls trip people up here. First, forgetting `cur = cur.right` after popping creates an infinite loop, because the same node is pushed and popped forever. Second, treating the inner `while` as `if` only descends one step at a time and misses the leftmost grandchildren; you must drain the entire left spine before the first pop.

## Preorder and postorder iteratively, in one paragraph each

**Preorder** is the easiest to convert. Push the root onto a stack, then loop: pop a node, record it, and push its right child *before* its left child so that the left child comes out first (LIFO):

```python
def preorderTraversal(self, root):
    if not root: return []
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right: stack.append(node.right)  # pushed first => popped last
        if node.left:  stack.append(node.left)   # pushed last  => popped first
    return out
```

**Postorder** has a sneaky shortcut. If you run the preorder template but push *left* before *right*, you get root → right → left, which is the *reverse* of postorder. So just collect into a list and reverse at the end:

```python
def postorderTraversal(self, root):
    if not root: return []
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.left:  stack.append(node.left)
        if node.right: stack.append(node.right)
    return out[::-1]
```

This trick dodges the explicit "have I visited the right subtree yet?" bookkeeping that a faithful left-right-root iterative postorder requires.

# BFS: level-order traversal

Depth-first goes as deep as it can before backing up. Breadth-first does the opposite: it visits everyone at distance `0` from the root, then everyone at distance `1`, and so on. The right data structure is a FIFO queue, because we want the children of the *first* node we enqueued to come out *before* the children of the second.

![BFS / level-order traversal: visit nodes one level at a time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-tree-traversal/fig3_bfs_levelorder.png)

The one trick worth knowing is the **level-size capture**: at the start of each outer iteration, snapshot how many nodes are currently in the queue. Those (and only those) belong to the current level; anything enqueued during their processing is one level deeper.

## LeetCode 102 — Level Order Traversal

```python
from collections import deque
from typing import List, Optional

def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if root is None:
        return []

    result: List[List[int]] = []
    queue: deque[TreeNode] = deque([root])

    while queue:
        # Capture the level boundary BEFORE we start enqueueing children.
        level_size = len(queue)
        level: List[int] = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(level)

    return result
```

Time is $O(n)$. Space is $O(w)$ where $w$ is the maximum width of the tree — for a complete binary tree the worst level holds about $n/2$ nodes, so worst-case space is $O(n)$.

The level-size pattern is also the foundation for many siblings: zigzag traversal (alternate the level direction), right-side view (take the last value of each level), minimum depth (return when you find the first leaf), and shortest path in an unweighted graph all reuse this scaffold unchanged.

# Two friendly DFS warm-ups

Both of the next two problems take less than ten lines, but they are perfect templates for postorder and inorder respectively.

## LeetCode 104 — Maximum Depth: postorder in disguise

The maximum depth of a tree is one more than the maximum depth of its deeper subtree. That is a recursive definition you can transcribe almost directly:

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if root is None:
        return 0
    left = self.maxDepth(root.left)
    right = self.maxDepth(root.right)
    return 1 + max(left, right)
```

This is postorder: we have to know the depths of *both* children before we can answer for the parent. The same shape solves "diameter of a binary tree", "balanced binary tree" (return `-1` as a sentinel for unbalanced), and "sum of subtree values" — only the combine step changes.

## LeetCode 98 — Validate BST: inorder, with a twist

It is tempting to write `node.left.val < node.val < node.right.val` and recurse. That check is wrong: it only compares neighbours, not the whole subtree. A node deep in the right subtree can still violate the BST property with respect to a distant ancestor.

The fix is to pass the *running interval* `(low, high)` that every value in the current subtree must fall inside:

```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def check(node, low: float, high: float) -> bool:
        if node is None:
            return True
        if not (low < node.val < high):
            return False
        return (check(node.left,  low,        node.val) and
                check(node.right, node.val,   high))

    return check(root, float("-inf"), float("inf"))
```

Equivalently, you can do an inorder traversal and verify that the values come out strictly increasing — this works because inorder of a BST is sorted (which is the whole reason inorder is interesting in the first place).

```python
def isValidBST_inorder(self, root: Optional[TreeNode]) -> bool:
    prev = float("-inf")
    stack, cur = [], root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        if cur.val <= prev:
            return False
        prev = cur.val
        cur = cur.right
    return True
```

# LeetCode 105 — Construct Binary Tree from Preorder and Inorder

This is the most-asked construction problem and the cleanest illustration of *what each traversal actually tells you*.

> **Two facts.** Preorder tells you which node is the **root** of every subtree (it is always the first element of the slice). Inorder tells you, given the root's value, **where the left subtree ends and the right subtree begins** (everything to its left is the left subtree, everything to its right is the right subtree).

That is sufficient to reconstruct the tree uniquely. Figure 5 shows the recursion on the running example, three levels deep.

![Build a binary tree from preorder + inorder](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/leetcode/binary-tree-traversal/fig5_build_from_pre_in.png)

The naive implementation is to re-scan the inorder array and re-slice both arrays at every recursive call — that's $O(n^2)$ time and very wasteful on memory. The standard production version uses a hash map from value to inorder index, then recurses on indices:

```python
from typing import List, Optional

def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """LeetCode 105. O(n) time, O(n) space."""
    inorder_index = {v: i for i, v in enumerate(inorder)}
    pre_iter = iter(range(len(preorder)))  # advances once per node

    def build(in_left: int, in_right: int) -> Optional[TreeNode]:
        if in_left > in_right:
            return None

        # The next preorder value is, by construction, the root of this subtree.
        root_pre_idx = next(pre_iter)
        root_val = preorder[root_pre_idx]
        root = TreeNode(root_val)

        # Where does it sit in the inorder array?
        mid = inorder_index[root_val]

        # Build LEFT first because preorder is root-LEFT-right; consuming the
        # iterator in this order keeps `next(pre_iter)` aligned with the
        # subtree we're currently building.
        root.left  = build(in_left, mid - 1)
        root.right = build(mid + 1, in_right)
        return root

    return build(0, len(inorder) - 1)
```

The single most error-prone detail is the order of the two recursive calls. Preorder is root-**left**-right, so the left subtree's root is the *next* preorder element after the current root; if you build the right child first, you'll consume the wrong slice of preorder values and quietly construct the wrong tree.

A close cousin reconstructs from postorder + inorder. The only changes are: the root sits at the *end* of postorder, and you must build the **right** subtree first (because postorder is left-right-**root**, and reading it backwards from the end gives root-right-left):

```python
def buildTreePost(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """LeetCode 106. O(n) time, O(n) space."""
    inorder_index = {v: i for i, v in enumerate(inorder)}
    post = postorder
    self._i = len(post) - 1

    def build(in_left: int, in_right: int) -> Optional[TreeNode]:
        if in_left > in_right:
            return None
        root_val = post[self._i]
        self._i -= 1
        root = TreeNode(root_val)
        mid = inorder_index[root_val]
        root.right = build(mid + 1, in_right)  # right FIRST
        root.left  = build(in_left, mid - 1)
        return root

    return build(0, len(inorder) - 1)
```

Why does this work but preorder + postorder *not* work? Because inorder is the only one of the three that gives you a left/right boundary; without it you cannot tell where the left subtree ends. The classic counter-example is the two trees `(root=1, left=2)` and `(root=1, right=2)`: their preorder and postorder are identical, but their inorders differ.

# Choosing between recursion and iteration

| | Recursive | Iterative |
| --- | --- | --- |
| Code complexity | Lower — mirrors the definition | Higher — you maintain the stack yourself |
| Space | $O(h)$ on the call stack | $O(h)$ on an explicit stack |
| Stack overflow risk | Real for very deep trees (Python: ~1000 default) | None — you control allocation |
| Readability | Usually clearer | Usually noisier |
| Performance | Comparable; constant-factor differences | Comparable |

**Practical advice.** Reach for recursion first; rewrite to iterative only when (a) the interviewer asks, (b) the tree can be pathologically deep, or (c) you need to bail out mid-traversal in a way that's awkward to express as a recursive return. The iterative inorder template is worth memorising because it appears verbatim in BST-iterator and $k$-th-smallest problems.

# Q&A

### Why does inorder of a BST come out sorted?

Because the BST invariant says *all* left descendants are smaller and *all* right descendants are larger, and inorder visits exactly in the order "everything smaller, then me, then everything larger". By induction over depth, every element is emitted in increasing order. This is also why inorder is the natural traversal for BST validation.

### Can I uniquely rebuild a tree from any two traversals?

Only some pairs work. **Preorder + inorder** and **postorder + inorder** both uniquely determine the tree (when values are distinct). **Preorder + postorder** does *not*: the example above (`1` with a single left child versus `1` with a single right child) shows two distinct trees with identical preorders and postorders. The principle is that you need one traversal that fixes the root (preorder or postorder) and another that fixes the left-right split (only inorder does this).

### What about duplicate values?

With duplicates, even preorder + inorder is no longer unique: `preorder = [1, 1]`, `inorder = [1, 1]` could be either of two trees. LeetCode side-steps this by guaranteeing distinct values in the constraints; in production code you'd disambiguate using node identity (e.g., `id(node)`) rather than value.

### Is Morris traversal worth learning?

Morris traversal achieves $O(1)$ extra space by temporarily re-pointing the rightmost node of each left subtree back to its inorder successor and undoing the link on the way back up. It is genuinely $O(n)$ time and $O(1)$ space, but it *mutates the tree mid-traversal*, which makes it unsafe in concurrent settings and harder to reason about. For interviews it is a nice "I know it exists" answer; for production, the small constant of an explicit stack is almost always worth the simplicity.

### Why is BFS time complexity $O(n)$?

Every node enters the queue exactly once and leaves exactly once, so the total number of queue operations is $2n$. The queue's worst-case *size* is $O(w)$, where $w$ is the maximum width of the tree, which is what dominates the space cost.

# Summary

The whole article reduces to two ideas you can carry into any tree problem:

1. **DFS is a single recipe — visit, recurse left, recurse right — and the four traversals (pre / in / post / level) are just choices about *when* you take the "visit" step.** Preorder commits before the children, postorder commits after, inorder commits between. Level-order swaps the LIFO call stack for a FIFO queue and processes everything at distance $k$ before anything at distance $k+1$.

2. **Recursion and an explicit stack are the same algorithm in different clothes.** Pushing onto the stack stands in for a recursive call; popping stands in for returning. Once you internalise this, "rewrite this recursive solution iteratively" stops being a separate skill and becomes a mechanical translation.

Construction problems (LeetCode 105 / 106) are an application of the same recipe: preorder or postorder hands you a root, inorder hands you a split point, and you recurse on the two halves.

**Checklist before the interview.** Recursive pre/in/post and BFS — yes. Iterative inorder using the explicit-stack template — yes. Build-from-preorder-and-inorder using a value-to-index hash map — yes. Maximum depth and BST validation as your "small but representative" warm-ups — yes. Once those five templates are muscle memory, most binary-tree LeetCode problems are five-minute exercises.
