---
title: LeetCode（一）—— 哈希表
tags: Leetcode
categories: Algorithm
date: 2023-05-12 12:00:00
mathjax: true

---

哈希表是一种在计算机科学中广泛应用的数据结构，其高效的查找、插入和删除操作使其在解决各种算法问题时表现出色。在本文中，我们将通过解析三道Leetcode经典问题——“两数之和”（Two Sum）、“最长连续序列”（Longest Consecutive Sequence）以及“字母异位词分组”（Group Anagrams），深入理解哈希表的强大功能，并探讨其在实际应用中的潜在场景。

<!-- more -->

## 一、两数之和（Two Sum）

### 题目描述
给定一个整数数组 `nums` 和一个目标值 `target`，请在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

### 解决思路
为了解决这个问题，我们可以使用哈希表来存储已经遍历过的元素及其对应的索引。在遍历数组的过程中，对于每个元素 `num`，我们检查 `target - num` 是否已经在哈希表中存在。如果存在，则说明我们找到了两个数，使得它们的和等于目标值；否则，将当前元素及其索引加入哈希表。

### 代码实现

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for i, num in enumerate(nums):
            if target - num in hash_map:
                return [hash_map[target - num], i]
            hash_map[num] = i
        return []
```

### 使用场景拓展

- **电子商务**：在购物车推荐系统中，可以通过类似的思路，快速查找两种商品组合的价格是否满足用户的预算。
- **金融科技**：在交易系统中，快速匹配买卖双方的交易请求，以确保交易的及时性和准确性。

## 二、最长连续序列（Longest Consecutive Sequence）

### 题目描述

给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。

### 解决思路

利用哈希集合去重和快速查找的特性，我们可以在 O(n) 的时间复杂度内找到最长连续序列。具体方法是，将所有元素存入哈希集合，然后遍历集合中的每个元素。如果该元素是一个序列的起点（即它的前一个元素不在集合中），则我们通过不断查找其后续元素，计算当前序列的长度。

### 代码实现

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        longest_streak = 0

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
```

### 使用场景拓展

- **社交网络**：在用户活跃度分析中，计算用户连续活跃的最长天数，以评估用户粘性。
- **生物信息学**：在DNA序列分析中，找出最长的连续基因序列，以进行基因组研究。

## 三、字母异位词分组（Group Anagrams）

### 题目描述

给定一个字符串数组，将字母异位词组合在一起。字母异位词指的是由相同的字母组成，但字母顺序不同的字符串。

### 解决思路

我们可以使用哈希表，将排序后的字符串作为键，原字符串作为值进行存储。这样，所有的字母异位词都将被归到同一个键下。最终，我们返回哈希表中所有值的列表。

### 代码实现

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = collections.defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            anagrams[key].append(s)
        return list(anagrams.values())
```

### 使用场景拓展

- **文本处理**：在大规模文本处理中，快速归类相似单词或短语，以优化搜索引擎的索引建立。
- **数据清洗**：在数据清洗过程中，识别并归类不同拼写但意义相同的词语，提升数据分析的准确性。

