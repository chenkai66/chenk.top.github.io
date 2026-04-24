---
title: "NLP (7): Prompt Engineering and In-Context Learning"
date: 2025-09-08 09:00:00
tags:
  - NLP
  - Prompt Engineering
  - LLM
  - In-Context Learning
categories: Natural Language Processing
series: NLP
part: 7
total_parts: 12
lang: en
mathjax: true
description: "From prompt anatomy to chain-of-thought, self-consistency and ReAct: a working theory of in-context learning, the variance you have to fight, and the patterns that scale to real systems."
disableNunjucks: true
series_order: 7
---

The same model can produce a sharp answer or a confident hallucination. The difference is rarely the weights -- it is the framing. A vague request like *"analyze this text"* gets you a generic summary; a prompt with a role, two clean examples, and a strict output schema gets you something a parser can consume. **Prompt engineering is the discipline of turning that gap into a repeatable system instead of a lucky shot.**

In-Context Learning (ICL) is the mechanism that makes this work. When you put a few examples inside the prompt, the model does not retrain; it conditions its forward pass on those examples and effectively *infers a task* from them. Understanding what ICL can and cannot do is the difference between a developer who fights the model and one who steers it.

This part is the seventh in the NLP series. It assumes you know roughly how a Transformer decoder generates tokens (Part 4) and what an autoregressive LM is (Part 6). Everything below is grounded in published behaviour -- but be warned: the literature on prompt engineering is unusually noisy, and most numbers are model- and dataset-specific. Treat the bars in the figures as illustrative shapes, not benchmark claims.

## What you will learn

- **Prompt anatomy**: the five composable blocks (system, instruction, examples, query, format spec) and what each one buys you.
- **Three paradigms**: zero-shot, few-shot, and chain-of-thought -- when each is the right choice, and what it costs in tokens.
- **A working theory of ICL**: why a non-trained model can still "learn" from in-prompt examples, and which signals it actually picks up.
- **The variance problem**: how much accuracy can swing from format and ordering alone, and how to measure it.
- **Self-consistency**: turning a stochastic decoder into an ensemble by sampling many reasoning paths.
- **ReAct**: interleaving reasoning with tool calls, the foundation of modern agents.
- **A small system**: prompt registries, A/B harnesses, and the discipline that keeps prompt sets from rotting.

## Prerequisites

- Familiarity with large language models -- see [Part 6: GPT and Generative Models](/en/nlp-gpt-generative-models/).
- Basic Python; comfort reading short snippets.
- Access to any LLM API (OpenAI, Anthropic, an open-weights model).

---

## 1. Anatomy of a prompt

A prompt is *a single text string the model conditions on*. Everything else -- "system" vs. "user" roles, function schemas, retrieval results -- is just structured text the API stitches into one sequence before tokenization. Treating a prompt as a flat string with named blocks is the cleanest mental model.

![Anatomy of a structured prompt](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig1_prompt_anatomy.png)

The five blocks below are not mandatory, but most production prompts contain a subset of them, in roughly this order:

1. **System / role.** Sets persona, refusal policy, tone, length budget. Stable across requests, so it caches well.
2. **Task instruction.** One sentence stating the goal in the imperative.
3. **Few-shot examples.** Demonstrations of input -> output pairs. The primary ICL signal.
4. **User query.** The actual input to be processed.
5. **Format spec.** A schema (JSON, regex-able tags, table) that pins the output shape.

A pragmatic prompt builder:

```python
from dataclasses import dataclass, field
from typing import Iterable

@dataclass
class Prompt:
    system: str = ""
    instruction: str = ""
    examples: list[tuple[str, str]] = field(default_factory=list)
    query: str = ""
    format_spec: str = ""

    def render(self) -> str:
        parts: list[str] = []
        if self.system:
            parts.append(f"[SYSTEM]\n{self.system}")
        if self.instruction:
            parts.append(f"[TASK]\n{self.instruction}")
        if self.examples:
            shots = "\n\n".join(
                f"Input: {x}\nOutput: {y}" for x, y in self.examples
            )
            parts.append(f"[EXAMPLES]\n{shots}")
        if self.format_spec:
            parts.append(f"[FORMAT]\n{self.format_spec}")
        if self.query:
            parts.append(f"[INPUT]\n{self.query}\nOutput:")
        return "\n\n".join(parts)
```

Two design notes that beginners often miss:

- **Order matters.** Examples placed *after* the format spec but *before* the query consistently work better than examples buried at the top -- recency biases the decoder.
- **Stable prefix, variable suffix.** Put everything that does not change (system, examples, format) at the top so KV-cache reuse and prompt-cache features can work. Variable input goes last.

### Four principles that survive contact with reality

These are the principles I would still teach today, after a lot of prompts in production:

1. **Clarity over cleverness.** Replace "analyze the text" with "classify the text into {positive, negative, neutral} and return JSON". You are competing with every plausible interpretation the model could pick up.
2. **Specificity buys determinism.** Say what *not* to do, and what to output when uncertain ("if you cannot answer from the document, return `{\"answer\": null}`"). Models honour negative constraints surprisingly well.
3. **Context completeness.** If the answer needs a definition the model could plausibly not have, include it. Cheaper than a wrong answer.
4. **Role assignment when relevant.** "You are a senior security reviewer" measurably narrows the distribution of outputs on code-review tasks. It is *not* a magic spell -- avoid it for generic tasks where it just adds tokens.

---

## 2. Zero-shot, few-shot, chain-of-thought

These are the three baseline framings every other technique builds on.

![Three prompting paradigms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig2_prompting_paradigms.png)

### Zero-shot

Describe the task; provide no examples. The model relies on whatever it learned during pre-training and instruction tuning.

```python
zero_shot = """Classify the sentiment of the sentence as positive, negative, or neutral.

Sentence: This movie has an excellent plot and outstanding acting.
Sentiment:"""
```

Use zero-shot when the task is something the model already knows well (sentiment, translation, summarization for short inputs) and you care about latency or cost. Its weakness is that the *output format* is unstable -- the model may emit "Positive sentiment", "POSITIVE", or a paragraph of analysis. Pin the format with one constrained sentence ("Reply with exactly one word from the set {positive, negative, neutral}.").

### Few-shot

Few-shot prompting puts $k$ examples in front of the query. This is the textbook setting for ICL.

```python
few_shot = """Classify sentiment as positive, negative, or neutral.

Sentence: The weather is beautiful today, sunny and bright.
Sentiment: positive

Sentence: This product is poor quality and overpriced.
Sentiment: negative

Sentence: It will rain tomorrow.
Sentiment: neutral

Sentence: The service at this restaurant is impressive.
Sentiment:"""
```

Three things examples actually do:

- **Task identification.** They disambiguate what task you mean. "Translate" might mean transliterate, paraphrase, or rewrite -- two examples nail it.
- **Format alignment.** The output side of each shot is a template. The model copies the template.
- **Label-space anchoring.** The set of labels in your examples becomes the model's effective output vocabulary, even if you never enumerate them.

The often-cited surprise: **the labels themselves matter less than you might think.** Min et al. (2022) showed that randomizing the gold labels in few-shot prompts barely hurts on many tasks -- what matters is the *distribution of inputs and the label space*. The takeaway is not "labels do not matter at all" (they do for hard tasks) but "do not over-engineer label correctness; engineer coverage and format".

### Chain-of-thought

For multi-step problems, ask the model to *write its reasoning before the answer*.

```text
Problem: A book has 120 pages. On day 1, 30 pages were read.
On day 2, twice as many as day 1 were read. On day 3, half of
the remaining pages were read. How many pages were read on day 3?

Let's think step by step.
1. Day 1: 30 pages.
2. Day 2: 2 x 30 = 60 pages.
3. Read so far: 30 + 60 = 90.
4. Remaining: 120 - 90 = 30.
5. Day 3: 30 / 2 = 15.

Answer: 15 pages.
```

The trick is mechanical, not mystical. Each generated reasoning token *changes the conditioning context for the next token*. The model is autoregressive, so writing intermediate state ("Read so far: 90") makes that state available to all later tokens -- including the final answer. Without it, the model has to compute every intermediate quantity in a single forward pass through a fixed-depth network. CoT effectively buys extra serial compute, paid in output tokens.

![Chain-of-thought reasoning flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig3_cot_flow.png)

A clean way to think about CoT probabilistically: for problem $x$ with answer $a$, marginalize over latent reasoning chains $z$:
$$P(a \mid x) \;=\; \sum_z P(z \mid x)\, P(a \mid x, z).$$
Greedy CoT picks one $z$ and hopes it is right. **Self-consistency** (Section 5) approximates the sum by sampling many $z$.

When CoT helps and when it does not:

- **Helps**: arithmetic, multi-hop QA, code reasoning, anything with intermediate state.
- **Neutral or hurts**: simple classification, retrieval, single-fact lookup. The reasoning preamble adds tokens, latency, and a chance for the model to talk itself into the wrong answer.
- **Modern models with built-in "thinking" modes** (long-context reasoning models trained on CoT-style traces) absorb most of this; explicit CoT prompting still helps but the gap is smaller than it was on the 2022 GPT-3 generation.

---

## 3. A working theory of in-context learning

Why does putting examples in the prompt change behaviour at all? The model's weights are frozen. Three complementary explanations -- none alone complete -- are the closest thing the field has to consensus.

**1. Implicit task inference.** The pre-trained LM has seen many tasks expressed in text during training (Q&A pairs, code-comment pairs, translation pairs). At inference, examples in the prompt act as a *posterior update* over which "task" the rest of the prompt is drawn from. This is the Bayesian view from Xie et al. (2022): few-shot examples sharpen $P(\text{task} \mid \text{prompt})$.

**2. Implicit gradient descent inside attention.** A line of work (Akyurek et al., von Oswald et al., 2022-2023) shows that attention layers can implement one-step gradient descent on a linear regression task encoded in the prompt. The mechanistic claim is strong only for toy settings, but the suggestive picture -- "attention is doing some kind of fast adaptation" -- is useful intuition.

**3. Pattern matching plus copy.** The simplest explanation: induction heads (Olsson et al., 2022) copy patterns from earlier in the context. Few-shot prompts give the model a pattern to copy.

The practical consequences are the same regardless of which story you prefer:

- **More examples help, with steeply diminishing returns.** Most of the win comes from the first 2-4.
- **Distribution matters more than correctness.** Examples that cover the input space beat examples that are individually clever.
- **Recency wins on ties.** The last example in the prompt has outsized influence.
- **Position-of-correct-answer biases exist.** On multiple-choice tasks, models systematically favour position A or the most recent option, depending on the model.

---

## 4. The variance problem

Here is the uncomfortable truth nobody mentions in the marketing material. **Prompt accuracy can swing 10-30 points based on choices that should not matter:** the order of your examples, whether you write `Q:` or `Input:`, whether you wrap the answer in quotes.

![Prompt sensitivity to format and ordering](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig5_prompt_sensitivity.png)

Lu et al. (2022) called this *order sensitivity* and showed that on classification tasks the same model with the same examples in different orders ranged from near-random to near-state-of-the-art. Sclar et al. (2024) extended this to format sensitivity -- swapping `Q:/A:` for `Question:/Answer:` produces double-digit accuracy swings.

What this means in practice:

- **Always evaluate prompts on a held-out set of at least 50-100 examples.** A single anecdote tells you nothing.
- **Run multiple seeds / orders** when comparing two prompts. Report the median, not the best.
- **Fix the prompt format early** and treat changes as a versioned event, not a tweak.

A minimal evaluation harness:

```python
import statistics, random
from typing import Callable

def evaluate(
    build_prompt: Callable[[str, list], str],
    model_call: Callable[[str], str],
    cases: list[dict],
    examples_pool: list[dict],
    k: int = 4,
    n_seeds: int = 5,
) -> dict:
    """Return median / spread accuracy across n_seeds shufflings."""
    accs = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        shots = rng.sample(examples_pool, k)
        correct = 0
        for case in cases:
            out = model_call(build_prompt(case["input"], shots))
            if out.strip() == case["expected"].strip():
                correct += 1
        accs.append(correct / len(cases))
    return {
        "median": statistics.median(accs),
        "min": min(accs),
        "max": max(accs),
        "spread": max(accs) - min(accs),
    }
```

If `spread > 0.05`, your prompt is unstable and any single-run number you report is noise.

### How many shots is enough?

![Accuracy vs number of in-context examples](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig4_shots_saturation.png)

The pattern is consistent across task families: the first few shots matter a lot, returns saturate around $k \approx 4$ to $8$, and beyond $k \approx 16$ extra examples mostly cost tokens. Two practical rules:

- For classification with $\le 5$ classes, $k = 2 \times \text{num classes}$ is a good default.
- For generation tasks, $k = 2$ to $3$ high-quality demonstrations beats $k = 10$ mediocre ones every time.

Pick examples that are **diverse across the input distribution** and **clean / unambiguous on the output side** -- the model will copy your format faithfully, including bugs.

---

## 5. Self-consistency: turn the decoder into an ensemble

A single CoT chain can take a wrong turn at step 2 and propagate the mistake. Self-consistency (Wang et al., 2022) addresses this with a one-line fix: **sample $k$ different reasoning chains, then majority-vote the final answers.**

![Self-consistency: many paths, one vote](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig7_self_consistency.png)

Formally, given problem $x$ and $k$ sampled chains $z_1, \dots, z_k$ each yielding answer $a_i$:
$$\hat{a} \;=\; \arg\max_a \sum_{i=1}^{k} \mathbb{1}[a_i = a].$$

This is a Monte Carlo approximation to the marginal $\sum_z P(z \mid x)\, P(a \mid x, z)$. It works because **errors tend to be diverse but correctness tends to be convergent** -- many wrong reasoning paths land on different wrong answers, while correct paths agree.

```python
from collections import Counter

def self_consistency(
    model_call,
    prompt: str,
    k: int = 8,
    temperature: float = 0.7,
    extract_answer=lambda s: s.strip().split()[-1],
) -> dict:
    """Sample k CoT completions, majority-vote the answers."""
    answers = []
    for _ in range(k):
        out = model_call(prompt, temperature=temperature)
        answers.append(extract_answer(out))
    counts = Counter(answers)
    top, n = counts.most_common(1)[0]
    return {"answer": top, "confidence": n / k, "votes": dict(counts)}
```

Two notes from running this in production:

- **Temperature matters.** Use $T \in [0.5, 0.9]$. At $T = 0$ all samples collapse to the same chain; the ensemble degenerates.
- **The vote ratio is your confidence signal.** A 5/5 unanimous answer is much more trustworthy than a 3/5 plurality. Surface this number to downstream consumers; it is one of the cheapest reliability metrics you get.

**Tree-of-thought** (Yao et al., 2023) generalizes this: instead of sampling independent linear chains, explore a tree of partial reasoning steps, prune low-scoring branches, and search. It is more powerful but more expensive; reach for it only when self-consistency plateaus.

---

## 6. ReAct: reasoning + acting

Self-consistency improves what the model can do with what it already knows. **ReAct** (Yao et al., 2022) addresses the harder case: when the model needs *external information or actions*. The pattern interleaves three blocks in the output:

- **Thought.** Free-form reasoning about the current state.
- **Action.** A structured tool call: `search("..."); calc("..."); read_file("...")`.
- **Observation.** The tool's output, fed back into the prompt for the next iteration.

![ReAct: interleave Thought, Action, Observation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig6_react_pattern.png)

The loop terminates when the model emits a final `Answer:` instead of an `Action:`. Modern agent frameworks (LangChain agents, OpenAI's function calling, Anthropic's tool use) are all variations on this template -- often with the structured tool call moved to a JSON-typed API field rather than parsed from free text.

A minimal implementation that captures the essentials:

```python
import re
from typing import Callable

class ReActAgent:
    def __init__(self, model_call: Callable[[str], str],
                 tools: dict[str, Callable[[str], str]]):
        self.model_call = model_call
        self.tools = tools

    def _system(self) -> str:
        names = ", ".join(self.tools)
        return (
            "You can use tools: " + names + ".\n"
            "Use this format strictly:\n"
            "Thought: <reasoning>\n"
            "Action: <tool>(<argument>)\n"
            "Observation: <tool result>\n"
            "...repeat...\n"
            "Answer: <final answer>"
        )

    def run(self, question: str, max_steps: int = 6) -> str:
        ctx = f"{self._system()}\n\nQuestion: {question}\n"
        for _ in range(max_steps):
            out = self.model_call(ctx, stop=["\nObservation:"])
            ctx += out
            if "Answer:" in out:
                return out.split("Answer:")[-1].strip()
            m = re.search(r"Action:\s*(\w+)\((.*?)\)", out)
            if not m:
                return "(agent could not parse action)"
            name, arg = m.group(1), m.group(2).strip()
            obs = self.tools[name](arg) if name in self.tools \
                  else f"unknown tool: {name}"
            ctx += f"\nObservation: {obs}\n"
        return "(max steps reached)"
```

Three things that matter for a ReAct agent in production:

- **Stop tokens.** Stop generation at `\nObservation:` so the model does not hallucinate the tool's response.
- **Tool error handling.** Wrap tool calls; pass the *exception message* back as an observation. The model will often correct itself.
- **A step budget.** Always cap iterations. Without a budget, agents loop until they bankrupt you.

---

## 7. Building a prompt system

A handful of strong prompts is a script. A *system* of prompts is what survives team turnover, model upgrades, and four months of A/B tests. Three habits separate the two.

### Treat prompts like code

Version them, code-review them, store them in the repo, not in a copy-paste doc. A minimal registry:

```python
from pathlib import Path

class PromptRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, str] = {}

    def get(self, name: str, version: str = "latest") -> str:
        key = f"{name}@{version}"
        if key not in self._cache:
            path = self.root / name / f"{version}.txt"
            self._cache[key] = path.read_text()
        return self._cache[key]
```

Tag each call site with `(prompt_name, version)`. When you change a prompt, bump the version; old code keeps using the old prompt until you migrate.

### Evaluate before you ship

Every prompt that goes to production needs three artefacts:

- A **golden set** of 50-200 inputs with expected outputs (or a graded rubric).
- An **automated evaluator** (exact match, JSON schema check, or LLM-as-judge with its own pinned prompt).
- A **regression CI step** that runs both the current and the candidate prompt and refuses to merge if accuracy drops.

```python
def regression_check(old_prompt: str, new_prompt: str,
                     cases, model_call, judge) -> bool:
    old_acc = mean(judge(case, model_call(old_prompt + case.input))
                   for case in cases)
    new_acc = mean(judge(case, model_call(new_prompt + case.input))
                   for case in cases)
    return new_acc >= old_acc - 0.01   # 1pp regression budget
```

### Combine techniques deliberately

The big wins usually come from stacking:

- **Role + few-shot + format spec** for any structured-output task.
- **CoT + self-consistency** for any reasoning task you cannot afford to get wrong.
- **ReAct + retrieval** for any task that needs facts the model does not have.

Avoid stacking for its own sake. Each block costs tokens, increases latency, and adds a place for things to go wrong.

```python
def build_advanced(question: str, examples: list[dict],
                   *, role: str = "", use_cot: bool = True) -> str:
    parts: list[str] = []
    if role:
        parts.append(f"You are {role}.")
    parts.append("Solve the following problem.")
    for ex in examples:
        parts.append(f"Q: {ex['question']}")
        if use_cot and "reasoning" in ex:
            parts.append(f"Reasoning: {ex['reasoning']}")
        parts.append(f"A: {ex['answer']}\n")
    if use_cot:
        parts.append("Think step by step, then state the final answer "
                     "on a line beginning with 'A:'.")
    parts.append(f"Q: {question}\n")
    return "\n".join(parts)
```

---

## Worked example: a sentiment classifier you can actually deploy

Putting it all together for a small but realistic task.

```python
import json, re
from collections import Counter

SYSTEM = (
    "You are a careful text classifier. Reply with valid JSON only, "
    "no commentary."
)

EXAMPLES = [
    ("This movie is great, with outstanding acting.",
     {"sentiment": "positive", "confidence": 0.95}),
    ("Battery dies in three hours and the app crashes.",
     {"sentiment": "negative", "confidence": 0.92}),
    ("The package arrived on Tuesday.",
     {"sentiment": "neutral", "confidence": 0.88}),
]

FORMAT_SPEC = (
    'Output exactly: {"sentiment": "<positive|negative|neutral>", '
    '"confidence": <0.0-1.0>}'
)

def build_prompt(text: str) -> str:
    shots = "\n".join(
        f"Text: {x}\nOutput: {json.dumps(y)}" for x, y in EXAMPLES
    )
    return (
        f"[SYSTEM]\n{SYSTEM}\n\n"
        f"[FORMAT]\n{FORMAT_SPEC}\n\n"
        f"[EXAMPLES]\n{shots}\n\n"
        f"[INPUT]\nText: {text}\nOutput:"
    )

def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None

def classify(model_call, text: str, *, k: int = 5) -> dict:
    """Self-consistency over a sentiment classifier."""
    prompt = build_prompt(text)
    labels, confs = [], []
    for _ in range(k):
        parsed = parse(model_call(prompt, temperature=0.5))
        if parsed and "sentiment" in parsed:
            labels.append(parsed["sentiment"])
            confs.append(float(parsed.get("confidence", 0.5)))
    if not labels:
        return {"sentiment": "unknown", "confidence": 0.0, "votes": {}}
    counts = Counter(labels)
    top, n = counts.most_common(1)[0]
    return {"sentiment": top,
            "confidence": n / len(labels),
            "model_confidence": sum(confs) / len(confs),
            "votes": dict(counts)}
```

Notice how every section from this article appears: a role (system), a format spec, three diverse few-shot examples, a parser that tolerates the model's noise, and self-consistency to suppress label flips. None of it is exotic; together it is the difference between a 60% and a 90% pipeline.

---

## FAQ

**Is a longer prompt always better?**
No. After a point, extra context distracts the model and bloats latency and cost. Start short, add only what measurably helps on your eval set.

**How many shots should I use?**
Two to five for classification, one to three for generation. More than eight rarely helps and often adds variance from format issues. Always test, do not assume.

**Does CoT help on every task?**
No. It shines on multi-step reasoning (math, logic, code, multi-hop QA). On simple classification or fact lookup it adds noise and tokens for no gain.

**What temperature should I use?**
For deterministic outputs (classification, extraction): $T \in [0.0, 0.2]$. For balanced generation: $T \in [0.5, 0.7]$. For self-consistency sampling: $T \in [0.7, 0.9]$ -- you *want* path diversity.

**Does prompt engineering replace fine-tuning?**
It replaces a lot of the cases that used to need fine-tuning, especially for instruction-following tasks. Reach for fine-tuning when you need (a) consistent behaviour at scale where prompt drift is unacceptable, (b) a smaller / cheaper model that matches the big one on your domain, or (c) a behavioural change the base model resists.

**Does example order matter?**
A lot more than it should. The recency bias is real -- the last example influences the model most. Always evaluate at multiple orderings and report the median.

**Should I worry about prompt injection?**
Yes, the moment your prompt includes any text from outside your control (user input, retrieved documents, tool outputs). Treat untrusted text as data and never let it modify your instructions; this is its own topic and we will return to it in the agents and safety material.

---

## Series Navigation

- **Previous**: [Part 6 -- GPT and Generative Language Models](/en/nlp-gpt-generative-models/)
- **Next**: [Part 8 -- Model Fine-tuning and PEFT](/en/nlp-fine-tuning-peft/)
- [View all 12 parts in the NLP series](/tags/NLP/)
