---
title: "Prompt Engineering Complete Guide: From Zero to Advanced Optimization"
date: 2025-01-12 09:00:00
tags:
  - Prompt Engineering
  - LLM
categories: Large Language Models
lang: en
description: "Master prompt engineering from zero-shot basics to Tree of Thoughts, DSPy, and automated optimization. Includes benchmarks, code, and a debugging toolkit."
disableNunjucks: true
---

The same model, two prompts: one gets 17% accuracy on grade-school math, the other gets 78%. The difference is not magic — it is prompt engineering. This guide shows you the techniques that work, the research behind them, and how to systematically optimize prompts for production.

## What you will learn

- **Foundations** — zero-shot, few-shot, many-shot, task decomposition, and the five-block prompt skeleton.
- **Reasoning techniques** — Chain-of-Thought, Self-Consistency, Tree of Thoughts, Graph of Thoughts, ReAct.
- **Automation** — Automatic Prompt Engineering (APE), DSPy, LLMLingua compression.
- **Practical templates** — structured output, code generation, data extraction, multi-turn chat.
- **Evaluation and debugging** — metrics, A/B testing, error analysis, the failure-mode toolkit.

**Prerequisites.** Basic Python; experience calling any LLM API. No math background required.

---

## Why prompt engineering matters

When OpenAI released GPT-3 in 2020, researchers quickly noticed something surprising: the same model produced wildly different results depending on how you phrased a request. A poorly worded prompt generated nonsense; a carefully crafted one solved complex reasoning tasks. This was not a bug. It is a fundamental property of how these models learn.

Traditional programming runs on exact instructions: write a function, specify inputs and outputs, and the computer executes deterministically. Language models work differently. They predict the most likely continuation of text given the patterns learned from trillions of tokens. Your prompt does not command the model — it sets up a context that nudges its probability distribution toward useful outputs.

The stakes are high. A well-engineered prompt can reduce API costs by 10x through more efficient context usage. It can boost task accuracy from 40% to 90% on complex reasoning benchmarks. For production systems handling millions of requests, these gains translate to real business value.

## Anatomy of a production prompt

![Anatomy of a production prompt: five composable blocks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig1_prompt_anatomy.png)

Almost every prompt in production breaks down into the same five blocks: **role**, **context**, **instruction**, **examples**, and **output format**. Treat them as a skeleton. Swap the body for each task; keep the bones consistent. Reusing this structure makes evaluation, caching, and version control dramatically easier.

## Foundational techniques

### Zero-shot prompting

Zero-shot means asking the model to perform a task without any demonstrations. You rely entirely on the model's pre-training to interpret and execute the request.

```python
prompt = """Classify the sentiment of this review:
"The movie was boring and predictable. I wouldn't recommend it."

Sentiment:"""
```

The model has seen no examples of sentiment classification in this prompt. It must infer from training that "boring" and "wouldn't recommend" indicate negative sentiment.

**Where zero-shot works well:** simple, well-defined tasks the model has seen many times during pre-training (translation, summarization, basic Q&A); tasks with clear conventions; rapid prototyping when you have no examples ready.

**Where it fails:** domain-specific jargon, ambiguous instructions, tasks needing precise output formatting.

The original GPT-3 paper (Brown et al., 2020) reported 59% zero-shot accuracy on natural language inference; few-shot raised it to 70%. The gap is the cost of dropping examples.

**Three optimization tips that almost always help:**

1. *Be explicit about the task.* Instead of "Tell me about this review," say "Classify the sentiment as positive, negative, or neutral."
2. *Specify the output format.* Add "Return only one word: positive, negative, or neutral" to suppress verbose answers.
3. *Add constraints.* "Ignore sarcasm and focus on literal sentiment" prevents the most common pitfall.

A production-ready template:

```python
def zero_shot_classify(text: str, labels: list[str]) -> str:
    return f"""Task: classify the following text into exactly one of:
{', '.join(labels)}

Text: {text}

Category (respond with only the category name):"""
```

### Few-shot prompting

Few-shot provides 2–10 examples before the actual query. This dramatically improves accuracy by establishing the pattern the model should follow.

```python
prompt = """Classify the sentiment of these reviews:

Review: "Absolutely loved this film! Best movie I've seen all year."
Sentiment: positive

Review: "It was okay. Nothing special but not terrible either."
Sentiment: neutral

Review: "Waste of time and money. Horrible acting."
Sentiment: negative

Review: "The cinematography was stunning and the story kept me engaged."
Sentiment:"""
```

The examples do three things at once: they pin down the output format (single word, lowercase), they cover edge cases (mixed reviews map to "neutral"), and they prime the model with the right semantic patterns.

**The mental model.** Few-shot examples are *soft conditioning*. The next-token mechanism looks for patterns in the demonstrations and applies them to the new input. You are essentially programming through demonstration.

**Choosing examples.** Liu et al. (2021) on "What Makes Good In-Context Examples?" found that:

- *Diversity beats volume.* Five diverse examples outperform twenty similar ones.
- *Hard examples help.* Include the edge cases the model is likely to mishandle.
- *Order matters.* Put the most relevant example closest to the query.

A simple selector that combines similarity with diversity:

```python
from sklearn.metrics.pairwise import cosine_similarity

def select_examples(query: str, pool: list[dict], k: int = 5) -> list[dict]:
    """Pick k examples that are relevant to the query but diverse from
    each other. Uses cosine similarity over precomputed embeddings."""
    q_emb = get_embedding(query)
    embs = [get_embedding(ex["text"]) for ex in pool]

    sims = cosine_similarity([q_emb], embs)[0]
    candidates = sims.argsort()[-k * 2:][::-1]   # 2k most relevant

    selected = [candidates[0]]
    for idx in candidates[1:]:
        if len(selected) >= k:
            break
        if all(cosine_similarity([embs[idx]], [embs[s]])[0][0] < 0.9
               for s in selected):
            selected.append(idx)
    return [pool[i] for i in selected]
```

On SuperGLUE, GPT-3 went from 69.5% (zero-shot) to 71.8% (one-shot) to 75.2% (32-shot). Diminishing returns kick in around 10–15 examples for most tasks.

### Many-shot prompting

Anthropic's 2024 work showed that 100K+ token contexts unlock *many-shot* prompting with hundreds of examples. This bridges the gap between few-shot prompting and traditional fine-tuning.

A typical scenario: you are building a code reviewer that catches company-specific anti-patterns. Instead of fine-tuning (which needs infrastructure, data pipelines, and ongoing maintenance), you place 200 reviewed examples directly in the prompt:

```python
prompt = """Review the following code for anti-patterns.

[Example 1]
Code: def get_data(): return db.query('SELECT * FROM users')
Issue: Raw SQL injection risk. Use parameterized queries.
Severity: High

[Example 2]
Code: result = api.call(); result.field
Issue: No error handling. Add try/except for network failures.
Severity: Medium

... (198 more examples) ...

Now review this code:
{user_code}

Issue:"""
```

**Why it works.** With hundreds of examples the model effectively learns a task-specific distribution — similar to fine-tuning, but without weight updates. More examples cover more edge cases, reducing ambiguity. Format consistency becomes near-perfect because the model has seen the pattern dozens of times.

Anthropic's findings: 500-shot prompting approaches fine-tuned performance on specialized tasks; gains plateau around 200–300 examples; works best with Claude's 200K window.

**Trade-offs.** A 200K-token prompt at GPT-4 list pricing is roughly $2 per request. Latency suffers. The fix is **prompt caching** (Claude, GPT-4): the static prefix is cached and reused across requests, so you pay full price once and a discounted rate after.

### Task decomposition

Complex tasks fail when you ask the model to do too much in one pass. Decomposition breaks a hard problem into simpler sub-problems with clearer success criteria.

Instead of "Analyze this legal contract and extract all obligations," do:

```python
sections = extract_sections(contract)               # Step 1: identify sections

obligations = []                                    # Step 2: per-section extraction
for section in sections:
    prompt = f"""Legal text: {section}

List all obligations. Format each as:
[Party] must [Action] [Conditions]"""
    obligations.extend(llm_call(prompt))

classified = classify_obligations(obligations)      # Step 3: classify by type
summary = summarize_obligations(classified)         # Step 4: synthesize
```

**Why it helps.** Each sub-task is simpler and easier to validate. You can inspect intermediate outputs. When something fails, you know exactly which step broke. GitHub Copilot Workspace uses this exact pattern: understand the codebase, identify affected files, generate per-file edits, synthesize a complete solution — each step driven by a specialized prompt.

Three patterns worth keeping in your toolbox:

```python
def etl_pattern(data, extract_p, transform_p):
    extracted = llm_call(extract_p.format(data=data))
    return llm_call(transform_p.format(data=extracted))

def map_reduce(items, map_p, reduce_p):
    mapped = [llm_call(map_p.format(item=i)) for i in items]
    return llm_call(reduce_p.format(items=mapped))

def validate_retry(prompt, validator, max_retries=3):
    for _ in range(max_retries):
        result = llm_call(prompt)
        if validator(result):
            return result
        prompt += f"\n\nPrevious attempt was invalid: {result}\nTry again:"
    raise ValueError("Max retries exceeded")
```

## Three paradigms compared

![Zero-shot, few-shot, and chain-of-thought prompting compared on accuracy and token cost](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig2_paradigm_compare.png)

The same arithmetic problem expressed three ways. Zero-shot is cheapest but weakest on multi-step reasoning. Few-shot pays for ~8x more tokens to lift accuracy by ~16 points. Chain-of-thought spends slightly fewer tokens than few-shot — and crushes both with 78.7% accuracy on GSM8K-class problems (Wei et al., 2022; Kojima et al., 2022).

## Advanced reasoning techniques

### Chain-of-Thought (CoT)

CoT asks the model to *show its work*: generate intermediate reasoning steps before producing the final answer. This single change yields massive improvements on math, logic, and multi-step reasoning.

![CoT reasoning: explicit steps cut error rates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig3_cot_flow.png)

Wei et al. (2022) found that adding "Let's think step by step" lifted GSM8K accuracy from 17.1% to 78.2%.

Without CoT:

```python
prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A:"""
# Often outputs: "10"  (wrong)
```

With CoT:

```python
prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step."""
# Outputs:
# Roger starts with 5 balls.
# He buys 2 cans, each with 3 balls -> 6 new balls.
# Total: 5 + 6 = 11.
# Answer: 11   (correct)
```

**Why CoT works.** The leading interpretability theory: language models perform implicit computation across the layers of a single forward pass, but each pass has a fixed compute budget. By generating intermediate tokens, the model gets *more* forward passes — one per generated token — and can spread the computation over them. It is the model's working memory.

**Three variants worth knowing.**

*Zero-shot CoT* — just append "Let's think step by step." Surprisingly effective across diverse tasks:

```python
def zero_shot_cot(question: str) -> str:
    return f"{question}\n\nLet's think step by step."
```

*Few-shot CoT* — provide demonstrations that include reasoning chains:

```python
prompt = """Q: If 5 machines take 5 minutes to make 5 widgets, how long
   would 100 machines take to make 100 widgets?
A: Let's think step by step.
   - 5 machines make 5 widgets in 5 minutes
   - so 1 machine makes 1 widget in 5 minutes
   - 100 machines make 100 widgets in 5 minutes
Answer: 5 minutes

Q: A farmer has 15 sheep and all but 8 die. How many are left?
A: Let's think step by step.
   - "all but 8 die" means 8 survive
   - so 8 sheep are left
Answer: 8

Q: {new_question}
A: Let's think step by step."""
```

*Structured CoT* — enforce a fixed reasoning template:

```
1. What is known: ...
2. What is unknown: ...
3. Relevant principles: ...
4. Step-by-step solution: ...
5. Final answer: ...
```

**Benchmark results from Wei et al. (2022):**

| Benchmark      | Baseline | CoT   | Δ      |
|----------------|----------|-------|--------|
| GSM8K (math)   | 17.1%    | 78.2% | +61pp  |
| SVAMP (math)   | 63.7%    | 79.0% | +15pp  |
| CommonsenseQA  | 72.5%    | 78.1% | +6pp   |
| StrategyQA     | 54.3%    | 66.1% | +12pp  |

**When CoT helps.** Multi-step reasoning, problems requiring intermediate calculations, tasks where the reasoning path matters for explainability, decisions with trade-offs.

**When CoT does not help.** Simple lookups, tasks where the model lacks the requisite knowledge (CoT cannot fix missing facts), and tasks where short answers are required (the reasoning tokens are pure overhead).

A reusable engine:

```python
class ChainOfThoughtEngine:
    def __init__(self, model, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def solve(self, question: str, mode: str = "zero-shot"):
        if mode == "zero-shot":
            prompt = f"{question}\n\nLet's think step by step."
        elif mode == "few-shot":
            prompt = self._build_few_shot_prompt(question)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        response = self.model.generate(
            prompt, temperature=self.temperature, max_tokens=512,
        )
        return self._extract_answer(response), response

    def _extract_answer(self, response: str) -> str:
        for marker in ("Answer:", "Therefore,", "The answer is"):
            if marker in response:
                tail = response.split(marker)[-1].strip()
                return tail.split(".")[0].split("\n")[0].strip()
        return response.strip().split("\n")[-1]
```

### Self-Consistency

A single CoT chain might wander into a wrong answer. Self-consistency (Wang et al., 2022) generates several chains and picks the *majority* answer.

![Self-consistency: sample many reasoning paths, vote on the answer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig5_self_consistency.png)

The intuition is purely Bayesian: if 7 of 10 different reasoning chains reach the same answer, that answer is much more likely correct than any single chain.

```python
from collections import Counter

def self_consistency(question: str, n: int = 5,
                     temperature: float = 0.7) -> tuple[str, float]:
    """Sample n reasoning chains and majority-vote on the final answer."""
    prompt = f"{question}\n\nLet's think step by step."
    answers = []
    for _ in range(n):
        response = llm_call(prompt, temperature=temperature, max_tokens=512)
        answers.append(extract_answer(response))

    most_common, count = Counter(answers).most_common(1)[0]
    return most_common, count / n
```

A real example:

```
Q: "If you overtake the person in 2nd place, what place are you in?"

Chain A: "You overtake 2nd, so you're now 2nd."           -> 2nd  ✓
Chain B: "You were behind 2nd, now ahead, so 1st."        -> 1st  ✗
Chain C: "Overtaking 2nd means taking their position."    -> 2nd  ✓
Chain D: "You pass the person in 2nd. You're now 2nd."    -> 2nd  ✓
Chain E: "You overtake 2nd, making you 1st."              -> 1st  ✗

Majority: 2nd place  (3/5 = 0.6 confidence)
```

Wang et al. (2022) reported on GSM8K: standard CoT 74.4% → self-consistency (n=40) 83.7%. On CommonsenseQA: 78.1% → 81.5%. The technique pays for itself on hard tasks.

**Cost.** Self-consistency multiplies inference cost by *n*. Pick *n* by stakes:

| Task criticality | Suggested *n* | Cost  |
|------------------|---------------|-------|
| Exploratory      | 3             | 3x    |
| Production       | 5             | 5x    |
| High-stakes      | 10–20         | 10–20x|

Adaptive variant — start cheap, escalate only when confidence is low:

```python
def adaptive_self_consistency(question: str, threshold: float = 0.7):
    answer, conf = self_consistency(question, n=3)
    if conf >= threshold:
        return answer, conf
    return self_consistency(question, n=10)
```

A more sophisticated **weighted vote** asks the model to also score each chain's logical quality and weights votes accordingly. It costs an extra inference per chain but discounts shaky reasoning automatically.

### Tree of Thoughts (ToT)

ToT (Yao et al., 2023) extends CoT by treating reasoning as *search through a state space*. Instead of a single chain, the model explores a tree, scores each branch, and backtracks from dead ends.

![Tree of Thoughts: branch, score, and backtrack on the Game of 24](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig6_tree_of_thoughts.png)

A simplified DFS implementation:

```python
def tree_of_thoughts(problem: str, depth: int = 3, breadth: int = 3):
    def explore(state: str, d: int) -> int:
        if d >= depth:
            return evaluate_solution(state)
        thoughts = generate_thoughts(state, k=breadth)
        scored = [(t, explore(state + "\n" + t, d + 1)) for t in thoughts]
        return max(scored, key=lambda x: x[1])[1]

    return explore(problem, 0)


def generate_thoughts(state: str, k: int = 3) -> list[str]:
    prompt = f"""Current reasoning:
{state}

Generate {k} possible next steps:
1."""
    response = llm_call(prompt, temperature=0.8, max_tokens=300)
    return [t.strip() for t in response.split("\n") if t.strip()][:k]


def evaluate_solution(state: str) -> int:
    prompt = f"""Rate this reasoning on a scale of 1-10 for logical
coherence, progress toward the solution, and likely correctness.

{state}

Score (integer 1-10):"""
    return int(llm_call(prompt, temperature=0, max_tokens=5))
```

**Game of 24 example.** Use 4, 9, 10, 13 with +, -, *, / to make 24:

```
Root: {4, 9, 10, 13}
├─ 13 - 9 = 4   -> {4, 4, 10}        v=8 (kept)
│  └─ 10 - 4 = 6   -> {4, 6}         v=9
│     └─ 6 * 4 = 24                  SOLVED
├─ 10 - 4 = 6   -> {6, 9, 13}        v=5 (explore later)
└─ 9 + 10 = 19  -> {4, 13, 19}       v=2 (pruned)
```

Yao et al. (2023) benchmarks:

| Task            | CoT   | ToT   | Δ     |
|-----------------|-------|-------|-------|
| Game of 24      | 7.3%  | 74%   | +66pp |
| Creative writing| 7.3   | 7.9   | +0.6  |
| Crosswords      | 15.6% | 78%   | +62pp |

**The cost.** Breadth-3 depth-4 search is ~80 LLM calls per problem. ToT pays off only when (a) there are multiple plausible solution paths and (b) self-evaluation is reliable for the task. Use it for combinatorial puzzles, planning, and constraint satisfaction. Skip it for straightforward Q&A.

A production-friendly best-first version uses a priority queue with a hard call-count cap so a single problem cannot blow your budget.

### Graph of Thoughts (GoT)

GoT (Besta et al., 2023) generalizes ToT to arbitrary DAGs. Thoughts can *merge* (combining multiple branches) or *iterate* (refining a single thought across rounds), enabling reasoning patterns trees cannot express.

A canonical example — multi-document summarization:

```
Documents -> per-doc summary  ┐
            per-doc summary  ┼-> merge themes -> final synthesis
            per-doc summary  ┘
```

Each per-document summary is independent and can run in parallel. The merge step combines them. This is a graph, not a tree.

```python
def graph_of_thoughts_summarize(documents: list[str]) -> str:
    summaries = [llm_call(f"Summarize:\n{d}") for d in documents]
    themes = [llm_call(f"Extract themes:\n{s}") for s in summaries]
    merged = llm_call(f"Merge these themes:\n" + "\n\n".join(themes))
    return llm_call(f"Synthesize a final summary:\n{merged}")
```

On a 32-number sorting task, Besta et al. reported 89% accuracy at 62% lower cost than ToT — the merge operations remove redundant exploration.

### ReAct (Reason + Act)

ReAct (Yao et al., 2022) interleaves *thinking* with *acting*. The model alternates between reasoning steps and tool calls, observing the result of each action before deciding what to do next.

```
Thought: I need the population of Paris.
Action: search("Paris population")
Observation: 2.16 million (2019)
Thought: Now I need Tokyo's population.
Action: search("Tokyo population")
Observation: 37.4 million (2021)
Thought: Tokyo is larger.
Action: finish("Tokyo's population is larger than Paris.")
```

ReAct fixes three things language models are bad at on their own: stale knowledge (training data has a cutoff), precise calculations, and access to private data. A minimal agent:

```python
class ReActAgent:
    def __init__(self, model, tools: dict, max_steps: int = 10):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        trajectory = [f"Task: {task}"]
        for _ in range(self.max_steps):
            response = self.model.generate(
                self._build_prompt(trajectory), temperature=0,
            )
            thought, action, action_input = self._parse(response)
            trajectory.append(f"Thought: {thought}")
            trajectory.append(f"Action: {action}[{action_input}]")

            if action == "finish":
                return action_input

            tool = self.tools.get(action)
            obs = tool(action_input) if tool else f"Unknown action: {action}"
            trajectory.append(f"Observation: {obs}")
        return "Step budget exhausted without finishing."
```

Performance on HotpotQA (multi-hop QA): standard prompting 28.7% → CoT 32.9% → ReAct 37.4%. On AlfWorld (interactive environment): 12% → 34%.

**Best practices.**

- *Document tools well.* The model picks tools by their docstrings.
- *Truncate observations.* Long search results can blow the context window.
- *Cap steps.* Always set a hard limit to prevent infinite loops.
- *Return descriptive errors.* Let the model recover instead of crashing.

## Prompts are not robust by default

![Prompt sensitivity: format and order can swing accuracy by 20+ points](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig4_prompt_sensitivity.png)

Same model, same examples, same task. Just changing the *format* of the demonstrations or the *order* of the few-shot examples can swing accuracy by 20+ points (Lu et al., 2022; Sclar et al., 2024). This is why empirical evaluation is non-negotiable. *Test your prompt under multiple orderings before declaring a winner.*

## Optimization and automation

Manual prompt engineering does not scale beyond a handful of tasks. The following techniques automate the process.

### Automatic Prompt Engineering (APE)

APE (Zhou et al., 2022) automates the search for the best prompt:

1. **Generate** candidate prompts using an LLM, given the task description and a handful of examples.
2. **Evaluate** each candidate on a validation set.
3. **Select** the highest-performing one.

```python
def automatic_prompt_engineering(task_description, train_examples,
                                 val_examples, num_candidates=20):
    meta_prompt = f"""Task: {task_description}

Examples:
{format_examples(train_examples[:5])}

Generate {num_candidates} different prompts that could solve this task.
Each prompt should use a different approach or phrasing.

Prompts:"""
    candidates = parse_prompts(
        llm_call(meta_prompt, temperature=1.0, max_tokens=2000)
    )

    results = []
    for prompt in candidates:
        correct = sum(
            normalize(llm_call(f"{prompt}\n\nInput: {x}\nOutput:",
                               temperature=0)) == normalize(y)
            for x, y in val_examples
        )
        results.append((prompt, correct / len(val_examples)))

    return max(results, key=lambda x: x[1])
```

Zhou et al. found APE-discovered prompts beating human-written baselines by 3–8 percentage points across many tasks. The key insight: APE explores phrasings humans would not try, optimizes directly on your data, and can test hundreds of candidates cheaply.

An *iterative* extension feeds the current best prompt back into the meta-prompt and asks for refinements — a kind of hill climbing in prompt space.

### DSPy: declarative prompts as code

DSPy (Khattab et al., 2023) treats prompting as a programming problem. Instead of hand-writing prompts, you write *programs that compose prompts*, and a compiler tunes them automatically.

The core abstractions:

- **Signatures** — typed input/output specs.
- **Modules** — composable prompt templates.
- **Optimizers** — automatic tuners that pick demonstrations and instructions.

A sentiment classifier:

```python
import dspy

class SentimentSignature(dspy.Signature):
    """Classify sentiment of text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")

class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SentimentSignature)

    def forward(self, text: str):
        return self.predictor(text=text)

dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))
classifier = SentimentClassifier()
print(classifier("This movie was amazing!").sentiment)  # "positive"
```

DSPy can automatically tune the underlying prompt by *bootstrapping* demonstrations from a training set:

```python
from dspy.teleprompt import BootstrapFewShot

train_data = [
    dspy.Example(text="Great product!", sentiment="positive"),
    dspy.Example(text="Terrible service.", sentiment="negative"),
    # ...
]

optimized = BootstrapFewShot(metric=exact_match).compile(
    SentimentClassifier(), trainset=train_data,
)
```

Multi-stage programs compose naturally:

```python
class MultiHopQA(dspy.Module):
    """Answer questions that require multiple reasoning steps."""
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.gen_query = dspy.ChainOfThought("question -> search_query")
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str):
        q = self.gen_query(question=question).search_query
        ctx = self.retrieve(q).passages
        return self.answer(context=ctx, question=question).answer
```

The DSPy compiler optimizes all three sub-prompts together. The trade-offs are real: a learning curve, less direct control over wording, and an upfront optimization cost. Use DSPy when you have a stable training set and a real evaluation metric.

### LLMLingua: prompt compression

LLMLingua (Jiang et al., 2023) compresses prompts to cut cost while preserving accuracy. A small LLM scores each token's importance; low-scoring tokens are removed.

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()
compressed = compressor.compress(
    original_prompt,
    rate=0.5,            # target 50% compression
    target_token=200,    # or specify exact budget
)
```

The underlying technique is conditional perplexity: remove a token, measure how much the perplexity of the next prediction increases, and keep only the tokens that move the needle.

Reported impact:

- Question answering at 2x compression: 2–3% accuracy drop, 50% cost savings, 1.4x latency improvement.
- RAG at 4x compression: 5–7% accuracy drop, 75% cost savings.

Best fit: long-context scenarios (RAG, document analysis) where the cost-vs-quality trade-off is worth it. Avoid for legal or medical text where every word matters.

A sketch of an *adaptive* compressor that allocates budget per section by priority:

```python
class AdaptiveCompressor:
    def __init__(self, base): self.base = base

    def compress(self, prompt: str, budget: int) -> str:
        sections = split_sections(prompt)
        budgets = {
            "instruction": int(budget * 0.3),  # never starve
            "examples":    int(budget * 0.4),
            "context":     budget - int(budget * 0.7),  # squeeze hardest
        }
        return merge({
            k: self.base.compress(sections[k], target_token=b)
            for k, b in budgets.items()
        })
```

## Practical templates

![Reusable prompt template library: same skeleton, swapped per task](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/prompt-engineering-complete-guide/fig7_template_library.png)

The figure above shows the same five-block skeleton specialized for six common tasks. The benefit is not aesthetic — it makes evaluation, caching, and version control dramatically easier.

### Structured output

Getting valid JSON out of an LLM is famously tricky. Three strategies, in order of robustness:

```python
def generate_structured(data: str, schema: dict) -> dict:
    """Schema-first prompting with retry on validation failure."""
    prompt = f"""Generate a JSON object matching this schema:
{json.dumps(schema, indent=2)}

Rules:
- All required fields must be present
- Use the specified types
- Enum fields must use one of the listed values
- No extra fields

Input data:
{data}

Output valid JSON only:"""
    response = llm_call(prompt, temperature=0)
    try:
        parsed = json.loads(response)
        validate_against_schema(parsed, schema)
        return parsed
    except Exception as e:
        return retry_with_feedback(prompt, response, str(e))
```

Strategy 2 — few-shot with valid examples — works when the schema is simple. Strategy 3 — provider-native function/tool calling — is the most reliable when available; the API guarantees the JSON is well-formed.

### Code generation with self-test

```python
def generate_code(task: str, language: str = "python", tests=None) -> str:
    prompt = f"""Write {language} code for this task.

Task: {task}

Requirements:
- Clear comments
- Handle edge cases
- Follow {language} best practices
{format_test_cases(tests) if tests else ''}

Provide complete, runnable code:
```{language}
"""
    code = extract_code_block(llm_call(prompt, temperature=0.3))
    if tests:
        results = run_tests(code, tests, language)
        if not all(r.passed for r in results):
            code = debug_and_fix(code, results, prompt)
    return code
```

The pattern is *generate → test → repair*. The repair prompt feeds the failing test back to the model with the original instructions intact.

### Multi-turn conversation management

Long conversations exceed the context window. The fix: keep a sliding window of recent messages plus an LLM-generated summary of the older ones.

```python
class ConversationManager:
    def __init__(self, model, system_prompt: str, max_tokens: int = 4000):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        context = self._get_context()
        response = self.model.generate(self._build_prompt(context))
        self.history.append({"role": "assistant", "content": response})
        return response

    def _get_context(self) -> list[dict]:
        ctx = [{"role": "system", "content": self.system_prompt}]
        used = count_tokens(self.system_prompt)
        for msg in reversed(self.history):
            t = count_tokens(msg["content"])
            if used + t > self.max_tokens:
                break
            ctx.insert(1, msg)
            used += t
        return ctx

    def summarize_old_turns(self) -> None:
        if len(self.history) < 10:
            return
        old = self.history[:6]
        summary = self.model.generate(
            f"Summarize concisely:\n{format_messages(old)}\n\nSummary:"
        )
        self.history = (
            [{"role": "system", "content": f"Earlier conversation: {summary}"}]
            + self.history[6:]
        )
```

## Evaluation and debugging

Prompt engineering is empirical. Without metrics you are guessing.

**Metrics, in order of cost.**

```python
def exact_match(pred, truth):
    return normalize(pred) == normalize(truth)

def f1(pred, truth):
    p = set(normalize(pred).split())
    t = set(normalize(truth).split())
    if not p or not t:
        return float(p == t)
    common = p & t
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(t)
    return 2 * prec * rec / (prec + rec)

def semantic_similarity(pred, truth, model="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer, util
    m = SentenceTransformer(model)
    return util.cos_sim(m.encode(pred), m.encode(truth)).item()

def llm_as_judge(pred, truth, criteria):
    return parse_score(llm_call(f"""Evaluate this output.
Task: {criteria['task']}
Expected: {truth}
Actual: {pred}

Score 0-10 with brief justification:"""))
```

Pick the cheapest metric that correlates with what you actually care about.

### A/B testing prompt variants

```python
class PromptExperiment:
    def __init__(self, test_set, metrics):
        self.test_set = test_set
        self.metrics = metrics

    def evaluate(self, prompt_fn):
        return {
            m.__name__: sum(
                m(llm_call(prompt_fn(x), temperature=0), y)
                for x, y in self.test_set
            ) / len(self.test_set)
            for m in self.metrics
        }

    def compare(self, variants: dict) -> "pd.DataFrame":
        import pandas as pd
        rows = [{"prompt_name": name, **self.evaluate(fn)}
                for name, fn in variants.items()]
        df = pd.DataFrame(rows)
        return df.sort_values(by=df.columns[1], ascending=False)
```

Always test on data the prompt designer has not seen.

### Debugging failing prompts

When prompts misbehave, run through this checklist:

| Symptom                         | Likely cause                | Fix                                       |
|---------------------------------|-----------------------------|-------------------------------------------|
| Vague or wandering output       | Ambiguous instructions      | Add specific constraints and examples     |
| Output ignores some requirement | Contradictory instructions  | Resolve the conflict, set priorities      |
| Output is wrong despite trying  | Missing context             | Provide grounding facts or retrieved docs |
| Format mismatch                 | No format spec              | Specify schema with an example            |
| Different answers each run      | Too complex for one pass    | Decompose into multiple steps             |

A small detector that flags the most common issues:

```python
class PromptDebugger:
    AMBIGUOUS = {"relevant", "appropriate", "good", "bad",
                 "some", "few", "many", "stuff", "things"}
    CONFLICT  = {"but", "however", "although", "except"}

    def check_ambiguity(self, prompt):
        return any(w in prompt.lower() for w in self.AMBIGUOUS)

    def check_conflicts(self, prompt):
        return sum(w in prompt.lower() for w in self.CONFLICT) >= 2

    def infer_format(self, text):
        t = text.strip()
        if t.startswith("{") and t.endswith("}"): return "json_object"
        if t.startswith("[") and t.endswith("]"): return "json_array"
        if "\n-" in t or "\n*" in t or "\n1." in t: return "list"
        return "prose"
```

### Error analysis

Bucket failures by mode. The categories below cover the vast majority of mistakes:

```python
def categorize_error(pred: str, truth: str) -> str:
    p, t = normalize(pred), normalize(truth)
    if not p:
        return "empty_output"
    if p in t or t in p:
        return "partial_match"
    p_w, t_w = set(p.split()), set(t.split())
    overlap = len(p_w & t_w) / max(len(p_w), len(t_w))
    if overlap > 0.5:
        return "semantic_error"
    if overlap > 0:
        return "partial_hallucination"
    return "complete_hallucination"
```

Then attack the largest bucket first.

## Common pitfalls

| Pitfall                        | Fix                                                                             |
|--------------------------------|---------------------------------------------------------------------------------|
| Vague instructions             | List concrete dimensions to optimize: clarity, length, voice, format.            |
| Assuming knowledge             | Include the code, document, or data the prompt refers to.                        |
| Overly long prompts            | Split, summarize, or use RAG. The "lost in the middle" effect is real.           |
| Ignoring output format         | Specify schema, units, and language explicitly with an example.                  |
| No validation                  | Wrap calls in a `validate → retry → fail` loop.                                  |

## FAQ

**Should I use higher or lower temperature?** 0 for tasks needing consistency (classification, extraction, math, code). 0.7–0.8 for creative tasks. 1.0+ rarely. Default to 0 for structured tasks, 0.7 for creative ones.

**How many examples in few-shot?** 2–3 for simple tasks, 5–7 is the sweet spot for most, 10+ only if the examples are diverse. Past 50 you should consider fine-tuning.

**When should I fine-tune instead?** When you have 1,000+ high-quality labeled examples, the task is highly specialized, latency or cost are critical, and you have plateaued with prompt engineering. Otherwise, prompt — it iterates 1000x faster.

**How do I prevent hallucinations?** Ground with retrieved context, instruct the model to say "I don't know," request quoted citations, lower the temperature, and add a verification pass.

**How do I handle long documents?** Chunk + map-reduce, retrieval-augmented generation, hierarchical summarization, or models with large context windows (Claude 3 200K, Gemini 1.5 1M). RAG is usually the right default.

**Do prompts transfer across models?** Universal techniques (clear instructions, few-shot, format specs, CoT) transfer well. Exact phrasing, format preferences (Claude likes XML), and tool-calling syntax do not. Always test on the target model.

**Can I automate optimization?** Yes — APE, DSPy, and genetic search all work. Start manual to understand the task, then automate.

**XML, JSON, or plain text for prompts?** Plain text for simple prompts. JSON for structured I/O. XML for complex multi-part prompts (especially with Claude). All three are fine — pick the one your downstream parser expects.

**CoT vs ToT vs GoT — when to use which?**

| Technique | Structure        | When to use                                    | Cost     |
|-----------|------------------|------------------------------------------------|----------|
| **CoT**   | Linear chain     | Multi-step reasoning, math, logic              | 1–2x     |
| **ToT**   | Tree search      | Multiple solution paths, planning, puzzles     | 5–50x    |
| **GoT**   | Arbitrary DAG    | Parallel processing, merging insights          | Varies   |

**The future?** Multimodal prompting, tighter automation (DSPy, APE), aggressive compression, meta-prompting (prompts that generate prompts), embodied agents. The skill will not disappear — it will shift from manual crafting to designing optimization objectives, evaluation harnesses, and orchestration.

## Closing

Prompt engineering started as trial-and-error and has matured into a discipline backed by research and reusable frameworks. The fundamentals — clear instructions, well-chosen examples, structured output — apply universally. Advanced techniques like Chain-of-Thought and Tree of Thoughts unlock capabilities that look impossible with naive prompting. APE and DSPy scale these practices to production.

But techniques alone are not enough. Effective prompt engineering requires:

- **Empiricism.** Test everything. What works on one model or task may fail on another.
- **Iteration.** Your first prompt will rarely be your best. Refine based on real failures.
- **Evaluation.** Without metrics you are guessing.
- **Context.** Understand the model's strengths, the task's requirements, and the trade-off between cost, latency, and quality.

Start simple. Measure constantly. Iterate relentlessly. The best prompt is the one that reliably solves your problem — not the cleverest one.

## References

- Brown et al., 2020. *Language Models are Few-Shot Learners.* NeurIPS.
- Wei et al., 2022. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Kojima et al., 2022. *Large Language Models are Zero-Shot Reasoners.* NeurIPS.
- Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Yao et al., 2023. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS.
- Besta et al., 2023. *Graph of Thoughts: Solving Elaborate Problems with Large Language Models.* AAAI.
- Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
- Zhou et al., 2022. *Large Language Models Are Human-Level Prompt Engineers.* ICLR.
- Khattab et al., 2023. *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.* arXiv.
- Jiang et al., 2023. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP.
- Liu et al., 2021. *What Makes Good In-Context Examples for GPT-3?*. arXiv.
- Lu et al., 2022. *Fantastically Ordered Prompts and Where to Find Them.* ACL.
- Sclar et al., 2024. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR.
- Min et al., 2022. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP.
