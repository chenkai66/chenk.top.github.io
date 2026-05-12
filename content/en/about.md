---
_build:
  list: never
  publishResources: true
  render: always
title: "About"
type: about
layout: about
translationKey: "about"
---

## Chen Kai
Engineer, writer — building long-running, production-grade agent systems at Alibaba Cloud.

### Current work
- **AI4Marketing**: A full-stack AI marketing platform that turns a single sentence into an end-to-end campaign — email, ad creative, short video, TTS — across cross-border e-commerce channels. I design the control plane and sub-agent orchestration: the controller is the *only* human-facing interface.
- **AI4Science**: An autonomous research agent system that accepts scientific questions, reads papers, designs & runs experiments, and returns structured reports. Pipelines run for hours or days — sustained by shared memory architecture and harness-driven skill evolution.
- **llm-elevator**: An internal performance enhancement layer (not publicly available), comprising prompt template engine, agent runtime harness, and adversarial evaluation framework. It does *not* modify models — instead, it lifts real-task success rates and stability of frontier LLMs. All upstream projects depend on it.
- **DaaS (Documents-as-Skill)**: Converts unstructured technical documentation into callable, verifiable, context-aware agent skills.
- **chenk.top**: A technical blog: {{< blog-stats >}}. Every post is written twice from scratch — no translations. Chinese favors concision; English favors exposition.

### The core question I keep returning to
**How do long-running systems maintain resilience amid failure, model swaps, cost pressure, and infrastructure migration?**
Concrete levers: dynamic token budget allocation across providers, compressing failures into reusable skills, type-safe shared memory across agents, and bridging the critical gap between prompt demo and production-ready observability, replay, rate limiting, and trust.

### Beliefs I code by
- Tools expire; engineering judgment endures.
- Documentation is more valuable than code.
- Premature abstraction is the most expensive instinct an engineer can have.
- A small system that runs stably for 30 days beats a flashy one that works for 30 minutes.
- Treat agents as real systems — with explicit cost, defined failure modes, and operational overhead — not as "talking prompts."

If you're also building intelligent systems meant to last, drop me a line. I reply within 24 hours.
