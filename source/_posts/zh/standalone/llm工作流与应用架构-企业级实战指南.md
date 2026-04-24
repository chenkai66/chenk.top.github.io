---
title: "LLM 工作流与应用架构：企业级实战指南"
date: 2024-05-09 09:00:00
tags:
  - Workflow
  - Architecture
  - RAG
  - LLM
categories: Large Language Models
lang: zh-CN
description: "从一次 API 调用到一个生产级 LLM 平台：工作流模式、RAG、模型路由、部署拓扑、成本杠杆、可观测性、企业集成——以及那些真正决定成败的取舍。"
disableNunjucks: true
---

绝大多数 LLM 教程，恰恰在真正有意思的工作开始之前就结束了。它们告诉你怎么调用 chat completion 接口，怎么挂一个向量库，怎么用 Streamlit 包成一个 demo——这些都没错，但都不是凌晨三点一万人涌入、每隔一条回答就出现幻觉时让你头疼的那部分。

这篇文章只谈 demo 之后的事。立场是明确的：一个生产级 LLM 系统，本质上是一个普通的分布式系统，只不过被强行塞了一个非确定性的组件进去——而工程上绝大部分的精力，都用来"圈住"这种不确定性。下面会按七个维度展开：应用分层、工作流模式、RAG/微调/Prompt 的取舍、部署拓扑、成本、可观测性、企业集成。每一节都尽量短、尽量具体，只讨论真正能撬动指标的那几根杠杆。

## 你将学到

- 如何把 LLM 应用拆成五层，让"非确定性"被关在一层里
- 四种工作流模式（chain、branch、loop、parallel），以及它们各自不该用在哪里
- 一个能扛住现实考验的"Prompt vs RAG vs 微调"决策规则
- 一个带语义缓存、LLM Gateway、多档模型池的部署拓扑
- 六根成本杠杆，以及拉动它们的正确顺序
- 一个把四个 LLM 专属信号补回去的可观测性方案
- 让大多数试点上不了线的六个企业集成"门槛"

## 阅读前提

你写过 REST 接口、用过异步 Python 或 Node、知道 Docker 大致是什么、调用过至少一家 LLM 厂商的 SDK。你不需要在生产里跑过向量数据库——文中涉及的工程做法都会自带足够的上下文。

---

## 1. LLM 应用的分层架构

![LLM 应用栈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig1_application_stack.png)

构建 LLM 产品最有用的一个心智模型是：把它当作一个普通的五层应用，只是其中**恰好有一层是概率性的**。自上而下：

1. **体验层**——Web、移动端、IDE 插件、企业 IM 机器人。流式 UX 应该住在这里，而不是模型里。
2. **Agent / 编排层**——规划器、工具路由、记忆、护栏。你的代码大头都在这。
3. **检索与上下文层**——RAG、语义缓存、会话存储，以及最后把消息列表组装出来的 prompt builder。
4. **模型服务层**——一个 LLM Gateway 顶在多个模型供应商前面，负责路由、降级、流式输出。
5. **工具与数据面**——SQL、搜索、代码执行、第三方 API、向量库。这一层是确定性的，可以像普通服务一样测试。

横切五层的，是那一列老朋友：认证授权、审计与 PII 脱敏、限流、可观测性、成本核算、评估。

这种分层之所以重要，是因为它告诉你**力气该花在哪儿**。如果一个团队把重试逻辑写在体验层、把 prompt 模板塞进模型层、把限流到处都来一份，那这个项目八成是上不了线的。把每个关注点压回它应该所在的层，模型层就能保持得很轻。

第二个推论：可靠性工作大部分都不是 LLM 工作。模型对你来说就是一个 HTTP 黑盒。真正有意思的问题——*正确的工具是否被调用了？参数对不对？是不是访问了正确租户的数据？是否在预算之内？*——全都活在编排和检索层，全是确定性的 Python 或 TypeScript，全都可以跑单测。

## 2. 工作流模式：能用最简单的就用最简单的

![四种编排模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig2_workflow_patterns.png)

Anthropic 在 [*Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents)（2024）里讲过一个值得反复体会的观点：很多人嘴里的 "agent"，其实用一个固定的小工作流就能解决得更好。工作流可预测、便宜、好评估。Agent——LLM 自己动态决定下一步——只在任务空间真的开放时才该用。

实际生产里，你只需要四种模式。

**Chain（链式）**。顺序流水线：抽取 → 转换 → 总结。当步骤确定、顺序已知时用它。实现就是一个函数列表，不需要框架。

**Branch（分支路由）**。先用一个小模型判一下意图，再分发：FAQ 走小模型，写代码走 tool-using 路径，硬推理走旗舰模型。这一种模式贡献了你**不换模型供应商**就能拿到的大部分成本节约——我们在自己的部署里观察到，加一个 LLM 路由 + 三档模型池，相比"全部走最大模型"通常能省 60–80% 的费用。

**Loop（反思循环）**。生成 → 评判 → 改写。评判方可以是 LLM-as-judge、正则、跑单测、或者一个 SQL dry-run。循环唯一的前提是必须有上限：迭代次数（一般 3 次就够）、token 总预算、一个不依赖另一次模型调用的终止条件。

**Parallel（并行 + 聚合）**。扇出到 N 个 worker，再 reduce。适合分块摘要、self-consistency 投票、多源整合。两条警告：并行调用会把成本乘以 N；reduce 那一步通常需要用最大的模型，因为综合比生成更难。

加任何一种模式之前都该问一句：*这一步，能不能用非 LLM 服务做？* 如果一个正则、一条 SQL、一个函数调用就能搞定，就别召唤模型。每一次 LLM 调用都是一个让延迟、成本、不确定性渗进来的口子。

## 3. RAG vs 微调 vs Prompt 工程

![Prompt / RAG / 微调 决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig3_rag_vs_finetuning.png)

这是任何 LLM 设计评审都会被问到的问题，答案其实简单：**用最便宜的、能把缺口填上的那个工具**。缺口分两种，一分钟就能诊断出来。

如果模型**风格或格式不对**，但事实大致是对的，那是**行为缺口**。先做 prompt 工程：few-shot 示例、CoT 脚手架、结构化输出约束。Schulhoff 等人的 [*The Prompt Report*](https://arxiv.org/abs/2406.06608)（2024）系统整理了那些跨模型代际仍然有效的做法。Prompt 都用尽了再考虑微调——LoRA 和 DPO 让微调便宜了很多，但迭代速度仍然远不如改 prompt。

如果模型**根本不知道**——你的产品、你的客户、你上个季度的故障——那是**知识缺口**，默认答案就是 RAG。RAG 在三个属性上明显胜过 prompt 和微调：可以引用来源（答案可审计）、几分钟就能刷新（导文档、重新 embed、完事）、能扩展到任何微调都吃不下的语料规模。Lewis 等人那篇 [*RAG 原始论文*](https://arxiv.org/abs/2005.11401)（NeurIPS 2020）至今仍是最干净的形式化定义；Karpukhin 等人的 [*Dense Passage Retrieval*](https://arxiv.org/abs/2004.04906)（EMNLP 2020）则给出了你真正会部署的那种 bi-encoder 检索器。

**RAG + 微调** 这种组合拳是真实存在的，但用得不多。只在以下几种情况才值得：语料稳定、领域窄、质量比迭代速度重要。例子：单一专科的医学问答、固定 schema 的内部工具、强监管下的金融摘要。这种场景下要么微调检索器、要么微调阅读器，RAG 仍然负责"新鲜度"，运维复杂度自己扛。

几条实战经验：

- 一个下午搓出来的"朴素 RAG"，在除了最容易的查询之外的题目上分数都会很难看。增加复杂度的合理顺序大致是：更好的 chunking → 混合检索（dense + BM25） → cross-encoder 重排 → query 改写 / HyDE。每加一步都会推高延迟和成本，所以**只有指标告诉你失败模式确实出现了，才加**。
- 长上下文模型并不能取代 RAG。Liu 等人那篇 [*Lost in the Middle*](https://arxiv.org/abs/2307.03172)（TACL 2024）已经说清楚：哪怕是 128k token 的窗口，模型对开头和结尾的注意力也远多于中间段。检索在精度和成本上仍然是赢的。
- 旗舰模型大多不开放微调。要微调就微调开源模型（Llama、Qwen、Mistral），需要旗舰能力的请求走 API。第 2 节那个 branch 模式，就是让这种"分而治之"在工程上跑得起来的关键。

## 4. 生产环境的部署拓扑

![生产部署拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig4_production_deploy.png)

一个生产 LLM 服务的形状没什么稀奇，**有意思的是哪几个盒子在干重活**。从左往右读：

**Edge（边缘）**。CDN 加 WAF，限流要够狠。LLM 接口对滥用者有特别强的吸引力——每个请求都直接花你的真金白银——所以边缘限流要比普通 API 更紧，而且必须**按身份**限流，不能只按 IP。

**API Gateway**。认证、按租户配额、请求级审计、决定一个请求走到哪个版本哪个应用的路由规则。**这一层的配额单位应该是 token 或美元，而不是次数**——一次 10 万 token 的查询可能比 1000 次短查询都贵。

**应用服务**。无状态的 FastAPI / Node 服务，挂在 HPA 后面。它们负责跑第 2 节里的工作流，但**不直接调模型**——它们调 LLM Gateway。

**语义缓存**。坐在 LLM Gateway 前面，是这张图上**杠杆率最高的盒子**。两层：第一层是规范化后的精确匹配缓存；第二层是把用户问题 embed 之后查向量索引，余弦相似度超过阈值（0.95 是一个站得住的默认值）就直接命中。在客服、FAQ、分析类工作负载上，30–60% 的命中率是常见的；开放性创作场景下能有 5–10% 已经不错。

**LLM Gateway**。负责模型选择、降级（主模型 5xx 或被限流时自动降到备模型）、按请求 token 预算控制、流式输出。把这一摊事**集中在一个服务里**，而不是散落在各应用代码里，是让你"一个下午就能换模型供应商"而不是"一个季度都换不完"的关键。

**模型池**。建议三档：自有 GPU 上的小模型（吃下最高频、最容易的请求）、开源中型模型（通用任务）、旗舰 API（硬骨头）。模型旁边还要配一个**代码 / 工具沙箱**——绝对不要把模型生成的代码跑在你自己的应用进程里。

**异步通道**。一个消息队列加一组 worker，处理所有不需要同步返回的事：文档摄入与 embedding、定时评估、成本与用量汇总、批量生成、再训练。

**遥测**。所有盒子都吐 OpenTelemetry trace 和 Prometheus 指标。第 6 节细讲。

一个常见的踩坑：跳过 LLM Gateway，让应用代码直接调模型。在你需要换供应商、加降级、加 token 预算的那天之前，这都没问题——但那天之后，每个服务都得改一遍。**第一天就把 Gateway 建起来**，哪怕它后面只挂一家供应商。

## 5. 成本优化

![成本优化杠杆](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig5_cost_optimization.png)

LLM 成本，是导致"开发环境跑得欢、被财务一刀砍掉"的头号原因。这里有六根杠杆，请按以下顺序拉，每一根都建立在前面之上。

1. **Prompt 压缩**。短 prompt 在三个维度上更便宜：输入 token、首 token 时延、（因为 attention 是平方复杂度的）基础设施压力。剥掉系统 prompt 里的废话，能用 ID 就别用名字，砍掉模型已经无法有效利用的旧对话历史。
2. **小模型路由**。第 2 节的 branch 模式：每个请求先分类，只把硬骨头送给贵模型。分类本身用一个 7B 模型就行，每次成本不到一分钱。
3. **语义缓存**。第 4 节讲过。决定它价值的是命中率，**第一天就要埋点**。
4. **批处理**。OpenAI、Anthropic、几乎所有厂商都有 batch API，价格大约是同步调用的一半，SLA 24 小时。所有不直接面向用户的活——夜间批量摘要、embedding 刷新、评估批跑——都应该走 batch 接口。
5. **INT8 / INT4 量化**。自托管模型的话，[GPTQ](https://arxiv.org/abs/2210.17323)（Frantar 等，ICLR 2023）这一类技术在大多数任务上能做到约 2× 的内存与延迟节省、质量损失可忽略。这是 GPU 成本占大头时该拉的杠杆。
6. **谈判或自托管**。摸清楚自己的流量画像之后，承诺消费折扣和自托管开源模型才会变成真选项。和旗舰 API 的盈亏平衡点，按工作负载算大概在每月 5000 万 – 1 亿 token 之间。

图右那张散点图是**质量 / 成本前沿**。从生产流量观察到三件事：本地小模型在通用任务上能拿到旗舰 60–65% 的质量，在窄任务上能逼近 95%；INT8 量化把你向左下方拉一点（更便宜、几乎一样好）；在任何一档之上叠 RAG，都是不换供应商时能拿到的最大单笔质量提升。**用你自己的评估画一张你自己的图，然后拿来做路由决策**。

关于 benchmark 的一句忠告：公开榜单（MMLU、MT-Bench、Arena）对具体产品的泛化性很差。自己做一个小型任务 eval（50–500 个 prompt，配参考答案或评分细则），每次换模型或改 prompt 就重跑，**信你自己的，不要信公开数字**。

## 6. 可观测性

![可观测性栈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig6_observability.png)

标准的可观测性工具栈——Prometheus、OpenTelemetry、Loki / Tempo / Grafana——对 LLM 服务都适用，只需要在 [Google SRE 的四金信号](https://sre.google/sre-book/monitoring-distributed-systems/)（latency、traffic、errors、saturation）之上**再加四个 LLM 专属信号**。

**单请求成本** 是事实上的第五个金信号。用 histogram 跟踪、对 p95 报警、按租户 / 接口 / 模型拆分。这个指标会变成财务在群里 @ 你的那条消息。

**幻觉率**。抽样 1–5% 的流量做 LLM-as-judge——让一个更强的模型判断这条回答是否被检索上下文支撑。绝对值是噪声，**趋势不是**。

**Groundedness / 引用命中率**。对 RAG 系统：回答里的事实陈述里，有多少能追回到一个被检索到的 chunk？这是用户感知幻觉的先行指标，而且计算很便宜。

**护栏触发率**。注入过滤器、PII 脱敏、内容审核，触发频率是多少？陡峰一般意味着有人找到了新绕过；缓涨一般意味着用户行为发生了漂移、过滤规则该重新调。

**缓存命中率与降级率** 收尾。缓存命中率告诉你语义缓存是不是在挣钱；分供应商的降级率告诉你**用户还没来抱怨**之前哪家上游正在劣化。

任何稍微复杂一点的工作流，分布式追踪都不是可选项。[Dapper 论文](https://research.google/pubs/pub36356/)（Sigelman 等，2010）至今仍是正确的心智模型。一个 RAG 请求的典型 trace 包含 `retrieve`、`rerank`、`compose_prompt`、`llm.generate` 以及任何工具调用的 span——没有这些 span，"这个请求慢了" 就是一句没法 debug 的话。

一个值得采纳的做法：**记日志前对 prompt 和 response 做哈希**。运营所需的可见性都还在（哪个模板、回答多长、哪个用户），但你不再需要无限期地把用户原文存进系统、在合规上承担风险。真的需要看原文 debug 时，走一条单独的、被审计的路径。

## 7. 企业集成

![企业集成模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/llm%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9E%B6%E6%9E%84-%E4%BC%81%E4%B8%9A%E7%BA%A7%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97/fig7_enterprise_patterns.png)

讲一句不那么好听的实话：试点上不了线，**几乎都不是模型质量的锅**——是模型周围这六个盒子的锅。每个盒子都对应一个采购方一定会问的问题，你必须有一个站得住的答案。

**Identity（身份）**。SAML / OIDC 单点登录、SCIM 自动化用户生命周期。没有这些，每接一个客户都是一个几周的项目。

**Authorisation（授权）**。功能层用 RBAC，数据层用 ABAC。对 RAG 而言意味着：**文档级别的过滤必须在检索层强制执行**——永远不要相信"我在 prompt 里告诉模型不要提文档 X"。租户看不见的文档，**检索器就不能返回**。

**Data residency（数据驻留）**。存储和推理都要支持区域路由。欧盟客户要求欧盟托管的模型 + 欧盟托管的向量库，没有讨价还价空间。这是一个**很难事后补救**的拓扑决策，要在第一个欧洲客户落地之前就定下来。

**BYOK 与 BYOM**。客户自己 KMS 里的密钥；可选的自托管模型端点——通过你的平台路由，但明文数据从不离开客户 VPC。第 4 节那个 LLM Gateway 是 BYOM 在工程上跑得通的前提。

**Audit 与 DLP**。一份不可篡改的"谁在什么时候做了什么"审计日志，加上一道在入参和出参两端做 PII 脱敏的 DLP。**主体访问请求（DSAR）工具**——能找出并删除某个用户的所有数据——是 GDPR 和好几条美国州法的硬要求；数据模型设计的时候就要把它做成一句 SQL 能搞定，而不是一次考古挖掘。

**Compliance（合规）**。SOC 2 Type II 是 B2B SaaS 的入场券。ISO 27001、HIPAA、FedRAMP 各自打开特定市场。它们都不是孤立的技术项目，而是**证据收集项目**——架构必须能撑起这些证据。好消息是：前面五个盒子做扎实了，认证大部分就是文书工作。

一个排序建议：**别试图第一天就把六个盒子全做完**。第一个付费客户之前，搞定 SSO、RBAC、审计日志；第一个受监管客户之前，加上数据驻留和 BYOK；正式认证留到有客户合同明确要求时再启动。

## 关于本文有意省略的内容

这篇文章里没有 600 行 FastAPI 脚手架来拼一个"完整的 RAG 系统"。这种代码在几十个开源仓库里都有，而且**老化得很快**。老化得慢的，是决策框架：这个关注点该归哪一层、这个任务该用哪种工作流、哪根杠杆能用最小的质量损失换最大的成本节约。代码是简单的部分，**工程判断才是难的部分**——这也是我们在这里试图压缩进来的东西。

如果你想要具体的参考实现，LangChain、LlamaIndex、Anthropic 官方 cookbook 都在持续维护，远比一篇静态文章新。代码去那边翻；取舍回这边看。

## 结语

一个生产级 LLM 应用，本质就是一个**调用了一个非确定性函数的普通分布式系统**。绝大部分工作都在确定性那一侧：把对的上下文交给模型，把对的请求路由到对的模型，把回来的东西观测得足够好——好到能在指标变化时知道"出了点什么"。

本文里的这些模式，都是那种被真实流量、真实客户、真实采购流程考验过、活下来的东西。它们不是唯一的答案，但是一个**讲得通的默认值**。大多数团队从一个讲得通的默认值开始，比从零自己重新发明一个，要走得更远。**先搭最小可工作的工作流、第一天就把可观测性铺上、在指标告诉你需要之前不要加复杂度**——这是我对每一个新启动的 LLM 项目的建议。

## 参考文献

- Lewis, P. 等. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. <https://arxiv.org/abs/2005.11401>
- Karpukhin, V. 等. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020. <https://arxiv.org/abs/2004.04906>
- Liu, N. F. 等. *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024. <https://arxiv.org/abs/2307.03172>
- Schulhoff, S. 等. *The Prompt Report: A Systematic Survey of Prompting Techniques*. 2024. <https://arxiv.org/abs/2406.06608>
- Anthropic. *Building Effective Agents*. 2024. <https://www.anthropic.com/research/building-effective-agents>
- Frantar, E. 等. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. <https://arxiv.org/abs/2210.17323>
- Sigelman, B. H. 等. *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*. Google Tech Report, 2010. <https://research.google/pubs/pub36356/>
- Beyer, B. 等. *Site Reliability Engineering*. O'Reilly / Google, 2016——金信号一章. <https://sre.google/sre-book/monitoring-distributed-systems/>
- Gao, L. 等. *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL 2023. <https://arxiv.org/abs/2212.10496>
- Hu, E. 等. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. <https://arxiv.org/abs/2106.09685>
