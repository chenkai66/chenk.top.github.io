---
title: "Terraform 实战（一）：为何 IaC 是唯一出路"
date: 2026-03-12 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - Infrastructure as Code
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 1
description: "Agent 系统是个移动靶——每个月都有新工具、新记忆库、新区域。手动点控制台撑不到第二个同事入职。本系列第一篇讲为什么要在阿里云上用 Terraform，盘点 alicloud provider 真正覆盖了哪些资源，并把它和 Pulumi、Crossplane、ROS 摆在一起对比，让你第一次就选对。"
disableNunjucks: true
translationKey: "terraform-agents-1"
---
过去十八个月，我在阿里云上交付了四个 Agent 系统。其中三个起步都是某人在控制台点出来的单台 ECS 上的 `tmux` 会话。这三个项目，通常在第二位工程师加入、生产环境资源告急，或安全团队索要网络拓扑图时，迫使我不得不在某个手忙脚乱的周末紧急重构。

第四个是从 `terraform apply` 开始的。它是唯一一个无需占用周末就能交付的系统。

这个系列就是为第四种模式准备的实战指南：如何用 Terraform  部署 AI Agent 系统在阿里云上真正需要的基础设施。这不是 Terraform 教程——网上有很好的教程，官方 `快速入门` 文档也覆盖了基础。这是一份面向同时满足‘正在运行 Agent’和‘部署在阿里云上’两个条件的高级工程师的实战手册。

八篇文章。最后交付一个真实可用的栈。第一篇讲讲为什么。

![Terraform for AI Agents (1): Why IaC Is the Only Sane Way to Ship Agents — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/illustration_1.png)

## What "an agent system" actually requires

聊基础设施之前，我们先明确 Agent 系统到底包含哪些组件——那些 `pip install langgraph` 的 README 通常会略过的部分：

![AI agent workloads running on cloud infrastructure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/wanxiang_agent_infra.png)


1. **A runtime** 跑 Agent 循环进程的运行时——通常是 Python 或 Node——并且能扛住重启
2. **A vector store** 语义记忆用的向量存储——文档嵌入、历史对话、工具输出
3. **A relational store** 存会话状态的关系型存储——轮次对话、工具调用 trace、用户身份
4. **An object store** 存制品的对象存储——生成的图片、PDF、截图、运行快照
5. **An LLM gateway** 统一持有 API Key 并执行每 Agent 配额限制的地方
6. **Outbound network** 用来调用 DashScope、OpenAI、Anthropic 或者你的爬虫目标
7. **Observability** Agent 运行是非确定性的，所以日志和 trace 不是可选项
8. **Secrets**  provider keys、OAuth tokens、OSS 凭证、数据库密码
9. **Cost control** 因为当 Agent 自我循环时，token 账单可能一夜之间翻 10 倍

这至少涉及九个相互交互的阿里云服务。每个服务都有独立的控制台入口、RAM 权限策略、可用区（Region）支持范围和网络配置。指望手动把这些全部连线，并且在三个月的演进后还能让 `dev`、`staging` 和 `prod` 保持一致，概率差不多是零。

## The console-vs-IaC moment

九个服务手动操作，等于九个漂移面——这种痛苦太普遍，我甚至画了一张标准图来描述它：

![Infrastructure as Code workflow transforming declarative configs into cloud resources](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/wanxiang_iac_workflow.png)


![Console clicks vs Terraform — where the divergence happens](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig1_console_vs_iac.png)

细看左列：每一步都看似合理，没有一个是愚蠢的 mistake——恰恰是聪明人用数月积累的微小合理决策堆叠而成的结果。右列走的是同一条路径，但每一步变更都通过可审查的配置文件记录在 Git 中。两列之间的 diff 就是“我交付了这个”和“凌晨 2 点我被 call 醒因为没人知道 `cn-beijing` 里跑着什么”的区别。

阿里云 Terraform 官方文档说得更委婉些：

> Console operations: Click and enter parameters step by step. Repeat manual steps — hard to ensure consistency. Rely on documentation and verbal agreements.
>
> Terraform: Describe the desired state of resources in configuration files. Configuration files are reviewable, shareable, and reusable. Store configuration files in version control. Changes are traceable and reversible.

第二段就是全部卖点。这个系列里的其他内容都是实现细节。

## What Terraform actually is, in two sentences

用两句话概括 Terraform 到底是什么：Terraform 是 HashiCorp 出品的开源声明式工具。你用 **HashiCorp Configuration Language (HCL)** 编写 `.tf` 文件来描述想要的云资源；Terraform 将该期望状态与 **state file** 中记录的 live state 进行 diff 并生成 **plan**；你 review plan；然后 `apply`；Terraform 将 plan 翻译成 provider API 调用。

需内化三点：

- **Declarative, not imperative.** 不要写‘创建实例’这样的命令式语句，而要声明‘存在一个符合该规格的实例’这一状态。如果实际状态未发生变化，重复执行配置将不产生任何变更。这让 Terraform 可以安全地在每次 commit 时从 CI 运行。
- **State is real.** `terraform.tfstate` 文件是从 HCL 资源地址到云实际资源 ID 的 JSON 映射。一旦丢失 state file，Terraform 就会认为所有资源都不存在。第二篇文章会讲把 state 放在哪里才持久——但影响远不止“别丢文件”，我们后面会回来说。
- **Plan before apply.** 这是杀手锏。每次变更都会在你动手*之前*字面展示什么会被创建、修改或销毁。养成把 plan 输出 paste 到 PR 描述里的习惯——未来的你会感谢自己。

## State as the agent stack's 物料清单

"State 是真实的”这点值得多写几句，因为对于 Agent 栈来说，state 文件兼任了你的库存清单。

我交付的每个 Agent 栈在某个时刻都被审计过——安全审查、财务核对云支出、或者新入职的 SRE 试图厘清当前运行的是什么。每次审计的核心问题都相同：当前有哪些资源？由谁创建？花费多少费用？

如果你的基础设施活在 Terraform state 文件里，这个问题 30 秒就能回答：

```bash
terraform state list | wc -l                                  # how many resources
terraform state list | awk -F. '{print $1"."$2}' | sort -u    # what kinds
terraform show -json | jq '[.values.root_module.resources[] | {addr:.address, type:.type}]'
```

对于我今天跑的四个 Agent 栈，这三条命令几秒钟就能生成一份综合清单。在用 Terraform 之前，同样的审计需要打开 ECS、VPC、RDS、OSS、RAM、KMS、SLS、ARMS、ACK、CloudMonitor、ALB 和 OpenSearch 的十二个控制台标签页——运气好按 tag 过滤，运气不好全靠直觉。

State 文件在供应链意义上就是一份 *物料清单*：每个资源都携带 provider 版本与 module 来源；当 alicloud provider 暴露 CVE（每年数次），几分钟内即可遍历所有项目 state 文件并检索相关条目：

```bash
for d in stack-*/; do
  (cd "$d" && terraform providers schema -json 2>/dev/null \
     | jq -r '.provider_schemas | keys[]' \
     | grep -F 'aliyun/alicloud' && echo "  in $d")
done
```

更深层的点：**state 把基础设施变成了数据**。一旦成了数据，你就可以针对它写工具。我有个小 Python 脚本遍历每个项目的每个 state 文件，生成一个 `(stack, resource_type, resource_id, region, monthly_cost_estimate)` 的 CSV。这个 CSV 会出现在我的月度成本会议上。没有 state 文件作为统一的结构化真相来源，这一切都不会存在。

反面是 state 文件很珍贵。丢了它，资源宇宙对 Terraform 就不可见了——下次 plan 时会得到 wholesale "resource already exists" 错误。第二篇文章专门讲把 state 放在不会消失的地方。

## What the Aliyun provider covers

只有当 provider 能真正创建你声明的东西时，state 才有用。云平台通过 **provider plug-ins** 与 Terraform 对话。官方 `alicloud` provider 是中国第一个官方 Terraform provider，由阿里维护。截至撰写时，它在大约六个域中提供了 **300+ 资源类型**：

![alicloud provider coverage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig2_provider_coverage.png)

根据官方文档，支持的类别包括：

- **Compute and containers**: ECS, ACK (Kubernetes), Function Compute, Auto Scaling
- **Networking**: VPC, SLB, ALB, NLB, NAT Gateway, Cloud Enterprise Network
- **Storage and databases**: OSS, NAS, ApsaraDB RDS, PolarDB, Redis, MongoDB
- **Security and management**: RAM, KMS, WAF
- **Observability**: SLS, ARMS, CloudMonitor
- **Big data and AI**: MaxCompute, PAI

这覆盖了我们九组件清单里的所有内容，包括每 LLM provider 的 keys（在第 6 篇文章中通过 KMS Secrets Manager 处理）。

一个最小的 HCL 示例， pinned provider 版本是因为你应该从第一天就 pin 住：

```hcl
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230"
    }
  }
}

provider "alicloud" {
  region = "cn-shanghai"
}

resource "alicloud_vpc" "main" {
  vpc_name   = "agents-prod"
  cidr_block = "10.20.0.0/16"
}

resource "alicloud_vswitch" "private_a" {
  vpc_id     = alicloud_vpc.main.id
  cidr_block = "10.20.1.0/24"
  zone_id    = "cn-shanghai-l"
}

resource "alicloud_security_group" "agent_runtime" {
  name   = "agent-runtime-sg"
  vpc_id = alicloud_vpc.main.id
}
```

三个资源，带有 Terraform 自动解析成正确依赖顺序的 `vpc_id` 引用。你不用说“先 VPC，再 vSwitch，再 SG"——你写出想要的，Terraform 构建 DAG。

## Modules: the unit of reuse

三个资源是玩具版本。真实栈有几百个资源，让几百个资源可管理的方法是 **modules**。module 只是一个接受输入并产生输出的 `.tf` 文件目录。一旦你有了 working pattern——一个带三个 vSwitch、一个 NAT 和安全组基线的 VPC——把它封装进 module，你就可以 Across `dev`、`staging`、`prod` 和 `intl-prod` 复制它而不用复制 HCL。

一个最简 module 调用：

```hcl
module "vpc" {
  source = "./modules/vpc-baseline"

  for_each = toset(["dev", "staging", "prod"])

  vpc_name   = "agents-${each.key}"
  cidr_block = "10.20.0.0/16"
  zones      = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]
}
```

`./modules/vpc-baseline/main.tf` 的主体包含实际的 `alicloud_vpc`、`alicloud_vswitch`、`alicloud_nat_gateway` 资源。调用者不需要知道——他们只想要一个有合理的默认配置的 VPC。这和 Python 函数 idea 一样，只是 applied to infrastructure。（只要迭代的是有意义的名称集，就用 `for_each` 而不是 `count`——`count` 在你删除中间项时会重新编号，导致 Terraform 销毁并重建无关资源。）

我们将在第 3 篇文章中构建这个 module，并在后续每篇文章中复用。
## IaC 到底能防止哪些 Agent 特有的故障模式

在把 Terraform 和其他工具拉出来对比之前，得先搞清楚你到底买的是什么。市面上讲 IaC 总爱提“一致性”和“可复现”——话没错，但听着没劲。折腾这套系统三年下来，最让我头疼的故障模式都跟 Agent 有关，而每一个都有对应的 Terraform 解法：

![Infrastructure drift detection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/wanxiang_drift_detection.png)


**1. 凌晨 3 点的 Token 泄露**。Agent 自己死循环了——停止条件写烂了、工具无限重试、规划状态 hallucinated——一夜之间烧掉 40,000 元的 LLM 预算。手动在控制台点出来的栈没有程序化的预算 guard，因为没人写。Terraform 栈从第一天就 provision 了 `alicloud_log_alert`（见文章 7），因为模块默认自带。告警的成本不过是计划里多一个资源；没有告警的成本是财务打来的电话。

**2. “谁拿了我的 Key"恐慌**。外包走了。他们手里有一份原型阶段用的 OpenAI key。控制台栈把 key 散落在三台 ECS 的 `.env` 文件和一个钉钉私聊里。轮换 key 得花半天时间 `grep -r OPENAI_API_KEY ~/projects` 然后祈祷。Terraform 栈把所有 key 都放进 KMS Secrets Manager（见文章 6），藏在 RAM role 后面，只有一个 canonical location——轮换就是编辑 `secrets.auto.tfvars` 后跑一次 `terraform apply`。

**3. 幽灵 NAT 账单**。Agent 话多；LLM 调用最话多。路由配错让它们走了错误区域的公网 NAT，出口账单翻三倍。控制台不会告诉你——它只会下个月在账单里显示那一行。Terraform 的 `alicloud_nat_gateway` 和它服务的 `alicloud_vswitch` 在同一个模块里，所以依赖关系在 plan 阶段就看得见。文章 3 把这设为默认。

**4. “生产环境在哪个 Region？”的问题**。阿里云有 30+ 个 Region。刚来两个月的工程师调试故障，不知道生产 RDS 在 `cn-shanghai` 还是 `cn-beijing`，只能去 grep 钉钉历史记录。Terraform 的答案是 `terraform output rds_endpoint`。五秒解决，永久可追溯。

**5. “周末 LLM 网关配置丢了”**。网关路由模型、holding quotas、记录流量。那是两台 ECS，有人 SSH 进去编辑 `/etc/litellm/config.yaml` 配的。服务器重启 `/etc/litellm` 就没了。Terraform 版本把配置放在 cloud-init 里（见文章 6）——每台实例起来配置都一样，无需人工步骤。

这些故障模式都不 exotic。我做过的项目里每一个都发生过。每一个都被 IaC 模式 *防止* 了，而不仅仅是 *恢复*。这才是 Agent 栈上用 Terraform 的真正理由——不是抽象的可复现性，而是一张具体的灾难清单，上面列出的事都不会发生。

## Terraform vs Pulumi vs Crossplane vs ROS

承认了 IaC 的价值，为什么偏偏是 Terraform？快速扫一眼 alternatives。都没错；选适合团队的，别搞宗教斗争：

![IaC tools compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig3_iac_tools_compare.png)

用完这四个之后我的 honest read：

- **Terraform** 是默认选项。Provider、模块和懂它的人生态最大。HCL 第一天觉得怪，之后就好了。除非你有特别强的理由反对，否则就选它。
- **Pulumi** 封装了 Terraform providers 但让你写 Python/TypeScript/Go。表达能力是实的——你能用循环、条件判断，还有 IDE 真正检查的类型。代价是调试：出问题的时候，你得 debug 两层（你的代码 → Pulumi → TF provider）。如果你的团队真的恨 HCL，这值得。
- **Crossplane** 活在 Kubernetes 里——每个云资源变成一个 CRD，你 `kubectl apply` 出一个 VPC。如果你已经是纯 Kubernetes  shop 且上了 GitOps，这很美；如果不是，这很痛苦。
- **ROS** (Resource Orchestration Service) 是阿里云的原生等价物。跟控制台深度集成，JSON 或 YAML 模板，不用装 provider 插件。只有当你 100% 永远只用阿里云，且运维团队偏好托管服务时才选这个。

阿里云文档本身对比得很公平：

> [Terraform 和 ROS] 都是声明式 IaC 工具。Terraform 是支持多云管理的开源第三方工具。ROS 是原生阿里云服务，与阿里云管理控制台深度集成。如果你需要多云支持或已经在其他地方使用 Terraform，请选择 Terraform。

对于一个调用多个 LLM provider 且将来可能需要美国 Region 或新加坡 Region 的 Agent 系统，对多云友好的 Terraform 是正确的默认值。

## 这个系列会做什么，不会做什么

会做的：

- 带你从 `terraform init` 到一个跑在阿里云上的完整 `research-agent-stack`，共八篇文章。
- 展示 VPC, ECS, ACK, OSS, RDS, OpenSearch, KMS, SLS 和 CloudMonitor 的真实、可运行的 HCL。
- 覆盖文档里没有的故障模式——state drift, locked tfstate, GFW provider 下载，Region 缺货。
- 最后给你一个可以 fork 的 starter repo。

不会做的：

- 教你 HCL 语法超出我们用到的部分。官方 HashiCorp 教程做得更好。
- 教你怎么写 Agent 本身。LangGraph, AutoGen, MetaGPT, Claude Code 都有系列教程了；选一个就行。
- 逐功能对比阿里云 against AWS 或 GCP。IaC 模式跨云通用；资源名字不一样而已。

## 接下来是什么

文章 2 是第一次 hands-on：安装 alicloud provider，挑选认证方法（三种选择——静态 AK/SK, AssumeRole, ECS RAM role——并不等价），在 OSS 上设置 remote state 并用 Tablestore 做 locking，以及 `dev`/`staging`/`prod` 的 workspace 模式。

如果你今天只做一件事，安装 Terraform（macOS 上 `brew install terraform`，或 follow 官方 `Install Terraform` 主题）并运行 `terraform version` 确认。系列其余部分假设你已经有了。

> **Real-world tip:** 从第一天就在 `required_providers` 里 pin 住 alicloud provider 版本，并用 `required_version` pin 住 Terraform 本身。provider 正在积极开发中，minor 版本之间的 breaking changes 罕见但不是零。pin 住版本意味着你周五的 `terraform plan` 在周一返回同样的结果。