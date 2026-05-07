---
title: "用 Terraform 给 AI Agent 上云（一）：为什么 IaC 是唯一靠谱的部署方式"
date: 2026-03-12 09:00:00
tags:
  - Terraform
  - 阿里云
  - 基础设施即代码
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 1
description: "Agent 系统是个移动靶——每个月都有新工具、新记忆库、新区域。手动点控制台撑不到第二个同事入职。本系列第一篇讲为什么要在阿里云上用 Terraform，盘点 alicloud provider 真正覆盖了哪些资源，并把它和 Pulumi、Crossplane、ROS 摆在一起对比，让你第一次就选对。"
disableNunjucks: true
translationKey: "terraform-agents-1"
---

过去十八个月我在阿里云上交付过四个 Agent 系统。其中三个一开始都是某个人在控制台点一阵之后留下的一台 ECS 上跑着 `tmux` 会话。这三个系统都各自经历过一个仓促的周末——第二位工程师入职那次、生产区域 GPU 缺货那次、安全团队来问网络拓扑图那次——把所有东西从头重建一遍。

第四个系统从第一天就是 `terraform apply` 起家的。也只有这一个，我没有为它失去过周末。

这个系列就是第四种模式的实战手册：怎么用 Terraform 在阿里云上把一个 AI Agent 系统真正需要的云基础设施搭起来。它不是 Terraform 入门教程——网上不缺好的入门，官方 `Get Started` 文档也覆盖了基础。它写的是"我跑 Agent"和"我在阿里云上跑"这两件事交集处的那本资深工程师剧本。

一共八篇。最后会落在一个真正能跑的完整 stack 上。第一篇先讲 why。

![用 Terraform 给 AI Agent 上云（一）：为什么 IaC 是唯一靠谱的部署方式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/illustration_1.jpg)

## 一个 Agent 系统到底需要哪些东西？

在聊基础设施之前，我们先明确一下一个 Agent 系统的核心组件——这些通常是 `pip install langgraph` 的 README 文件里不会提到的内容：

1. **运行时环境**  
   这是承载 Agent 主循环的地方，通常用 Python 或 Node.js 实现。它需要能够应对服务重启的情况，确保进程持续运行。

2. **向量存储**  
   用于保存语义记忆，比如文档的嵌入表示（embeddings）、历史对话记录、工具输出等数据。

3. **关系型数据库**  
   存储会话状态，包括每一轮对话的上下文、工具调用的追踪记录以及用户身份信息。

4. **对象存储**  
   用来存放生成的各种文件，比如图片、PDF 文档、截图以及运行时的快照。

5. **LLM 网关**  
   统一管理 API 密钥，并为每个 Agent 设置调用配额，避免资源滥用。

6. **外网访问能力**  
   提供出网通道，用于调用 DashScope、OpenAI、Anthropic 等外部服务，或者抓取目标网站的数据。

7. **可观测性支持**  
   Agent 的运行本质上是非确定性的，因此日志和分布式追踪（traces）是必不可少的，而不是可选项。

8. **密钥管理**  
   包括第三方服务的 API 密钥、OAuth 令牌、OSS 凭证以及数据库密码等敏感信息。

9. **成本控制机制**  
   因为当 Agent 不小心进入自循环时，Token 消耗可能会在一夜之间暴涨 10 倍。

这至少涉及九个不同的阿里云服务，它们之间还有特定的交互方式。每个服务都有独立的控制台页面、RAM 权限配置、地域范围限制以及网络设置。试想一下，经过三个月的迭代后，手动配置的 `dev`、`staging` 和 `prod` 环境还能完全一致的概率，几乎为零。
## 控制台与 IaC 的分水岭

这个痛点太普遍了，我已经总结出一个经典的对比图：

![控制台操作 vs Terraform —— 差异从何而来](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig1_console_vs_iac.png)

仔细看左边那一列。每一步都看似合理——没有明显的低级错误。这些步骤其实是聪明人在几个月的时间里，基于一个个小而合理的决策逐步积累出来的结果。右边那列走的是同样的流程，但每一步都在 git 中留下了记录。两者的差距，其实就是“我完成了交付”和“凌晨两点被叫起来救火，因为没人清楚 `cn-beijing` 上到底运行了什么”的区别。

阿里云 Terraform 官方文档用更委婉的方式描述了这一点。在《What Is Alibaba Cloud Terraform?》中提到：

> 控制台操作：通过界面一步步点击并输入参数。重复手动操作——难以保证一致性。依赖文档和口头沟通。
>
> Terraform：通过配置文件描述资源的期望状态。配置文件可评审、可共享、可复用。将配置文件存储在版本控制系统中，变更可追溯、可回滚。

第二段话就是 Terraform 的核心价值主张。至于本系列的其他内容，不过是围绕这个核心的具体实现细节罢了。
## 用两句话解释 Terraform 是什么

Terraform 是 HashiCorp 推出的一款开源声明式工具。你通过编写 `.tf` 文件（使用 **HashiCorp Configuration Language (HCL)**）来描述所需的云资源；Terraform 会将这些期望的状态与存储在 **状态文件** 中的实际状态进行对比，生成一个执行计划（**plan**）；你审核这个计划后，运行 `apply` 命令，Terraform 便会将计划转化为对云服务提供商 API 的调用。

需要牢记的三点是：

- **声明式而非命令式。** 你不用写“创建一个实例”，而是描述“这里应该有一个这样的实例”。如果配置没有变化，重复执行同样的代码不会产生任何操作（no-op）。这正是 Terraform 可以安全地集成到 CI/CD 流程、每次提交都自动运行的原因。
- **状态文件至关重要。** `terraform.tfstate` 文件是一个 JSON 格式的映射表，它将你的 HCL 资源定义与云平台上的实际资源 ID 对应起来。一旦丢失状态文件，Terraform 就会认为这些资源不存在了。后续我们会讨论如何将状态文件存储在一个可靠的远程位置。
- **先规划再执行。** 这是 Terraform 的核心亮点。每次变更前，它都会明确告诉你具体会发生什么——哪些资源会被创建、修改或销毁。养成将 `plan` 输出粘贴到 PR 描述中的习惯吧，未来的你会感激现在这么做的自己。
## 阿里云 Provider 的功能覆盖范围

云平台通过 **Provider 插件**与 Terraform 进行交互。官方的 `alicloud` Provider 是中国首个正式发布的 Terraform Provider，由阿里巴巴维护。截至目前，它已经支持 **300 多种资源类型**，涵盖大约六大领域：

![alicloud provider 覆盖范围](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig2_provider_coverage.png)

根据官方文档《What Is Alibaba Cloud Terraform?》的介绍，支持的类别包括：

- **计算与容器**：ECS、ACK（Kubernetes）、函数计算（Function Compute）、弹性伸缩（Auto Scaling）
- **网络**：VPC、SLB、ALB、NLB、NAT 网关、云企业网（Cloud Enterprise Network）
- **存储与数据库**：OSS、NAS、RDS、PolarDB、Redis、MongoDB
- **安全与管理**：RAM、KMS、WAF
- **大数据与 AI**：MaxCompute、PAI

这基本上涵盖了我们之前提到的九个核心组件，除了“可观测性”相关的部分（如 SLS、ARMS、CloudMonitor，其实也支持，只是未列在这份简短清单中）以及针对不同 LLM 提供商的密钥管理部分（这部分会在第六篇文章中通过 KMS Secrets Manager 解决）。

以下是一个最简化的 HCL 示例，直接摘自阿里云官方 ECS 实践文档：

```hcl
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

这个例子中定义了三个资源，其中通过 `vpc_id` 引用建立了资源之间的依赖关系。Terraform 会自动解析这些依赖，并生成正确的执行顺序。你不需要明确指定“先创建 VPC，再创建 vSwitch，最后创建安全组”，只需描述目标状态，Terraform 会帮你构建出对应的有向无环图（DAG）。
## 模块：基础设施复用的基本单元

在使用 Terraform 的早期阶段，最重要的习惯之一就是学会使用 **模块（module）**。简单来说，一个模块就是一个包含 `.tf` 文件的目录，它通过输入参数生成输出结果。当你设计出一个可靠的模式——比如一个带有三个 vSwitch、NAT 网关和安全组基线的 VPC——可以将其封装为模块，然后在 `dev`、`staging`、`prod` 和 `intl-prod` 等环境中轻松复用，而无需复制粘贴 HCL 代码。

以下是一个简单的模块调用示例：

```hcl
module "vpc" {
  source = "./modules/vpc-baseline"

  vpc_name   = "agents-${var.env}"
  cidr_block = "10.20.0.0/16"
  zones      = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]
}
```

在这个例子中，`./modules/vpc-baseline/main.tf` 文件中定义了实际的 `alicloud_vpc`、`alicloud_vswitch` 和 `alicloud_nat_gateway` 资源。调用者不需要关心这些实现细节，他们只需要一个配置合理的 VPC。这就像在编程中调用一个函数一样，只不过这里的“函数”是用来构建基础设施的。

我们将在第三篇文章中详细讲解如何构建这个模块，并在后续的文章中反复使用它来搭建不同的环境。
## Terraform、Pulumi、Crossplane 和 ROS 的对比

在正式决定之前，快速了解一下其他选项。这些工具没有绝对的对错，选择时应该考虑团队的实际需求，而不是盲目追求某种“信仰”：

![四种 IaC 工具对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig3_iac_tools_compare.png)

在深入使用过这四个工具之后，我的真实感受如下：

- **Terraform** 是目前最稳妥的默认选择。它的生态最为庞大，无论是 provider、module 还是社区支持都非常成熟。HCL 语法刚开始可能会让人觉得有点奇怪，但用一天就能适应。除非你有特别的理由不用它，否则选 Terraform 准没错。
- **Pulumi** 基于 Terraform 的 provider 构建，但它允许你用 Python、TypeScript 或 Go 来编写代码。这种灵活性带来了真正的表达能力——比如循环、条件判断，以及 IDE 能实时检查的类型系统。不过，代价是调试复杂度增加：出问题时你需要跨越两层（你的代码 → Pulumi → TF provider）来排查。如果你的团队实在受不了 HCL，可以考虑这个。
- **Crossplane** 完全运行在 Kubernetes 中，所有的云资源都被抽象为 CRD，你可以通过 `kubectl apply` 来创建 VPC 等资源。如果你的团队已经全面拥抱 Kubernetes 并且采用 GitOps 流程，这种方式会显得非常优雅；但如果你还没到那个阶段，学习和迁移成本可能会让你头疼。
- **ROS**（资源编排服务）是阿里云原生的解决方案，与阿里云控制台深度集成，模板支持 JSON 或 YAML 格式，也不需要额外安装 provider 插件。只有当你完全扎根于阿里云，并且运维团队更倾向于使用托管服务时，才建议选择 ROS。

阿里云官方文档的 FAQ 中有一段很中肯的对比说明：

> Terraform 和 ROS 都是声明式的基础设施即代码（IaC）工具。Terraform 是开源的第三方工具，支持多云环境管理；而 ROS 是阿里云原生服务，与阿里云管理控制台深度集成。如果你需要多云支持，或者已经在其他地方使用了 Terraform，那么 Terraform 是更好的选择。

对于一个需要调用多个 LLM 提供商、未来可能扩展到美国或新加坡区域的 Agent 系统来说，支持多云的 Terraform 显然是更合理的默认选择。
## 这个系列的内容范围：做什么，不做什么

我们会做：

- 用八篇文章的篇幅，带你从 `terraform init` 开始，最终在阿里云上部署一个完整的 `research-agent-stack`。
- 提供真实可用的 HCL 配置代码，涵盖 VPC、ECS、ACK、OSS、RDS、OpenSearch、KMS、SLS 和 CloudMonitor 等资源的创建与管理。
- 分享文档中未提及的实际问题和解决方案，比如状态漂移（state drift）、tfstate 文件锁定、GFW 下载 Terraform Provider 的限制，以及某些区域资源库存不足的情况。
- 在系列结束时，提供一个可以直接 fork 的初始代码仓库，帮助你快速启动自己的项目。

我们不会做：

- 深入讲解 HCL 语法。除了我们在实践中用到的部分，更多内容可以参考 HashiCorp 官方教程，它们讲得更系统、更全面。
- 教你如何编写 Agent 本身。目前已经有针对 LangGraph、AutoGen、MetaGPT 和 Claude Code 的专门系列文章，你可以根据需求选择适合的教程。
- 对比阿里云与 AWS 或 GCP 的具体功能差异。虽然不同云服务的资源名称可能不同，但基础设施即代码（IaC）的设计模式是通用的，学习这些模式更有价值。
## 接下来的内容

第二篇文章将带你进行第一次动手实践：安装 alicloud provider，选择适合的认证方式（静态 AK/SK、AssumeRole 和 ECS RAM role 这三种方式并不完全等价），配置基于 OSS 和 Tablestore 的远程状态存储与锁定机制，以及如何使用 workspace 模式来管理 `dev`、`staging` 和 `prod` 环境。

如果你今天只能完成一件事，那就先安装 Terraform（在 macOS 上可以通过 `brew install terraform` 安装，或者参考官方文档《Install Terraform》中的步骤），然后运行 `terraform version` 确认安装成功。后续文章默认你已经完成了这一步。

> **实战建议：** 从一开始就为 `required_providers` 中的 alicloud provider 锁定版本号。这个 provider 正在快速迭代中，虽然小版本之间的破坏性变更很少，但并非完全没有。锁定版本的好处是，你在周五运行的 `terraform plan` 结果，周一还能保持一致，避免意外问题。
## State：Agent 栈的物料清单

![用 Terraform 部署 AI Agent（一）：为什么 IaC 是唯一靠谱的交付方式 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/illustration_2.jpg)

我经手过的每个 Agent 栈，迟早都会经历一次审计——可能是安全团队的审查，可能是财务对云支出的核对，也可能是新来的 SRE 想搞清楚到底线上跑着什么。每次问的问题都大同小异：**“有哪些资源？谁创建的？花了多少钱？”**

如果基础设施的状态存储在 Terraform 的 state 文件里，回答这些问题只需要 30 秒：

```bash
terraform state list | wc -l                            # 资源总数
terraform state list | awk -F. '{print $1"."$2}' | sort -u   # 资源类型
terraform show -json | jq '[.values.root_module.resources[] | {addr:.address, type:.type}]'
```

我现在管理的四个 Agent 栈，运行这三条命令几秒钟就能生成一份完整的资源清单。而在没有 Terraform 的时代，完成同样的审计需要打开十几个控制台页面——ECS、VPC、RDS、OSS、RAM、KMS、SLS、ARMS、ACK、CloudMonitor、ALB、OpenSearch——运气好还能靠标签筛选，运气不好就只能凭经验瞎猜。

state 文件不仅是基础设施的快照，它还是一份**供应链意义上的物料清单**。每个资源都记录了 provider 的版本、module 的来源，甚至在较新的 Terraform 版本中还有内容哈希值。假设某天 alicloud provider 爆出了 CVE-2025-XXXXX，你可以在几分钟内横扫所有项目，找出受影响的部分：

```bash
for d in stack-*/; do
  cd $d
  terraform providers schema -json 2>/dev/null | jq -r '.provider_schemas | keys[]' \
    | grep -F 'aliyun/alicloud' && echo "  位于 $d"
  cd ..
done
```

这不是假设，而是现实。`alicloud` provider 每年都会发布几次破坏性变更或安全相关的更新。如果没有清单，你根本不知道哪些栈受到影响；而有了 Terraform，你只需要简单地 grep 一下。

更深层次的意义在于：**state 把基础设施变成了数据。** 只要它是数据，你就可以围绕它构建工具。我写了一个简单的 Python 脚本，遍历所有项目的 state 文件，生成一份 CSV 表格，内容包括 `(stack, resource_type, resource_id, region, monthly_cost_estimate)`。这份表格是我们每月成本会议的核心输入。没有 state 文件作为统一的结构化数据源，这一切都无法实现。

当然，state 文件的价值也意味着它的脆弱性。一旦丢失，Terraform 就再也无法感知到那些资源的存在——下一次执行 `terraform plan` 时，你会看到一堆“资源已存在”的错误。本系列的第二篇文章会详细讨论如何妥善保存 state 文件，而后续内容也都基于这一前提展开。
## IaC 真正能预防的 Agent 特有故障

通常，IaC 的宣传点都集中在“一致性”和“可复现性”上。这些确实重要，但它们远远低估了 Terraform 在 **Agent 栈** 上的实际价值。在运行这类系统三年后，我发现最让人头疼的故障几乎都与 Agent 相关：

**1. 凌晨三点的 Token 泄漏危机。**  
某个 Agent 因为停止条件写错、工具无限重试或者状态管理出问题，陷入死循环，结果一夜之间烧掉了 ¥40,000 的 LLM 预算。如果是手动通过控制台搭建的环境，通常不会有预算防护机制，因为没人专门去写这个逻辑。而用 Terraform 搭建的栈，从第一天起就自带 `alicloud_log_alert`（详见第七篇），因为模块默认集成了这一功能。多一条资源的成本，远低于没有它时财务部门半夜打来的问责电话。

**2. “谁拿了我的密钥？”恐慌。**  
一位外包员工离职了，他手里有一份当初用来做原型的 OpenAI 密钥副本。如果环境是通过控制台手动搭建的，这个密钥可能散落在三台 ECS 实例的 `.env` 文件中，甚至还有人通过 Slack 私聊发过。轮换密钥时，你得花半天时间运行 `grep -r OPENAI_API_KEY ~/projects`，然后祈祷没有遗漏。而在 Terraform 栈中，所有密钥都存储在 KMS Secrets Manager 中（详见第六篇），并通过 RAM 角色进行管理，集中在一个地方——轮换密钥只需修改 `secrets.auto.tfvars` 文件，然后执行 `terraform apply` 即可。

**3. NAT 账单突增的幽灵问题。**  
Agent 本身就很“话痨”，尤其是调用 LLM 时流量更大。如果路由配置错误，流量可能会通过错误区域的公网 NAT 出口，导致出网账单翻三倍。控制台不会主动提醒你这个问题，只能等下个月账单出来才发现异常。而在 Terraform 中，`alicloud_nat_gateway` 和它服务的 `alicloud_vswitch` 位于同一个模块中，依赖关系在 `terraform plan` 阶段就能清晰看到。这种设计在第三篇中有详细说明，并且是默认配置。

**4. “生产环境到底在哪个区域？”的困惑。**  
阿里云有 30 多个可用区。一个两个月前入职的工程师，在排查生产环境故障时，搞不清楚 RDS 是在 `cn-shanghai` 还是 `cn-beijing`，最后不得不翻钉钉聊天记录来找答案。而使用 Terraform 时，只需运行 `terraform output rds_endpoint`，五秒钟就能得到确切答案，并且可以永久追溯。

**5. “LLM 网关配置丢失”的周末噩梦。**  
网关负责路由模型请求、管理配额、记录流量日志等功能。有人通过 SSH 登录到两台 ECS 实例，手动编辑了 `/etc/litellm/config.yaml` 文件。结果服务器重启后，配置文件被清空，服务中断。而 Terraform 的实现方式是将配置写入 cloud-init（详见第六篇）——每台实例启动时都会自动加载相同的配置，完全不需要手工干预。

这些问题并不罕见，每一个我都曾在实际项目中遇到过。更重要的是，这些问题不是靠 IaC 事后补救的，而是直接被 **预防** 了。这才是 Terraform 在 Agent 栈上的真正价值所在——不是抽象的“可复现性”，而是一份实实在在的“灾难预防清单”。
