---
title: "用 Terraform 给 AI Agent 上云（一）：为什么 IaC 是唯一靠谱的部署方式"
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

过去十八个月我在阿里云上交付过四个 Agent 系统。其中三个一开始都是某个人在控制台点一阵之后留下的一台 ECS 上跑着 `tmux` 会话。这三个系统都各自经历过一个仓促的周末——第二位工程师入职那次、生产区域 GPU 缺货那次、安全团队来问网络拓扑图那次——把所有东西从头重建一遍。

第四个系统从第一天就是 `terraform apply` 起家的。也只有这一个，我没有为它失去过周末。

这个系列就是第四种模式的实战手册：怎么用 Terraform 在阿里云上把一个 AI Agent 系统真正需要的云基础设施搭起来。它不是 Terraform 入门教程——网上不缺好的入门，官方 `Get Started` 文档也覆盖了基础。它写的是"我跑 Agent"和"我在阿里云上跑"这两件事交集处的那本资深工程师剧本。

一共八篇。最后会落在一个真正能跑的完整 stack 上。第一篇先讲 why。

![用 Terraform 给 AI Agent 上云（一）：为什么 IaC 是唯一靠谱的部署方式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/illustration_1.jpg)

## 一个 Agent 系统到底需要哪些东西

在聊基础设施之前，先把一个 Agent 系统的核心组件列清楚——这些通常是 `pip install langgraph` 的 README 里不会提的：

1. 运行时——承载 Agent 主循环的进程，通常是 Python 或 Node，且需要在重启后存活
2. 向量存储——保存语义记忆，包括文档 embedding、历史对话、工具输出
3. 关系型数据库——保存会话状态、逐轮对话、工具调用追踪、用户身份
4. 对象存储——存放产物，比如生成的图片、PDF、截图、运行快照
5. LLM 网关——统一持有 API key，按 Agent 维度强制配额
6. 出网通道——调用 DashScope、OpenAI、Anthropic，或抓取目标站点
7. 可观测性——Agent 运行是非确定性的，所以日志和 trace 不是可选项
8. 密钥管理——provider key、OAuth token、OSS 凭证、数据库密码
9. 成本控制——因为 Agent 一旦自循环，token 账单一夜之间能涨十倍

至少九个阿里云服务，互相之间还要按特定方式接好。每个服务都有自己的控制台页、自己的 RAM 权限、自己的地域范围、自己的网络。手工把这些接齐，再让 `dev`、`staging`、`prod` 在三个月迭代后还彼此一致——概率约等于零。

## 控制台和 IaC 的分水岭

九个服务手工接，就是九个漂移面。这个痛点太普遍了，我画了一张固定对比图：

![控制台操作 vs Terraform —— 差异从何而来](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig1_console_vs_iac.png)

仔细看左边那一列。每一步都说得通——没有低级错误。这是聪明人在几个月里基于一连串小而合理的决策累积出来的样子。右边走的是同一条路，但每一步都在 git 里留了痕迹。两者的差距，就是"我交付了它"和"凌晨两点被叫醒，因为没人知道 `cn-beijing` 上跑着什么"的区别。

阿里云 Terraform 官方文档说得更克制：

> 控制台操作：通过界面一步步点击并输入参数。重复手动操作——难以保证一致性。依赖文档和口头沟通。
>
> Terraform：通过配置文件描述资源的期望状态。配置文件可评审、可共享、可复用。将配置文件存储在版本控制系统中，变更可追溯、可回滚。

第二段就是整个核心价值主张。本系列剩下的内容，都是围绕这个主张的实现细节。

## 用两句话解释 Terraform 是什么

Terraform 是 HashiCorp 推出的开源声明式工具。你用 HashiCorp Configuration Language（HCL）写 `.tf` 文件描述想要的云资源；Terraform 把这份期望状态和记录在状态文件里的实际状态做 diff，生成 plan；你审 plan、`apply`，Terraform 把 plan 翻译成对 provider 的 API 调用。

需要内化的三点：

- 声明式而非命令式。你不写"创建一个实例"，你写"存在一个这种形状的实例"。配置没变就是 no-op。这是 Terraform 可以在 CI 里每次提交都跑的根本原因。
- 状态是真实存在的。`terraform.tfstate` 是一个 JSON map，把你的 HCL 资源地址映射到云上真实的资源 ID。状态文件丢了，Terraform 就以为什么都不存在。第二篇会讲怎么把状态文件放在一个不会丢的地方——但状态的意义远不止"别丢文件"，下文会展开。
- 先 plan 再 apply。这是杀手级特性。每次变更前你都能看到将要新建、修改、销毁哪些资源的字面 diff。养成把 plan 输出贴到 PR description 的习惯，未来的你会感激你。

## State：Agent 栈的物料清单

"状态是真实存在的"这一点值得单独一节，因为对 Agent 栈来说，state 文件兼任你的资源清单。

我经手的每个 Agent 栈迟早都会被审计——可能是安全评审，可能是财务核对云开销，也可能是新来的 SRE 想搞清楚到底跑着什么。每次问的问题都一样：有什么资源、谁建的、花多少钱。

如果基础设施在 Terraform state 里，回答这三个问题只需要 30 秒：

```bash
terraform state list | wc -l                                  # 资源总数
terraform state list | awk -F. '{print $1"."$2}' | sort -u    # 资源类型
terraform show -json | jq '[.values.root_module.resources[] | {addr:.address, type:.type}]'
```

我现在管的四个 Agent 栈，跑这三条命令几秒钟就能出一份完整清单。在 Terraform 之前，同样的审计要打开十二个控制台 tab——ECS、VPC、RDS、OSS、RAM、KMS、SLS、ARMS、ACK、CloudMonitor、ALB、OpenSearch——运气好靠 tag 过滤，运气不好靠记忆。

state 也是供应链意义上的物料清单。每个资源都带着 provider 版本和 module 来源。alicloud provider 一年发几次破坏性变更或安全相关更新，CVE 一旦下来，你能在几分钟内 grep 所有项目，找出哪些栈受影响：

```bash
for d in stack-*/; do
  (cd "$d" && terraform providers schema -json 2>/dev/null \
     | jq -r '.provider_schemas | keys[]' \
     | grep -F 'aliyun/alicloud' && echo "  位于 $d")
done
```

更深一层：state 把基础设施变成了数据。一旦它是数据，你就能围绕它写工具。我有一个小 Python 脚本，遍历所有项目的 state 文件，输出一份 CSV——`(stack, resource_type, resource_id, region, monthly_cost_estimate)`。这份 CSV 是每月成本会议的输入。没有 state 这个统一的结构化数据源，这一切都不存在。

代价是 state 很贵重。一旦丢失，Terraform 就再也看不见那些资源——下次 plan 会得到一堆"资源已存在"的错误。第二篇会专门讲怎么把 state 放在一个不会丢的地方，本系列后续都默认你已经做到了。

## alicloud provider 覆盖了什么

state 有用的前提是有 provider 真能创建你声明的东西。云平台通过 provider 插件和 Terraform 对话。官方 `alicloud` provider 是中国第一个正式的 Terraform provider，由阿里巴巴维护。截至本文，它支持 300+ 资源类型，覆盖大致六个领域：

![alicloud provider 覆盖范围](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig2_provider_coverage.png)

按官方文档的口径：

- 计算与容器：ECS、ACK（Kubernetes）、函数计算、弹性伸缩
- 网络：VPC、SLB、ALB、NLB、NAT 网关、云企业网（CEN）
- 存储与数据库：OSS、NAS、RDS、PolarDB、Redis、MongoDB
- 安全与管理：RAM、KMS、WAF
- 可观测性：SLS、ARMS、CloudMonitor
- 大数据与 AI：MaxCompute、PAI

我们前面那张九组件清单，全都覆盖到了，包括各家 LLM provider 的 key（第六篇用 KMS Secrets Manager 处理）。

一个最小 HCL 示例，并且从第一天就锁定 provider 版本——这是一定要做的：

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

三个资源，通过 `vpc_id` 引用建立依赖。Terraform 自己解析依赖、生成正确的执行顺序。你不需要写"先 VPC 再 vSwitch 再 SG"，写出目标状态，Terraform 帮你建 DAG。

## 模块：复用的基本单元

三个资源是玩具版本。真实栈有几百个，让几百个保持可控的方法是模块。一个 module 就是一个 `.tf` 目录，吃输入、吐输出。一旦你设计出一个稳定的模式——比如三个 vSwitch + NAT + 安全组基线的 VPC——就把它包成 module，在 `dev`、`staging`、`prod`、`intl-prod` 反复使用，不再复制 HCL。

最简调用：

```hcl
module "vpc" {
  source = "./modules/vpc-baseline"

  for_each = toset(["dev", "staging", "prod"])

  vpc_name   = "agents-${each.key}"
  cidr_block = "10.20.0.0/16"
  zones      = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]
}
```

`./modules/vpc-baseline/main.tf` 里是真正的 `alicloud_vpc`、`alicloud_vswitch`、`alicloud_nat_gateway`。调用方不需要关心实现，他们只想要一个有合理默认值的 VPC。和 Python 函数一个意思，只是作用对象是基础设施。（只要迭代对象是有意义的命名集合，优先用 `for_each` 而不是 `count`——`count` 在你删掉中间元素时会重排索引，导致 Terraform 销毁并重建一堆无关资源。）

第三篇会构建这个 module，并在后续每一篇里复用它。

## IaC 真正能预防的 Agent 特有故障

在比较 Terraform 和它的替代品之前，先把"你具体在买什么"钉住。通用 IaC 宣传聚焦在"一致性"和"可复现性"——没错，但太弱。三年下来，最让我痛的故障都是 Agent 形状的，每一个都对应一个 Terraform 形状的修复：

1. 凌晨三点的 token 泄漏。Agent 自循环——停止条件写错、工具无限重试、planner 状态幻觉——一夜烧掉 ¥40,000 的 LLM 预算。控制台搭的栈没有可编程预算护栏，因为没人写。Terraform 栈从第一天就带 `alicloud_log_alert`（第七篇），因为 module 默认包含。多一条资源的成本，远低于财务半夜打电话的成本。

2. "谁拿走了我的 key" 恐慌。一个外包离职，他手里有一份原型时期复制的 OpenAI key。控制台搭的栈把那个 key 散落在三台 ECS 的 `.env` 文件和一条钉钉私聊里。轮换一次是半天 `grep -r OPENAI_API_KEY ~/projects` 加祈祷。Terraform 栈里每个 key 都在 KMS Secrets Manager（第六篇），后面挂着 RAM 角色，唯一一个权威位置——轮换就是改 `secrets.auto.tfvars` 然后 `terraform apply`。

3. 幽灵 NAT 账单。Agent 本来就话痨，调 LLM 更话痨。路由配错，流量从错误地域的公网 NAT 出去，出网账单翻三倍。控制台不会告诉你这件事，只会下个月在账单里出现一个新条目。Terraform 里 `alicloud_nat_gateway` 和它服务的 `alicloud_vswitch` 在同一个 module，依赖关系在 plan 阶段就看得见。第三篇会把这种结构定为默认。

4. "生产到底在哪个区？" 阿里云有 30+ 地域。两个月前入职的工程师在排障时，搞不清楚 prod RDS 是在 `cn-shanghai` 还是 `cn-beijing`，得翻钉钉历史。Terraform 的答案是 `terraform output rds_endpoint`，五秒钟，永久可追溯。

5. "我们丢了 LLM 网关配置"的周末。网关负责路由模型、持有配额、记录流量，是两台 ECS——有人 SSH 上去手改了 `/etc/litellm/config.yaml`，重启之后 `/etc/litellm` 被清空，服务挂了。Terraform 版本把配置写进 cloud-init（第六篇），每台实例启动时都自带相同配置，零手工步骤。

这些故障没有一个是异国情调的。每一个都在我经手的项目上发生过。每一个都是被 IaC 模式预防掉的，而不是事后恢复掉的。这才是 Terraform 在 Agent 栈上的真正卖点——不是抽象的可复现性，而是一份具体的"不会发生的灾难"清单。

## Terraform、Pulumi、Crossplane、ROS 怎么选

接受了 IaC 的论点，为什么偏偏是 Terraform？快速过一下其他选项。没有绝对对错，按团队契合度选，不要按信仰：

![四种 IaC 工具对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/01-why-terraform-for-agents/fig3_iac_tools_compare.png)

把这四个都用过之后，我的真实判断：

- Terraform 是默认。provider、module、会用的人都最多。HCL 用第一天觉得怪，第二天就习惯了。除非你有强理由不用，否则就选它。
- Pulumi 在 Terraform 的 provider 之上套了一层，让你用 Python/TypeScript/Go 写。表达力是真的——你有循环、条件、IDE 真能查类型。代价是调试链路：出问题时要穿过两层（你的代码 → Pulumi → TF provider）。如果团队真心讨厌 HCL，值得。
- Crossplane 活在 Kubernetes 里——每个云资源都是一个 CRD，你 `kubectl apply` 出 VPC。如果你已经是纯 Kubernetes + GitOps 的团队，很优雅；否则痛苦。
- ROS（资源编排服务）是阿里云原生方案，和控制台深度集成，模板用 JSON 或 YAML，不需要装 provider 插件。只有当你 100% 长期绑死阿里云、运维团队偏好托管服务时再选。

阿里云官方文档的 FAQ 给出的对比挺中肯：

> Terraform 和 ROS 都是声明式 IaC 工具。Terraform 是开源的第三方工具，支持多云管理；ROS 是阿里云原生服务，与阿里云管理控制台深度集成。如果你需要多云支持，或已经在其他地方使用了 Terraform，那么 Terraform 是更好的选择。

对一个会调用多家 LLM provider、未来可能要新加坡或硅谷地域的 Agent 系统来说，多云友好的 Terraform 才是合理的默认。

## 这个系列做什么、不做什么

会做：

- 用八篇文章带你从 `terraform init` 走到一个真能跑的 `research-agent-stack`，部署在阿里云上。
- 给出 VPC、ECS、ACK、OSS、RDS、OpenSearch、KMS、SLS、CloudMonitor 的真实可用 HCL。
- 覆盖文档里没有的故障模式——state drift、tfstate 锁死、GFW 拉 provider、地域库存不足。
- 系列结束时给一个可以 fork 的初始仓库。

不会做：

- 教 HCL 语法，超出我们用到的范围。HashiCorp 官方教程做得更系统。
- 教你怎么写 Agent 本身。LangGraph、AutoGen、MetaGPT、Claude Code 的系列都已经有了，挑一个看。
- 把阿里云和 AWS、GCP 做逐项功能对比。IaC 模式跨云通用，资源名字不通用。

## 接下来

第二篇是第一次动手：装 alicloud provider；选认证方式（静态 AK/SK、AssumeRole、ECS RAM 角色——这三种并不等价）；把 state 放在 OSS 上、用 Tablestore 做锁；以及 `dev`/`staging`/`prod` 的 workspace 模式。

如果今天只能做一件事，就是装上 Terraform（macOS 上 `brew install terraform`，或按官方 `Install Terraform` 步骤），然后跑 `terraform version` 确认。后续文章默认你已经装好了。

> 实战建议：从第一天就在 `required_providers` 里给 alicloud provider 锁版本，并在 `required_version` 里给 Terraform 自身锁版本。provider 在快速迭代，小版本之间的破坏性变更很少，但不是零。锁定版本意味着你周五跑的 `terraform plan`，周一还能拿到一样的结果。
