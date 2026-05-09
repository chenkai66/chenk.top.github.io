---
title: "OpenClaw 快速上手（二）：安装到第一次对话，十分钟搞定"
date: 2026-04-04 09:00:00
tags:
  - openclaw
  - installation
  - tui
  - dashscope
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 2
description: "在 macOS 或 Ubuntu 上装好 OpenClaw，接入模型服务商，启动 TUI，十分钟之内得到一个能用的 Agent。顺便解决那个浪费最多时间的 Node 版本坑。"
disableNunjucks: true
translationKey: "openclaw-quickstart-2"
---

![OpenClaw 安装与首次对话流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_1.png)

## README 说五分钟，实际要十分钟

OpenClaw 的 README 写着 "5 minutes to your first agent"。如果你的 Node 版本恰好满足要求，这个说法大致成立。但现实是：大部分人的机器上跑着 Node 18 或 20，装完之后 gateway 启动直接报语法错误——因为代码里用了 optional chaining 的某些新写法，需要 Node 22.16 以上才能解析。

光是排查这个问题就能吃掉五分钟。所以我说十分钟更诚实。

这篇文章的目标：从零开始，到你在终端里和 Agent 完成第一次真实对话，全程不超过十分钟。我会把每个坑点提前标出来，你跟着走就行。


## 前置条件

开始之前，确认这几样东西：

### Node.js v22.16+

这是最关键的一条。OpenClaw 的 gateway 代码使用了一些需要 Node 22.16 才支持的语法特性。低于这个版本，gateway 进程启动就会崩溃，报错信息类似：

```
SyntaxError: Unexpected token '?.'
```

检查你当前的版本：

```bash
node --version
# 如果输出 v18.x 或 v20.x，需要升级
```

如果版本不够，用 nvm 升级（下面会详细说）。

### 2GB 可用内存

Gateway 本身占用不大（约 200MB），但 Agent 在处理大文件或多轮对话时，内存峰值可能到 1.5GB。2GB 是安全线。

### LLM API Key

你需要至少一个模型服务商的 API Key。推荐选择：

- **DashScope（通义千问）**：国内首选，有免费额度
- **百炼 Coding Plan**：200 元/月，8 个模型随便用
- **Anthropic**：效果最好，但需要海外支付方式

具体怎么拿 Key，后面有专门一节讲。

## 升级 Node：用 nvm，别用 sudo

如果你的 Node 版本不够，最干净的升级方式是 nvm。

### 安装 nvm（如果还没有）

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

装完之后重新打开终端，或者手动 source：

```bash
source ~/.zshrc   # macOS
source ~/.bashrc  # Ubuntu
```

### 安装 Node 22

```bash
nvm install 22
nvm use 22
nvm alias default 22
```

验证：

```bash
node --version
# v22.16.0 或更高
```

第三行 `nvm alias default 22` 很重要——没有它的话，下次开终端又会切回旧版本。

## 安装 OpenClaw

一行命令：

```bash
npm install -g @anthropic-ai/openclaw@latest
```

注意包名是 `@anthropic-ai/openclaw`，带 scope 前缀。这是因为 npm 上 `openclaw` 这个名字早就被占了（一个废弃的 2017 年的包），Anthropic 用了自己的 org scope 来发布。

安装完成后验证：

```bash
openclaw --version
```

如果能正常输出版本号，说明安装成功。

## 安装失败的常见问题

### npm 权限错误

症状：

```
npm ERR! Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'
```

原因：你可能之前用 `sudo npm install -g` 装过东西，把全局目录的 owner 改成了 root。

解决方案：**用 nvm，不要用 sudo**。nvm 管理的 Node 把全局包装在用户目录下（`~/.nvm/versions/`），不需要 root 权限。如果你已经在用 nvm 但仍然报错，检查一下是不是 shell 里有 alias 指向了系统 Node：

```bash
which node
# 应该输出类似 /Users/yourname/.nvm/versions/node/v22.16.0/bin/node
# 如果输出 /usr/local/bin/node，说明 nvm 没生效
```

### node-gyp 编译失败（macOS）

症状：一大堆 C++ 编译错误，提到 `node-gyp`。

原因：OpenClaw 依赖的某些 native 模块需要编译，而你的 macOS 没装 Xcode Command Line Tools。

```bash
xcode-select --install
```

弹窗点确认，等它装完（大约 5 分钟），再重新 `npm install -g @anthropic-ai/openclaw@latest`。

### 网络超时（国内用户）

症状：

```
npm ERR! network timeout at: https://registry.npmjs.org/@anthropic-ai/openclaw
```

解决方案：切换到 npmmirror：

```bash
npm config set registry https://registry.npmmirror.com
npm install -g @anthropic-ai/openclaw@latest
```

装完之后如果你想切回官方源：

```bash
npm config set registry https://registry.npmjs.org
```

或者只在安装时临时指定：

```bash
npm install -g @anthropic-ai/openclaw@latest --registry=https://registry.npmmirror.com
```

### 安装成功但命令找不到

症状：`npm install -g` 显示成功，但 `openclaw` 命令报 `command not found`。

原因：nvm 的 bin 目录没在 PATH 里。通常是因为 shell 配置文件里 nvm 的初始化脚本位置不对。

检查：

```bash
echo $PATH | tr ':' '\n' | grep nvm
```

如果没有输出，手动在 `~/.zshrc`（或 `~/.bashrc`）末尾加上：

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

然后 `source ~/.zshrc`。

### 版本冲突：全局 vs 本地

如果你之前在某个项目里 `npm install openclaw`（不带 -g），项目目录下的 `node_modules/.bin/openclaw` 可能会覆盖全局版本。用 `which openclaw` 确认你执行的是全局安装的那个。

## 获取 API Key

### DashScope（推荐国内用户）

DashScope 是阿里云的模型服务平台，通义千问系列模型都在这上面。

1. 打开 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)
2. 用阿里云账号登录（没有就注册一个，实名认证需要几分钟）
3. 左侧菜单 "API-KEY 管理" -> "创建新的 API-KEY"
4. 复制保存

免费额度：通义千问开源模型（qwen-turbo、qwen-plus）每月有 100 万 tokens 的免费额度，对于学习和个人使用完全够了。

### 百炼 Coding Plan

如果你想用更多模型、更大的配额：

- 价格：200 元/月
- 包含：qwen-max、qwen-plus、qwen-turbo 等 8 个模型
- 入口：[bailian.console.aliyun.com](https://bailian.console.aliyun.com)
- API Key 和 DashScope 通用

### Anthropic

Anthropic 的 Claude 模型在代码和推理任务上效果最好，但获取 API Key 对国内用户有门槛：

- 需要海外信用卡（Visa/Mastercard）
- 或者用香港 VPS 做转发代理
- 注册地址：[console.anthropic.com](https://console.anthropic.com)
- 最低充值 5 美元

如果你有条件，Claude 3.5 Sonnet 搭配 OpenClaw 的效果确实是目前最好的。但 DashScope 的 qwen-max 也完全能用，不必强求。

## Onboard 向导

安装完成后第一次运行任何 OpenClaw 命令，会自动进入 onboard 向导。你也可以手动触发：

```bash
openclaw onboard
```

向导会依次问你：

1. **Agent 名字**：给你的 Agent 起个名，纯标识用途，随便填
2. **你的名字**：Agent 在对话中会用这个称呼你
3. **模型服务商**：选 DashScope / Anthropic / OpenAI 等
4. **API Key**：粘贴你刚才拿到的 Key
5. **默认模型**：比如选 `qwen-max` 或 `claude-3-5-sonnet`

完成后配置写入 `~/.openclaw/openclaw.json`。你随时可以手动编辑这个文件来改配置。

## 启动 Gateway

```bash
openclaw gateway start
```

正常输出应该是：

```
[INFO] Gateway starting on port 18790...
[INFO] Agent loop initialized
[INFO] Provider: dashscope (qwen-max)
[INFO] Gateway ready. Listening at http://localhost:18790
```

如果看到 `SyntaxError` 或者进程立刻退出，99% 是 Node 版本问题，回到上面 "升级 Node" 那节。

Gateway 启动后会常驻后台。它是 TUI 和 Web Dashboard 的后端——所有 Agent 逻辑都在 gateway 里跑。

## TUI 第一次对话

打开另一个终端窗口（gateway 需要保持运行），启动 TUI：

```bash
openclaw tui
```

你会看到一个终端界面，底部有输入框。下面我用三条测试消息来验证各项功能是否正常。

### 测试一：简单问候

输入：

```
你好，自我介绍一下
```

预期：Agent 会回复一段自我介绍，提到它的名字（你在 onboard 时设的那个）和基本能力。这条消息验证的是 LLM 连接是否正常。

如果这里超时或报错，检查 API Key 是否正确、网络是否通畅。

### 测试二：读取文件

输入：

```
读一下 ~/.openclaw/openclaw.json 的内容
```

预期：Agent 会调用文件读取工具，把配置文件的内容展示给你。你能看到你刚才填的 provider、model 等信息。这条消息验证的是工具调用能力。

在 gateway 的日志里，你会看到类似：

```
[TOOL] read_file: ~/.openclaw/openclaw.json
[TOOL] result: { "provider": "dashscope", "model": "qwen-max", ... }
```

### 测试三：创建文件

输入：

```
在 /tmp 目录下创建一个 hello.txt，内容写 "OpenClaw is working"
```

预期：Agent 调用写文件工具，创建文件，然后确认完成。你可以 `cat /tmp/hello.txt` 验证。

这三条消息如果都正常通过，恭喜——你的 OpenClaw 已经完全可用了。

## 更多可以尝试的操作

基本功能确认之后，可以试试这些更有意思的：

### 读取网页

```
帮我看看 https://news.ycombinator.com 首页现在有什么热门内容
```

Agent 会调用 web fetch 工具抓取页面，然后给你总结。

### 执行 Git 命令

```
看看当前目录的 git log，最近 5 条
```

Agent 调用 shell 工具执行 `git log --oneline -5`，把结果返回给你。

### 多步骤任务

```
在 /tmp 创建一个 project 目录，里面放一个 main.py，写一个简单的 Flask hello world，然后告诉我怎么运行它
```

这条指令需要 Agent 执行多个步骤：创建目录、写文件、给出说明。你能在 gateway 日志里看到完整的工具调用链：

```
[TOOL] shell: mkdir -p /tmp/project
[TOOL] write_file: /tmp/project/main.py
[AGENT] Composing response...
```

### 观察 Gateway 日志

Gateway 的日志是理解 Agent 行为的最好窗口。每次工具调用都会打印：

- 调用了什么工具
- 传了什么参数
- 返回了什么结果
- LLM 的思考过程（如果模型支持 thinking）

后续文章会深入讲 gateway 的内部机制。

## Web Dashboard

OpenClaw 自带一个 Web 界面，跑在 gateway 同一个端口上：

```
http://localhost:18790
```

浏览器打开就能看到。Dashboard 提供：

- 对话历史浏览
- Agent 状态监控
- 配置在线编辑
- 工具调用的可视化时间线

日常使用 TUI 就够了。Dashboard 在调试复杂问题、给别人演示、或者你想看完整的工具调用时间线时比较有用。

![Web Dashboard 界面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_2.png)

## 安装后的目录结构

安装和 onboard 完成后，`~/.openclaw/` 下的文件结构：

```
~/.openclaw/
├── openclaw.json    # 主配置文件（provider、model、API key 等）
├── workspace/       # 工作目录和记忆存储
├── skills/          # 技能定义文件
├── agents/          # Agent 配置（多 Agent 时用到）
└── sessions/        # 会话记录（对话历史）
```

几点说明：

- `openclaw.json` 是最核心的文件，几乎所有配置都在这里
- `workspace/` 里存放 Agent 的长期记忆，跨会话保留
- `skills/` 目前为空，后续文章会讲怎么自定义技能
- `sessions/` 会随着使用逐渐增大，可以定期清理

## 架构概览

理解了目录结构之后，再看一下运行时的数据流：

![运行时数据流：TUI - Gateway - LLM Provider，Gateway 同时调用本地系统](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/fig1_dataflow.png)

工作流程：

1. 你在 TUI 输入消息
2. TUI 通过 HTTP/WebSocket 发给 Gateway
3. Gateway 里的 Agent Loop 把消息连同上下文发给 LLM Provider
4. LLM 返回文本回复，或者返回工具调用请求
5. 如果是工具调用，Gateway 在本地执行（读文件、跑命令等），把结果再送回 LLM
6. 循环直到 LLM 给出最终回复
7. 最终回复通过 WebSocket 推回 TUI 展示

这个 "LLM 思考 -> 调用工具 -> 拿到结果 -> 再思考" 的循环就是所谓的 Agent Loop。后面的文章会拆开讲它的每一个环节。

## 小结

到这里你应该已经：

- 装好了 Node 22.16+
- 全局安装了 OpenClaw
- 配置了 DashScope（或其他 provider）的 API Key
- 启动了 Gateway
- 在 TUI 里完成了第一次对话

下一篇我们深入 Gateway 的内部实现：它怎么管理 Agent Loop，怎么调度工具，怎么处理并发请求。理解了 Gateway，你就能根据自己的需求去定制 Agent 的行为了。

---

*本系列下一篇：[OpenClaw 快速上手（三）：让 Agent 循环跑起来的六个层](/zh/openclaw-quickstart/03-architecture/)*
