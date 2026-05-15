---
date: 2023-06-26 09:00:00
title: "paper2repo： GitHub Repository Recommendation for Academic Papers"
description: "paper2repo（WWW 2020）双塔 GCN：引用图与协同标星图分头训练，靠余弦桥接对加 WARP 排序损失打通跨塔嵌入空间。"
tags:
  - Recommend System
categories: 论文笔记
lang: zh
mathjax: true
disableNunjucks: true
translationKey: "paper2repo-github-repository-recommendation"
---

读论文时最折磨的瞬间之一是：方法看懂了，想复现原作者的代码，结果论文里的 "code available at" 要么根本没提，要么链接已失效（404），要么指向一个空仓库。退而求其次去 GitHub 搜，能命中的基本都是名字起得规范、README 写得用心的项目；冷门方法、起名随意的工程则很难找到。

paper2repo（WWW 2020）将这一过程系统化：将论文摘要和 GitHub 仓库放入同一嵌入空间，在该空间中，“论文 · 仓库”的内积表示相关度，并据此进行排序。

真正需要讲清楚的，不是它采用的 CNN 编码器或 GCN——这两者在 2020 年已是推荐系统与图学习中的标准配置。真正有意思的设计是：论文和仓库分别活在两张完全不同的图上（论文这边是引用图，仓库那边是协同标星 + 标签共现图），如果两个塔分头训练 GCN，得到的两套嵌入会落在两个互不相干的 $d$ 维空间里，跨塔做内积没有意义。 paper2repo 的解法是以一批已知的论文-仓库匹配对（即论文正文中明确提供 GitHub 链接的样本）作为跨模态桥梁，并引入余弦相似度约束，强制对应桥接对的嵌入方向对齐。模型的其他组件本质上都服务于使该约束能够端到端优化。


---

## 你将学到什么
- 如何用三种非对称信号（引用、协同标星、标签共现）拼出一张联合异构图
- 为何两座彼此独立的 GCN 塔仍需共享同一度量空间？余弦约束如何实现这种对齐？
- WARP 排序损失，以及它为什么和归一化嵌入是天然搭档
- 如何用 “拉格朗日 → 乘性” 这一招把约束权重 $\lambda$ 这个超参从训练里彻底消掉
- 推理时模型怎么用，论文报告的 HR@10 / MAP@10 / MRR@10 在工程意义上意味着什么

## 前置知识

- 熟悉 GCN 的层间传播：$H^{(l+1)} = \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)})$
- 熟悉推荐系统的常见指标： HR@K、 MAP@K、 MRR@K
- 知道余弦相似度、 hinge 损失，以及 margin 排序损失大致在做什么


---

## 系统总览

![paper2repo system architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig1_system_architecture.png)

图 1 从左右两边、自上而下读。模型是两座并行的塔。左塔把论文摘要喂进 CNN，再用 GCN 在引用图上传播，输出论文嵌入 $h^p$。右塔对仓库描述和标签做同样的事，在仓库关联图上传播，输出 $h^r$。两座塔不共享任何权重，但训练阶段被两股跨塔的力量拽在一起：一是桥接对上的余弦对齐约束，二是用论文-仓库正样本对喂的 WARP 排序损失。推理阶段两座塔各自独立运行，排序就是 $h^p$ 和预先算好的仓库嵌入索引做一次稠密内积。

## 联合异构图

![Heterogeneous graph: papers, repos, users, bridged pairs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig2_heterogeneous_graph.png)

模型设计中首个关键决策，是为每座塔选择适配的图结构。论文这边有一个干净的现成信号——引用——它是有向、稀疏、语义明确的。仓库这边没有可以直接对应的东西， paper2repo 用两种隐式信号把它造出来。

**论文引用图。** 节点是论文，边是引用关系（按无向处理）。节点特征是 CNN 在摘要上算出的向量。

**仓库关联图。** 节点是仓库，边来自两个来源：

- *协同标星边。* 被同一个用户同时星标的两个仓库连一条边。这是一种噪声较大但构建成本极低的隐式协同信号（即用户同时星标两个仓库）。注意：用户不作为节点显式建模于仓库图中，仅通过协同标星行为间接诱导出边（图 2 中点状橙色虚线所示）。
- *标签共现边。* 两个仓库共享一个标签，且这个标签的 TF-IDF 分数高于阈值（论文取 0.3）的话，连一条边。这个阈值用来过滤掉 "python"、"deep-learning" 这类太泛的标签，保留真正有主题区分度的那部分。

仓库节点的特征是 CNN 编码的描述向量加上标签词向量的均值，再投影到同一个维度做融合。

**桥接对。** 一小批论文-仓库对是已经标好的——论文正文里直接放出 GitHub 链接的那些。这是整个对齐机制唯一能看到的监督信号。数据集里 7,571 个仓库中，只有 2,107 个属于桥接仓库。其他所有仓库要被检索出来，靠的都是经由两张图传递过去的间接信号。

## 文本编码

两座塔的文本编码用的是同一套 CNN-over-words 配方：

1. **分词** 把描述、摘要或标签序列切成 $\{x_1, \dots, x_n\}$，用预训练词向量（论文用 GloVe）把每个 token 映成 $d$ 维向量。
2. **多窗口卷积** 用窗口大小 $h \in \{2, 3, 4\}$ 各 $k$ 个卷积核：
   $$
   c_i = \sigma(W \cdot x_{i:i+h-1} + b)
$$
3. **时间维最大池化** 把每个卷积核压成一个标量，拼起来得到固定长度的特征向量。
4. **标签** 因为是无序集合，把所有标签的词向量取平均，再过一个全连接层，投影到和描述特征同样的维度。
5. **特征融合** 把描述特征和标签特征加和或拼接，得到最终的仓库表示。

论文摘要走同一个编码器，只是没有标签那个分支。两个编码器的输出就是各自 GCN 的输入节点特征 $H^{(0)}$。

## 受限 GCN

标准 GCN 一层是
$$
H^{(l+1)} = \sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\,H^{(l)}\,W^{(l)}\right)
$$
其中 $\tilde A = A + I$ 加了自环，$\tilde D$ 是它的度矩阵。在每张图上跑 $L$ 层，就分别得到论文嵌入 $h^p$ 和仓库嵌入 $h^r$。这套嵌入对图内任务（论文这边的引用预测、仓库这边的协同标星预测）是有用的，但它们活在两个互不相关的 $d$ 维空间里——跨塔做内积没意义。

“受限”两个字解决的就是这件事。对每一对桥接论文和桥接仓库 $(i, i)$，要求
$$
h^{p}_{i}\!\cdot h^{r}_{i} \;\geq\; 1 - \delta, \qquad \delta \approx 10^{-3}.
$$
两边嵌入都做了 $\ell_2$ 归一化，所以内积等于余弦相似度，这条约束等价于：桥接对的论文-仓库嵌入应该在方向上几乎重合。两座塔不共享任何参数，唯一能满足这个约束的方式，就是逼着 GCN 和编码器学出方向对齐的坐标轴。

## 两股训练力

![WARP ranking + cosine alignment constraint](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig3_embedding_objectives.png)

图 3(a) 展示排序力。归一化之后所有嵌入都活在单位球面上。对一个 “论文-正样本仓库” 对 $(p, r^+)$，损失把 $h^{r^+}$ 朝 $h^p$ 的方向拉，把任何落在正样本附近 margin $\gamma$ 之内的负样本 $r^-$ 推开。图 3(b) 展示对齐力。如果没有跨塔约束，两座 GCN 各自训练得到的 “论文云” 和 “仓库云” 会以椭球状漂在 $\mathbb{R}^d$ 的不同区域里，互不接壤。约束把桥接对捆在一起，相当于把整片仓库云拽进论文云的坐标系。

### WARP 排序损失

WARP （Weighted Approximate-Rank Pairwise）是一种带 rank-aware 权重的 margin 排序损失。对正样本对 $(p, r^+)$ 和采样得到的负样本 $r^-$：
$$
\ell(p, r^+, r^-) = L\!\left(\mathrm{rank}(p, r^+)\right) \cdot \big[\gamma - h^p\!\cdot h^{r^+} + h^p\!\cdot h^{r^-}\big]_+
$$
$[\cdot]_+ = \max(0, \cdot)$ 是 hinge， $L(k) = \sum_{j=1}^{k} 1/j$ 是一个非递减权重——当正样本目前排在很靠后的位置时，权重变大。这里的 rank 是估出来的：不停采样负样本，直到撞见一个违反 margin 的，看采了多少次。

两点工程经验。第一， WARP 单条正样本要配很多负样本，原始论文的做法是无放回采样直到撞见 margin 违例。第二，归一化嵌入下内积就是余弦，取值在 $[-1, 1]$，所以 margin $\gamma$ 是有直观意义的（典型值 0.1 上下）。

### 把拉格朗日乘子换成乘性因子

完整目标是 WARP 损失加上对齐约束。教科书式的拉格朗日写法是
$$
\min \sum_{(p,r^+,r^-)} \ell(p, r^+, r^-) \;+\; \lambda \cdot C_e
$$
其中 $C_e$ 是平均对齐误差
$$
C_e = \frac{1}{|B|}\sum_{i \in B} \big[(1 - \delta) - h^{p}_{i}\!\cdot h^{r}_{i}\big]_+,
$$
$|B|$ 是桥接对数量， $\lambda$ 控制两项的权衡。问题是训练过程中两项的量级会不停漂移，固定一个 $\lambda$ 要么把排序损失压死，要么让约束放空。 paper2repo 把加性的拉格朗日换成乘性的：
$$
\mathcal{L} = \left(\sum_{(p,r^+,r^-)} \ell(p, r^+, r^-)\right) \cdot (1 + C_e).
$$
归一化嵌入下 $C_e \in [0, 2]$ 是有界的，作为乘性因子量级也合适。约束被满足时（$C_e \to 0$）乘子塌缩到 1，损失退化为纯 WARP；约束被违反时整个排序损失被放大，逼优化器先把对齐修好。整套机制不再需要 $\lambda$ 这个超参。

## 训练流程

**正样本。** 每个桥接论文-仓库对算一个正样本。为了扩充正样本，论文还把那些被同一批用户高频共星标的仓库对 $(r, r')$ 也算作 “相关”——很多用户同时星了 $r$ 和 $r'$，就把 $r'$ 当作 $r$ 的正样本。这种扩充是对称的，喂进的是仓库这边的排序损失。

**负样本。** 在整个仓库池里均匀采样。 WARP 本身会一直采到撞见 margin 违例为止，所以有效的负样本数量会随着当前正样本的难度自动调整。

**优化器。** Adam，统一更新 CNN 编码器、 GCN 权重、投影层。 GCN 在论文里只用 2 层——再深就开始过平滑了。

**输出。** 对齐好的论文与仓库嵌入，加上预先算好的仓库嵌入索引，供线上检索用。

## 推理流程

![Recommendation flow at query time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig4_recommendation_flow.png)

线上来一篇新论文，左塔跑一遍： CNN 编码摘要，再用 GCN 在它已知的引用邻居上传播一次（如果这篇论文还没被引用，就只用编码器输出）。结果是单个向量 $h^p$。排序就是 $h^p$ 和预算好的仓库嵌入矩阵做一次稠密矩阵-向量乘， top-K 是一次 argpartition。整套系统没有重排器、没有二段过滤——这是个有意识的工程取舍：方便上线，方便做消融。

图 4 那个示意 shortlist 反映的是典型的排序模式：榜首基本被那些 “话题接近”（文本相似度高）+ “在关联图里位置好”（GCN 平滑得好）的仓库占据。两个信号在论文话题清晰、有公认参考实现时互相加成；在话题冷门时会分歧——这种情况下关联图几乎是唯一可靠的信号，模型实际上退化为 “社区平时会和这类工作放在一起的是哪些仓库”。

## 实验

![paper2repo vs seven baselines on HR@10, MAP@10, MRR@10](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig5_evaluation_results.png)

**数据。** 32,029 篇论文，来自 Microsoft Academic （2010-2018 顶会顶刊）； 7,571 个 GitHub 仓库，其中 2,107 个是桥接仓库。论文把桥接对切成训练 / 验证 / 测试三份用作对齐监督，剩下的图结构作为无监督的辅助信号。

**指标。** HR@K （前 K 是否命中相关仓库）、 MAP@K （按 rank 加权的精度）、 MRR@K （首个相关仓库 rank 的倒数）。统一在 K = 10 上评估。

**基线。** 七个跨域或图感知推荐方法： BPR、 MF、 LINE、 NCF、 CDL、 KGCN、 NSCR。它们都没有显式的跨塔对齐目标。

图 5 里这种排布正好是 “对齐约束在干实事” 该有的样子。完全不利用图结构的 BPR、 MF 垫底；只用一侧结构的 LINE、 NCF 上来一截；两侧结构都用、再加外部知识图谱的 CDL、 KGCN、 NSCR 接近 paper2repo。 paper2repo 拉开一个稳定的差距，**HR@10 上的差距最大**——这个指标只看 “正确仓库有没有进 shortlist”，不在意它是不是排第一。这正好对应了对齐约束在干的事：把桥接仓库拽进嵌入空间里 “对” 的那个区域，但区域内部的精排留给排序损失继续打磨。

## 局限

**仓库冷启动。** 一个新仓库没人星标、最多挂一两个标签，在关联图里几乎没有边。 GCN 平滑出来的嵌入和孤立的文本特征差不多——大致等价于 TF-IDF 检索的水平。除非有人手工补一对桥接关系，否则约束帮不上忙。

**桥接对稀缺。** 整个对齐机制建立在一层薄薄的有监督桥接对上。在论文-代码挂钩稀少的领域（早期论文、工业界的应用 ML 文章）里，可对齐的监督信号根本不够。

**全图 GCN 的算力开销。** 这个量级（约 4 万节点） 2 层 GCN 跑得动，但显然延伸不到真实世界中数百万级论文、数千万级仓库的规模。下一步自然是 GraphSAGE、 ClusterGCN 这类基于采样的 GNN。

**静态快照。** 整张图是当成静态处理的，但现实里引用按月增长、星标按天增长。要部署的话，时序 GNN 或流式索引才更合适。

## 上线时的双塔检索

训练讲到“我们得到了对齐的 embedding”就停了，但真正能跑起来的系统还要解决三个细节。原论文略过了它们，工程上躲不掉。

**索引选型。** 7,571 个 repo 的规模下，在 GPU 上做暴力的稠密矩阵-向量乘是亚毫秒级的， CPU 上 `numpy.dot` 也够用。一旦把规模拉到 GitHub 真实的 3 亿仓库量级，选型就开始重要： HNSW （`hnswlib`）在 $M=32$、$ef\_construction=200$ 的配置下，对 1000 万条向量单机大约 5 ms/query 能给出 0.99 召回。 paper2repo 里向量被 $\ell_2$ 归一化、相似度是余弦， FAISS `IndexFlatIP` 是个诚实的基线，`IndexHNSWFlat` 是生产目标。

**刷新节奏。** 引用和 star 是持续累积的。每季度重训一次的 GCN 对论文塔够用——引用边变化得慢； repo 塔需要更勤快，因为 star 变化很快， tag 共现的 TF-IDF 阈值也会随语料整体分布漂移。一个稳的拆法： repo 索引每晚重建、论文索引每周重建、跨塔对齐每月微调一次。对齐是最慢的那个旋钮，它要等 bridged 对增长——而 bridged 对的增长依赖于新论文里作者写上 "code available at" 这种链接，这事完全不在你的控制范围内。

**冷启动恢复回路。** 新建的 repo 没有边，只能靠 TF-IDF / CNN 编出来的特征向量站着，这本质上等价于关键词搜索能给你的东西。运营上有两招比较顺手：第一，对度分布末尾 20% 的 repo，把 README 加 tag 的 TF-IDF 余弦当作兜底排序器；第二，跑一个定期的“自桥接”任务，把置信度高于阈值的 top-1 论文-repo 预测晋升进 bridged 集合，让对齐监督的规模随时间扩张。第二个动作放在论文里叫“半监督”，放到生产系统里就是一个不断长训练数据的定时任务。

## 跟现代双编码器检索器的对比

paper2repo 的设计早于 dense passage retrieval、 ColBERT 和 CLIP 这一波跨模态对齐方法。今天再做一遍，会变成什么样？值得想一下。

纯双编码器方案（用 sentence-transformer 在 bridged 对上跑 InfoNCE 微调）会把 GCN 整个去掉，只靠文本编码器。对 README 写得很丰富的 bridged repo 来说效果接近，架构也简单得多。双编码器一直输的地方是**stub repo**： README 只有一行的项目几乎没有文本信号，而**图信号**——共同 star、 tag 共现——恰恰能把它的 embedding 拽进一个有意义的邻域里。冷启头部 repo 上两者差距不大；真正拉开差距的是那批文本稀薄的长尾 repo——这部分价值正是 GCN 复杂度换来的。

如果今天重写 paper2repo， CNN 编码器会换成 sentence-transformer， WARP 会换成 InfoNCE，受限 GCN 这个骨架大概率保留，但全图传播会换成 GraphSAGE 那种邻居采样。最经久不衰的反倒是那个乘性拉格朗日的小 trick——它本质上是一个动态 loss 平衡方法，在现代多任务文献里以各种不同的名字反复出现。

## 总结

paper2repo 真正有意思的点不在 GCN，不在 WARP，也不在 CNN 编码器——这三样到 2020 年都已经是标配了。有意思的是 *受限* 这两个字：意识到两座互相独立的塔可以靠一小批桥接对加上损失里一个乘性因子捆在一起。这个套路可以干净地迁移到很多 “两侧结构都很丰富、但桥接信号很薄” 的跨平台场景：论文-数据集、论文-作者、商品-评论、查询-文档（点击日志当桥）等等。文本编码器换、图类型换，但这个约束机制不变。

最值得做的三个延伸方向。一是给论文这边加上作者 / 期刊 / 机构节点，给仓库这边加上贡献者 / 组织节点，把每座塔的图升级成真正的异构图；二是把全批 GCN 换成基于采样的 GNN，扩到 4 万节点以上的规模；三是把静态桥接集合换成自训练循环，从高置信的 top-1 预测里持续挖出新的桥接对喂回训练。

## 参考文献

[1] Shao, H., Sun, D., Wu, J., Zhang, Z., Zhang, A., Yao, S., Liu, S., Wang, T., Zhang, C., & Abdelzaher, T. (2020). paper2repo: GitHub Repository Recommendation for Academic Papers. *Proceedings of The Web Conference 2020*, 580-590.

[2] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

[3] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to Large Vocabulary Image Annotation. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*, 2764-2770.

[4] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*.
