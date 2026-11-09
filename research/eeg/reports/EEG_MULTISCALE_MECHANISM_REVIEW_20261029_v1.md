# EEG 局部信息通路：机制复审与锚定注意力原型

2026-10-29，America/Los_Angeles。本报告不改已登记的编码器、D40、ordinary 或原 prefix v1；不启动远端 GPU。新增锚定注意力仅为另存的未训练原型。**当前没有新图片收益、创新充分或 SOTA 结论。**

## 研究问题应怎样收窄

有价值的问题是：在同一全局 EEG 条件 c、同一图像历史和同一预训练生成器下，未池化的 EEG 信息是否还有可用于重建的增量？“用了基础模型”“增加 EEG token”及“采用自回归生成”本身都已有直接近邻，不能合并成一个新颖性结论。AVDE 已将预训练 EEG 编码器与多尺度自回归图像生成连接起来；本轮应把它作为可复核的骨架和对照。[AVDE 原论文](https://arxiv.org/html/2602.22555v1)

RPA 提醒我们认真比较视觉监督：其强投影基线并不依赖更大的 EEG 编码器；所报告 LaBraM 检索值也低于其投影基线。其图像重建另有语义、结构先验与 img2img 阶段，因此检索数字不是重建正确率，也不能直接与本轮单图协议排名。[RPA 原论文及附录 F、I](https://arxiv.org/html/2609.31698v1)

NVOL 已采用中间视觉层和分层生成；其 CSLS 检索还使用候选图库和 EEG 查询间的相似度结构。这不能作为 EEG-only 自由生成的正常输入。中间层、分层及“神经可见性”均不应重新包装成空白方向。[NVOL 原论文](https://arxiv.org/html/2609.02582v1)

## 当前 EEG memory 实际是什么

现有真实输入是 `63×1×200`：每个电极只有一个覆盖约一秒的 patch。编码器的卷积会处理该段波形，但这不是已验证的多时窗 EEG 层级，也不能据图像的 coarse-to-fine 次序宣布对应生物视觉加工时间。

令编码器池化前 token 为 t_j，现有末端归一化为 N，已有输出头为 W∈R^{1024×200}、b。当前接口为

\[
h_0=N(\operatorname{mean}_j t_j),\quad h_j=N(t_j),\quad c=Wh_0+b,
\]

\[
H_j=c+W(h_j-\bar h),\qquad\bar h=\operatorname{mean}_j h_j.
\]

因此 H 的均值为 c，但局部差异位于同一输出头的至多 200 维列空间中。它们不是 1024 个相互独立的新测量通道。全局训练目标只监督池化输出；逐 token 应用同一个 N/W 并未自动赋予每个电极输出可靠的视觉语义。真实训练后的局部范数、共同分量、留出增量仍须检查。

这还给出一个可另行验证的工程等价式。若后续希望减少约 5.12 倍 memory 存储，可只缓存 δh_j=h_j−h̄，并复用已有投影 A：

\[
A H_j+b_A=Ac+b_A+AW\delta h_j.
\]

它不增加可学习参数。浮点运算结合顺序会变化，不能未经 logits／像素回放就宣布逐位等价；已登记的 materialized-memory 导出不因此改变。

## 为什么“中心化 prefix”仍会改变原生路径

对同一 image query q，把原生图像 K/V 与 EEG K/V 直接拼接，注意力严格分解为

\[
O_{cat}=(1-\lambda_q)O_{img}+\lambda_q O_E,
\qquad\lambda_q=\frac{Z_E(q)}{Z_{img}(q)+Z_E(q)}.
\]

即使所有 H_j=c，新增的重复 K/V 仍会改变 softmax 分母；一般 O_cat≠O_img。输入 memory 均值相同，不等于生成器原响应不变。因此原 prefix v1 中“重复 c、相同 token 数”的对照必不可少。这一注意力混合问题及 prefix 的表达限制已有研究，本报告的分解不是原创定理。[统一的参数高效迁移视角](https://arxiv.org/abs/2110.04366)、[ICLR 2024 prefix 能力与限制](https://proceedings.iclr.cc/paper_files/paper/2024/hash/18a367688479c1b08b001584218a4443-Abstract-Conference.html)

另一个容易无效的控制是仅在单条 query 内打乱 EEG memory token 的排列。当前 decoder 不给这些 memory 添加新的位置编码；它将其作为集合读取，理论上对 token 轴置换不变。CPU 实测也符合这一点。必须交换不同 query 的局部内容，不能把无效的顺序置换当成破坏 EEG 信息的控制。编码器自身的电极位置已经进入各 token 向量，这是另一件事。

## 更明确的改动：只注入相对同一 c 的注意力创新

新原型保留原生图像注意力自己的归一化，另计算 EEG 响应相对同全局条件锚点的差。令 z_j=A H_j+b_A，z_0=Ac+b_A；T_c 是当前原生 block 的条件 LayerNorm／affine，设

\[
u_j=T_c(z_j),\quad u_0=T_c(z_0),\quad
\Delta v_j=W_V(u_j-u_0).
\]

若当前原生注意力没有 key L2 归一化，取 Δk_j=W_K(u_j−u_0)；若启用了原生 per-head key 归一化，则取

\[
\Delta k_j=\operatorname{normalize}_{head}(W_Ku_j+b_K)
-\operatorname{normalize}_{head}(W_Ku_0+b_K).
\]

使用原生 query 的实际 bias、per-head 归一化／倍率和 attention scale s，定义

\[
R_E(q)=\sum_j\operatorname{softmax}_j(sq^\mathsf T\Delta k_j)\Delta v_j.
\]

公共 key 平移在 softmax 中消去，公共 value 在响应相减中消去，因此这是同 Q 下的 EEG 注意力减去同 c 锚点响应。新增路径为

\[
O_{new}=O_{img}+W_O R_E(q).
\]

原有输出投影的 bias 只在 O_img 中出现一次。该总响应继续经过原生条件残差门、DropPath 和 FFN。所有 W、归一化、门与 FFN 都是原模型已经存在的参数，没有新增可学习层或 α 扫描。

H_j=c 时 Δk、Δv 严格归零。实现另显式旁路，使 logits、原生采样随机流及最终像素回到原路径。各层的 EEG K/V 只由静态 EEG memory 和 c 生成；图像可以读取 EEG，EEG K/V 不读取真实或生成图像，原生 KV cache 只保存图像历史。

这不是已校准后验、最优采样公式或 EEG 生理不确定性。它回答的是一个更窄的架构问题：能否在保留全局路径的情况下，把 query 可用的局部 EEG 差异单独交给生成器？差分注意力、并行／解耦注意力均有已有研究；相减操作本身不能支撑新颖性。[Differential Transformer](https://arxiv.org/abs/2410.05258)、[视觉参数高效迁移中的并行注意力](https://arxiv.org/abs/2210.00788)

## 小扰动分析及需要避免的解释

在固定 q 下，一般有

\[
R_E(q)=\overline{\Delta v}+s\,\operatorname{Cov}(\Delta v,\Delta k)q+
\text{higher-order terms}.
\]

原始 H−c 的均值为零，经过条件 LayerNorm 后 Δv 的均值却不必为零。对平滑的 T_c，在常量 memory 附近，上述均值项和协方差项通常都是原始中心化扰动的二阶项。因此不能只称它为“协方差注入”；也不能由非零响应、二阶公式或梯度直接宣布恢复了图像细节。

这里的通道 memory 差异不等于重复试次的测量噪声。D26 的有限重复非线性偏差修正、D27 的重复差分频谱、D31 的删一决策敏感性及 D39 的测量约束，均不能改名后成为本方法的支持证据。新路径不重估同一 R4 的独立信息量，不把 patch 范数作为已校准 SNR。

## 已完成的实现检查

新增文件：[锚定注意力原型](../EEG_anchored_attention_20261029_v1.py)、[CPU 检查入口](../check_EEG_anchored_attention_CPU_20261029_v1.py)。原 prefix v1 没有修改。实现采用原 block 的功能组合，没有替换源文件、安装 kernel hook 或修改原生 cache 语义。

[22 项 CPU 检查](../audit/EEG_anchored_attention_20261029_v1/CPU_contract_actual_v1.json)在作者真实 VAR/VQ 模块的两 block、宽度 64 随机模型上通过。包括：

- 关闭及 H=c 的 logits／像素精确回退，参数对象集合不变；
- 固定 c 时局部差异可见，未来图像输入不能影响较早尺度；
- 原生 cache 长度为 1、5、14 个图像 token，未混入 EEG token；成功及异常都释放 cache；
- cached 与完整 teacher-forward 在同一生成历史上的 logits 最大差约 4.66×10⁻⁹；
- 独立 float64 的完整 EEG 响应减锚点复算，在普通 K 和原生 per-head L2 K/Q 下最大误差约 3.25×10⁻⁹／5.19×10⁻⁹；
- 同一次条件 dropout 同时移除 global/local，梯度到达局部 memory 和原 QKV。

测试模型有 226,976 个原 decoder 参数，新增学习参数 0；为检查可见信息通路，临时随机注意力门初始化为 0.1。没有加载真实预训练生成器、使用真实 EEG 或生成研究效果图。本轮 CPU 契约不构成实际全尺寸 GPU profile。

当前版本要求 attention/projection dropout 为零，且不使用 fused residual layout；原生 FFN、残差门和采样仍复用。创新分支会增加 query 投影及 EEG attention 的计算，不能以“零新增参数”声称零成本。

## 公开 VAR 到 AVDE 的加载边界

公开 FoundationVision VAR 使用离散类别的 `class_emb`；当前 AVDE VAREEG 使用 1024 维 EEG 的 `embed_proj`，二者不是同一个张量或语义接口。[公开 VAR 作者源码](https://raw.githubusercontent.com/FoundationVision/VAR/main/models/var.py)与本地锁定的 [AVDE VAREEG](../reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69/models/var_eeg.py)可以直接核对。

实际权重准备完成后，必须逐项检查真实 checkpoint：明确列出剔除的 `class_emb.weight`、重新初始化的 `embed_proj.weight/bias`，其余所有预期张量及 buffer 的键、形状、dtype 和数值均须核验。这里只指出源代码中的预期差异，尚未替代真实权重检查。不能宽泛 `strict=False` 后称全部加载成功，不能把类别 embedding 行当成 EEG 维度，也不能使用查询类别选择原生类别条件。

## 建议的下一项有限实验

先完成真实全尺寸 GPU 契约。随后若另行登记训练，应冻结同一 EEG 编码器及同一真实 OOF bank，比较原生 global、锚定 own-memory、训练时同概念 donor-memory 三臂；三臂原参数数量、数据、初始化、最终轮及优化预算相同。新增通路会改变函数族，参数数量相同不等于容量相同，因此 own／donor 配对机制控制仍不可省略。正常推理只用自身 EEG。锚定 H=c 与原生 global 在函数上相同，计算成本可单独 profile，不需要为同一函数额外训练一个重复基线。原 prefix v1 已有四臂设计不能据此静默改动。

固定 c 下的 post-fit local-memory 干预可以补充检查路径是否使用配对信息；同概念 donor 并不是真正从 p(H|c) 抽样的条件随机化检验，因此结果也不能直接称为条件互信息或因果效应。

在当前尺度历史 S_k=R_{<k} 下，有限模型的平均对数似然差为

\[
\mathbb E[\log q_1-\log q_0]
=I(R_k;H\mid S_k,c)
+\mathbb E\,\mathrm{KL}(p(R_k\mid S_k,c)\|q_0)
-\mathbb E\,\mathrm{KL}(p(R_k\mid S_k,c,H)\|q_1).
\]

因此正差还可能来自模型逼近差异。若保存每尺度平均 log probability，联合序列差必须按各尺度 token 数加权求和；十尺度等权平均只是另一个诊断定义。不能在 Dcheck 看完后挑正向尺度作为新方法。

是否继续最终由固定面板的真实 EEG-only 图片、同概念刺激区分和匹配控制决定。若无图片增量，停止该实例；若有支持，再扩多被试并完成同权限的完整官方 R80／公开指标。RPA 等强视觉监督、原生 AVDE、HIVE+ordinary 和本轮单图成本都需分列，不能以检索、loss 或随机模型工程成功代替 SOTA。
