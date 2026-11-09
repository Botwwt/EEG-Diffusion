# D37：EEG 时间贡献经冻结 ordinary 的导数进入原生 IP 注意力

## 完整400图片终态：当前实例停止

2026-10-28 16:11:15 America/Los_Angeles，固定 80→400 图片队列全部十三阶段 exit0；16:17:06 独立 CPU 闭合。主臂 DINO2AFC 为 **0.792769**，旧 ordinary 为 **0.793885**，配对差 **−0.001115**、描述性概念重采样 95% 区间 **[−0.008346, +0.005247]**。区间上界低于登记的 +0.01 实用门槛，按原规则记 `STOP_INSTANCE / DESCRIPTIVE_PRACTICAL_FUTILITY`。同概念 Top1 由 **27.75% 降至 25.00%**，差 −2.75 个百分点、区间 [−5.25, −0.50] 个百分点。没有新增有效方法、机制增量或 SOTA 结论。

这是 sub-01、物理 R4、40 概念各十刺激的同一 Dcheck400 开发面板；已多轮曝光，单生成 seed／query，不能当独立确认。三候选均使用同一旧 ordinary，新增训练／SGD／学习参数均 0；每图仍为原生 512×512、四步 Euler ancestral、CFG0、IP1、四次 UNet。额外方向导数、K/V 与注意力计算必须计入成本。正式旧 D 仍停止，第二卡及锁已释放；另一张卡上的 X01 跨框架 ordinary 原队列持续训练。

### 六臂完整结果与固定比较

| 分支 | DINO2AFC | 目标余弦 | 400 图库 Top1 | 同概念十选一 Top1 | 同概念 2AFC |
|---|---:|---:|---:|---:|---:|
| 原生 parent（复用） | 0.769455 | 0.253861 | 5.25% | 21.75% | 0.631944 |
| 旧 ordinary（复用） | 0.793885 | 0.261653 | 6.50% | 27.75% | 0.656389 |
| 旧 risk（复用） | 0.790476 | 0.260688 | 5.75% | 24.50% | 0.660278 |
| 正确 EEG 条件算子 | 0.792769 | 0.261463 | 6.50% | 25.00% | 0.654722 |
| 正确 EEG 展平残差控制 | 0.793327 | 0.261357 | 6.75% | 25.25% | 0.658333 |
| 同概念错配方向控制 | 0.800984 | 0.262982 | 6.25% | 24.00% | 0.660833 |

80 条时，三候选 DINO 为 0.845802／0.846147／0.861059，旧 ordinary 为 0.858083；主臂−ordinary 为 −0.012281、区间 [−0.031078, +0.003833]。未触发原“均值≤−0.02 且区间全负”的明确损害提前停止条件，因此全部三臂扩到 400；没有把 80 条判成收益。原 80 条图片行和特征字节完整保留，160／240／320 条中间点没有评分。

400 条主臂−展平控制的 DINO 为 −0.000558、区间 [−0.002864, +0.001442]；主臂−同概念错配为 −0.008214、区间 [−0.017049, +0.000746]，**两项均跨零**，不支持所声称的配对机制。错配控制的 DINO 均值虽更高，同概念 Top1 仍低于 ordinary；它还读取同概念另一 query 的 EEG 及控制标签，不能改选成正常推理赢家。全部臂和五组固定比较保留，不挑控制结果或重新扫描半径。

重采样固定 1000 次、seed1729，以完整概念及各十刺激为单位，图库保持原 400 张。区间是本开发面板的描述统计，不覆盖训练 seed、多轮候选或可选停止的不确定性。DINO2AFC 不等于对象正确比例；本轮未新算完整 SEED／ObjectF1，也没有质量非劣或跨被试结论。见[完整400评分](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen400/score/completion.json)、[全部逐query指标](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen400/score/all_per_query_metrics.json)、[80条原判定](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen80/score/completion.json)。

### 实际执行、失败与复核范围

事前锁定[图片科学方案](../configs/temporal_conditional_attention_20261028_v1/image_spec_v1.json)及[实际执行绑定](../configs/temporal_conditional_attention_20261028_v1/image_execution_v1.json)。所有 400 条原生／ordinary 条件在出图前匹配旧 D09 bank；真实 R4 前向相对旧缓存的最大差为 4.89e−6／9.54e−7。CPU 为 400×5 方向真实完成冻结 ordinary/A/P JVP，主体 6.347 秒、包装 8.520 秒；方向预算与原工程相同。相同时间方向经过真实七十层原生 K/V，不使用查询目标图构造新条件。

首次全400 CPU 准备因 ordinary 接口要求 γ／β 显式匹配 batch 长度，在 5.692 秒后停止。v2 只将同一 affine 展开到 batch 维，失败源码、终态和空输出目录保留；没有改 EEG、方向、半径、seed 或损失。另一次注册包装的 Path／字符串错误发生在准备调用前，亦保留。两次本地慢速中继均停止并保留部分文件，后改用已有直连源服务器路径；成功输入传输 6.235 秒不代表整轮研究时长。见[唯一 CPU 接口修正](../configs/temporal_conditional_attention_20261028_v1/CPU_preparation_repair_v2.json)。

新增 **1200 张研究图**（三臂各400）及五个新进程的 ordinary 关闭回放，共 1205 张、**4820 次实际 UNet**；之前五张 T 工程图另计。parent／ordinary／risk 的旧 1200 图和特征直接复用。五次回放均逐像素回到同一旧 ordinary 参考；每张研究图的 subject、stimulus、array_row、目标路径、条件与噪声 SHA、文件／RGB SHA 及完整分母已复核。宿主 UNet 权重摘要和旧 ordinary SHA 前后不变；实际特征阶段还重编码 old ordinary 锚点确认 DINO 坐标一致。

独立 CPU 另重算 **12000 个标量指标**及五组固定比较的 1000 次概念 bootstrap，最大标量误差 0；目标自配对 DINO2AFC=1。它是独立评分实现与文件／像素核验，使用同一 NumPy 后端和既有 DINO 特征，**没有独立重跑 UNet 或第二个图像编码器**。见[独立评分](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen400/score/independent_CPU_audit.json)、[全量图像审计](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen400/image_audit.json)、[最终闭合](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/image_queue_v1/independent_closure_CPU.json)、[78份实际原产物SHA](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/fetch_manifest.json)。

七个 GPU 作业包装合计 **1390.556 秒**；完整图片控制器从 15:47:23 至 16:11:15，**1431.821 秒（23分51.8秒）**。输入恢复、失败包装、设计实现和最终文档另计，不能称整轮 5–10 分钟。图像阶段最大额外 CPU 9001160704 字节、GPU allocated 8767719424 字节；含本臂硬链接的逻辑输出 878666965 字节、唯一 inode allocated 494456832 字节。独立终态确认 GPU 空闲且独占锁释放，原图和失败均保留。

### 从实际变化收窄下一问题

完整400图，条件算子相对 ordinary 的 DINO 归一化特征更新能量为 0.131313，目标误差方向的两倍对齐为 0.130934，平方误差净变 +0.000379；202／400 条目标余弦改善。条件／展平两条更新方向的平均余弦为 0.959732，但并非同一张图或完全同一特征。新增通路确实影响图片，不能用“分支没有生效”解释失败；这些事后特征分解也不是新的训练目标或成功门槛。见[固定400特征变化分解](../audit/temporal_conditional_attention_20261028_v1/image_actual_20261028T231927571633Z/candidate/screen400/feature_delta_analysis.json)。

下文的局部展开显示零均值时间方向首先通过交叉二阶矩影响响应。另有一个**精确性质**：同时置换五个已计算 K/V 方向的列表顺序，条件与展平算子的输出均不变；FP64 合成核验误差分别 3.33e−16／2.22e−16。因此这个注意力读取器将已计算贡献作为集合，没有单独建模贡献之间的先后方向。原 W_t 仍包含原始时间坐标，**不能由此声称翻转原始 EEG 不变或全部时序信息丢失**。见[列表置换核验](../audit/crossdisciplinary_20261028_v1/D37_time_permutation_actual_CPU_v1.json)和[代码](../check_BrainHIVE_temporal_permutation_CPU_20261028_v1.py)。

下一问题可以收窄为“在已经保留原生总条件后，真实 EEG 的有向跨窗关系是否有条件增量”，先检验可识别性和匹配控制，再决定架构。这里没有据此启动新模型、采样强度或新图片；不能从经典注意力、局部秩或排列性质直接宣称下一机制有效。

---

以下公式和 15:12 工程快照保留其原始含义；上方完整图片结果已更新所有旧“尚未出图”。

这是一个与旧 ordinary 分开开关的架构候选。旧 ordinary 的权重、强度 1、半径 0.25 和原实现不动；本候选另读同一 query 的物理 R4 EEG，经宿主已有权重构造五个时间贡献。主候选新增学习参数为 0，不改变采样算法或每图四次 UNet 调用；增加 CPU 方向导数及每层注意力、K/V 投影计算。当前的工程验证不能证明图片收益或方法原创性。

## 从第一层压缩问题到可计算的旁路

记物理四次平均 EEG 为 x∈R^(17×250)，原生第一层和后续网络为

$$h=Wx+b,\qquad c=f(h).$$

Wv=0 时，原生 c 对 v 不可见；这只是一项结构事实，D22 的高斯旁路已经失败，不能由零空间存在推定 EEG 还有可恢复的视觉信息。本轮不重跑该回归，而把已有 W 按五个固定时间窗拆开：每窗 50 点，边界为 0/50/100/150/200/250，在 250 Hz 下各 200 ms。它们不是已识别的生理阶段，也不表示刺激事件已去卷积。

$$h_t=W_t x_t,\qquad r_t=h_t-\frac15\sum_s h_s,\qquad d_t=J_f(h)r_t.$$

这里的 f 使用真实原生残差 GELU 网络和输出 LayerNorm affine。实现按 GELU 与 LayerNorm 的精确导数求 JVP，不将新特征放进任意学习坐标。由于 Σr_t=0，亦有 Σd_t=0。若各窗 W_t v_t 抵消，总 c 可以不变，而这组时间贡献可以改变；这是可见性检查，不是生理有效性或图片增量证据。

用冻结链路 Φ(c)=P(A(ordinary(c;γ,β))) 继续传播：

$$T_k=\Phi(c)_k,\qquad D_{k,t}=J_\Phi(c)d_t.$$

每个原生 token 内先对时间去均值，再将五个方向的 RMS 范数限制在 0.25||T_k|| 内。这是执行前固定的局部信任预算，不按 Dfit、Dcheck 或图片分数选强度，不保证线性化准确或图像质量。

通过已有的每层 K/V 权重得到 δK_(k,t)、δV_(k,t)。没有另训时间投影、添加可学习 token 或改变旧 ordinary。代码中保留的 656128 参数 `TemporalReadout` 只是早期合成检查草稿；在首次服务器运行前已退出主方案，实际 CPU/GPU 主候选均不实例化它。

## 保持原生四 token 边际的闭式解

对当前层、当前状态下的真实 query q，原生四 token 注意力为

$$a_k=\operatorname{softmax}_k(qK_k^\top/\sqrt d),\quad
s_{k,t}=q\delta K_{k,t}^\top/\sqrt d.$$

如果直接把每个 token 的五个方向展平归一化，残差通路的四 token 边际会变成 a_k Z_k / Σ_j a_j Z_j，其中 Z_k=Σ_t exp(s_(k,t))。它同时改变了时间读取和原生 token 分配。

本轮把“在每个原生 token 内读取时间证据”写为带固定边际的经典 KL 投影：

$$\min_{\pi\ge0}\sum_{k,t}\pi_{k,t}\log\frac{\pi_{k,t}}{a_k/5}
-\sum_{k,t}\pi_{k,t}s_{k,t},\qquad
\sum_t\pi_{k,t}=a_k.$$

对 π 求导并消去每个 k 的拉格朗日乘子，得到

$$\pi_{k,t}=a_k\frac{\exp(s_{k,t})}{\sum_u\exp(s_{k,u})},\qquad
\delta O=\sum_{k,t}\pi_{k,t}\delta V_{k,t}.$$

原处理器先完整计算原生输出，再将 δO 经同一冻结输出线性层的无 bias 部分加入，避免重复计算原生 bias。零 δV 和显式关闭都应精确回到原图。固定边际只在同一当前 q 下成立；上游新增响应会改变后续 q，不能声称整个反向轨迹或输出图像保持不变。

匹配控制 `flat_residual` 保留同一原生输出和同一时间残差，只把残差通路的边际换为 a_k Z_k / Σ_j a_j Z_j。两种算子都在零值修正时严格回退，不使用本来就会扰动基线的普通 token 拼接作为唯一控制。

## 已有近邻与本轮实际问题

[IP-Adapter 原论文](https://arxiv.org/abs/2308.06721)已有解耦图像交叉注意力，[Hierarchical Attention Networks](https://aclanthology.org/N16-1174/)早已有层级注意力；JVP 和带边际的 KL 投影亦是经典工具。仅添加 EEG token、用层级 softmax 或写出这个解，不能作为原创性结论。[EEG-RegionalDiff 的出版摘要与章节预览](https://www.sciencedirect.com/science/article/abs/pii/S003132032501252X)还已有 EEG 表征、视觉监督和区域语义扩散适配的近邻；本轮只查阅其摘要／章节预览和作者仓库入口，未完成全源码或完整协议审计。

本候选真正待检验的是：现成 EEG 编码器丢掉的“时间贡献如何相互抵消”，经已训练生成链路的局部导数传输后，能否在保留原生 token 边际的条件下提供图片增量，以及该增量是否超过展平归一化与错配时间证据控制。它不解卷积事件，不把 EEG 时延直接等同扩散时间，不保证与 ordinary 的效果相加。去掉 ordinary 后可定义 Φ=P∘A，因此结构上可分别开关；两个背景下的真实图片交互尚未检验。

## 已取得的 CPU 证据

本地／服务器十二项数学和接口检查通过。真实输入为 sub-01 完整 T 索引中均匀取出的 32 条，来自 full-T 父编码器，属于 FIT-only 工程输入；不是旧 oracle T32，不称概念 OOF。32×5 个原生输出方向与 FP64 自动求导逐项比较，最大误差 1.9984e−15；原生 FP32 前向最大差 7.1526e−7。160 个冻结 ordinary/A/P JVP 真实完成，全部 70 层、140 个原生 K/V 权重完成 CPU 运算；这些 Q 为合成 Q，不是 UNet 轨迹。

合成 W 核扰动使 c 的相对变化约 9.62e−16，时间方向相对变化约 0.102277，经冻结 ordinary/A/P 后约 0.091861。它只验证这条路径能区分被总体投影抵消的人工方向。最大 token RMS 相对预算为 0.25000003。CPU 主体 11.432 秒、包装 13.563 秒、额外 CPU 峰值 1772216320 字节，无 CUDA 初始化、训练更新或图片。

首次 CPU 工程调用漏传作者类加载器第二个源文件，3.512 秒后失败；v1 文件、错误和终态保留。v2 只改为显式传入 models_brain.py 与 layers.py，不改数学、数据或门槛。见[真实 CPU 原产物与 SHA](../audit/temporal_conditional_attention_20261028_v1/actual_20261028T220217144222Z/fetch_manifest.json)、[API 修正回执](../audit/crossdisciplinary_20261028_v1/D37_CPU_API_repair_v2.json)。

## UNet 工程状态（15:12历史快照）

首次 GPU 作业完成精确冻结加载后，旧采样观测器因只接受原生类名而拒绝新封装，包装 50.285 秒、exit1，未运行一次 UNet、未产出图片。原失败与源码保留。v2 增设显式观测器，仍要求每图四次 UNet、280 次原生 K 和 280 次原生 V；开启新通路另核验各 280 次时间 K/V 投影，四个 ancestral 噪声逐项与旧参考比对。原生处理器、scheduler、seed、冻结加载器及原采样源码均未修改。见[观测器修正范围](../audit/crossdisciplinary_20261028_v1/D37_GPU_observer_repair_v2.json)。

v2 已在第二卡真实完成五张固定 T 工程图：关闭、条件算子零值、展平算子零值、正确条件算子和正确展平算子。三条回退均逐像素匹配旧 ordinary，非零两臂最大 RGB 变化均为 208，后者只证明像素发生变化。全部 70 层、每图四次 UNet、原生 K/V 各 280 次，以及开启时新增 K/V 各 280 次已实际观测；每图初始噪声、四个 ancestral 噪声与原参考一致。宿主全部 UNet 权重前后摘要一致，旧 ordinary SHA 和强度 1 不变。主体 63.022 秒、包装 64.029 秒 exit0，额外 CPU 峰值 9338716160 字节，GPU allocated 峰值 8742685696 字节。见[真实 GPU 原产物与 SHA](../audit/temporal_conditional_attention_20261028_v1/GPU_actual_20261028T221033430913Z/fetch_manifest.json)。独立 CPU 解码五张原图、重算文件／RGB SHA、三路像素差、20 个实际噪声返回及调用记录，0.408 秒通过；它没有独立重算 UNet 数值响应。首次 CPU 复核误用了 `pixel_sha256` 字段，失败保留；v2 只按实际 `generated_pixel_sha256` 字段重读，没有重跑 GPU、图片或模型。见[独立 CPU 核验及 GPU／锁释放](../audit/temporal_conditional_attention_20261028_v1/independent_GPU_engineering_CPU_v2.json)。尚无 Dcheck 图片、效果评分、机制增量或 SOTA 结论。跨框架 [X01 ordinary](ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_EXECUTION_20261028_v1.md)继续使用另一张卡，不受本轮作业影响；正式旧 D 保持停止。

实现：[时间 JVP 与闭式注意力](../brainhive_temporal_conditional_attention_20261028_v1.py)、[原生 IP 接入与观测器 v2](../brainhive_temporal_IP_binding_20261028_v2.py)、[真实 CPU 工程](../profile_BrainHIVE_temporal_conditional_attention_CPU_20261028_v2.py)、[真实 GPU 工程](../profile_BrainHIVE_temporal_IP_binding_GPU_20261028_v2.py)。

本轮两个 GPU 包装合计 114.314 秒，包含首次未进入 UNet 的失败。首次真实 CPU 启动至独立像素核验终态 1124.315 秒，包含修复、实现与交接；首次设计和最终文档另计，不以 63 秒代替整轮开发时间。
上述“后续／尚未”属于 15:12 的工程历史快照。随后另行登记固定 Dcheck400 EEG、同概念错配时间方向及展平控制的真实图片比较，强度不由 T 工程图挑选。最新科学结果在后续实际更新中给出。

## 图像运行中的算子阶数分析

给定同一个当前 Q 和原生四 token 权重 a，令 λ 同时缩放新增 K/V 方向。λ 只用于以下合成算子检验，不是图片配方中的强度候选；实际图片继续使用锁定的半径 0.25，没有 λ 扫描。

$$R(\lambda)=\sum_k a_k\sum_t\operatorname{softmax}_t(\lambda s_{k,t})\,\lambda\delta V_{k,t}.$$

由于时间 K/V 方向均去均值，ΣδK=ΣδV=0，展开得

$$R'(0)=0,\qquad
R(\lambda)=\lambda^2\sum_k a_k H_kq+O(\lambda^3),\qquad
H_k=\frac1{5\sqrt d}\sum_t\delta V_{k,t}\delta K_{k,t}^{\mathsf T}.$$

所以新增路径在小扰动下首先读取 K/V 交叉二阶矩，而不是线性读出每个时间方向。每个 token 的 H 秩至多四，四 token 加权和的秩至多十六。原生 K/V 独立，H 一般不是对称正定矩阵，不能把它当作必然向正确视觉目标收缩的更新。

同样因为 Σs=0，展平控制的原生 token 边际改变从 λ² 开始；它再乘上本来从 λ² 开始的残差，故展平与条件残差的差从 λ⁴ 开始。这是固定 Q 的局部结论，不能直接外推为半径 0.25 下真实 UNet 或图片差异必然很小。

合成 FP64 自动求导检验的一阶导约 −3.33e−16，二阶系数误差 1.11e−15；λ 减半时两种残差之差的范数比为 15.927，符合四阶局部行为。该检查没有读取 EEG 值或图片得分，也没有新增图片候选。数学检查本地完成于 22:51:39 UTC；服务器解释记录于 22:52:39 UTC，此时 80 条评分文件已经存在，但记录程序不读取分数。它是运行中分析，不是新增事前成功门槛。见[数学检查](../audit/crossdisciplinary_20261028_v1/D37_attention_order_math_local_v1.json)和[检查源代码](../check_BrainHIVE_temporal_attention_order_CPU_20261028_v1.py)。

[现代 Hopfield 网络原论文](https://arxiv.org/pdf/2008.02217v1)已给出 softmax 注意力与连续状态联想记忆的联系。这里只核对其更新公式与对应假设；本轮独立 K/V、随 UNet 状态变化的 Q 和生成残差不满足直接套用该记忆更新收敛结论的条件。不能给经典注意力换一个记忆名称便宣布新颖性或生理机制。
