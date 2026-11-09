# D17：跨脑软监督的原生编码器训练与真实图片结果

**2026-10-27实际完成，停止本次固定实例，不扩400。** 原始EEG神经教师通过了可区分性门槛，但没有转成相对强对照的图片增量。教师、30个编码器的完整训练、240张图片及评价，GPU主体共320.85秒（5分21秒）；三个控制器共375.21秒（6分15秒）。首次控制器到最后运行终态1123.44秒（18分43秒），包含阶段间实现；首次设计与最后CPU/文档交付另计。

## 1. 与旧实验有什么实质区别

D16改的是冻结c之后的内部响应修正，未胜强对照。这轮把干预前移到原生BrainMLP的训练监督：其他被试的原始EEG仅在训练时提供同概念刺激混淆关系，推理只使用当前query的17×250 R4 EEG。原BrainMLP、A/P/IP及四步Turbo架构不变，没有新增推理头。

复用D04十份完整T物理mean4输入、概念折、PCA64/MCCA16原实现，十个HIVE full-T的25轮最终权重、固定F缓存、D09的240张强对照图及400目标DINO特征。本轮另对**全部132300条T的17×250数值**与原63通道R4缓存逐值核验一致，并导出8200条Dfit与400条Dcheck小型EEG缓存，避免后续每次重读十份大原始文件。该CPU准备实际20.40秒。[完整输入回放](../audit/crossbrain_supervision_20261027_v1/actual/candidate/inputs/completion.json)

D04此前检验sub-01→教师A、由独立B评价；D05修改原始EEG，D06修改latent轨迹。D17把相同基础神经坐标用于**所有十被试的训练专用软监督**，不是再次宣称MCCA或跨被试对齐本身是新算法。视觉监督的神经可见性、适应性及跨被试目标均有直接近邻，不能凭这份组合声称论文创新。[NVOL](https://arxiv.org/abs/2609.02582)、[适应性视觉监督](https://arxiv.org/abs/2609.24109)、[跨被试视觉目标](https://arxiv.org/abs/2609.36971)

## 2. 教师、控制和固定训练

对每个当前被试，排除其全部EEG；其余九人按固定规则分成4人A、5人B。两个概念折中，PCA、MCCA、白化及A→B Procrustes全部只拟合另一折概念。对本折每个完整十刺激概念，分别去除A/B的概念均值、归一化，计算对称跨组余弦矩阵，再按固定温度0.1行softmax，形成十选一软分布。它是训练目标，**不是校准后的神经后验**。

全部20个被试×折的十图Top1为52.69%–55.91%，正确减同概念行错排余弦为0.31488–0.34031；每项都通过事前Top1≥15%、余弦差≥0.01的前置门槛。GPU主体13.70秒，没有F、图像、Dfit/Dcheck EEG或UNet加载。每个query的教师都排除了当前被试及当前概念；这不排除全部共享刺激/采集伪迹，也不是正常推理可以读取其他人的EEG。[教师冻结协议](../configs/crossbrain_supervision_20261027_v1/teacher_spec.json)、[实际全部读数](../audit/crossbrain_supervision_20261027_v1/actual/candidate/teacher/completion.json)

每个被试的三个模型从同一个原HIVE full-T检查点、同一个logit scale开始，保持原类和forward，6,452,224参数及训练用一个logit scale。原25轮后另固定10遍完整T，每模型140更新、132300次该被试query暴露；每臂合计十被试1400更新、1323000次query暴露。100个完整概念组一批，即1000 query，末批230；三臂使用相同组顺序及同seed重置的dropout随机流。

| 训练臂 | 固定目标 |
|---|---|
| visual-only | 对称batch InfoNCE |
| crossbrain-soft | 同一InfoNCE＋0.1×KL(神经教师分布‖学生同概念F分布) |
| shuffled-crossbrain-soft | 同一InfoNCE＋0.1×KL，仅在同概念内固定错排教师行 |

错排保留列质量和行熵的集合，只改变对应关系。学生同概念分布温度0.1，概念分组仅在训练/评价使用。视觉目标为冻结确定性原生F缓存，与原父编码器的随机dropout教师不同，三臂完全匹配此变化。AdamW lr=1e−4、betas=(0.9,0.995)、wd=0.05，LayerNorm/bias不衰减，clip=1、140步cosine；BF16模型前向、FP32 loss，seed1729。GPU拟合主体60.03秒；固定最后第十遍，不按Dfit选epoch、模型或强度。[拟合前协议](../configs/crossbrain_supervision_20261027_v1/fit_spec.json)

## 3. 全部8200条Dfit与运行日志

以下是原生1024维F坐标的编码器读数，**不是D15标准化F线性预测的0.94量级，也不是图片分数**。

| 全8200条Dfit | 原生F MSE↓ | 同概念十图log loss↓ | 十图Top1↑ |
|---|---:|---:|---:|
| 原25轮父编码器 | 1.757140 | 1.927848 | 38.1707% |
| 视觉微调 | 1.763840 | 1.936013 | 37.8780% |
| 正确跨脑监督 | 1.763814 | 1.936195 | 37.9024% |
| 错排跨脑监督 | 1.764096 | 1.938288 | 37.7683% |

正确跨脑相对视觉微调的MSE差仅−0.0000253，描述性95%区间跨零；同概念log loss反而高0.0001822，区间[0.0000515,0.0003049]。相对原25轮父编码器，MSE高0.006675、log loss高0.008347，两个区间都全正。相对错排教师有部分特征预测优势，不能替代相对纯视觉和强生成对照的图片增量。[全部32800条逐query读数](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/fit_CPU_audit_v2/Dfit_per_query.csv.gz)

![完整T和全8200条Dfit曲线](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/fit_CPU_audit_v2/actual_T_and_Dfit_curves.png)

三臂训练InfoNCE从约2.572降至2.279；但Dfit同概念损失从原1.92785升至1.93601/1.93620/1.93829。这是三臂共同的训练/开发背离，不能简单归为正确神经教师本身不含信息，也没有据此选较早epoch或调大/调小辅助权重。正确监督的新c相对原生平均改变约8.55%，范数约32.011，没有条件数值爆炸的迹象；这些读数不证明唯一失败原因。

## 4. 真实图片，按原规则停止

全部400×3最终条件在第一张新图片前锁定。子集为原D09 Dcheck400图库的前80条、sub-01、8完整概念×10刺激；每query每臂一张512×512、四步、CFG0/IP1，与原生/ordinary/risk逐query共用初始及四次祖先噪声。新增三臂240张及两张原生路径回放，968次UNet；三组强参考240张和目标特征全部复用。原生生成器及先验权重冻结。[出图前协议](../configs/crossbrain_supervision_20261027_v1/image_spec.json)

| 80条实际图片，同一400目标图库 | DINO2AFC↑ | 目标余弦↑ | 同概念十图Top1↑ |
|---|---:|---:|---:|
| 原生，复用 | 0.823183 | 0.271002 | 17.50% |
| ordinary，复用 | 0.858083 | 0.274514 | 23.75% |
| risk，复用 | 0.852381 | 0.277069 | 23.75% |
| 视觉微调 | 0.819079 | 0.272359 | 22.50% |
| 正确跨脑监督 | 0.825783 | 0.273929 | 23.75% |
| 错排跨脑监督 | 0.827412 | 0.272775 | 25.00% |

正确跨脑−risk为**−0.026598，[−0.039697,−0.013401]**，8/8概念均负，触发事前均值≤−0.02且区间上界<0的明确损害停止。相对ordinary为−0.032299，区间全负；相对视觉微调+0.006704但区间跨零，相对错排监督−0.001629且区间跨零。局部同概念Top1相对原生增加不能抵消强对照下的失败。**不扩400，不追加epoch/学习率/教师权重扫描，不挑三臂最大值作为赢家。**[完整配对读数](../audit/crossbrain_supervision_20261027_v1/actual/candidate/stage80/completion.json)、[全部480条图片评分](../audit/crossbrain_supervision_20261027_v1/actual/candidate/stage80/per_query.csv.gz)

![每概念固定首刺激，全部六种方法](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/images80_CPU_audit/fixed_first_stimulus_8concepts.png)

该图没有按效果挑样本，不是全量对象准确率或感知质量非劣评价。DINO2AFC不是对象正确率；单被试、八概念、单训练seed及反复开发曝光的区间仅为描述性，不构成独立确认或SOTA。

## 5. 完整核验、成本与下一问题

全部132300个教师软目标CPU重算，最大概率差3.58e−7；20项组成员和概念排除关系、40个固定query的原始EEG教师前向回放通过。全部24600条Dfit、1200条Dcheck的新编码器前向在CPU完整重放，最大相对L2=1.35e−6；不是小样本代替全量。8200条目标值/身份、30个优化器每参数140步和100个完整训练遍历核验。全部240图的条件、身份、初始/祖先噪声、字节/像素、512 RGB及480条评分/全部bootstrap闭合；两张原生回放字节与像素一致，242张文件/像素均唯一，恒色研究图0。[教师CPU复核](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/teacher_CPU_audit_v2/summary.json)、[完整编码器CPU复核](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/fit_CPU_audit_v2/summary.json)、[图片CPU复核](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/images80_CPU_audit/summary.json)

两个后验CPU分析初稿失败均保留：一处比较了GPU/CPU不同归约产生的FP32 Top1均值，最大差5.96e−8，20项整数命中数全同；v2改核验整数命中。另一处CSV假定旧Dfit schema有query_id，实际没有；v2导出全部实际身份字段，并重跑整个分析。未改任何模型、目标、图片或科学门槛，没有GPU重试；原stderr由SSH观察、未单独落完整原日志，不能伪称原审计一次通过。[实际纠正范围](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/postfit_CPU_audit_corrections.json)

| 实际范围 | 秒 |
|---|---:|
| 完整物理输入CPU回放及新缓存 | 20.40 |
| 全部十被试/两概念折神经教师 | 13.70 |
| 30个原生编码器完整10遍训练与Dfit | 60.03 |
| 240新图＋两回放＋DINO/评分 | 247.11 |
| GPU主体合计 | 320.85 |
| 三次GPU包装合计 | 327.23 |
| 三个控制器合计，含源校验 | 375.21 |
| 首次控制器至最后运行终态，含阶段间实现 | 1123.44 |

三个GPU作业均正常退出；峰值额外CPU8.412GiB、GPU allocated8.176GiB，10/20GiB上限不变。最终74个科学源SHA不变，十个旧父模型和缓存原SHA不变。所有新旧模型、优化器、输入银行及负结果保留在持久盘；大数组和数GB检查点仅同步SHA/shape/身份，不搬回本机。D与本轮全部PID退出，GPU及锁释放。[完整运行闭合](../audit/crossbrain_supervision_20261027_v1/actual/server_audit/actual_scientific_closure.json)

下一问题改为**保持旧编码器、从原生c出发学习确定性状态传输**：D12的独立高斯起点影响曾约为换EEG的15.8倍，D17又显示继续InfoNCE微调损害开发泛化。拟检验只接收当前状态与时间的全局速度场，从实际c传输至F，而非从独立高斯重新生成条件或再次更新EEG编码器。此类分布传输本身已有[Rectified Flow](https://arxiv.org/abs/2209.03003)先例；下一设计仍须真实图片与错配耦合控制。

这是未实现的下一假设，不是新效果。连续ODE的可逆性、有限步数值回放、语义配对和图片质量必须分开；可逆也可能压缩可用距离，同一视觉目标的多被试重复不能被一一映射精确合并后又声称完全无信息损失。先核验实际坐标、数值往返与刺激差异，不将数学性质直接当作防塌缩或论文创新。持续目标仍未完成。
