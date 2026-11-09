# 研究路线与训练目标更换：EEG 可预测视觉目标、原生先验与粗尺度边际似然

2026-10-29，America/Los_Angeles。用户最新要求是改变研究路线和训练目标，不继续死磕旧配方。本文记录实际撤下、更换依据、实现及两卡有限队列；没有新方法图片收益或 SOTA 结论。

## 已经撤下的内容

旧局部 EEG 锚定 AR 主臂在 **7425/8270 更新、118784 次 query 暴露**处整步保存停止；已完成八整轮，第九轮位置12944。独立 CPU 核验覆盖221个模型张量、203组 Adam、顺序及 CPU/CUDA/噪声 RNG。保留检查点 SHA `c3cec79f9349f44b69b69ce133adc8c54e30866c1dfe3c17526893da2e1df2f8`。后续 donor 臂未启动。

原图片、损害判读等待器和本地跨机接续器已显式退役，旧 handoff v4 仅作历史，不能据其自动恢复。中止主臂和未启动 donor 都是 **UNMEASURED_NOT_NEGATIVE**，不能补填科学负结果。原生固定10轮模型已完成，及其80图、后续诊断证据均保留。最早 ordinary 的权重、实现和强度1保留；旧正式 HIVE D 仍停止。

实际撤下证据：[旧卡](../audit/crossdisciplinary_20261028_v1/old_task_AR_user_route_withdrawal_old_v1_20261029T120932150511Z.json)、[新卡](../audit/crossdisciplinary_20261028_v1/D37_single_AR_user_route_withdrawal_new_v1_20261029T120919399249Z.json)。

## 为什么改变目标

完整200条 Dfit 的旧原生固定末轮诊断已执行及独立复算。实际采样前缀下，正确 EEG 相对同概念错配在最前两个尺度的平均 log-likelihood 增量为 **+1.184148、+0.459591 nats/token**；后面多数尺度反向。按680个 token 加权后的增量为 **−0.278725**，描述性区间全负。正确 EEG 相对零条件的加权增量为 **−1.901950**；最后四个尺度200/200 query 均未胜零条件。

实际采样前缀相对正确图像前缀的正确条件 NLL 恶化约 **+7.382965 nats/token**。CFG 在本诊断的未截断概率下继续恶化；不能据此宣称关闭 CFG 或 EEG 会改善实际图片。诊断使用冻结的一个旧模型，不代表 EEG 的信息极限，也不证明所有 AR 路线无效。正确图像前缀本身能解释目标，因此正确前缀下 EEG 增量消失也不能单独证明条件塌缩。

此前完整第六轮训练 CE 主臂/原生为4.507054/4.434751，梯度范数比中位数0.971049；没有证据把失败简单归于梯度更大。新的解释是一个**待检验的目标失配假设**：让低信噪比 EEG 预测所有细尺度图像内容，同时更新整套生成先验，可能不利于保留可泛化的视觉条件。它需要匹配对照和真实图片验证。

完整原输出和 CPU 复核引用保存在本地 `audit/crossdisciplinary_20261028_v1/D37_single_new_route_native_ELBO_new_profile_actual_v1_20261029T122300531042Z.json` 的 `previous_actual_prefix_CPU` 和 `previous_actual_prefix_summary`，远端完整数组、200张实际采样图片及两张回放均保留于 `AR_Dfit200_actual_prefix_20261029_v1`。这200张是冻结旧基线诊断图，不是新方法效果。

## 新的统一问题与推导

令 $E$ 为真实63电极 R4 EEG，$Y_1,\ldots,Y_{10}$ 为原生 VQ 多尺度离散 token，$z$ 为生成先验的原生1024维条件。保留公开生成器全部已有权重，移除旧 EEG 接入时引入的随机1024→1024映射，恢复原生条件坐标与公开 class embedding 所定义的尺度。

冻结的归一化因果分解满足

\[
p_G(Y\mid z)=\prod_{k=1}^{10}p_G(Y_k\mid Y_{<k},z),\qquad
\sum_{Y_{3:10}}p_G(Y\mid z)=\prod_{k=1}^{2}p_G(Y_k\mid Y_{<k},z).
\]

所以只训练前两个完整尺度的 **1+4=5 个 token**，有精确的边际概率含义。这一尺度选择在新拟合和图片之前锁定，受到旧模型诊断启发，但不是 EEG 生理带宽定理，当前不搜索尺度数。

真实 EEG 编码器输出对角高斯 $p_\theta(z\mid E)$，每个允许 T 图像有一个仅训练时存在的高斯 $q_i(z)$。目标为

\[
\mathcal L_i=\frac15\left\{
\mathbb E_{q_i}\!\left[-\sum_{k=1}^{2}\log p_G(Y_{ik}\mid Y_{i,<k},z)\right]
+D_{KL}\!\left(q_i(z)\Vert p_\theta(z\mid E_i)\right)\right\}.
\]

这是相应 EEG 条件粗尺度负对数似然的变分上界。NLL 和 KL 都是每 query 的 nats，**一起除以5**，没有另调 beta。原型先在全部680 token 上验证同一 ELBO 工程接口，后另存 v2 改成此精确边际目标；不能把 v1 工程检查写成 v2 正式训练完成。

若允许任意分布，固定 EEG 模型时的理想更新为

\[
q_i^*(z)\propto p_G(Y_{i,1:2}\mid z)\,p_\theta(z\mid E_i).
\]

它同时要求视觉解释能力与 EEG 可预测性。实际实现限制为对角高斯，用重参数梯度近似优化，并没有计算出精确后验。固定教师分布时，任意条件对角高斯的总体 KL 投影满足逐坐标矩条件

\[
m(E)=\mathbb E[\mu_i\mid E],\quad
v(E)=\mathbb E[\sigma_i^2+(\mu_i-m(E))^2\mid E].
\]

这里是理想总体最优条件，不是本轮神经网络的闭式已求解答案。其意义是让 EEG 同时学习可预测方向和残余变化；是否校准、能否恢复对象，仍须留出检验。

推理时只读 EEG，由 $p_\theta(z\mid E)$ 抽一个条件，再由冻结原生先验生成全部十个尺度。训练图像的 $q_i$、真实 token、概念和目标图都不进入正常推理。不选最好 latent、图片或生成 seed。

## 两卡匹配实验与固定预算

| 项目 | 旧卡主臂 | 新卡匹配对照 |
|---|---|---|
| 教师 E 步的 KL 参考 | 当前 EEG 条件分布 | 固定原生条件边际高斯 |
| EEG 的 M 步 | KL(q_image ‖ p_EEG) | 完全相同 |
| EEG 初始化、结构、T 数据、顺序、预算 | 相同 | 相同 |
| 冻结 VAR / VQ | 全部保留 | 全部保留 |

两者都训练同一个真实 EEG 主干和高斯输出头（约623万参数），没有在 ordinary 后加新头。图像变分表只在 T 内训练存在。该对照检验“视觉目标接受 EEG 可预测性约束”是否提供增量，不能只凭有了概率输出就宣布机制成立。

固定 sub-01 完整 T：13230刺激、1323概念、真实63电极 R4，full-T 归一化。既有 full-T EEG 初始化已见过 T，这不是全链 OOF 或全十被试完成。seed1729，10遍，batch64，每遍207个 EEG 更新，共2070更新／132300次主 query 暴露；每个 query 每遍4个教师更新，共40次。教师 Adam lr0.03，EEG AdamW lr1e−4、wd0.01；只用固定末轮。匹配臂同序、同初始化。

先做64条真实数据工程检查，通过才自动进入完整 T。每个包装900秒、主体800秒、20GiB GPU /10GiB额外CPU、保留2GiB盘缓冲。每整轮保存完整模型、两种优化器、顺序和 RNG；最多三个有限分段，不因运行失败无限重试。断点源文件和科学配方按 SHA 固定。

已实际启动控制器：旧卡 **PID287281**，新卡 **PID71366**。运行回执见 `audit/crossdisciplinary_20261028_v1/native_marginal_{old,new}_actual_launch_v2_*.json`；这表示有限队列真实启动，不能替代后续拟合完成回执。

## 已通过与尚待完成

v1 全 token 接口：13项 CPU 数学/接口检查；旧卡和新卡各完成真实公开权重、真实 EEG 的 E/M 梯度工程检查，全部暂时更新重置，原生 identity 前向和 null 条件像素回放通过。旧卡主体30.426秒，新卡修复后32.877秒。新卡第一次缺 SciPy 私有搜索路径的失败保留，只修环境，没有改科学输入和损失。

v2 精确边际与逐图优化器：8项 CPU 检查通过，包括完整原生前向与截短因果前向的 logits/梯度一致、后续真 token 不影响粗尺度、未访问图像参数不随其他图的动量漂移、恢复一致。初版 Adam 偏置修正幂使用 float32，与独立 Torch Adam 三步相差约6.11e−7；改成 float64 标量幂后通过，原失败另存。真实64条资源检查、全 T 训练和实际图片须各自读真实完成回执。

后续固定同一 Dcheck400 的80→400图片规则；80只停止明确损害，其余扩大。主臂需相对匹配视觉教师和无 EEG 边际生成对照的 DINO增量≥0.01、配对描述性区间下界>0，同概念Top1不降，才考虑多被试与完整指标。初期比较仍是已曝光开发集与单训练seed，不等于独立确认或官方 SOTA。旧原生 AR 的 CFG/条件接入不同，不能不匹配协议就直接排名。

条件变分学习与逐图 latent 优化已有明确近邻：[Sohn 等，NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)、[Bojanowski 等，ICML 2018](https://proceedings.mlr.press/v80/bojanowski18a.html)。本轮不声称这些公式原创。论文贡献候选必须来自 EEG 特定目标耦合的必要性、匹配机制增量和真实图片收益，当前均未建立。

实际代码：[边际与逐图优化器](../EEG_native_marginal_core_20261029_v2.py)、[训练](../train_EEG_native_marginal_20261029_v2.py)、[有限队列与独立CPU复核](../queue_EEG_native_marginal_20261029_v2.py)、[CPU检查](../audit/EEG_native_conditional_variational_20261029_v1/marginal_CPU_actual_v2.json)。
