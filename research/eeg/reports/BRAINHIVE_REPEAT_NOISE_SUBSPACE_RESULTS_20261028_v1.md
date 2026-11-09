# D23：原生第一层的重复度量约束，完整400条结果

2026-10-28 02:10:07，America/Los_Angeles，完整图片和独立CPU复核结束。D23相对原生有积极开发信号，但没有稳定超过最早的ordinary，且同概念刺激Top-1下降；固定判定为 **STOP_INSTANCE**。不扩大至多被试或全V，不据此追加步长、epoch或强度扫描。正式D仍停止。

## 实际改动和训练范围

本轮直接改变已有BrainMLP的4250→1024第一层权重，保留原bias、残差MLP、输出LayerNorm和logit_scale。训练后把参数折回原`proj_in.weight`，每模型仍为6,452,224参数，没有新增推理参数。生成继续使用同一个旧ordinary检查点、强度1和原生A/P/IP/Turbo四步。

由角色内四次物理重复估计正则化协方差代理N。它包含重复内生理变化，不是已知的生理噪声真值。对原权重W0，约束问题与参数化为：

$$\min_W L_{\mathrm{InfoNCE}}(f(We+b),F(I)),\qquad WNW^\top=W_0NW_0^\top,$$

$$N=LL^\top,\quad W_0NW_0^\top=SS^\top,\quad W=S Q L^{-1},\quad QQ^\top=I.$$

使用切梯度$G_T=G-\operatorname{sym}(GQ^\top)Q$、固定步$\Delta=-0.032G_T/\|G_T\|_F$和Cholesky回缩。这个闭式步只优化局部线性目标，不保证实际图片改善。完整推导、协方差收缩、信息权限和既有文献近邻见[原协议](BRAINHIVE_REPEAT_NOISE_SUBSPACE_PROTOCOL_20261028_v1.md)。Stiefel优化和重复性滤波不作为原创定理。

sub-01的full/oof0/oof1各训练三臂：正确重复度量约束、坐标固定置换的同谱度量约束、正确度量但不约束。9模型各完整允许T十遍，共810更新、793800次主query暴露；训练循环19.493秒，拟合程序全段38.270秒。这里是单被试完整T，**不是全十被试T**。OOF噪声估计与拟合只用本角色FIT概念；旧ordinary本身使用全部T，因此不把整个系统称为完全OOF。

正常推理仅用当前query的物理mean4。Dfit820只作最终固定读数，没有选择epoch或强度；400条条件在新图之前全部锁定。各臂均单seed、单图、相同初始与四步ancestral噪声、同一400目标图库。普通修正仍是最早的同一检查点，SHA `a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03`，没有重训ordinary。

## 完整400条图片结果

40概念、每概念10刺激、sub-01。三臂共1200张新研究图，三组旧强参考共1200张直接复用。所有方法均在同一400目标图库评分。

| 方法 | DINO2AFC | 目标余弦 | 图库Top-1 | 同概念10选1 |
|---|---:|---:|---:|---:|
| 原生parent | 0.769455 | 0.253861 | 5.25% | 21.75% |
| 旧ordinary | 0.793885 | 0.261653 | 6.50% | 27.75% |
| 旧risk | 0.790476 | 0.260688 | 5.75% | 24.50% |
| 正确重复度量约束 | **0.797362** | 0.262440 | 6.00% | 23.50% |
| 置换度量约束 | 0.774267 | 0.247885 | 2.75% | 20.50% |
| 正确度量、不约束 | 0.796454 | 0.262577 | 6.00% | 24.25% |

| 主臂减参考 | DINO均值差 | 描述性概念重采样95%区间 |
|---|---:|---|
| 原生parent | +0.027907 | [0.006636, 0.048673] |
| 旧ordinary，事前主参考 | +0.003477 | [−0.005046, 0.011078] |
| 旧risk | +0.006886 | [−0.005085, 0.021178] |
| 置换度量约束 | +0.023095 | [−0.000529, 0.046730] |
| 正确度量、不约束 | +0.000909 | [−0.005835, 0.007908] |

相对ordinary的同概念Top-1差为**−4.25个百分点**，区间[−7.75, −1.24375]个百分点。主臂未达到事前+0.01和区间下界大于零的强参考门槛，也没有证明约束机制胜过不约束。不能把“所有组合相对原生提升”归因于新增第一层约束：旧ordinary本身已有收益。

80条时主臂0.849969，ordinary0.858083，相对ordinary−0.008114、区间跨零，按既有“只提前停止明确损害”规则扩400。完整400使用同一图库但包含更多query，不能把80/400的绝对均值下降当成训练退化。完整结果固定停止，没有挑控制臂或最优子集改判。见[全读数](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/stage400_recovered_v2/readout.json)和[全部2400行评分](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/stage400_recovered_v2/per_query.csv)。

## 从实际日志学到的内容

full主臂逐epoch平均InfoNCE从2.777948降到2.677272，不约束从2.778650降到2.669057；两OOF角色同样下降。Dfit最终主臂/不约束的ordinary条件MSE为1.592149/1.591785，同概念F Top-1均39.024%。二者非常接近，不支持仅靠该约束获得额外刺激信息的解释。只有最终Dfit读数，没有逐epochDfit曲线，不能断言后期过拟合或挑较早epoch补救。见[完整33条训练事件](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/fit/training_log.jsonl)与[只读日志汇总](../audit/repeat_noise_subspace_20261027_v1/postclosure_log_readonly_summary_v1.json)。

正确约束的full噪声Gram相对误差为2.61e−8，不约束为2.59%；主臂确实实现了所声明的代数约束。但图片差仅+0.000909、区间跨零，**约束实现正确与约束产生收益是两件事**。旧共同输入核的full响应相对值由约1.16e−15变为0.04265，说明已有权重确实重新选择了此前不可见的方向；这个核仅用于训练后的探针，未参与拟合。它不能证明新见方向有可泛化的视觉信息。[九模型权重与度量证明](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/fit/fit_proofs.json)。

下一问题应同时保护刺激间可辨几何与生成宿主坐标的一致性；仅保持第一层的重复噪声Gram不足以约束后续非线性语义。当前没有据此拟合下一模型，也没有在已曝光的400条上调参。后续若改变内部坐标，必须推导相邻权重如何一起变换，并核对正常推理等价性和强参考增量。

## 时间保护、补缺与独立复核

首次400扩展在有限时间检查处停止，保留1159/1200行，未完成DINO扩展特征。原失败回执和旧源码全部保留；新执行版本先逐条验证1159行的身份、图片SHA、条件和噪声，再仅补41张。919张已生成扩展图和原80条240张均未重出。随后编码全部960张扩展图，原80条特征逐位保留，最后由独立CPU程序核对全部1200图及2400行六方法评分、重采样与判定。[原失败](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/stage400/failure.json)、[补缺完成](../audit/repeat_noise_subspace_20261027_v1/actual/candidate/stage400_recovered_v2/completion.json)、[独立图片复核](../audit/repeat_noise_subspace_20261027_v1/actual/server/images400_CPU_audit_v2/summary.json)。

另有全部43350条native/ordinary条件CPU前向、9组第一层折叠权重及其余原生权重逐位核验。协方差CPU独立复核覆盖16个完整输入列和32个hidden Gram行，不能称为全部4250列/1024行CPU重算。[拟合CPU核验](../audit/repeat_noise_subspace_20261027_v1/actual/server/fit_CPU_audit/summary.json)。图片复核重新计算存储DINO特征对应的分数，不是CPU重新运行DINO视觉编码器。

五次GPU包装含失败共**1240.794秒**；训练主体仅19.493秒，不能用它代表整轮。首次profile控制器至最终CPU阶段闭合**7598.561秒，2小时6分39秒**，含连接、阶段间实现和工程补缺，首次设计及最终文档另计。全部拥有的进程退出，闭合观察时GPU空闲、GPU锁可用；未恢复旧D。[科学闭合与全部作业耗时](../audit/repeat_noise_subspace_20261027_v1/actual/server/scientific_closure.json)。

结果仍是单被试、单训练seed、已曝光开发面板的描述性证据。没有多被试全V确认、独立机制增量或SOTA结论。固定原图展示为前8概念各首刺激、全部六方法，未按分数挑图：[原图对照](../audit/repeat_noise_subspace_20261027_v1/actual/server/images400_CPU_audit_v2/fixed_first_stimulus_8concepts.png)。
