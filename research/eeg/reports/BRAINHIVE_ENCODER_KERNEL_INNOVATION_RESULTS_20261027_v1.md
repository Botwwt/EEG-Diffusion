# D22：从编码器不可见方向到条件生成解，实际增量仍未通过

2026-10-27，America/Los_Angeles。**共同零空间的结构事实通过了真实模型核验；其正确配对视觉预测优于随机投影和错排控制，但仍差于原 c-only，因此当前解析实例停止，不进入 Dcheck 或图片。**这不是已成立的架构贡献。正式 D 保持停止。

## 1. 先确定后处理无法修复什么

[D21](BRAINHIVE_CONDITIONAL_INNOVATION_DIAGNOSTIC_RESULTS_20261027_v1.md)实际表明：直接使用全部原始 EEG 的条件高斯更新，MSE 恶化 4.50%；更新方向有弱正对齐，能量却过大。下一项没有从已曝光 Dfit 挑一个新缩放系数，而是先问哪些 EEG 输入差异在现有条件通路中根本不可见。

实际 BrainMLP 为 c=f(We+a)，W∈R^(1024×4250)。若 Wv=0，则

$$c(e+v)=f(We+Wv+a)=c(e),\qquad h(c(e+v))=h(c(e)).$$

第二个等式对任何只接收 c 的后处理 h 成立。它定位了改变信息通路的必要位置，但不证明 v 对应生理上可能的 EEG，也不证明图像与 v 有关。

对每个被试的 full/oof0/oof1 三个真实 W，堆叠为 \(\mathcal W\in\mathbb R^{3072\times4250}\)。秩上界给出 \(\dim\ker\mathcal W\ge1178\)。完整 QR 分解 \(\mathcal W^\top=QR\) 的末 1178 列 U 满足

$$\mathcal WU=0,\qquad U^\top U=I_{1178},\qquad b_e=U^\top e.$$

1178 来自固定结构的维数保证，未依据留出图片扫描。U 只保证是共同核的一个子空间；没有证明它等于整个共同核，也没有证明 (c,b_e) 能完整恢复 e。

## 2. 如何由同一个问题连到生成状态

在声明的联合高斯工作模型下，先条件于已有 c，记

$$u\mid c\sim\mathcal N(m_c,S_c),\quad
\begin{bmatrix}u\\b_e\end{bmatrix}\Bigm|c
\sim\mathcal N\left(\begin{bmatrix}m_c\\\mu_b\end{bmatrix},
\begin{bmatrix}S_c&C_b\\C_b^\top&R_b\end{bmatrix}\right).$$

这里 u 是标准化后的真实 VAE PC，而不是原生 F 或 c。令 \(H=C_b^\top S_c^{-1}\)、\(R_\eta=R_b-C_b^\top S_c^{-1}C_b\)。图像后验均值由同一个二次问题给出：

$$\arg\min_u\frac12\|u-m_c\|_{S_c^{-1}}^2+
\frac12\|b_e-\mu_b-H(u-m_c)\|_{R_\eta^{-1}}^2.$$

配方或求正规方程得到

$$\boxed{m_{cb}=m_c+C_bR_b^{-1}(b_e-\mu_b)},\qquad
\boxed{S_{cb}=S_c-C_bR_b^{-1}C_b^\top}.$$

再加入同坐标中的生成状态 \(z=\alpha u+\eta_z\)、\(\operatorname{Cov}(\eta_z)=D\)，得到

$$S_{cbz}=\left(S_{cb}^{-1}+\alpha^2D^{-1}\right)^{-1},\qquad
m_{cbz}=S_{cbz}\left(S_{cb}^{-1}m_{cb}+\alpha D^{-1}z\right).$$

计算中以 Cholesky 解线性方程，不显式求逆。标准化 PC 的 \(D=\sigma^2\operatorname{diag}(1/\mathrm{PC\_std}^2)\)。因此编码器遗漏坐标 b_e、图像方向 C_b、剩余方差 S_cb 和状态更新来自同一个模型；若条件交叉协方差为零，则回到 c-only。

**以上是经典高斯求解，不是新定理。**其最优性要求模型及矩统计正确，不保证真实 EEG 或 HIVE 轨迹满足，更不保证对象图片质量。当前没有将它接进 Turbo 并声称精确后验。

## 3. 实际固定队列与权限

第一段真实运行严格加载全部 30 个原模型，在 FP64 核验完整 U 的残差及正交性，共同核相对残差最大 1.42×10⁻¹⁵。FP32 对每模型 16 条真实输入加入固定相对范数 0.1 的核内扰动，原生／旧 ordinary 输出相对变化最大约 1.38×10⁻⁶。另一次独立 CPU 使用新扰动完成 480 次挑战、每模型固定 32 列核检验和 800 组投影行检查。CPU 没有重做完整 QR。

全十被试全部 T 和 820 条 Dfit 的 1178 维坐标已导出，并建立相同维数、固定 seed 的随机正交输入投影。共同核占中心化 T EEG 能量 31.37%–65.76%，不等于视觉信息或生理 SNR。结构核验 GPU 包装／主体为39.154／36.987秒，独立CPU主体24.181秒；两个控制器全段分别57.546／43.069秒，首次设计及传输另计。

第二段实际固定队列为：每被试完整 13230 条 T，各拟合正确共同核、随机投影、同概念错排共同核三个解析模型，共 30 个；再读全部 8200 条 Dfit×4 时刻×5 臂，最后独立 CPU 复核。共用 D21 的 128 维 T-VAE 基底、48.9456% 方差范围、c、latent、状态噪声及固定 0.1 协方差收缩，没有重新选择 PC、强度、时刻或最佳 epoch。c-only 的评分与 D21 完成数值对齐。本次 z 是允许的真实 Dfit latent 的前向加噪状态，图像仅用于诊断监督；没有将其伪装成正常 EEG 推理可读取的输入，也没有验证实际 HIVE 生成轨迹上的适用性。

共同核使用了完整 T 上训练的 full 权重；ordinary 修正头也曾在全部 T 上训练。因此仅原生条件来源是 OOF，整个核基底或修正流程不是完全 OOF。Dfit 已曝光，本次不是独立确认。没有读取 Dcheck/V/官方效果或新增外部图像。

## 4. 全量结果：结构存在不等于有益更新

| 固定臂 | NLL／标准化 PC 维 | 原生 PC MSE |
|---|---:|---:|
| c-only | 0.602761 | 7.903725 |
| 正确共同核 | 0.605691 | 7.951354 |
| 同维随机投影 | 0.609269 | 8.020295 |
| 同概念错排共同核训练 | 0.608603 | 8.027487 |
| 正确模型、推理共同核错配 | 0.608691 | 8.026248 |

正确共同核相对 c-only 的 MSE **恶化 0.6026%**，10/10 被试均恶化。MSE reduction −0.047629，描述性 subject/concept 95% 区间 [−0.062035, −0.035565]；NLL improvement −0.002930，区间 [−0.003942, −0.002118]。虽小于 D21 的损害，仍不满足准入门槛。

正确共同核优于随机投影，MSE reduction +0.068941，区间 [0.055645, 0.082207]；相对错排训练为 +0.076133，区间 [0.058251, 0.097918]。这支持该表示在本工作模型中不同于同维随机表示；**不能改判它带来了相对原 c 的有效增量，更不能写成重建图片更好**。

保持 `STOP_KERNEL_GAUSSIAN_INSTANCE_NO_IMAGES_OR_DCHECK`。没有把“正确胜错排”替换为主门槛，也没有据此搜索缩放、rank 或新的 shrinkage。

## 5. 复核与解释上限

首次队列完成全部 GPU 拟合、预测与冻结读数后，CPU 审计在第一被试末批因读取越过 13230 边界而触发形状断言。原源码、失败日志、控制器记录均保留。v2 只把 CPU 切片上界限制在本被试范围，模型、GPU 结果、科学配方及判读没有修改；只另跑 CPU 复核。

v2 CPU 已复算全部132300条 T ordinary 条件、164000组128维预测／协方差／评分和固定判读；最大预测、评分绝对误差分别约1.42×10⁻¹⁴、5.69×10⁻¹⁴。全部30个模型的128维图像条件正规方程通过，EEG正规方程和条件增益只挑战每模型预定16列，不能写成CPU重新拟合全部1178列。

D22拟合GPU包装25.420秒、主体23.388秒；v2 CPU主体36.807秒。计入首次CPU失败的两个控制器全段合计149.414秒；**首次控制器至CPU最终闭合574.056秒（9分34秒）**，含修复和阶段间传输，首次设计及最终文档另计。不能把初次CPU失败写成完整队列一次通过。全部本轮进程已退出，GPU及独占锁在闭合观察时释放。没有新增 UNet 或研究图片，没有正式 D 恢复，没有图片增益、独立确认或 SOTA 结论。

零空间网络已有[直接文献近邻](https://arxiv.org/abs/1806.06137)，高斯条件解也已有大量先例。当前只建立了真实架构的可验证约束和一次失败的条件增量实例。后续若继续，必须改变本次实际失败的建模环节，并以相同强基线检验；不能将这个固定投影换个名字再作为新贡献。

证据：[结构与 CPU 核验](../audit/encoder_kernel_20261027_v1/actual/server/scientific_closure.json)、[D22 科学闭合](../audit/encoder_kernel_innovation_20261027_v1/actual/server/scientific_closure.json)、[全部 164000 条评分](../audit/encoder_kernel_innovation_20261027_v1/actual/server/all164000_Dfit_scores.csv.gz)、[v2 CPU 审计](../audit/encoder_kernel_innovation_20261027_v1/actual/server/CPU_audit_v2/summary.json)、[固定配方](../configs/encoder_kernel_innovation_20261027_v1/candidate_spec.json)。
