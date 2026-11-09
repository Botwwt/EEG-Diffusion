# D20：真实原生误差配对训练 native IP 权重，80 条图片停止

**固定实例停止，不扩 400。** 三臂真实训练和 240 张新图已完成；正确误差配对相对事前固定 ordinary 的 DINO2AFC 为 **−0.0200188，描述性 95% 区间 [−0.0297664,−0.0098340]**，触发原明确损害规则。原生误差监督没有胜过错误负例控制。最终 CPU 复核闭合于 **2026-10-27 19:13:11，America/Los_Angeles**；图片 GPU 终态为 18:02:35，二者不是同一时间。[真实闭合回执](../audit/error_pair_lora_20261027_v1/final_actual/server/scientific_closure_v2.json)

## 实际改动与信息权限

D19 的图像域打分优化失败后，本轮不再优化生成终点，而是训练原生 IP 注意力 K/V 的 rank2 权重增量。固定原 BrainMLP、最早 ordinary 修正检查点（SHA a66869e5…9e11cc03，强度 1）、原 A/P 和其余 UNet 权重。70 个 IP 层共 140 个 K/V 投影，每臂新增 906240 参数，增量 Frobenius 范数不超过原投影的 5%。原权重指纹不变。

复用 D19 的 sub-01 T400、40 个概念的真实图像／原生 parent 图片 VAE 银行和 Dfit200；**不是全部十被试 T**。每臂固定完整 T400 一遍、batch2 对、200 更新、400 次主 query 暴露，seed1729、AdamW lr1e−4、betas(.9,.99)、wd.01、clip1。固定最终头，不根据 Dfit 挑 epoch。既有 ordinary 本来训练于完整十被试 OOF-T，三臂之间计算匹配，不声称与旧强参考的训练成本相等。

正图为真实刺激，负图为相同 query 的旧 parent 生成图；它们是研究配对，不是逐张人工确认偏好的标签。普通修正生成器为冻结训练参考。令 e± 为学生噪声 MSE，r± 为冻结参考噪声 MSE，Δ±=e±−r±，目标为

$$L=\mathbb E e_+ +0.1\operatorname{MSE}(\epsilon_{student},\epsilon_{reference})
 +\mathbb E\operatorname{softplus}(100(\Delta_+-\Delta_-)).$$

positive-only 将最后项乘零，但仍计算同样两个分支。sameconcept-wrong-error 只把负图替为同概念下一刺激的旧 parent 图片，正图、EEG 条件、噪声与时间保持一致。三臂初始化、完整覆盖、顺序和随机流匹配。[固定配方](../configs/error_pair_lora_20261027_v1/fit_spec.json)

该目标与 [Diffusion-DPO](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html) 有直接近邻，本轮不是原创偏好学习定理，也不是 Turbo 精确似然或后验。改变了注意力权重这一事实不能代替方法创新或图片增量。

## 完整训练及开发日志

| 固定臂 | 全 T400 更新／暴露 | 训练主体秒 | Dfit200 正图 MSE，初始→最终 | 最终偏好代理 |
|---|---:|---:|---:|---:|
| paired-error | 200／400 | 326.52 | 0.177800→0.170001 | 0.486864 |
| positive-only | 200／400 | 315.07 | 0.177800→0.160701 | 0.456629 |
| sameconcept-wrong-error | 200／400 | 312.98 | 0.177800→0.174518 | 0.599601 |

全部最终 MSE 低于初始，偏好代理初始约 0.693147；positive-only 的偏好数值仅为完整报告的只读代理。它们没有转化为强参考图片收益，不能只凭这些读数宣布有效，也不能未经分析统一归为后期过拟合。[全部拟合和 420 项权重界 CPU 复核](../audit/error_pair_lora_20261027_v1/final_actual/server/fit_CPU_audit/summary.json)

## 同一 400 目标图库的真实 80 条图片

| 方法 | DINO2AFC | 目标余弦 | 同概念 Top1 |
|---|---:|---:|---:|
| parent，直接复用 | 0.823183 | 0.271002 | 17.50% |
| ordinary，直接复用 | 0.858083 | 0.274514 | 23.75% |
| risk，直接复用 | 0.852381 | 0.277069 | 23.75% |
| paired-error | 0.838064 | 0.268760 | 23.75% |
| positive-only | 0.831015 | 0.277323 | 25.00% |
| sameconcept-wrong-error | 0.855890 | 0.273723 | 21.25% |

正确配对−错误负例控制的 DINO 差为 −0.0178258，区间 [−0.0355929,−0.0019048]；正确配对−positive-only 为 +0.0070489，区间跨零。正确配对虽高于 parent，仍未胜强参考，未建立配对机制。三臂全报告，不从错误负例组挑赢家晋级。[完整图片和配对评分](../audit/error_pair_lora_20261027_v1/final_actual/candidate/stage80/completion.json)

每 query 每臂一张，原生 512×512、4 次 UNet、CFG0、IP1，固定相同初始及四次实际祖先噪声。LoRA 运算有额外成本，同 UNet 次数不等于计算完全相同。复用旧三组 240 张，新三组 240 张；成功 profile 两张、正式图片阶段一张回放，首次失败 profile 另外两张回放也保留，没有 best-of-N。

![固定八概念首刺激，全部六种方法](../audit/error_pair_lora_20261027_v1/final_actual/server/images80_CPU_audit_v2/fixed_first_stimulus_8concepts.png)

展示按原登记概念首刺激固定取样，不按效果筛选；不用于估算对象正确率或感知非劣。所有区间均为已曝光、单被试、八概念的描述性区间，不是独立确认。

## 失败、修复及实际成本

首次工程 profile 在 positive-only 的 t999 出现一个 K 投影梯度为零，严格断言停止；此前仅做四次临时工程更新，正式训练更新为 0。精度回放证实 FP32 主权重先转 FP16 再进入 BF16 autocast 会产生该下溢；v2 仅在训练中让 FP32 主权重直接进入 BF16 autocast，原 FP16 推理路径保留。同一真实批次的总体梯度范数变化约 0.3%，修正后 3 臂×4 时刻的 140 个投影均有非零梯度，原权重及零增量像素一致。[修正原始记录](../audit/error_pair_lora_20261027_v1/final_actual/server/explicit_precision_correction_v2.json)

工程 profile 装入 [0,10,20,30] 四个 T400 位置，但优化实际只使用前两个 [0,10]；后两个未参与更新，不写成四个完整训练 query。

原 CPU 图像审计在所有身份／图片／噪声／评分重算后，制作展示图时误读 D19 的 Dcheck `target` 字段。D19 原本未导出该字段。v2 使用原 D09 已冻结目标元数据，重新执行全部检查，**没有重训、重出图或更改科学门槛**。240 张图均通过 512 RGB、文件／像素 SHA、query、条件、token、模型和噪声复核，480 条六臂分数及全部 bootstrap 精确重算，恒色图 0。原失败保留，不将原控制器写为全自动成功。[完整 v2 CPU 复核](../audit/error_pair_lora_20261027_v1/final_actual/server/images80_CPU_audit_v2/summary.json)

三臂训练主体合计 **954.57 秒**，全部六次 GPU 包装（含失败 profile）合计 **1893.10 秒，约 31 分 33 秒**；第一次 GPU 包装至图片终态 **2328 秒，38 分 48 秒**，含阶段间实现与交接。最终展示审计修复和后续理论推导到 19:13:11 才科学闭合，不能把整轮称为 5–10 分钟。拟合／图像包装和全部失败均保留在[实际作业回执](../audit/error_pair_lora_20261027_v1/final_actual/server/scientific_closure_v2.json)。

全部 D20 PID 与正式 D 原 PID 已退出，GPU 无进程，独占锁实际可获取。停止本实例，不自动扩 400、加 epoch 或扫描 rank/lr。用户提出进一步从底层逻辑推导，下一问题及已执行的纯 CPU 原型见[条件创新推导](BRAINHIVE_CONDITIONAL_INNOVATION_DERIVATION_20261027_v1.md)；它尚不是新的已启动 GPU 队列。
