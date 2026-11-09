# D39：由 EEG 重复测量约束原生编码器的生成式再训练

> **21:15:31实际终态：四臂/十组完整400已闭合，停止当前实例。** 主候选+ordinary 0.781353对同轮ordinary 0.793490，差−0.012137、区间全负；原生背景+0.006429、区间跨零。全部4000图、20000标量及15组bootstrap CPU核验，另下载真实特征在本机复算，排名指标一致。没有组件新收益；D40按原队列接续，旧ordinary不变。见[完整结果与日志诊断](BRAINHIVE_MEASUREMENT_PRIOR_RESULTS_20261028_v1.md)。以下为此前执行快照。

> **2026-10-28 19:11更新**：19:11:46实际复查：unanchored/isotropic/measurement已各完成827更新，permuted_measurement为592／827，GPU36983运行。原训练29455与图片等待31949继续；尚无研究图片效果。新增D40等待器37447排在本轮完整十组图片/评价及31949退出之后，不修改本轮源码或科学规则。

## 18:12 实际更新：十组图片执行器已绑定，等待原四臂训练

2026-10-28 18:10:34 America/Los_Angeles，图片等待控制器 **31949／start_ticks1468422969** 实际启动；18:12:42核对同一进程在线，原训练控制器29455／start_ticks1468253113与GPU31574在线。**unanchored已完成827更新／13230暴露，isotropic为224／827；measurement及置换臂尚未开始。** 首臂13份自然断点已由原训练队列CPU审计，固定最终模型SHA为`bb039256…6ff5253c`。完整四臂尚未闭合，新研究图片仍0。

本轮补齐了此前缺少的条件导出、真实生成、特征提取、评分、独立CPU复算及有限编排。新[执行绑定](../configs/measurement_prior_20261028_v1/image_execution_v1.json)固定162项源SHA；原训练源、训练方案及17:42的图片科学规则原字节未改，ordinary与正式旧D状态未改。旧policy中的“执行器尚未绑定”保留为拟合前历史，由这个单独文件登记工程完成。

等待器核对原训练进程的PID、启动标识和命令，只有四臂成功闭合且原控制器退出后才能申请GPU；训练失败或同一进程消失而无成功闭合时停止，不重训、不自动重试，不重置原训练截止时间。图片控制器总有限上限39600秒，训练后的评价段最多18000秒；每GPU段900秒、20query×十臂，每CPU阶段600秒，评价2GiB及剩余8GiB保护。当前等待没有启动新GPU作业。

正常条件仅由真实单被试R4 EEG导出。五种编码器使用同一CPU批次后端，old ordinary分别开／关；完成后将独立NumPy64回放全部400×十组原生及ordinary条件，不用旧预测替代新导出。每个GPU生成段先回放原有T工程图并要求像素相同，再生成研究图，保存实际条件、token、初始与四次ancestral噪声、权重和像素SHA。**新执行器的这项GPU回放尚未运行，CPU工程通过不代表完整新GPU管线已实测。**

原卡400张目标JPEG逐个匹配原D09元数据SHA；原始目标元数据、D09输入完成及目标编码完成的来源链随包绑定。两文件17104145字节直接传到第二卡，逐文件SHA通过，实际1.671秒，无新增目标图片或模型更新。生成结束后另用冻结DINO重编码全部400张原目标，要求回放原目标缓存，随后才读图像指标；不会仅凭数组尺寸或自配对1宣布目标身份正确。

服务器11个固定门槛边界、合成十组4000标量、15个1000次概念bootstrap及零交互情形复算通过；另真实T32原编码器与一个合成参数变化模型的原生／ordinary条件回放通过，最大误差3.56e−6。CPU核验3.848秒，新拟合／GPU／图片均0。T32工程不是400条实际候选条件或图片效能，后者仍待执行。

所有十组先完成80条再评分；双背景明确损害才停止，否则保留全部80图和特征、十组一起扩400，不看160／240／320效果。固定报告8个“各学习臂−原编码器”的背景对比、6个测量机制控制和1个ordinary交互差分，共15项；两项主背景比较及400门槛不变。交互差分只描述组合行为，不能由区间跨零宣称统计独立，也不能用控制赢家替换主方法。

实现：[共享契约与判读](../brainhive_measurement_images_20261028_v1.py)、[EEG条件与原目标导出](../prepare_BrainHIVE_measurement_images_CPU_20261028_v1.py)、[原生出图及DINO](../generate_BrainHIVE_measurement_images_GPU_20261028_v1.py)、[固定评分](../score_BrainHIVE_measurement_images_CPU_20261028_v1.py)、[独立复算](../audit_BrainHIVE_measurement_images_CPU_20261028_v1.py)、[有限等待与编排](../queue_BrainHIVE_measurement_images_CPU_20261028_v1.py)。

证据：[服务器CPU检查](../audit/measurement_prior_20261028_v1/actual_20261029T011304966647Z/repository/audit/measurement_prior_20261028_v1/image_executor_CPU_checks_v1.json)、[真实等待启动](../audit/measurement_prior_20261028_v1/actual_20261029T011304966647Z/repository/audit/measurement_prior_20261028_v1/actual_image_queue_launch_v1.json)、[目标包传输](../audit/measurement_prior_20261028_v1/actual_20261029T011304966647Z/repository/audit/crossdisciplinary_20261028_v1/D39_targets_pull_actual_completion_v1.json)、[首臂固定最终模型回执](../audit/measurement_prior_20261028_v1/actual_20261029T011304966647Z/candidate/arms/unanchored/completion.json)、[等待进程与训练在线](../audit/crossdisciplinary_20261028_v1/bridge_actual_D39_bound_pipeline_actual_wait_and_training_20261029T011240410946Z.json)、[本次取回SHA](../audit/measurement_prior_20261028_v1/actual_20261029T011304966647Z/fetch_manifest.json)。所有新增效能结论仍未成立。

2026-10-28 17:42:17 America/Los_Angeles，第二张卡正式启动四臂完整 T 队列，控制器 **29455**。17:46:48 实际首臂 unanchored 为 **144／827 更新、2304 次暴露**，GPU 29461 在线；其首个 64 更新／1024 query 断点已独立 CPU 核验。**四臂尚未完成，没有新方法的研究图片或效果结论。** 原 ordinary 不参与这轮训练图，权重与强度不变，留到评价时独立开关。

## 日志怎样决定改动位置

[D38](BRAINHIVE_ORDERED_AREA_RESULTS_20261028_v1.md) 的加入时序交互读出在两折均降低 FIT 误差、提高 held 误差；它说明当前估计器存在泛化落差，不能据此认定 EEG 没有时序信息。[D23](BRAINHIVE_REPEAT_NOISE_SUBSPACE_RESULTS_20261028_v1.md) 重选第一层后虽胜原生，却未胜同一旧 ordinary，重复约束相对不约束也未成立。

本轮直接训练原 BrainMLP 的输入投影和残差块，让真实冻结生成器的损失回传到 EEG 编码器；用原 EEG 重复差异规定允许改变的功能方向。原生拓扑、输出 affine、A／P／IP／UNet 不变，不新增推理模块。待检验的是**真实 EEG 测量方向是否优于同强度或同谱的通用约束**，不能由反传可用或矩阵非零宣布创新成立。

## 从物理重复到功能变化的约束

对 sub-01 完整 T 的 13230 刺激、1323 概念，读取真实四次重复，令原编码器 pre-affine LN 输出为 u₀：

$$\bar x_j=\frac14\sum_{r=1}^4x_{jr},\qquad
d_{jr}=J_{u_0}(\bar x_j)(x_{jr}-\bar x_j)/\sqrt{4(4-1)},$$
$$N=\frac1{13230}\sum_{j,r}d_{jr}d_{jr}^{\top},\qquad
C=0.9\frac{N}{\operatorname{tr}(N)/1024}+0.1I,\qquad
Q=\frac{C^{-1}}{\operatorname{tr}(C^{-1})/1024}.$$

1／12 的均值方差因子依赖重复独立、可交换及局部线性近似；本轮没有证明这些生理假设。全 T 聚合的 N 不是逐 query 的生理 SNR，也不是校准过的真实后验。trace 归一化只保留方向比例，取消了绝对噪声尺度；tr(Q)／1024=1 匹配环境空间中的各向同性期望惩罚，不代表每个可行 LN 位移的惩罚相等。

由同一个全 T 原生父模型初始化四臂，记 δ=uθ−u₀，固定 τ=0.25：

$$L_\theta=L_{\epsilon}^{native}+0.1L_{token}^{native}
+\frac1{2\cdot1024\tau^2}\operatorname{mean}_{j}\delta_j^TQ_{arm}\delta_j.$$

| 臂 | Qarm | 检验内容 |
|---|---|---|
| unanchored | 0 | 直接再训练本身 |
| isotropic | I | 通用功能保持约束 |
| measurement | 真实 Q | EEG 重复测量方向 |
| permuted_measurement | ΠQΠᵀ | 相同特征值、固定随机坐标置换的方向控制 |

四臂共享原始权重、T 顺序、t／噪声流、优化器、训练遍数。预期是在测量较稳定的方向更少改变、在变化较大的方向允许更多改变；这只是工作假设，测量变化方向不必等于语义误差方向。τ 是软约束尺度，不能写成 ordinary 的严格半径。

## 闭式局部解及它的适用范围

将真实生成残差在当前点线性化为 r+Jδ。实际编码器只能实现 δ≈HΔθ，其中 H 为参数到 u 的局部 Jacobian。令 U 为 range(H) 的正交基，写 δ=Ua，则局部二次问题为

$$\min_a\|r+JUa\|^2+\frac{a^TU^TQUa}{2D\tau^2},$$
$$a^*=-\left(U^TJ^TJU+\frac{U^TQU}{2D\tau^2}\right)^{-1}U^TJ^Tr,\qquad\delta^*=Ua^*.$$

已有代码中的 U=I 是无约束功能坐标代理，不能称为真实 LN 网络的精确更新。参数形式的正规方程可能因参数对称性而奇异；不能直接声称存在普通逆。新增[可行子空间代数核验](../audit/crossdisciplinary_20261028_v1/D39_feasible_local_math_v1.json)使用秩亏参数映射，与增广最小二乘解最大差 2.98e−16。**实际训练使用原非线性编码器的 AdamW，没有构建巨大的 H／U，也没有精确执行该闭式解。** 该推导说明目标怎样联动测量方向与生成梯度，不是全局最优或图片收益证明。

## 原生训练预算与 ordinary 的关系

四臂各额外完整遍历 sub-01 全 T 一次，batch16、末批14，各827更新／13230暴露，合计3308更新／52920暴露。训练原有四个 core 张量，共 **6450176 参数**；原 LN affine 2048 参数冻结，正常推理仍只用一个 6452224 参数编码器。AdamW lr1e−5、wd0.01、默认 betas／eps，clip1，固定最后一次，不用 Dfit／Dcheck／V 选模型。编码器拟合时保持 eval 模式关闭 dropout；这不是原父模型 InfoNCE 配方的逐项复现。

训练链路是 **EEG→原生编码器→A→P→IP／UNet**；旧 ordinary 不在训练图中。旧 ordinary SHA `a66869e5…9e11cc03`、强度1保持不变。使用的是已有 full-T 父模型的额外生成式微调，**不是十被试全 T，也不是整条链路 OOF 或独立确认**。原正式 D 不恢复。

每64更新追加不可覆盖的模型／AdamW／CPU与CUDA和噪声 RNG／覆盖与顺序／日志前缀断点；每 GPU 作业最多256更新、900秒，控制器最多21600秒，单臂2GiB、GPU20GiB、额外CPU10GiB。准入预留训练8GiB、评价2GiB及8GiB余量；这些是资源边界，完整训练耗时尚未实测。

## 实际核验与保留的失败

全 T CPU 统计用时9.420秒、包装12.025秒；全部52920个 JVP 向量的协方差与精度矩阵独立重算通过，相对误差3.57e−16／5.02e−15。另16条原始 EEG×4 的 NumPy64 JVP 回放最大差1.19e−6；只对这个子集独立重跑原始前向，不能称全部52920条原始计算都独立重做。CPU审计本体5.760秒、包装6.434秒。

先完成 ordinary 串联工程，再将正式训练分支固定为独立原生路径；两轮各8个临时更新全部丢弃，各两张 T 工程图，共4张，研究效果图为0。独立分支 GPU 工程本体87.780秒、包装89.294秒，训练循环11.177秒；四个 core 张量都有真实 ε loss 梯度，两张初始模型回放图逐像素一致，固定生成器与旧 ordinary 未变。峰值额外CPU9226768384字节、GPU9744268800字节。另行 CPU 解码两图、核对实际噪声、八个更新记录通过；它不是独立 UNet 梯度复算。

v2 分发器因旧 stdout 以 xb 打开而在 GPU 子进程启动前退出；真实退出码未观测，不能补成0或1。已核验其 PID 消失、GPU receipt和输出目录均不存在，失败证据保留。v3只使用唯一日志名，同一原生分支源码／数据／目标／seed／预算完成工程核验。原生 T latent 867350000字节含原始回执直接传输，逐文件原 SHA 一致，实际52.767秒；没有重建或改动原 latent。

17:45:39 的首64更新 CPU 审计核验真实77435540字节断点 SHA `2daa4066…4ca9952`，四个 core 权重均改变、输出 affine 逐位不变，模型／AdamW／全部 RNG／顺序覆盖和提交日志前缀一致；抽样16条条件前向改变 RMS=0.598386。核验期间后续日志正常增长，未中断 GPU；没有独立重放64次 GPU 更新。

证据：[原统计完成](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/candidate/CPU_v1/completion.json)、[独立CPU](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/candidate/CPU_v1/independent_CPU_audit.json)、[独立原生工程](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/candidate/GPU_engineering_v3/completion.json)、[两图及八更新审计](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/candidate/GPU_engineering_v3/independent_CPU_audit.json)、[首断点独立核验](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/repository/audit/measurement_prior_20261028_v1/first64_independent_CPU_audit_v1.json)、[预GPU失败闭合](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/repository/audit/measurement_prior_20261028_v1/GPU_v2_dispatch_failure_closed.json)、[取回SHA清单](../audit/measurement_prior_20261028_v1/actual_20261029T004557189665Z/fetch_manifest.json)、[17:46训练在线原始回执](../audit/crossdisciplinary_20261028_v1/bridge_actual_D39_documentation_training_live_20261029T004645599259Z.json)。

## 固定图片比较及尚未完成部分

[训练方案](../configs/measurement_prior_20261028_v1/training_spec_v1.json)和[十组因子图片规则](../configs/measurement_prior_20261028_v1/image_policy_v1.json)均在正式拟合前锁定。五种编码器（原始＋四臂）×ordinary关闭／开启，共十组；所有组使用相同 CPU 条件导出与原生生成后端，重新生成配对原始编码器基线，旧缓存只作次级参照。固定原Dcheck400、400图库、512／4步／CFG0／IP1、每query一图及相同旧query种子。

80条只在两个预定 measurement 背景比较都明确损害（DINO均差≤−.02且描述95%上界<0）时停止，否则全部十组扩400，不读取中间160／240／320效果。400主比较为 measurement＋ordinary 对原始编码器＋ordinary，要求DINO≥+.01且区间下界>0、目标余弦下界>0及同概念Top1均差≥0；native背景同门槛分列。真实方向还需胜 isotropic、置换和不约束，不能把普通微调收益冒称测量机制。

**17:46时的历史状态：当时控制器只负责训练和断点审计，十组图片执行器尚未绑定、没有自动出图作业；现已由上方18:12执行更新取代。** 后续需完成固定规则的执行绑定和真实配对评分，不重新选择训练配方。当前没有方法有效、ordinary 可加、跨被试或 SOTA 结论。

## 文献近邻限制原创主张

噪声与导数正则化已有[Bishop的经典工作](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/)；各向异性二次保持约束见[EWC第2节公式3](https://arxiv.org/html/1612.00796v2#S2)；功能先验也见[Pan等人的工作](https://arxiv.org/abs/2004.14070)。本轮阅读范围分别为作者摘要、EWC相关推导、功能先验摘要，未做全领域源码复现；不能说换到功能空间或加入 EEG 协方差便已足够新颖。本轮 inverse-noise 变化约束也不能直接冒充 Bishop 输入噪声定理的同一个公式。[推导与检索边界记录](../audit/crossdisciplinary_20261028_v1/D39_derivation_and_literature_scope_v1.json)保存具体范围。

另一张旧卡上的[X01 ATMS／标准SDXL等价ordinary](ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_EXECUTION_20261028_v1.md)于17:46:05实际19680／24807更新（79.3%）、314872暴露，两遍已完成；原控制器240115继续，最新19456断点已CPU审计。六个运行源与原HIVE ordinary均未变，尚无跨框架效果图片。
