<!-- EEG_ALL20_FITS_AND_IMAGE_RECOVERY_V329 -->
> **2026-11-11 17:23 America/Los_Angeles：20个拟合全部完成，图片已恢复运行。** 15编码器3375更新、五个各自ordinary12405更新及Dfit校准闭合。12:01出图前CPU eval注意力NaN导致队列停滞，本次已定位并保留失败；数学CPU后端全8300条件复核通过，20模型SHA不变，最大缩放误差4.526e-6。GPU已保存107图，首图原生像素/祖先噪声RNG逐位一致；新骨干完整图片分数仍待评价，不能称新方法收益。
> [实际故障、完整复核与当前队列](../research/eeg/reports/EEG_BACKBONE_COMPLETE_MATRIX_AND_LONG_CAMPAIGN_20261111_v325.md)
<!-- END_EEG_ALL20_FITS_AND_IMAGE_RECOVERY_V329 -->

<!-- EEG_ALL15_ENCODERS_COMPLETE_V327 -->
> **2026-11-11 09:30 America/Los_Angeles：15个25轮编码器全部完成。** 共3375更新和3307500次query暴露，五份真实OOF银行已闭合；全部训练日志独立复核通过。各自ordinary已接续，第一个MLP头第3遍2368/2481更新；其他头与8300新图仍待完整运行。同参数分离注意力的训练loss低于联合注意力，是优化线索，最终仍按完整表征及图片判断。原科学配方和单GPU链保持。
> [完整训练日志分析、公式及待验证结论](../research/eeg/reports/EEG_BACKBONE_COMPLETE_MATRIX_AND_LONG_CAMPAIGN_20261111_v325.md)
<!-- END_EEG_ALL15_ENCODERS_COMPLETE_V327 -->

<!-- EEG_BACKBONE_LONG_CAMPAIGN_V325 -->
> **2026-11-11：骨干与各自ordinary的完整长队列已实际运行。** 前轮三组2481更新/3320图片闭合；DINO未确认胜过ordinary，SSIM收益保留。新MLP17/锁定作者ATM63/Joint63/同参数Separated63/Joint17，共15个25轮编码器、5个独立ordinary、8300新图及13组14读数。新MLP三角色675更新已完成并核验，GPU链已接ATM。附完整T/Dfit/Dcheck的148800条条件诊断，避免只看图片分数猜原因。单1729，本研究单GPU，其他项目不变。
> CPU真实EEG/梯度缓存/BN/RNG检查通过；表征测量独立参考误差5.56e-17。原依赖/诊断精度失败均保留、训练科学源未改。既有attention/GradCache不能直接作为原创，尚无新骨干图片收益或SOTA/CCFA结论。
> [完整结果、骨干公式、对照、任务规模及边界](../research/eeg/reports/EEG_BACKBONE_COMPLETE_MATRIX_AND_LONG_CAMPAIGN_20261111_v325.md)
<!-- END_EEG_BACKBONE_LONG_CAMPAIGN_V325 -->

<!-- EEG_VISITATION_CURRENT_V320 -->
> **2026-11-10 17:58 America/Los_Angeles：完整新结果与下一轮已挂单GPU。** P_abs_free相对ordinary的SSIM+0.039040区间正，DINO−0.002788区间跨零；相对旧joint P的DINO+0.014949区间正，是恢复线索，尚未确认第二个语义胜出方法。五组训练与4150图片、前轮3320图均已闭合，完整逐query与日志复核通过。
> 新方向检验保护所用的真实EEG采样状态：native轨迹、匹配时间/sigma的forward状态、同概念错配轨迹三组各完整T827更新；另有冻结同参数更新范数控制，合计3320新图、9组14读数。CPU修正版5123621通过，GPU5123622实际运行轨迹前缀；拟合尚未开始。单seed1729，不加推理模块，无SOTA/CCFA保证。
> [结果、完整公式、控制、来源与执行边界](../research/eeg/reports/EEG_PRESERVATION_RESULTS_AND_SAMPLING_VISITATION_20261110_v320.md)
<!-- END_EEG_VISITATION_CURRENT_V320 -->

> **2026-11-10 05:12 America/Los_Angeles：约12小时的单GPU串行实验链已提交。** 当前v316先完成；随后五组固定完整T训练（4135更新）及4150张全830图片、14指标评价。比较实际EEG域绝对/差异响应保护，与原生P真实均值/随机方向约束，不增推理模块。CPU工程已过，新的GPU训练尚在依赖等待；不把已有工具声称为原创或预设好效果。
> [完整公式、五组对照、文献边界和时间估算](../research/eeg/reports/EEG_OVERNIGHT_TWELVE_HOUR_CAMPAIGN_20261110_v317.md)

> **2026-11-10 04:50 America/Los_Angeles：P/IP完整图片结果与下一项实际实现。** 九组训练、7470新图、29组全830×14指标已闭合。九组DINO均低于ordinary、SSIM均升；回放/正确配对减轻退化但没有胜强基线。固定联合P DINO0.754428 vs ordinary0.772164；SSIM0.398201 vs0.329790。冻结两个既有P的共同/逐EEG函数干预已通过全T均值、两图像素和CPU复核，单GPU正在出3320新图，随后全量评价；这是机制诊断而非新成功方法。
> [完整结果、日志分析、公式及下一步队列](../research/eeg/reports/EEG_DOWNSTREAM_FULL_RESULTS_AND_FUNCTIONAL_NEXT_STEP_20261110_v315.md)

> **2026-11-09 17:52 America/Los_Angeles：架构定位的完整T实验已完成。** 两项数值诊断及独立全预测回放共502740行/3016440单元通过。原始EEG线性读出未胜原生c；真实图像F仍保留强H14/VAE身份，RBF读出反而较弱。当前不以“骨架选错/融合抹掉身份”作为新论文主张。已有P/IP单GPU完整图片研究继续，没有新增图片收益或自动组件搜索。
> [设计、全部关键结果、限制和路线决定](../research/eeg/reports/EEG_ARCHITECTURE_DECISIVE_STUDIES_20261109_v312.md)

> **2026-11-09 17:16 America/Los_Angeles：创新充分性审视与完整T新诊断。** ordinary有完整R80相对原生收益，但未确认对F-only/训练公共方向的额外主语义增量；第一有效模块不能直接等同充分主创新。新增全13230/六折可逆坐标诊断无可靠ridge增量；普通头MSE下降但刺激Top1从23.02%降至19.52%。冻结七干预进一步显示径向调整可降MSE却不改变排名。它们是完整诊断结果，不是第二成功方法。原P/IP单GPU链仍在IP_replay，未出新效能图。
> [全部积极线索、ordinary对照与新诊断](../research/eeg/reports/EEG_LATEST_IDEAS_AND_ORDINARY_ASSESSMENT_20261109_v311.md)

> **2026-11-09 08:25 America/Los_Angeles：完整判断标准更新。** C三组/4980图及20组全部830条共同评价完成；C_pair DINO0.773282，ordinary0.772164，差区间跨零，尚无第二确认方法。新增完整条件—图片诊断CPU5104557及本机16600行复算完成：A的整体F余弦大升仍语义下降，C限制漂移后恢复语义；不据相关性宣称唯一因果。现有九组P/IP继续原单GPU串行链，不重复开训。完整一遍T不等于收敛、单sub03R4不等于跨架构或十被试R80。后续须有独特机制或可区分下一步的完整分析；论文候选仍待实现和验证。
> [完整方案、最新结果和近邻边界](../research/eeg/reports/EEG_PAPER_QUALIFICATION_AND_FULL_SCOPE_20261109_v307.md) · [完整C结果](../research/eeg/evidence/EEG_C_ANCHORED_ALL830_RESULTS_20261109_v308.json)

# EEG 图像重建：当前方法与证据

更新：2026-11-09 UTC。此页反映当前研究；2026-10-18 的响应蒸馏提案属于早期方案。

当前有 ordinary 相对原生 Brain-HIVE 的完整正向结果。第二个独立方法、跨宿主语义收益、GPU 端到端加速和 SOTA 尚未确认。工程恢复、特征检索和实际图片评价分别记录。

## 1. ordinary：保留的第一条方法线

原生链路为 EEG → BrainMLP → 条件 c → 原生残差映射 A → IP 投影 P → SDXL-Turbo。ordinary 在 c 与 A 之间修正方向，冻结 EEG 编码器和原生生成器，通过生成器反向传播训练修正头。

撤销原编码器 LayerNorm 的真实 affine：u=(c−β)/γ。将预测方向去均值、去径向分量得到 d，再计算：

$$\Delta=\rho\|u\|\frac{d}{\sqrt{\|u\|^2+\|d\|^2}},\quad
u'=\frac{\|u\|}{\|u+\Delta\|}(u+\Delta),\quad c'=\beta+\gamma u'.$$

训练目标为冻结生成器的噪声预测 MSE，加 0.1 倍原生 A/P token 角误差。修正头 525,568 参数，半径 0.25；零初始化和关闭路径回到原生条件。保持 u 的范数不保证图像质量或全部中间分布不变。

全 1654 概念训练版本、单 seed 1729、固定最后检查点，在十被试 × 200 刺激 = 2000 query 的 R80 评价中：

| 指标 | 原生 HIVE | ordinary | 配对差及描述性 95% 区间 |
|---|---:|---:|---|
| DINO2AFC | 0.872769 | 0.891198 | +0.018430 [0.006081, 0.030192] |
| SEED | 0.182698 | 0.190705 | +0.008007 [0.001090, 0.014995] |
| SSIM | 0.328023 | 0.353494 | +0.025472 [0.021907, 0.029209] |
| CLIP | 0.796234 | 0.810430 | +0.014196 [0.001847, 0.026074] |

R80 指每个刺激的输入 EEG 物理平均 80 次重复记录；不是 80 张候选图片或 80 次挑选生成。每 query 一张 512×512 图片、四步采样，同 query 噪声配对。

ordinary 相对受限 F-only 的额外 DINO/SEED 收益未确认，相对自己的训练公共方向的额外主语义收益也未确认。SSIM 等次级指标仍有正向增量。因此，目前不能把提升全部归因于逐 query 自适应方向，也不能把 ordinary 称为已确认的原创机制或 SOTA。

[完整结果及限制](../research/eeg/reports/BRAINHIVE_FULL1654_ORIGINAL_ORDINARY_R80_RESULTS_20261101_v1.md) · [公共方向对照](../research/eeg/reports/HIVE_TRAIN_ONLY_MEAN_FIELD_R80_RESULTS_20261101_v1.md)

## 2. 两条 EEG 编码端假设的完整比较与后续

此前按 F 检索间隔做硬约束的实例后期停止接受更新，未取得第二项方法的完整图片收益。当前保留这个问题线索，改为两条有匹配对照的 EEG 编码端实验：

| 方向 | 实际改动 | 控制 |
|---|---|---|
| 配对能量 | 原生 EEG 编码器已有权重，直接区分同概念不同刺激在冻结生成器中的误差 | 相同预算普通训练、反向配对项 |
| 条件空间残差 | 在原生输入投影使用另46电极的T内条件残差，冻结旧17电极路径与MLP | 相同容量的原始额外电极、同概念错配残差 |

第一条目标为 Lpos + τ softplus((Lpos−Lneg)/τ) + 0.1 token误差；同图像latent/噪声/时刻比较正确EEG与同概念另一刺激EEG。第二条先求 B=CNP(CPP+λI)⁻¹、r=(eN−μN)−B(eP−μP)，再用零初始化U更新原生第一层 h=Wold eP+b+Ur。它们分别检验训练目标与额外测量输入，不修改采样器。

六组各固定一遍完整T、827更新，单seed1729。六组均已完成827更新，并通过CPU模型/Adam/RNG/覆盖检查；全部830条×7编码器独立CPU重放亦通过。完整T训练诊断中，正确配对的同概念Top1为82.8949%，普通组77.6342%，净多696条；原生full-T为99.3802%。因此这是减轻训练内区分损失的线索，不是超过原生或图片增益。

完整830条Dcheck已比较14组：原生、匹配ordinary、六个编码器各自有/无冻结ordinary。组合是显式交互测试，不是为每个新编码器重新拟合ordinary，也不预设效果相加。完整结果见下方最新更新。后续仍需同一原则在HIVE/ATM等匹配宿主上验证；相同1024维不能直接交换权重。

对比扩散、ridge回归和线性折叠有已有近邻，公式不单独作为原创贡献。完整图片、机制控制和强63电极基线等证据决定是否值得扩展。[推导、真实状态与全部边界](../research/eeg/reports/BRAINHIVE_TWO_EEG_CANDIDATES_EXECUTION_20261109_v280.md)

### 下一步 原生 EEG 锚定训练

由完整T训练诊断继续登记三个C分支：训练整个原生EEG编码器，以同一输入的冻结原生输出为锚，在真实LayerNorm坐标把角度偏离限制在约14°内；正常推理需要两个EEGencoder前向，仍为四次UNet。它能改变从原始EEG提取的差异，超出只看压缩c的后处理，但不是一个已经确认的新创新。

CPU5099014真实输入/梯度/角度范数边界检查、CPU5099289核方向的函数通路检查已通过。三组各固定827更新，排在当前原六组的完整图片评分之后；随后新增六组×830=4980图，复用原14组图片/特征，联合20组评价。数学界限不保证F排名或图片质量，原始核方向构造也不证明自然EEG有用信息。

[推导、已挂完整队列与边界](../research/eeg/reports/EEG_ANCHORED_ENCODER_DERIVATION_AND_QUEUE_20261109_v298.md)

### 完整830条新结果与下游接续

现已完成原六候选、原生及ordinary共14组的11620张图片和14项读数。新匹配ordinary DINO为0.772164，原生为0.755191，差+0.016973、描述性概念区间[+0.000409,+0.032275]；同概念Top1与SSIM也有正向差。A_pair为0.588413，相对A_positive0.568880有增量，但仍低于原生。A/B的SSIM改善伴随语义明显下降，不能混为有效第二方法。

C真实首两步已经完成。原CPU审计误拒绝有效16字节CUDA随机状态，v302仅修schema并恢复CPU RNG检查；独立复核通过后接续原断点，未重训首段。

按用户新授权，另挂五组原生P与四组原生IP训练。固定EEG和ordinary，每组完整13230条T一遍827更新，用目标匹配对照检验EEG配对能量与原视觉条件回放；随后7470张新图，复用现有20组图/特征形成29组完整比较。CPU工程已通过，新GPU拟合排在C完整评价后，尚无下游新收益。P/IP参数量不同，位置间比较不称容量匹配；回放/对比工具也不直接当原创。

[完整执行、全部对照与限制](../research/eeg/reports/EEG_DOWNSTREAM_OVERNIGHT_EXECUTION_20261109_v305.md) · [完整架构公式](../research/eeg/reports/EEG_COMPLETE_ARCHITECTURE_AND_NOVELTY_20261109_v299.md)

### 论文主张与创新复核

当前P/IP九组定位为适配强基线和机制对照。主候选收拢到EEG可恢复视觉内容与下游生成保持的共同设计：先在隔离概念中验证视觉方向的可预测性，再检查实际生成图是否保留这些证据，比较同秩PCA、仅预测性、仅生成敏感性和普通P/IP适配。该候选延续11月3日草案，尚未完成新数值验证或新颖性证明；SAMGA、Structured Visual Target Learning、RPA、DCR等近邻必须区分。已排队数值实验不变。

[完整论文主张与近邻区别](../research/eeg/reports/EEG_PAPER_MAIN_CLAIM_AND_NOVELTY_REVIEW_20261109_v306.md)


## 3. GPU 适配：实现候选，尚无完整加速结论

640 个间隔对完整 HIVE 的显式 FP64 Jacobian 仅矩阵就约 30.77 GiB。无矩阵乘法 J(Jᵀλ) 能避免存储完整 J，但其实际总显存、求解迭代数和端到端收益仍需测量。

目前合成 CPU/GPU 数值检查已通过；真实训练后的 HIVE 仅完成两 query、完整 T 图库的 Jv/Jᵀw/Gram 支持核验。没有两宿主完整训练或生成器的加速倍数。固定条件的 IP K/V 缓存也属于已知优化，尚不能单独当作第三项创新。

## 4. 新集群当前完成到哪里

已从固定公开版本恢复原生视觉教师输入和 F 权重，在新集群训练 sub-03 的 full/oof0/oof1 三个 BrainMLP。每个 25 轮，共 675 更新；真实 T 折外条件 13,230 条及 full-D 条件 1,650 条已导出，14,880 条独立 CPU 前向复核通过。

这是原生基线重建。全部13230条T原生VAE缓存和独立核验已完成；新匹配ordinary也已完成3遍2481更新/39690暴露，Dfit820条按原固定五强度选择1，旧ordinary不覆盖。六组EEG候选及全部830×14=11620张配对图片、DINO/论文七项/EffNet评分和终态CPU复核已真实挂依赖链，设备与包装校验失败保留后，现已恢复实际出图，已完成11620/11620张和完整14项读数，结果见上方更新。

六评价权重及五离线包已在集群恢复，CPU5098561完整SHA和严格模型检查通过；评价依赖在单独目录，未改变训练环境。评分人工并列样例的初版失败保留，修正版以作者原始NumPy运算顺序通过CPU5098719。ObjectF1、CapSim和完整SEED尚缺，不能称原完整评价门槛已通过。

[匹配训练与完整T诊断](../research/eeg/reports/BRAINHIVE_PKU_MATCHING_ORDINARY_PROGRESS_20261109_v263.md) · [生成接口与输入验证](../research/eeg/reports/BRAINHIVE_PKU_GENERATOR_INPUTS_RESULTS_20261109_v257.md) · [原生重建结果](../research/eeg/reports/BRAINHIVE_PKU_NATIVE_REBUILD_RESULTS_20261109_v242.md) · [当前历史总记录](../research/eeg/BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md)

## 5. 本仓库保存什么

`research/eeg/` 保存代码、配置、推导、结果报告及少量关键证据；`provenance/research_snapshot_20261109.json` 列出原文件与公开副本哈希及访问元数据脱敏。

原始 EEG、模型权重、优化器、逐图大特征、传输归档和私人登录信息不进入 Git。大文件在用户的集群共享存储归档，完成全文件 SHA 校验后才清理本地副本。历史实验脚本保留原部署假设，不等同于可直接运行的统一软件包。
