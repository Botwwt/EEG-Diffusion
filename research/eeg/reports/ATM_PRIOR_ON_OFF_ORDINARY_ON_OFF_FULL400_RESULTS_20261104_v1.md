# 原生ATM：prior与冻结ordinary的完整400条结构比较

完整400条、四臂1600张图片已闭合，其中原parent/ordinary共800张直接复用，新direct/direct_ordinary共800张。绕过当前prior的direct相对原生parent，DINO2AFC从0.718459升至0.788477，差+0.070019、描述性95%区间[+0.032911,+0.105907]，31/40概念正向；SSIM和平均颜色误差也改善。这里推理输入仍只有EEG，没有使用oracle_F或查询真图。

同时，将原来在prior输出上训练的冻结ATM ordinary直接搬到原始EEG条件后，DINO由0.788477降至0.752130，差−0.036347、区间全负；SSIM却升高0.040502，40/40概念正向。这说明两个改动的语义收益不能默认相加，必须保留指标间的取舍。普通旁路是强结构基线，本项不因此成为已确认的第二个创新。

## 实际输入与两个开关

sub-01、真实物理R4、63电极；同一full ATMS编码器、同一h94 IP与SDXL-Turbo、512×512、4步、CFG0、IP1、相同query噪声和400目标图库，40概念各10刺激。四种输入分别为prior(c)、ordinary(prior(c))、c和ordinary(c)。每query单图；direct没有目标标签、图库检索、真图输入、缩放或截断。所有模型及强度全部冻结，新训练/模型/校准均0。

parent/ordinary使用原生50步DDPM prior，每query100次prior网络调用；direct两组为0次。四组均4次UNet。因此这是固定实际路径与成本的比较，不能声称同训练/推理计算量或作者官方完整复现。

本项冻结ATM ordinary为34e294b4…939844、525568参数、2481更新/39690暴露、强度1/半径0.25，和原HIVE a66869e5…9e11cc03不同。改变输入却没有重新拟合这个头，因此不能推断在direct条件上另训的ordinary也会同样降低DINO。原HIVE头和源码不改。

## 全部八项指标

平均颜色误差越低越好；其余指标越高越好。DINO2AFC是图库排序比例，不是对象正确率。同概念Top1是在十张同概念图中命中具体刺激的比例。

| 指标 | prior | prior+ordinary | direct | direct+ordinary |
|---|---:|---:|---:|---:|
| target_cosine | 0.241252 | 0.249887 | 0.289243 | 0.273875 |
| DINO2AFC | 0.718459 | 0.727306 | 0.788477 | 0.752130 |
| gallery_top1 | 0.025000 | 0.025000 | 0.032500 | 0.015000 |
| within_concept10_top1 | 0.160000 | 0.180000 | 0.170000 | 0.170000 |
| within_concept10_two_afc | 0.604167 | 0.595000 | 0.613889 | 0.602778 |
| PixCorr_RGB | 0.089263 | 0.074780 | 0.081026 | 0.073967 |
| SSIM_Y | 0.238045 | 0.281651 | 0.260900 | 0.301402 |
| mean_RGB_absolute_error | 0.128527 | 0.121961 | 0.109331 | 0.102332 |

## 事前固定的五个比较

所有差值均为左项减右项。交互为 `(direct_ordinary-direct)-(ordinary-parent)`。下表完整报告5×8项；区间来自固定seed1729、1000次概念bootstrap，是已曝光开发面板上的描述性区间，不能覆盖训练seed、被试或顺序筛选的不确定性。

| 比较 | 指标 | 均值差 | 描述性95%区间 | 正差概念数 |
|---|---|---:|---|---:|
| direct-parent | target_cosine | +0.047991 | [+0.029337, +0.065336] | 32/40 |
| direct-parent | DINO2AFC | +0.070019 | [+0.032911, +0.105907] | 31/40 |
| direct-parent | gallery_top1 | +0.007500 | [-0.015000, +0.032500] | 8/40 |
| direct-parent | within_concept10_top1 | +0.010000 | [-0.027500, +0.047500] | 14/40 |
| direct-parent | within_concept10_two_afc | +0.009722 | [-0.018056, +0.036111] | 20/40 |
| direct-parent | PixCorr_RGB | -0.008237 | [-0.024509, +0.005958] | 19/40 |
| direct-parent | SSIM_Y | +0.022856 | [+0.013054, +0.032782] | 31/40 |
| direct-parent | mean_RGB_absolute_error | -0.019196 | [-0.028292, -0.009241] | 11/40 |
| direct_ordinary-ordinary | target_cosine | +0.023988 | [+0.009406, +0.040641] | 25/40 |
| direct_ordinary-ordinary | DINO2AFC | +0.024825 | [-0.013445, +0.066407] | 25/40 |
| direct_ordinary-ordinary | gallery_top1 | -0.010000 | [-0.025000, +0.002562] | 2/40 |
| direct_ordinary-ordinary | within_concept10_top1 | -0.010000 | [-0.052500, +0.032562] | 12/40 |
| direct_ordinary-ordinary | within_concept10_two_afc | +0.007778 | [-0.018618, +0.033896] | 23/40 |
| direct_ordinary-ordinary | PixCorr_RGB | -0.000813 | [-0.018400, +0.015984] | 19/40 |
| direct_ordinary-ordinary | SSIM_Y | +0.019752 | [+0.012862, +0.025898] | 34/40 |
| direct_ordinary-ordinary | mean_RGB_absolute_error | -0.019629 | [-0.028471, -0.011195] | 10/40 |
| direct_ordinary-direct | target_cosine | -0.015368 | [-0.023987, -0.005722] | 11/40 |
| direct_ordinary-direct | DINO2AFC | -0.036347 | [-0.056272, -0.015688] | 10/40 |
| direct_ordinary-direct | gallery_top1 | -0.017500 | [-0.040000, +0.005000] | 3/40 |
| direct_ordinary-direct | within_concept10_top1 | +0.000000 | [-0.032500, +0.037562] | 12/40 |
| direct_ordinary-direct | within_concept10_two_afc | -0.011111 | [-0.034451, +0.011951] | 19/40 |
| direct_ordinary-direct | PixCorr_RGB | -0.007058 | [-0.019689, +0.006180] | 12/40 |
| direct_ordinary-direct | SSIM_Y | +0.040502 | [+0.035174, +0.046379] | 40/40 |
| direct_ordinary-direct | mean_RGB_absolute_error | -0.006999 | [-0.012413, -0.001830] | 13/40 |
| ordinary-parent | target_cosine | +0.008635 | [+0.000507, +0.017624] | 29/40 |
| ordinary-parent | DINO2AFC | +0.008847 | [-0.011098, +0.029826] | 22/40 |
| ordinary-parent | gallery_top1 | +0.000000 | [-0.017500, +0.017562] | 5/40 |
| ordinary-parent | within_concept10_top1 | +0.020000 | [-0.020000, +0.060000] | 17/40 |
| ordinary-parent | within_concept10_two_afc | -0.009167 | [-0.033063, +0.012243] | 18/40 |
| ordinary-parent | PixCorr_RGB | -0.014483 | [-0.029266, +0.000865] | 15/40 |
| ordinary-parent | SSIM_Y | +0.043606 | [+0.038119, +0.049446] | 40/40 |
| ordinary-parent | mean_RGB_absolute_error | -0.006566 | [-0.010494, -0.002476] | 11/40 |
| interaction_ordinary_without_prior_minus_with_prior | target_cosine | -0.024003 | [-0.034189, -0.013111] | 9/40 |
| interaction_ordinary_without_prior_minus_with_prior | DINO2AFC | -0.045194 | [-0.069414, -0.019320] | 10/40 |
| interaction_ordinary_without_prior_minus_with_prior | gallery_top1 | -0.017500 | [-0.050000, +0.012500] | 8/40 |
| interaction_ordinary_without_prior_minus_with_prior | within_concept10_top1 | -0.020000 | [-0.070000, +0.037500] | 13/40 |
| interaction_ordinary_without_prior_minus_with_prior | within_concept10_two_afc | -0.001944 | [-0.034444, +0.032535] | 20/40 |
| interaction_ordinary_without_prior_minus_with_prior | PixCorr_RGB | +0.007424 | [-0.010237, +0.023555] | 18/40 |
| interaction_ordinary_without_prior_minus_with_prior | SSIM_Y | -0.003104 | [-0.010206, +0.003919] | 14/40 |
| interaction_ordinary_without_prior_minus_with_prior | mean_RGB_absolute_error | -0.000433 | [-0.006177, +0.004822] | 19/40 |

这些结果支持的范围：

- direct相对parent的整体DINO、SSIM和颜色误差有正向支持，但图库Top1、同概念Top1/2AFC区间仍跨零；未证明刺激细节更准确。
- direct+ordinary相对原ordinary的DINO均值+0.024825但区间跨零；SSIM和颜色误差改善。
- direct上冻结ordinary带来语义下降和结构改善。DINO交互−0.045194、区间[−0.069414,−0.019320]；SSIM交互区间跨零，结构收益在两个输入域都出现。不是所有ordinary贡献都失效，也不能把SSIM增加叫作语义成功。
- 固定八组原图仍有明显对象替换，例如柠檬变成柜子/金属箱、豆芽变成混合沙拉。它们不是按分数挑选，也不能由八例估计完整对象正确率。

## 80到400以及独立复核

80阶段direct/parent的DINO为0.851629/0.765476；完整400降至0.788477/0.718459，方向保持，绝对值及效应量变化。按事前规则，两种绕过配置一起扩400；未查看160/240/320中间分数，未按80挑赢家或追加强度/seed扫描。

全部400条条件在出图前锁定。原两臂400条件、全部评价特征和80图片前缀逐位复用。full编码器原始角色表与真实split独立join，13230个T刺激/1323概念且Dcheck400不在拟合集；历史V模型选择曝光仍明确保留，不能称独立确认。真实八条EEG CPU/GPU编码差4.50e−7；全部400条头输出独立NumPy64差1.9133e−6。

服务器全1600图检查身份、目标、实际FP32/FP16条件、初始和四次ancestral噪声、像素SHA、解码及跨臂路径。无精确像素重复组、近空间常数图为0。这是空间完整性，不能等同正确对象或感知质量。

本地完整接收122个文件、33328561字节；独立复核两阶段9600个语义值、19200个配对值、80个bootstrap区间，均值/区间最大差2.78e−17；float64特征点积与原float32分数最大差4.60e−7。固定32图文件/解码像素及低层指标重算最大差5.72e−15，全部八行对照已实际查看。本地没有重新执行所有神经编码，也没有解码所有1600张图；原全量图片留在OLD。

## 耗时、失败与真实退出

控制器从2026-11-04T05:56:44.792461UTC至2026-11-04T06:28:06.457071+00:00，实际1874.615秒（约31分15秒）；13个GPU包装合计1705.971秒。设计实现、接收及本地复核另计，不把整轮称作5分钟。原09:56:44UTC截止没有重置。

分片manifest实查2026-11-04T06:41:02.759243+00:00：原控制器653049、全部13个包装与子进程均已退出，GPU进程列表为空。这是该观测时刻的事实，不保证未来空闲。没有重挂科学队列。

首次单次大响应在约31MB截断、exit−1，完整失败wire保留；后来只读导出改为带catalogue/SHA的五个分片，每个真实SSH命令exit0，并非把旧失败改成成功。接收v2又将Windows CRLF文件SHA和实际read_text标准化后的执行文本SHA误作相同，在创建结果目录前拒绝；v3显式核验两种SHA，数值复核主体与v1逐字相同。失败、旧源码与原始wire均保留，训练/生成/评分未因此重跑。

512MiB本项目预算、旧50GiB底线/8GiB缓冲及128MiB支持、256MiB true-forward、256MiB条件曝光等旧承诺不变。未删除文件、放宽资源、联系第二卡或恢复正式D/撤下的27个拟合。已用容量以各阶段资源回执为准，终态没有另作完整stat，未释放本项未用未来额度。

## 对下一项工作的约束

当前证据要求以后任何prior改造先超过direct这一更便宜且实际更强的EEG基线；不能只与现有较弱prior比较。原prior使用本研究T/R4配方，结果不能外推成作者官方全量先验无用，亦不能仅凭路径比较唯一归因于截断、采样或训练目标。

下一项应先定位EEG可辨的视觉变化在prior哪一阶段受损，并把结构偏好和刺激辨别分开分析；若另训ordinary-direct必须另存版本，并保留同视觉目标/公共方向对照。该新拟合尚未准入或启动，本报告未因SSIM/DINO其中一项不达标丢弃其他证据。残差扩散、x0/v/EDM预条件、可观测/零空间与扩散对比目标均有既有近邻；经典公式或去掉模块不冒称原创。

本轮是单被试、单seed、反复曝光的开发400图库，尚无十被试R80完整确认、完整语义评价器比较或第二创新/SOTA/CCF A结论。oracle_F另属读取当前真图的HIVE诊断，本轮四臂都不是oracle。

[完整400分数](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/ATM_prior_factorial_complete_20261104_v1/actual/readout400.json)、[80初筛](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/ATM_prior_factorial_complete_20261104_v1/actual/readout80.json)、[独立复核](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/ATM_prior_factorial_complete_20261104_v1/independent_local_audit.json)、[完整接收来源](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/ATM_prior_factorial_complete_20261104_v1/receiving_completion.json)、[固定八组原图](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/ATM_prior_factorial_complete_20261104_v1/fixed_eight_prior_factorial_examples.png)、[接收勘误](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/ATM_prior_factorial_receiver_v2_failure_and_v3_transport_erratum_20261104_v1.json)、[编码器原始拟合权限](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/ATM_prior_factorial_actual_original_encoder_permissions_local_20261104_v1.json)、[多领域近邻复核](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/reports/EEG_PRIOR_RESIDUAL_CONTRASTIVE_AND_OBSERVABILITY_REVIEW_20261104_v1.md)。
