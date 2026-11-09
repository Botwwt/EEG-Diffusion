# D40真实新增电极通路：完整400条结果与独立复核

**2026-10-29 00:03:24，America/Los_Angeles，实际闭合。** 三臂各完成sub-01全部T的13,230刺激／1,323概念一遍、827更新；八组3,200张真实图片、固定11项比较及原CPU审计全部完成。主候选加旧ordinary的DINO为0.786497，匹配ordinary为0.793490，差−0.006992、描述性95%区间跨零；同概念Top1从27.75%降至23.25%，差值区间全负。按原门槛记录 **STOP_INSTANCE**，不扩完整V、不扫描强度、不从控制分支替换主候选。

本报告读取原终态，不新增GPU、图片或拟合。全部59份原始文件共12,596,211字节已取回并核对原SHA；另在本地独立重算16,000个逐query指标及11×1,000次整概念bootstrap，原判定一致。

## 1. 实际比较的改动

原父模型仍读取后部17电极。新增46电极经过T内真实重复测量的工作协方差，形成条件创新Z；第一层改为 h=W₀vec(P)+b₀+Bvec(Z)，原残差MLP、输出LN、A/P/IP/UNet均冻结。B含11,776,000个可训练参数，零初始化，每步投影到固定Frobenius半径1.8279817948。它增加输入与容量，不能把收益预先归于噪声机制。完整工作假设、Schur推导和边界见[原执行协议](BRAINHIVE_SPATIAL_INNOVATION_EXECUTION_20261028_v1.md)。

三臂为边际白化的额外电极marginal、正确条件创新measurement，以及训练时同概念错配创新permuted_measurement。三臂均在正常推理使用当前query自己的63电极，不读取图像或概念标签挑选donor。三臂各完整T一遍，共2,481更新／39,690次query暴露；seed1729、batch16、末批14、固定最后模型。

ordinary未进入本轮训练图，独立开关；其原检查点SHA为 `a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03`，强度1、半径0.25均未变。正常推理EEG编码器总参数18,228,224，仍为原生512×512、4步Euler ancestral、CFG0、IP1、每query单图且同噪声。

这不是十被试或全链OOF：父编码器已在T拟合，旧ordinary也使用过T。本轮是已曝光Dcheck400图库上的单被试开发比较，不能作为独立确认或SOTA。

## 2. 八组完整400条结果

基线是本轮重新生成的匹配original/native与original/ordinary，使用同一400目标图库。不能用旧16600／1660图库或旧400生成缓存的分数替代。DINO2AFC也不是对象正确百分比。

| 编码器／背景 | DINO2AFC | 目标余弦 | 同概念Top1 | 同概念2AFC | 全图库Top1 |
|---|---:|---:|---:|---:|---:|
| original__native | 0.768421 | 0.253667 | 21.50% | 0.630278 | 5.25% |
| original__ordinary | 0.793490 | 0.261526 | 27.75% | 0.657500 | 6.25% |
| marginal__native | 0.749123 | 0.246414 | 21.25% | 0.608889 | 4.75% |
| marginal__ordinary | 0.780802 | 0.256020 | 24.25% | 0.635833 | 3.50% |
| measurement__native | 0.748904 | 0.243782 | 19.75% | 0.618333 | 3.75% |
| measurement__ordinary | 0.786497 | 0.259946 | 23.25% | 0.642222 | 4.50% |
| permuted_measurement__native | 0.764461 | 0.249456 | 19.75% | 0.616944 | 4.25% |
| permuted_measurement__ordinary | 0.784185 | 0.258126 | 26.75% | 0.645556 | 5.50% |

主门槛仍为measurement+ordinary相对匹配ordinary：DINO差至少+.01且描述性区间下界>0，目标余弦差区间下界>0，同概念Top1均值差≥0。本轮均未形成原定积极支持。原生背景主差−0.019518，区间[−0.041738,+0.000264]；ordinary背景主差−0.006992，区间[−0.023760,+0.008988]。ordinary背景同概念Top1差−4.50个百分点，区间[−8.50,−0.74375]个百分点。

80条时measurement/native相对匹配native曾为+0.013722、区间跨零；measurement/ordinary为−0.004950。按事前规则八组全部扩400，前80图与特征保留，没有读取中间160／240／320的效果或更换模型。完整400没有保持80条的原生背景正均值。

## 3. 固定11项比较与机制

下表保留全部登记比较。区间以40个完整概念为单位、1,000次、seed1729；反复开发后的描述性区间不覆盖训练seed或跨被试不确定性。

| 固定比较 | DINO均值差 | 描述性95%区间 | 同概念Top1差（百分点） |
|---|---:|---|---:|
| marginal_vs_original__native | -0.019298 | [-0.037989, -0.002135] | -0.25 |
| measurement_vs_original__native | -0.019518 | [-0.041738, +0.000264] | -1.75 |
| permuted_measurement_vs_original__native | -0.003960 | [-0.024142, +0.015881] | -1.75 |
| measurement_vs_marginal__native | -0.000219 | [-0.019912, +0.018998] | -1.50 |
| measurement_vs_permuted_measurement__native | -0.015558 | [-0.038503, +0.006674] | +0.00 |
| marginal_vs_original__ordinary | -0.012688 | [-0.029640, +0.004871] | -3.50 |
| measurement_vs_original__ordinary | -0.006992 | [-0.023760, +0.008988] | -4.50 |
| permuted_measurement_vs_original__ordinary | -0.009305 | [-0.022672, +0.004482] | -1.00 |
| measurement_vs_marginal__ordinary | +0.005695 | [-0.015312, +0.025085] | -1.00 |
| measurement_vs_permuted_measurement__ordinary | +0.002312 | [-0.017985, +0.020058] | -3.50 |
| ordinary_interaction | +0.012525 | [-0.013809, +0.039361] | -2.75 |

正确条件创新相对同概念错排控制的DINO差：native为−0.015558、ordinary为+0.002312，两项区间均跨零。正确相对边际控制也未通过原机制组合条件，两个背景的机制支持标记均为false。不能据此声称条件噪声消除有效，也不能改选较好的控制分支作为本方法。

结果反对当前工作协方差、预算和一遍拟合所定义的这个实例；它不证明额外电极没有信息、BrainHIVE已经达到上限，或任何63电极结构都无效。仅凭训练损失下降或输入存在条件创新，不能推出可用图片增量。

## 4. 完整性、独立复核与时间

服务器逐图核验全部3,200张图像、条件、目标及噪声；3,200个唯一image inode，像素重复和恒色图均为0，未删样本。服务器独立重算16,000标量和全部11项bootstrap，最大指标差0。来源：[原图身份审计](../audit/spatial_innovation_20261028_v1/final400_received_v1/004ed39653_image_audit.json)、[服务器独立审计](../audit/spatial_innovation_20261028_v1/final400_received_v1/df12fdac73_independent_CPU_audit.json)。

本地取回后另重算全部8×400×5指标、11项整概念区间与原判定：排名指标完全一致，逐query目标余弦最大误差3.576278687e-07，区间最大误差1.269346103e-08。这是独立评分实现与不同本地NumPy环境的回放，不是重新提取视觉特征、UNet或像素的独立复现。见[本地复核](../audit/spatial_innovation_20261028_v1/final400_received_v1/local_independent_rescore_v1.json)与[原始文件清单](../audit/spatial_innovation_20261028_v1/final400_received_v1/manifest.json)。

35个D40有限GPU包装全部退出0，合计9777.690秒（约2小时43分）；训练控制器4512.554秒，图片至独立评分控制器5460.290秒。原生命周期从等待D39开始至最终终态17573.236秒（4小时52分53秒），包含等待，不能当作训练本体。首次设计、CPU准备、传输和最终文档另计。

2026-10-29 00:18:22实际只读复查：原生命周期PID37447已终态，GPU无D40进程。该观察不代表第二卡之后没有运行其他已授权任务。正式旧D保持停止，旧ordinary保留。

## 5. 固定证据身份

- 最终分数：`7cc1be2d5bdf0884e25185862254302c7ed0e438a47f58509e87c5f98bb7f3f8`，[完整分数与11项区间](../audit/spatial_innovation_20261028_v1/final400_received_v1/2dffbee78e_completion.json)。
- 服务器独立审计：`310abaa702ef6a23077638739454aec1ae6d9bcc93c4c6d46740b237ad4f4625`。
- 正确创新最终编码器：`8b988cebcc046419b66f964c0097042aa0267270f84de00e05d4ef9d6c47c363`；三臂最终模型与原训练预算见各臂completion及[训练闭合](../audit/spatial_innovation_20261028_v1/final400_received_v1/3b34d8e2b9_completion.json)。
- 图片执行：`f6476814850f0bf4391fe422b6df75fae9b8282ff5f398ecdfc34c3a70858c0c`；最终逐query结果：[3200行CSV](../audit/spatial_innovation_20261028_v1/final400_received_v1/7732399585_per_query.csv)。

本轮没有组件成功、机制成立、感知质量非劣、跨框架通用性或SOTA结论。
