# ordinary 损失对照完整400：生成监督与刺激区分的证据

**2026-10-29 11:21:08，America/Los_Angeles，全部训练和图片比较闭合。** 原 ordinary 保持不变。F-only/token-only两个525568参数匹配头各完成全部十被试OOF-T三遍、24807更新；两组各400张新图及旧三组1200张参考闭合。ordinary在DINO上的增量相对廉价对照尚不明确，同概念Top1则有正向配对支持。当前不是新方法胜出或SOTA。

## 完整固定400结果

sub-01物理R4，40概念×10刺激，同一400图库；512、原生四步、CFG0/IP1、每query单图、同种子，无best-of-N。

| 方法 | DINO2AFC | 400图库Top1 | 同概念十选一Top1 | 同概念2AFC |
|---|---:|---:|---:|---:|
| parent | 0.769455 | 5.25% | 21.75% | 0.631944 |
| 原ordinary | **0.793885** | **6.50%** | **27.75%** | **0.656389** |
| 仅原始F角误差 | 0.786310 | 5.00% | 22.50% | 0.637222 |
| 仅原生A/P token角误差 | 0.787281 | 4.25% | 21.00% | 0.628889 |

| ordinary减对照 | 均值差 | 描述性95%概念bootstrap区间 |
|---|---:|---|
| F-only，DINO2AFC | +0.007575 | [−0.007247,+0.023421] |
| token-only，DINO2AFC | +0.006604 | [−0.008888,+0.023484] |
| F-only，同概念Top1 | **+5.25个百分点** | **[+2.00,+9.00]个百分点** |
| token-only，同概念Top1 | **+6.75个百分点** | **[+2.75,+11.00]个百分点** |
| token-only，同概念2AFC | +0.027500 | [+0.006660,+0.051118] |

F-only与token-only相对原生DINO分别+0.016855（区间[−0.001538,+0.034823]）及+0.017826（[+0.000550,+0.036735]）。廉价视觉对齐能解释一部分整体收益，不能声称只有生成监督有效。ordinary相对两者的DINO区间均跨零；同概念Top1有支持，但这是单被试、单seed、已曝光面板的描述性结果，不覆盖训练随机性和筛选/多重比较，也不是完整语义评价。新对照尚无ObjectF1/SEED或完整R80比较。

80条时原生/ordinary/F-only/token-only为0.823183/0.858083/0.848841/0.851504。两组均未触发事前明确损害条件，全部扩400。原160张80阶段图片逐位保留，首80特征按原版本冻结；没有按80结果选模型或改强度。400阶段两个控制实例均未胜原ordinary，按原规则停止，不自动扩R80或扫描loss权重。

证据：[完整400分数及全部固定对比](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/images/score400.json)、[80分数](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/images/score80.json)、[本机独立复算](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/independent_local_statistics.json)。原ordinary的完整十被试2000条R80正结果见[原报告](BRAINHIVE_ORDINARY_R80_FULL_RESULTS_20261029_v1.md)，这次没有重训、覆盖或改判它。

## 对照究竟匹配到什么

两组都保留原head结构、输出affine撤销/恢复、球面半径0.25、零方向初始化、seed1729。初始化张量摘要一致，三遍query顺序SHA逐一匹配原ordinary；每组132300真实OOF-T条件、三遍396900次暴露、batch16末批12、24807次AdamW更新，lr1e-4/wd0.01/clip1。旧A/P冻结，原FP16参数字节保持，token训练BF16 autocast。

\[
L_F=0.1\operatorname{mean}[1-\cos(c',F(I))],\qquad
L_P=0.1\operatorname{mean}[1-\cos(P(A(c')),P(A(F(I))))].
\]

原ordinary为$L_\epsilon+0.1L_P$。F-only和token-only不运行UNet/VAE或扩散噪声，因此**计算量没有匹配**。不能将这项差异唯一归因于某个新颖机制，也不能把非显著的DINO差异称为等价。

各组固定末轮后，只按原Dfit8200的token角误差网格{0,.25,.5,.75,1}校准；两新组均选1。没有读取Dcheck或官方目标来拟合、调强度或挑epoch。新头另存，原ordinary SHA保持`a66869e5…9e11cc03`。完整模型/Adam最终步数、3×132300覆盖、最终头与提交状态一致性CPU检查通过。

## 日志对下一步的实际约束

全部8200条Dfit、同一组真实条件和F目标的CPU描述性读数：

| 方法 | F余弦 | 原始F MSE | 同概念F Top1 | 已冻结Dfit token角误差 |
|---|---:|---:|---:|---:|
| parent | 0.122178 | 1.757140 | 38.1707% | 0.412668 |
| ordinary | 0.201561 | 1.599122 | 37.5122% | 0.217180 |
| F-only | **0.271993** | **1.458127** | 38.1707% | 0.251084 |
| token-only | 0.219200 | 1.563645 | 37.8902% | **0.195395** |

F-only的F匹配最好，token-only的token匹配最好，却都没有在这次真实图片的同概念Top1上胜过ordinary。这支持继续研究“宿主实际生成行为与静态目标对齐的差别”，不能把向量误差更小直接当作更好的重建。没有逐epoch留出曲线，不归为已证明的后期过拟合；不同训练目标的loss也不能直接排名。

另一项完整T132300条CPU测量将原ordinary方向分解为总体方向与EEG变化项。令$\Pi_u$为原去均值/去径向投影，$d_0(u)$为原head输出的中心化方向，$\mu=\mathbb E_Td_0(u)$，则

\[
d(u)=\Pi_u\mu+\Pi_u[d_0(u)-\mu].
\]

总体项能量相对完整方向为0.866462，变化项0.133845，交叉项−0.000308；完整与总体方向的平均余弦0.940845，半径使用比例中位数0.995727。代数误差1.42e−14。**这些不是图片收益的解释比例**：投影依赖当前EEG，两个分量不正交，球面映射非线性；不能因此说EEG没有作用或86.6%收益来自公共偏移。

这使下一项机制问题变得具体：在相同query、相同修正角度和噪声下，保留总体方向或保留样本变化方向，各自能保住多少图像收益。若仅比较原始分量而不匹配角度，会混入幅度变化。可预先定义$r=\|d(u)\|$，$q$取上述两个投影分量，并使用

\[
\Delta_q=\rho\|u\|\frac{r}{\sqrt{\|u\|^2+r^2}}\frac{q}{\|q\|},\qquad
u'_q=\|u\|\frac{u+\Delta_q}{\|u+\Delta_q\|}.
\]

对非零分量，其角度与原ordinary相同；零分量须事前定义明确回退。总体均值只从T估计，两分支仍可能使用当前EEG的几何和原head幅度，不能称为完全EEG无关的基线。该后续目前是推导，**尚未注册/派发GPU或生成效果图片**；经典投影与分解不是新创新点。

## 真实执行与核验

两组训练循环分别188.683/242.829秒，训练控制器505.202秒。图片控制器含等训练退出共921.791秒。15个GPU包装全部成功，charged合计1307.441秒；首次训练控制器17:59:12.990UTC至图片终态18:21:08.225UTC为1315.234秒，约21分55秒，首次设计/实现和最终独立复核/文档另计。

全部800张新图的身份、原始像素SHA、唯一inode、条件及FP16条件SHA、初始与四次ancestral噪声经独立CPU审计通过；像素重复0。全部800条/819200个条件标量由独立NumPy64公式复算，最大相对误差1.63e−7，范数和角度上界通过。十张工程回放分别逐像素回到旧parent/ordinary图片。DINO模型及图库沿用冻结SHA。

独立排名/加权bootstrap与原程序均值、区间最大差1.11e−15。初版审计在F-only减risk的同概念2AFC“正向概念数”处因约1e−17的数学零均值求和顺序差异退出；失败保留。v2只将零边界的计数歧义显式列出，原分数、区间和判定没有修改。全部955份元数据/逐query指标/程序按原SHA接收，本机另一种加权einsum复算260个统计标量，最大差1.11e−16；没有另行重跑视觉编码器或GPU训练。

初次训练派发脚本引号错误、图片注册错误依赖路径也均在相应训练/图片任务启动前修复，失败保留，见[执行历史](BRAINHIVE_ORDINARY_LOSS_CONTROLS_EXECUTION_20261029_v1.md)。这次没有把失败当成科学负结果或隐瞒为首轮全通过。

证据：[全部文件接收](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/receiving_completion.json)、[完整图片审计](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/images/independent_all_images_and_statistics_CPU_audit.json)、[全部条件复算](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/images/independent_all800_conditions_CPU_audit.json)、[训练最终状态审计](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/images/full_training_CPU_audit.json)、[T方向分解](../audit/HIVE_ordinary_loss_controls_complete_20261029_v1/population/completion.json)。

## 新颖性与研究边界

[Brain-Adapter的出版社摘要](https://www.sciencedirect.com/science/article/pii/S1746809426001059)已有EEG与IP-Adapter表征对齐，[SAMGA](https://arxiv.org/abs/2604.17782)已有适应EEG的视觉监督构造。因此不能以“加头”“对齐token”或“学习视觉目标”的名称主张原创。本轮出版社全文访问受403限制，未完成逐式全文比较；检索没有证明不存在相同方法。

本轮增加的是可复核的机制证据，不是又一个已成功的新方法。下一步应优先验证总体/样本特异方向的因果差别及训练随机性，并保留已有EEG目标耦合方向的实际正向线索。所有GPU只使用旧卡；训练305234、图片306675及本地观察器均已退出，最后18:24UTC观察GPU计算进程为空，第二卡未接触，原正式D仍停止。CCF A/SOTA目标继续，未达成。
