# D24：联动原生权重的坐标变换，闭式解与真实图片状态

> **完整400条已于2026-10-28 03:41:43，America/Los_Angeles闭合并停止。** 正确／同概念错配／仅affine的DINO为0.764912／0.761623／0.774655；原生／同一旧ordinary／risk为0.769455／0.793885／0.790476。主臂相对ordinary为−0.028972，描述性95%区间[−0.051305, −0.006597]，同概念Top-1由27.75%降至23.00%。固定判定STOP_INSTANCE／LARGE_HARM。1200张新图及旧三组1200张参考全部保留；末段因旧断言把240条缓存仍按80条检查而失败，v3只对480张现有图片补特征、汇总评分，新增图片和UNet为0，独立CPU核验全部1200图／2400行评分通过。GPU包装含失败共1275.212秒，首次拟合控制器至全量CPU闭合2939.431秒（48分59秒），首次设计及文档另计；全部进程退出、GPU及锁可用。下文80条与队列文字均为此前时间快照。[完整400读数](../audit/coordinate_gauge_20261028_v1/actual/candidate/stage400_scored_v3/readout.json)、[全量CPU核验](../audit/coordinate_gauge_20261028_v1/actual/server/images400_CPU_audit_v3/summary.json)、[闭合及全部失败费用](../audit/coordinate_gauge_20261028_v1/actual/server/scientific_full400_closure.json)。

2026-10-28，America/Los_Angeles。九个原生模型的闭式拟合、全部43350条条件CPU复核、三臂240张80面板图片及独立评分复核均已完成。主臂DINO0.846053，原生0.823183，旧ordinary0.858083；没有超过强参考。按事前“80条只停止明确损害”规则扩400，实际控制器PID228966于03:19:11启动。400终态尚未完成，不能据此宣布收益或停止当前方法。正式D仍保持停止。

## D23的日志怎样改变本轮问题

D23确实重新选择了第一层输入方向，并保持声明的重复噪声Gram，但没有证明相对不约束或旧ordinary的图片增量；同概念刺激区分亦下降。D24改问：现有EEG信息是否可以在保留其内部几何的条件下，重新对齐到生成宿主的视觉坐标？

本轮联动改变输入投影、残差MLP两端和输出LayerNorm affine。**函数上仍等价于一次正交输出坐标映射，不是新容量，也不新增EEG信息。** 第一层输入核满足ker(RW)=ker(W)，因此不能声称补回D22所诊断的不可见方向。把旋转吸收进多层权重本身不构成架构创新；本轮是一个明确、可证伪的坐标假设。

## 从同一个变换推导各层应怎样一起修改

原生单残差块写作

$$a=W_e e+b_e,\qquad h=a+W_2\operatorname{GELU}(W_1a),\qquad c=\beta_E+\gamma_E\operatorname{LN}_0(h).$$

取$RR^\top=I$且$R\mathbf1=\mathbf1$。保持均值的正交变换同时保持方差，因此在原eps=1e−6下有$\operatorname{LN}_0(Rh)=R\operatorname{LN}_0(h)$。同一个R要求以下权重联动：

$$\boxed{W_e'=RW_e,\quad b_e'=Rb_e,\quad W_1'=W_1R^\top,\quad W_2'=RW_2.}$$

GELU的输入保持不变，新的残差输出为Rh。将最后LN的affine换成冻结视觉F的真实$\gamma_F,\beta_F$，得到

$$\boxed{c'=\beta_F+\gamma_F R\left[(c-\beta_E)/\gamma_E\right].}$$

若漏改相邻权重，这个关系一般不成立。本轮实际零变换回放和联动前向已验证，不把任意旋转穿过GELU。

角色内允许FIT刺激的去affine EEG/视觉矩阵为X、Y。求解

$$\min_R\|XR^\top-Y\|_F^2,\qquad RR^\top=I,\ R\mathbf1=\mathbf1.$$

令B为$\mathbf1^\perp$的1024×1023正交基；对$(YB)^\top(XB)=U\Sigma V^\top$作SVD，则

$$\boxed{R^*=BUV^\top B^\top+\mathbf1\mathbf1^\top/1024.}$$

这是经典正交Procrustes解，不作为原创定理。它保留去affine后EEG向量间的内积、距离和夹角；不保证affine后的宿主几何、EEG与F的刺激匹配或图片收益。

## 实际拟合、权限与匹配控制

使用最早的sub-01原生full/oof0/oof1模型，不使用D23训练后的权重。各角色分别用自己的FIT概念求R：full13230刺激，两OOF角色6610/6620刺激；held概念不用于相应编码器和R拟合。原父模型与R使用同角色训练数据，不能把全系统宣称为无训练内偏差的OOF校准。

三臂为正确配对R、同概念内固定循环错配目标R、R=I但同样换视觉affine。前两臂具有相同EEG输入和视觉目标多重集合；第三臂隔离affine替换的作用。共6个闭式旋转、3个identity控制、9个最终原生模型；每模型仍6,452,224参数，SGD更新为0。全部允许FIT各参与一次闭式统计，没有epoch、步长或强度搜索。

新条件直接进入原生A/P/IP/Turbo四步，不再叠旧ordinary。旧ordinary仍作为事前固定主参考，使用同一最早检查点，SHA `a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03`、强度1，没有重训。正常推理仅用当前query的物理mean4，不读目标F、目标图或概念标签。全部400条件在新图前锁定。

GPU拟合全程序12.567秒，其中6次闭式SVD主体5.661秒；独立CPU程序15.965秒，核验全部43350条条件、9组折叠权重及6个闭式目标最优值。最大前向相对误差约1.052e−6，权重绝对误差1.164e−10，最优值相对差约1.918e−13。实际FIT交叉矩阵均为1023满秩。见[拟合证明](../audit/coordinate_gauge_20261028_v1/actual/candidate/fit/fit_proofs.json)、[CPU复核](../audit/coordinate_gauge_20261028_v1/actual/server/fit_CPU_audit_v2/summary.json)和[九项数学检查](../audit/coordinate_gauge_20261028_v1/local_math_checks.json)。

## 80条真实图片及固定扩展判定

sub-01、8概念、80query，同一400目标图库。三臂240张新图，旧三组240张直接复用；相同query初始及四步ancestral噪声，512×512、每query单图、无best-of-N。

| 方法 | DINO2AFC | 目标余弦 | 同概念10选1 |
|---|---:|---:|---:|
| 原生parent | 0.823183 | 0.271002 | 17.50% |
| 旧ordinary | 0.858083 | 0.274514 | 23.75% |
| 旧risk | 0.852381 | 0.277069 | 23.75% |
| 正确配对坐标变换 | 0.846053 | 0.273205 | 20.00% |
| 同概念错配坐标变换 | 0.840132 | 0.280721 | 15.00% |
| 仅视觉affine | 0.835746 | 0.278014 | 22.50% |

主臂相对ordinary的DINO差−0.012030，描述性95%区间[−0.046816, 0.020056]；相对原生+0.022870，区间[−0.007678, 0.051667]。当前均未形成稳定增量。主臂也没有证明胜过两个匹配控制。按既有v2规则，均值未达到−0.02的提前损害门槛，固定判定为**EXPAND400**，不能因尚未胜强参考而现场改成停止。

全部240图身份、条件、初始和四步噪声、字节/像素、480行六方法评分及重采样已独立复核。两条原生工程像素回放也完全一致。见[完整读数](../audit/coordinate_gauge_20261028_v1/actual/candidate/stage80/readout.json)、[图片CPU核验](../audit/coordinate_gauge_20261028_v1/actual/server/images80_CPU_audit/summary.json)、[固定原图展示](../audit/coordinate_gauge_20261028_v1/actual/server/images80_CPU_audit/fixed_first_stimulus_8concepts.png)。

首个CPU入口因未设置锁定依赖的搜索路径而缺少safetensors，失败日志保留。v2只设置原运行环境再调用未修改的v1核验；没有重拟合、换模型或提高误差容限。80阶段GPU包装两次共269.816秒；首次拟合控制器至80图片/CPU闭合1187.077秒（19分47秒），含阶段间实现及CPU入口修复，首次设计和后续文档另计。[真实80阶段闭合](../audit/coordinate_gauge_20261028_v1/actual/server/scientific_stage80_closure.json)。

## 400扩展的工程安排与创新边界

D23的960张扩展图曾触发单作业时间保护。本轮将相同固定400拆为80→240和240→400两个GPU作业：每段只新增480图，逐段保留图片与DINO特征；240中间态禁止计算科学分数。末段完成后统一核验全部1200图和2400行评分。每GPU作业上限780秒、每轮900秒，CPU另限240秒，外层有限2400秒。旧80图及特征直接复用，不重训、不换种子、不选择中间赢家。

03:24:49的[实际进程快照](../audit/coordinate_gauge_20261028_v1/actual_400_observation_20261028T102449Z.json)确认控制器PID228966和首段GPU子进程PID228975仍在线，首段已提交576/720行（含原80条240行），不是仅凭锁文件判断运行。

400真实结果仍待完成；实际状态见`audit/coordinate_gauge_20261028_v1/actual_images400_launch_v2.json`及同目录后续回执。本文不把队列预算写成已经完成。

[QuaRot](https://arxiv.org/abs/2404.00456)和[SpinQuant](https://arxiv.org/abs/2405.16406)已有联动旋转/权重吸收近邻。本轮拟合及80阶段结束后的进一步检索还发现[SCORE](https://arxiv.org/abs/2608.19134)：它使用正交变换进行无目标标签的跨被试EEG坐标恢复与图像检索。本轮是带角色内T监督的同被试生成接口，任务和权限不同，但不能把“保关系、恢复坐标”这个核心原理称为原创，也不能把改动折进多层权重作为创新证明。该文献发现不改变已经锁定的方法、控制或图片门槛。

当前只有单被试、单seed、已曝光开发面板结果，尚无稳定强参考增量、独立机制、跨架构或SOTA结论。后续方向必须受完整图片证据和已有近邻共同约束，不能仅凭闭式解成立继续添加同类旋转变体。
