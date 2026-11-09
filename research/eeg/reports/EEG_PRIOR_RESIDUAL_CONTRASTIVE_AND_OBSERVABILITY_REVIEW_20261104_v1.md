# 下一prior方向的直接文献近邻与适用前提

本文服务于正在执行的原生ATM prior开关×冻结ordinary开关图片比较。它不修改已锁定的六个程序、模型、数据、seed、资源上限或80→400规则，也不包含新训练、新图片或新方法效果。新方案是否值得拟合仍须依据这组正常EEG推理的完整图片；旁路本身不是第二项创新。

## 已核对的直接近邻

| 文献与实际阅读范围 | 已有方法内容 | 对我们下一方案的约束 |
|---|---|---|
| [RDDM，CVPR2024官方摘要](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Residual_Denoising_Diffusion_Models_CVPR_2024_paper.html)；PDF访问403，未声称全文核对 | 将残差与随机噪声的作用分开 | 单纯增加残差路径、区分确定性与多样性不能作为首创 |
| [DRDD，arXiv v1第3.2–3.4节](https://arxiv.org/html/2606.01048v1)；[CVPR2026官方记录](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_Decoupled_Residual_Denoising_Diffusion_Models_for_Unified_and_Data_Efficient_CVPR_2026_paper.html) | 固定噪声域内先做语义残差映射，再单独去噪 | 噪声与语义映射解耦已有非常直接的结构近邻，不能换成EEG输入后直接称第二创新 |
| [Observable and Null-Space Electrophysiological Source Imaging with Diffusion Priors，arXiv v1第III-A至III-C节](https://arxiv.org/html/2610.27461v1) | 由已知lead-field划分行空间与零空间，分别处理测量支持与先验补全 | 此任务恢复皮层源，不是EEG→图像；其精确测量保持依赖已知物理前向算子，不能直接套给学习到的图像潜变量映射 |
| [DCR，arXiv v1第4.1–4.3节](https://arxiv.org/html/2603.04803v1)；[CVPR2026官方记录](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Guiding_Diffusion-based_Reconstruction_with_Contrastive_Signals_for_Balanced_Visual_Representation_CVPR_2026_paper.html) | 在同一带噪状态下，对不同视觉条件产生的预测噪声做对比，包含增强条件及真实噪声正项，生成模型冻结 | 直接给冻结生成器的噪声响应增加对比损失已有近邻；视觉增强也不能无条件替换成任意EEG扰动 |
| [MCLR，arXiv v1第3.2–3.3节及理论前提](https://arxiv.org/html/2603.22364v1) | 通过条件似然比强化条件区分，并讨论加权目标与CFG的联系 | “把CFG的对齐作用移入训练”“正确/错误条件似然比”本身均不是新贡献。本文不将该预印本标为ICML主会论文 |

这些是针对下一候选的排重，不是声称已穷尽文献、方法等价、其他论文无效，或论文已有指标能自动迁移到我们的EEG任务。没有将别人的实验数值记为我们自己的结果。

## 对公式成立条件的检查

对已知线性测量 `y=Gx+noise`，若 `P_N` 确为G的零空间投影，则 `G P_N=0`。向x加入该空间的补全不会改变预测测量。这是已知前向算子的性质。

当前EEG→视觉系统没有相同的已知图像→EEG算子。若用拟合算子B代替G，只能保证 `B P_N(B)=0`，无法推出 `G P_N(B)=0`。实际上

```text
G P_N(B) v = (G-B) P_N(B) v
||G P_N(B) v|| <= ||G-B||op * ||P_N(B) v||.
```

例如G=[1,0]、B=[1,d]、v=[-d,1]，有Bv=0却有Gv=-d。这里是自行列出的经典代数反例，不是新EEG实验或原创定理。因此，未来若采用学习到的视觉可观测子空间，必须验证前向模型误差、概念外泛化与实际EEG配对作用；不能仅因投影矩阵幂等或有非零零空间就写成“保留真实EEG证据”。

已有D22编码器输入零空间说明的是：某些原始EEG输入差异对固定编码器不可见。它与上述图像/皮层源前向测量的零空间不是同一个对象，也没有识别视觉潜变量的物理可观测部分。

将噪声预测写成解析直通加视觉预测，也有精确的经典关系。对 `z=sqrt(a)*f+sqrt(1-a)*epsilon`，定义

```text
epsilon_prediction = (z-sqrt(a)*D(z,e))/sqrt(1-a)
MSE_noise = a/(1-a) * MSE_visual.
```

改变参数化或时间权重能改变优化问题的条件数和关注区域，不能凭代数等价性证明EEG信息变多或产生新的方法族。[已有完整T噪声诊断](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/reports/ATM_TRUE_FORWARD_EEG_SCORE_RESULTS_20261103_v1.md)还表明噪声MSE与刺激区分会出现反向变化，后续仍须同时保留两者。

## 与当前证据怎样衔接

本轮prior是当前T-only/R4范围下按作者结构训练的系统，不是作者全部66160条单次EEG特征训练协议的复现。150轮对应的更新数、条件质量与公开全训练存在差异。若旁路图片更好，也只能先说明它胜过当前这个固定prior；在宣布架构层面的普遍结论前，还需给强已知prior目标与合理训练预算相同的比较机会。不能看到旁路更好就把原方法整类判无效。

后续训练应区分三件事：修复强基线的参数化/训练匹配、借鉴已知条件对比目标，以及验证EEG特有机制带来的额外收益。只有第三项在完整图片中胜过前两项及冻结ordinary，才有依据推进独立第二方法的贡献主张。不存在单靠更多公式、更多参数或引用更多领域即可保证发表的结论。

当前唯一正在执行的新图片研究仍是[固定2×2实际队列](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/reports/ATM_PRIOR_ON_OFF_ORDINARY_ON_OFF_ACTUAL_EXECUTION_20261104_v1.md)。未因本文自动添加GPU任务、重训候选、扩大搜索或接触第二卡。
