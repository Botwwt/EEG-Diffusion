# 新父编码器的匹配 ordinary：真实训练、完整 T 诊断与后续目标

更新：2026-11-09T06:36:08.369045+00:00。新匹配 ordinary 的训练已开始；本轮没有新图片效能结果，也没有第二项创新或 SOTA 结论。

## 1. 已完成的输入与训练工作

- GPU5097501 完成全部13,230条 T 图像的原生 VAE mode 缓存，867,041,408字节，52分块；程序792.80秒、峰值allocated1.03GB。缓存仅在共享存储，不下载到D盘。
- CPU5097655独立复核全文件SHA、全部13230行/52块、有限性及两张原图的CPU VAE重放，45.61秒通过。最大逐元素误差分别0.0006292和0.0002350，满足事前固定的 atol=rtol=0.0005 联合容差；没有宣称CPU/GPU或旧缓存逐位相同。
- 新匹配ordinary使用原525,568参数头、半径0.25、原epsilon MSE+0.1 token角误差、AdamW1e-4/weight decay0.01、batch16、单seed1729。sub-03的13230条真实上游OOF条件及各自输出affine，固定3遍、2481更新/39690暴露，末批14。它是单被试匹配基线重建，不是十被试训练或新创新方法。
- GPU5097673完成首4次真实更新/64暴露，峰值allocated11.93GB。CPU5097685独立解码模型、Adam、RNG结构、完整已提交覆盖和四个loss，检查点SHA `1ab7257c8ac48e1291d621bde545df096126ee2ce172fd7326948bf099b1db1a`。没有独立CUDA训练轨迹重放的主张。
- 同一训练从该断点续跑，初始接续作业5097843。当前回执日志最新953/2481更新，已保存断点896，已进入第二遍。有限队列最多五段，每段20分钟，数值程序在1000秒附近安全保存；每段实际CPU断点检查通过才提交下一段，afterok依赖保证最多一张GPU。全量拟合、后续分段提交与最终校准/图片评价须各以实际终态为准，不能用本次挂队代替。

旧ordinary权重、旧父模型及历史结果均不覆盖。新head训练会遍历整个T，因此组合不能称全链路OOF。

输入权限措辞勘误：训练程序把包含全部16540训练F的已有容器读入CPU，再仅取T的13230行送入训练。程序中 `no_D_V_or_official_test_reads` 应理解为不把这些划分送入训练前向/loss/选点，不能理解为容器字节零访问。官方200图不在该容器中。详见[勘误](../configs/PKU_matching_ordinary_provenance_wording_erratum_20261109_v261.json)，数值源码/目标/数据索引未改。


**实际接续分支复核：2026-11-09 06:38 UTC后**：首分段5097843完整退出0:0，在1303更新/20846暴露处保存；独立CPU模型/Adam/RNG/全覆盖检查通过。第二分段5097979已真实RUNNING并从相同SHA断点恢复。自动提交分支已实际发生，完整2481更新及最终校准/图片效果仍未完成。

## 2. 第512次更新的完整T诊断

CPU5097880在固定的已保存512更新/8192暴露状态上，保留独立检查点副本，对全部13230条T计算两臂FP64完整图库及同概念十图检索，21.51秒完成。

| 指标 | 原生OOF条件 | 当前ordinary | 含义 |
|---|---:|---:|---|
| 与正确F匹配余弦 | 0.105840 | 0.183543 | 全13230条均提高，尚非实际图像指标 |
| 同概念Top-1 | 31.6478% | 31.3152% | 324条由对变错、280条由错变对，净少44条 |
| 全13230图库Top-1 | 0.5291% | 0.4762% | 18条由对变错、11条由错变对 |
| 同概念最难负例间隔均值 | -0.020035 | -0.019954 | 均值改善不保证阈值附近判对数增加 |
| 原始条件均值占二阶能量 | 0.03731% | 4.87581% | 公共分量增长，不足以单独宣称塌缩 |

8192条已暴露、5038条尚未轮到的T分层均保留。后者也属于允许训练T，不能包装为独立确认集。该诊断不读D/V效果、不改变训练、不挑中间epoch、不触发单指标停止。

全体匹配余弦增加与Top-1下降可以同时发生：正确相似度和最难负例相似度可以一起移动；平均间隔由所有样本决定，Top-1主要取决于各自间隔是否跨零。这里的成对翻转是实测，不能据此认定最终图片一定下降，也不能据token loss下降宣布图片成功。

## 3. 旧日志对下一次设计的约束

旧连续约束实例的第8遍共664次尝试、只接受25次；最后接受在第323次，第5至第8遍共332次尝试没有新增参数更新。第6与第8遍全部13230行/39690值及NPZ逐位相同。这解释了该实现继续追加epoch为何没有变化，不证明EEG没信息或全局最优。528次尝试存在可行候选但不满足要求的响应下降，111次未找到登记的可行候选；不能将它们全归为QP数值失败。

下一项设计需要检验“生成中配对EEG证据是否增加”，并重新评估把原F图库每query间隔当作不可下降硬约束的必要性。当前ordinary为完整参考继续运行。后续先做同噪声/同query的实际图片比较，保留DINO、语义、结构及同概念区分，再决定新目标。没有因此自动新增第二方法或重跑旧约束配方。

## 4. 已核对的近邻与信息权限

教师随学生调整并非空白：[CVPR2023 Cross-Modal KD](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Enhanced_Multimodal_Representation_Learning_With_Cross-Modal_KD_CVPR_2023_paper.html)讨论弱教师捷径，[C²KD CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)采用定制和双向蒸馏。因此，仅联合训练视觉F与EEG、或增加对齐loss，不足以支持原创性。

去噪能量用于判别也已有[Diffusion Classifier ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html)；音视频的[CMMD ECCV2024第2.2节](https://www.microsoft.com/en-us/research/uploads/prod/2024/10/ECCV_2024_avgenl.pdf)已用错配负例构造对比扩散目标。移植去噪NCE公式不能直接称第二项创新，单时刻MSE差也不能冒称精确后验对数比。

[JMVR预印本v1](https://arxiv.org/html/2603.19667v1)的主配置同时输入EEG与文本，表1另列EEG-only的JMVR*。其主配置表值不能直接作为我们EEG-only的同权限竞争结论；多尺度与按步门控本身也已有明确近邻。以上文献审查不代表复现其结果或确认其结论。

本轮可执行源：[训练](../PKU_train_matching_ordinary_GPU_20261109_v259.py)、[断点检查](../PKU_audit_matching_ordinary_CPU_20261109_v260.py)、[有限接续](../PKU_continue_matching_ordinary_GPU_20261109_v261.py)、[全T诊断](../PKU_probe_matching_ordinary_T_CPU_20261109_v262.py)。
