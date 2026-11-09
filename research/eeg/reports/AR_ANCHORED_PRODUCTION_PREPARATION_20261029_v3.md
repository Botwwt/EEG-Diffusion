# EEG anchored attention：三臂固定训练与生产入口准备

此记录是实现和准入前证据，不是正式训练或图片收益。此前 v1/v2、未训练的 concat-prefix 原型与旧 ordinary 全部保留。当前问题是：在相同 pooled EEG 条件 (c) 下，保留刺激自身的未池化 EEG 差异，能否改善真实生成图片。

锚定分支让原图像注意力独立归一化，再叠加同一已有 Q/K/V 对 EEG memory 与重复 (c) 的响应差。它不会因增加 EEG token 而改变图像注意力的 softmax 分母；不新增可学习参数。每层使用原 conditional LayerNorm 与实际 Q/K 头内归一化，不能把它简化为原始线性 K 的协方差公式。memory 当前是 63 个电极、各一秒窗口的表征，不能写成已实证的时间层级。

固定三臂为 native_global、paired_anchored、donor_anchored。训练集是 sub-01 的全部 13230 刺激、1323 概念，使用真实概念排除 OOF EEG bank；EEG 编码器和原生 4096 码 VQ 冻结。三臂从相同公开 VAR 骨干与相同 seed 的原有 EEG projection 开始，训练全部 310308096 个原有 VAR 参数。donor 只在训练时以同概念另一刺激的 (H_d-c_d) 替换局部差异，保留自己的 (c) 与图像目标；推理全部只使用自己的 EEG，不选择 donor、不读标签或图库。

每臂固定十个完整 T epoch、8270 次 AdamW 更新、132300 次 query 暴露，只保留固定末轮模型作为效果比较。逻辑 batch16 由八个 micro2 累积，末批14由七个 micro2 累积；每个 micro 的 mean token CE 乘以该 micro query 数除以逻辑 batch query 数，再反传，一次统一裁剪和 AdamW 更新。680 个码位置等权。micro2 是正式拟合前定义的随机配方，不声称与随机 physical batch16 逐位等价。三臂使用相同分段、query 顺序、条件 dropout 与 DropPath RNG 约定。

核心 v3 仅增加 native_global 的 `memory_bank=None` 接口，micro2 数学沿用 v2。37 项核心 CPU 检查已通过，包括 actual reduced author VAR 的 full/micro 梯度比较、无 memory native 与旧真实 bank native 的完整训练状态/RNG 一致、donor 权限及完整 optimizer-step 恢复。新增 16 项生产边界 CPU 检查覆盖 DRAFT 在创建输出/CUDA 前退出、实际 OLD profile 元数据契约、tuple 身份 join 和不可变日志前缀。首次草稿错误要求 canonical EEG row 含 `subject`，经实际 schema 复核已在拟合前修正；主体范围来自固定 sub-01 campaign，row join 使用实际存在的 array_row、stimulus_id、concept_id。

OLD 的真实预训练全形状 logical16/micro2 profile 已闭合：三个臂各三次临时更新，全部丢弃并恢复初值；GPU allocated 峰值 7963361280 字节、额外 CPU 峰值 6505562112 字节。真实 native 模型/203 份 Adam state/cursor/CPU、CUDA、shuffle RNG 的保存、加载、恢复全部一致，payload 为 3725863150 字节；序列化8.146秒、mmap加载0.050秒、恢复2.093秒。另真实 B1 接口18项检查通过，关闭分支与 (H=c) 的 logits/像素逐位回退、真实 Q/K 归一化 cache/full 最大 logit 误差约 (7.63\times10^{-6})。这些是工程检查，临时更新不是正式候选模型，工程图不是效果图。

生产入口为 [producer](../train_AR_anchored_complete_T_GPU_20261029_v1.py)、[finite queue](../queue_AR_anchored_complete_T_CPU_20261029_v1.py) 和 [纯 CPU 契约](../AR_anchored_training_contract_CPU_20261029_v1.py)。[三臂协议 v3](../configs/AR_anchored_training_DRAFT_20261029_v3/protocol.json) 与 [准入模板](../configs/AR_anchored_training_DRAFT_20261029_v3/admission_TEMPLATE_OLD.json) 均保持 DRAFT，不自动启动。计划 NEW 运行 native_global，OLD 顺序运行 paired、donor；每张卡仍必须绑定自己的真实预训练 profile、共同初始 tensor digest、实际输入/source SHA 和剩余输出预算。native 不读取或要求 3.4GB 局部训练 bank。

每个 arm 只保留两个新建 resume payload 和一个固定 final，日志和 commit/retirement 元数据全部保留。写下一个断点前验证当前断点，只允许退休本 store 的旧 previous，当前断点始终保留；写入中断保留 pending 和上一个 current，拒绝自动清理/重试。任何旧结果和唯一旧检查点均不在退休范围。恢复先验证从 update1 到当前断点的所有已提交 attempt 日志前缀，未提交的失败日志保留且不冒充完成工作。任何 micro 中途失败都回到上一完整 AdamW 更新；不保存半批梯度或把已消费一半的 RNG 当作完整步骤。

最终 checkpoint 路径为 `M/AR_anchored_training_20261029_v1/{arm}/final_model.pt`。其 `decoder` 字段是原 VAR state_dict，无 wrapper 前缀；同时记录 arm、admission SHA、epoch10、8270更新、132300 query 暴露、EEG/VQ bank completion SHA、共同初值和训练后 tensor digest。生成端须核对该 arm 完成回执和 final 文件 SHA，不能拿临时 profile 权重出研究图。

图片规则在拟合前已经固定：native256、十尺度680码、guidance1.5、top_k600、top_p0、more_smooth=False、每 query 单张无损 PNG，全部臂共享固定 SHA 派生 query seed。实际 Dcheck400 身份/400目标图库已在 CPU 闭合，80条阶段仅按事前明确损害规则停止，否则三臂都扩400并保留80前缀；400完整前不读部分效果。paired 必须相对 native 和 donor 都达到 DINO 均值至少 +0.01、描述区间下界大于零、目标余弦均值正向、同概念 Top1 均值不降。该 DINO 快筛不是完整 SOTA 评价。

解释上限保持明确：(H=c) 回退只针对**同一当前 VAR 权重**，训练后的 paired 关闭局部分支不会恢复另训的 native 模型或原公开 VAR 图片。相同参数量不等于相同函数容量。结构零点不推出 (E[R\mid c]=0)、条件信息增量或神经不确定性校准；same-concept donor 还会破坏 (c) 与局部统计的兼容性，paired 胜 donor 仅支持此控制下的刺激配对作用，不能单独排除 (c) 可预测的局部矩或容量效应，也不是精确 CRT/knockoff/因果检验。

本次实现没有启动远端 GPU、正式拟合或新增研究图片。正式执行由主控制器在两台主机实际资源和来源闭合后单独派发。
