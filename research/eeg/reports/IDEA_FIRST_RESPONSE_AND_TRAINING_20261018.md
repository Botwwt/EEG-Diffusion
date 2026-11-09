# 完整响应检查通过，已开始实际图片适配训练

2026-10-19 05:13 UTC 的实际状态。一个训练 seed1729，总上限40 GPU小时；全部大文件留在服务器。

## 已完成的独立生成器响应检查

全部十被试、83个 Dcheck 概念、8,300个 query 已完成。每个 query 使用预先固定的一组 latent 噪声和一个时间步，各模拟组使用四个既定条件抽样。实际124,500次 UNet 样本计算、7,800次 forward，运行2,578.90秒。

比较同一 noisy target 下，模拟条件与真实 EEG 条件经冻结生成器后的欧氏范数 energy score，越低越好。以下为 Conditional 减去对照，区间按被试和概念交叉重采样；十个被试的差值方向均为负。

| 对照 | 配对均值差 | 95% 区间 |
|---|---:|---:|
| Iso | −3.722951 | [−4.087413, −3.377276] |
| Moment | −1.803536 | [−2.066195, −1.544242] |

原生条件、初始投影器和完整冻结 UNet 三项相对判据均已通过，支持进入匹配图片实验。这些分数没有把输入向量归一化；也不是图片指标、重建准确率或完整采样轨迹的等价性检验。目标图像加噪只用于这个有真值的诊断，正式生成仍只读取 EEG。

二样本 MLP 的 Dcheck AUC 仍约0.986，说明模拟与真实条件之间还有明显差异。因此不声称已经精确拟合观测通道、证明 KL 很小或获得生理置信度。当前结论只涉及相对保真度优势。

## 正在运行的真实训练

已启动 `IDEA1_RealPaired_train`，实际 worker PID104727；05:13:47 UTC 已核对96次优化更新、1,536个训练 query，分支梯度非零。该时间点尚无新版本正式图片。

- 训练条件：真实两折折外 EEG 预测与本图配对；不用真图条件替代 EEG。
- 模块：从同一公开预训练权重开始，仅训练原生 ImageProjection 的8,400,896个参数，冻结 IP 注意力、主 UNet、VAE 和文本编码器。
- 目标：SDXL 检查点对应的图像 latent 噪声预测 MSE；相同图像噪声、时间步、数据顺序和条件 dropout 规则用于所有适配组。
- 固定训练量：全部十被试132,300个 T query，一遍完整 query pass，batch16，共8,269次更新；不根据 V 图片或最后测试挑检查点。完成固定训练量不自动证明收敛。

随后按相同科学配方执行 Conditional。执行优先级 v2 已将完整 RealPaired 生命周期作为先行承诺的匹配参照，避免额外 Frozen 图片预算推迟候选训练；候选优先出图。每个获完整预算的版本仍必须完成全部16,600个 V query、一张图/query、全部逐 query 指标、配对核查与报告。

Frozen、Clean、Iso、Moment、Conditional、RealPaired 以及 EEG 依赖、质量和确认要求全部保留。尚未执行或预算不足的控制仍是未确定；不能把部分组完成当作完整机制成立或 SOTA。当前尚未建立新 idea 的实际图片提升。

证据：`audit/idea_first_full_UNet_response_summary_v1.json`、`audit/idea_first_interface_actual_training_started_v1.json`、`configs/idea_interface_campaign_plan_v2.json`。完整响应逐 query CSV 与训练日志在服务器。
