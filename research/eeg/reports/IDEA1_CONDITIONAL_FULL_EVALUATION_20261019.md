# EEG 误差适配：同编码器完整图片比较

仅比较新 patience50 编码器和相同 SDXL512/20步/CFG5/IP1 下的结果。旧 patience20 图片另表保存，不混入此处差值。
一个训练 seed1729；每方法全部十被试、16,600 个 V query、每 query 一张。V 是开发数据，bootstrap 区间不覆盖训练 seed 变异。

| 方法 | 图片/预期 | 状态 | DINO2AFC | SEED | ObjectF1 |
|---|---:|---|---:|---:|---:|
| Frozen | 0/16600 | NOT_RUN_OR_INCOMPLETE | — | — | — |
| RealPaired | 0/16600 | NOT_RUN_OR_INCOMPLETE | — | — | — |
| Conditional | 16600/16600 | COMPLETE_FULL_IMAGE_METRICS_AND_IDENTITY | 0.724959 | 0.130436 | 0.112616 |
| Moment | 0/16600 | NOT_RUN_COMPLETE_LIFECYCLE_BUDGET_NOT_FUNDED | — | — | — |
| Iso | 0/16600 | NOT_RUN_COMPLETE_LIFECYCLE_BUDGET_NOT_FUNDED | — | — | — |
| Clean | 0/16600 | NOT_RUN_COMPLETE_LIFECYCLE_BUDGET_NOT_FUNDED | — | — | — |

## 完整配对差值

尚无同编码器完整配对图片比较。

## 解释边界

- 必需缺项：Frozen、RealPaired、Moment、Iso、Clean
- 五组只训练已有原生投影器8,400,896参数；这不是完整349,188,096参数IP分支训练或完整ATM公开复现。
- 固定配方为一个完整132,300训练query pass，真实覆盖每个被试—刺激一次；完成该预算不自动证明收敛或容量足够。
- Conditional 的原生/投影相对能量判据通过，仍不证明模拟分布等于真实分布；强分类器仍能区分两者。
- DINO2AFC不是对象准确率。ObjectF1使用COCO+gender82类及原空检测约定，需要同时看检测覆盖和真实图片。
- 尚缺普通噪声或RealPaired控制时，不能把增益归因为独立EEG误差机制；缺失/未收敛/失败不能填零或写成已完成负结果。
- 正确、置换、缺失EEG控制及完整盲评材料仍是任务要求；人工评分未实施不能声称画质非劣。
- 官方200刺激曾暴露；本表不宣布SOTA、未暴露最终确认或训练seed稳定性。
