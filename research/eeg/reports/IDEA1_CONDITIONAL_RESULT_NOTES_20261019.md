# Conditional 首个完整图片结果：尚无同协议增量结论

Conditional 已完成固定训练、全部十名被试的 16,600 张 V 图片、四类评价特征、完整逐 query 评分、低层指标，以及评分前后两次来源核查。评分后 16,600 条目标路径与生成路径均通过核对，失败为零。当前只有 Conditional 完成这一套新编码器下的全量评价；相同设置的 RealPaired 已进入全量生成，尚无完整配对差值。

## 实际数值

以下为一个固定训练 seed 1729 的结果。区间按被试和概念重采样，不覆盖训练 seed 变异；完整真实图像图库为 1,660 张。

| 指标 | 均值 | 95% 被试/概念区间 |
|---|---:|---|
| DINO2AFC | 0.724959 | [0.695030, 0.753469] |
| DINO cosine | 0.262566 | [0.251116, 0.274323] |
| DINO Top1 | 0.009940 | [0.006143, 0.014459] |
| DINO Top5 | 0.043012 | [0.029934, 0.057530] |
| EffNet | 0.078140 | [0.065702, 0.089548] |
| CapSim | 0.200553 | [0.188918, 0.212338] |
| ObjectF1 | 0.112616 | [0.093369, 0.133325] |
| SEED | 0.130436 | [0.117282, 0.143336] |

低层补充指标：PixCorr_RGB 0.068224，SSIM_Y 0.282315，平均 RGB 绝对误差 0.115082。全部 16,600 张存在且可解码，缺失、损坏/字节不符和重复 JPEG hash 组均为零。这些检查不证明对象正确或感知画质非劣。

## 数值表示什么

DINO2AFC 是生成图对正确原图相对于其他图库图片的相似性偏好，不是 72.50% 的对象重建准确率。DINO Top1/Top5 是该 1,660 图图库内的图像检索指标，也不是对象类别准确率。当前骨干为 DINOv3 ViT-S/16，名称中的 2AFC 不表示使用 DINOv2。

ObjectF1 使用固定 COCO+gender82 检测类别和原空检测约定，不是 THINGS 全概念识别准确率。原图空检测计数为 50/16,600 个 query，生成图为 8/16,600；这是逐 query 计数。blank proxy 为零也不能替代人工质量评价。人工盲评尚未实施。

本实现的 SEED 是逐 query 的 EffNet、CapSim 和 ObjectF1 三项算术平均，随后进行汇总。见实际 [评分入口](../score_full_manifest.py) 和 [锁定指标实现](../locked_metrics.py)，没有为本次结果修改指标。

指标自检回执：原图自配对 DINO2AFC 为 1.0，随机目标标签为 0.502356。后者使用 seed1729 从图库有放回抽取标签，不是严格双射置换，也不是打乱或移除 EEG 的生成实验；不能据此声称已证明 EEG 依赖。

## 成本、来源与保留范围

Conditional 自身四段 GPU 作业包装计时合计约 8.76 GPU 小时，包含训练、T40 诊断出图、全量生成和评价特征提取。完整特征提取回执为 8,027.31 秒，GPU 作业包装为 8,030.30 秒。前置编码器、OOF、验证、profiling、历史与失败作业继续计入全局 40 GPU 小时账本，不能只用本方法成本代替总成本。

完整图片、特征块和 16,600 行指标/来源 CSV 留在服务器。小型摘要、每名被试结果、回执和此报告同步到本地。

- [完整指标及每名被试摘要](../audit/idea_interface_Conditional_V_metrics_summary_v1.json)
- [低层指标和文件检查](../audit/idea_interface_Conditional_V_low_level_summary_v1.json)
- [评分后全部16600条来源核查](../audit/idea_interface_Conditional_V_identity_after_v1.json)
- [全量特征完成回执](../audit/idea_interface_Conditional_V_features_completion_v1.json)
- [完整阶段、成本与产物 SHA 绑定记录](../audit/idea_interface_Conditional_completed_full_evaluation_progress_v1.json)
- [自动生成的统一比较表快照](IDEA1_CONDITIONAL_FULL_EVALUATION_20261019.md)

## 当前结论

目前只能确认本版本已有完整且可追溯的图片评价，不能据此宣布适配成功、达到 SOTA 或独立机制增量。原先固定 T40 中的明显错类记录继续保留。完整 V 图库已建成并通过实际浏览检查；固定顺序的首个橡果 query 生成了棍状物，DINO2AFC 仍为0.964436，说明相对图库偏好不能代替对象正确性。此次局部目视观察不冒充全部 V 的人工判断；详见[完整图库与浏览核查](IDEA1_CONDITIONAL_GALLERY_QA_20261019.md)。

下一项正式证据是同一新编码器、EEG、查询种子、SDXL 设置和采样成本下的 RealPaired 完整结果及配对差值。Frozen、Clean、Iso、Moment、正确/置换/缺失 EEG、质量非劣和独立确认仍需各自实际证据。缺项不填零，不写成被否定，也不通过调整评价或反复训练来兑现目标。当前结果不证明模型容量不足。
