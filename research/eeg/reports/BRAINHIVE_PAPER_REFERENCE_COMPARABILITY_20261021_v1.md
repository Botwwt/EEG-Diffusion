# Brain-HIVE 原生 R80 与论文表值的描述性参照

## 2026-10-30：加入CogCapPro强参照，先核对指标和实际默认代码

本轮阅读两份原始论文，下载作者仓库`cf3fc5bc6e80106f767f902de7f5a2bf78972f60`全部34份Python/配置/Markdown/文本源，共254519字节；Git blob SHA1和逐文件SHA256均实际核验。另从已闭合ordinary实际执行的source manifest核对评价runtime大小/SHA、原utils_eval加载位置和L14断言，确认本地检查的原源码与真实执行链相连；初版静态读数保留，v2补齐这项来源核验。只解析文本/AST，没有导入执行作者代码、下载权重、启动模型或读取当前候选图片效果。[完整源码范围](../audit/HIVE_external_reconstruction_benchmark_scope_20261030_v1/actual_all_author_text_source_manifest.json)、[实际静态与来源核对](../audit/HIVE_external_reconstruction_benchmark_scope_20261030_v1/actual_metric_scope_and_static_source_audit_v2.json)。

| 参照 | 公开数字与范围 | 当前可比较边界 |
|---|---|---|
| 本轮旧ordinary | 10被试、2000 query、单1729；CLIP-L/14为0.808575，SSIM为0.353079 | 已闭合实际结果；源码用特征Pearson相关及严格小于的二选一识别计数，非CLIP余弦均值，也非DINO |
| CogCapPro原论文表III | SSIM0.398、Inception0.779、CLIP0.830、SwAV0.553 | 原文描述十被试数据及一般五次重复，但该重建行的精确被试/seed/图片聚合和CLIP评价backbone尚未独立落实；不得据训练backbone推定评价backbone或直接相减 |
| 单被试CogCapPro改编表2 | 单被试200目标、十seed；CLIP-L/14为0.870、CLIP-H/14为0.903、SSIM0.409，5步/CFG0 | 不可将H14分数当L14，也不可将200×10seed当十被试2000条；是否同本轮sub-01、评价预处理与生成重复统计仍未核对 |

来源：[CogCapPro §IV-A/表III](https://arxiv.org/html/2603.12722v1)、[单被试改编 §4/表2/局限](https://arxiv.org/html/2605.23996v1)。未取得两文逐query原分数，不计算跨论文配对区间或宣称同协议排名。较高SSIM和语义表值用于识别应复现的强参照，不是已测得的同成本差距。本文旧ordinary均值仅从已闭合author_completion读取，未再次运行评价器或重算2000条。

当前作者代码还需要明确的复现选择：`data/eeg.py:51`在被试内返回同一test_loader作val和test；`align/main.py`使用`diffusion_clip_embeddings_test.pt`及val_loader构建对齐验证，`SimpleAlignPipe.train`按验证余弦保存best。该静态路径有测试选择权限问题，不能作为T内校准方案直接运行；不据它声称论文历史运行发生相同选择。被试内主编码器默认早停看total_loss，不误写成全部阶段均按test accuracy选best。另有默认`model_type=diffusion`，而论文介绍STH-Align；`batch_generate.py:168`显式15步，与上述单被试改编5步及本轮4步不同。34份文本中未定位到可确立表III完整图像评价定义的实现，仅有EEG检索评价；这是评价来源未落实，不推断作者从未实现过它。

后续如实际进入此基线，须分别登记作者默认与同成本配置、独立T内校准及固定单seed/最终检查点，完整核验权重、输入和正常EEG推理权限。更改默认步骤或划分必须报告，不能一边改协议一边称原表严格复现。本次只读源码与参照核对**没有准入新基线训练或改变当前五组实验**，ordinary收益、原停止实例与全部数值失败继续保留。

## 2026-10-30：ordinary完整七项指标与论文其他配置的描述性参照

重新从已闭合2000条原始CSV计算，并核验原接收SHA和逐query身份/汇总一致；无新训练、出图或评价器前向。对照[原文表10–13](https://arxiv.org/html/2602.07495v1)，四种配置全部保存在[28项结构化参照](../audit/HIVE_ordinary_paper_configurations_reference_20261030_v1/actual_full2000_seven_metric_comparison.json)。下表保留匹配配置和语义双路配置：

| 指标 | 论文匹配三路配置 | 论文H14+B32 | ordinary原2000条 |
|---|---:|---:|---:|
| PixCorr ↑ | 0.195 | 0.187 | 0.208655 |
| SSIM ↑ | 0.336 | 0.339 | 0.353079 |
| AlexNet(2) ↑ | 0.843 | 0.836 | 0.840332 |
| AlexNet(5) ↑ | 0.905 | 0.908 | 0.905143 |
| Inception ↑ | 0.756 | 0.783 | 0.761226 |
| CLIP ↑ | 0.808 | 0.814 | 0.808575 |
| SwAV ↓ | 0.554 | 0.547 | 0.554004 |

匹配配置上的接近或提高，不等于超过论文所有配置。H14+B32在多项语义指标上仍有更高表值；表2另有CognitionCapturer-Depth的SSIM参考0.370。它们尚不是本轮实际复现的配对比较。逐指标最优包络仅用于说明差距，不能拼成一个真实模型。

作者十套重建指标均值与我方每query一图、组件训练覆盖和已曝光benchmark等边界保持。没有作者逐query数据，不能给出跨论文配对区间。三位小数的半末位敏感性是假设就近取整的精度检查，不是置信区间；AlexNet(5)/SwAV相对匹配表值的微差不可据此当成可靠胜负。仍无SOTA结论。

---

记录于 2026-10-21（America/Los_Angeles）。只绑定已经闭合的原生 R80 汇总分数与已发表表值；没有重新生成图片、重新评分、训练或读取尚未闭合的组件 V 分数，也不据此选择组件、强度、检查点或生成种子。

## 参照版本与配置

参照为[论文 v1 的表 2 EEG Ours 和表 13 Ave](https://arxiv.org/html/2602.07495v1)。二者对应 **H14+B32+VAE**，不是表 10 的 H14 单分支。实际 `prior_H14` 目录名不能代替配置：绑定的原始 `fusion_encoder/config.json` 明确包含 B32=512、H14=1024、VAE=1024，SHA 为 `4f72dcc0d21583817047b5987a5589213abecbb706a8c261431e298650a422b3`。

我方分数来自已经完成的 published 原生模型：10 被试、每被试完整 16540 张训练图、R80 测试、200 张候选、共 2000 query、每 query 一张图。它不等于当前组件的 protected T/R4 任务；后者必须继续完成同条件原生、普通、风险三组比较。

## 七项论文表值

“差值”是我方减论文，SwAV 越低越好，因此该行正差也表示我方较差。论文只保留三位小数；这里没有作者逐 query 数据，不能计算配对区间或宣称差异显著。

| 指标 | 论文表值 | 我方原生 R80 | 差值 |
| --- | ---: | ---: | ---: |
| PixCorr ↑ | 0.195 | 0.189344 | -0.005656 |
| SSIM ↑ | 0.336 | 0.328023 | -0.007977 |
| AlexNet(2) ↑ | 0.843 | 0.834430 | -0.008570 |
| AlexNet(5) ↑ | 0.905 | 0.900867 | -0.004133 |
| Inception ↑ | 0.756 | 0.742595 | -0.013405 |
| CLIP ↑ | 0.808 | 0.796234 | -0.011766 |
| SwAV ↓ | 0.554 | 0.564242 | +0.010242 |

这是数值参照，尚未逐项对齐论文统计协议。源码返回八项指标，论文这两张表只列七项：EfficientNet 距离没有论文表值，不补填；DINO2AFC、ObjectF1 和 SEED 也不冒充论文主表指标。论文正文 §4.2 的 PixCorr 写作 0.186，与表 2/13 的 0.195 不一致；本报告明确采用表值并保留该差异。

## 统计协议与解释边界

发布的 `build_reconstruction.py:279–299` 分别评价十套重建后平均指标，既不是像素平均，也不是 best-of-10。我方按用户要求只生成一张。论文 v1 没有明确主表训练 seed 的聚合数量；当前 `train_clip_intra.sh` 的三个 seed 配置是 SynCLR，不能据此认定 H14 完整组合主表一定平均三个训练 seed。旧记录中“作者多 seed 原表”的表述应按此勘误理解；已训练源码和历史回执不改写。

本轮 25 epochs、batch1024、学习率 5e-4 与论文描述对应；warmup0 遵循发布 YAML，而论文文字为10步。实际依赖版本及已有兼容修复另有记录，不能称环境与源脚本逐项相同。

这些原生结果在数值上接近论文参照，同时仍低于该表；它们不证明当前组件增益、对象正确率或 SOTA，也不消除官方测试已曝光的限制。当前行动仍是闭合 R4 三组图片比较和冻结功能诊断，没有因此重训、扩大模型、减少正式数据或改换测试协议。

## 可复核文件

- [结构化证据与输入 SHA](../audit/BrainHIVE_paper_reference_R80_descriptive_actual_20261021_v1.json)
- [自动计算的完整精度差值表](BrainHIVE_native_R80_paper_reference_descriptive_20261021_v1.csv)
- [已有原生分数](BrainHIVE_native_R80_comparison_20261020_v10.csv)
- [已有原生图片与逐 query 评价闭合](../audit/BrainHIVE_native_R80_full_closure_20261020_v10.json)

表值由两名代理分别核对原论文；差值由已保存 CSV 计算，没有新增评价结果或候选配置。
