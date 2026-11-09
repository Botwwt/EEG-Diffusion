# Conditional 完整 V 生成与评分前配对核查完成

截至 2026-10-19 14:16 UTC，Conditional 已生成全部 16,600 张 V 图片，覆盖十名被试、每名 1,660 个 query。评分前的完整来源与保存核查通过，失败为零。现已进入原定评价特征提取，尚无正式逐 query 图片分数或配对增益结论。

## 实际完成范围

| 项目 | 已核对的实际结果 |
|---|---|
| 生成范围 | 完整 V，16,600 query，10 被试各 1,660；每 query 一张，无新增种子或输出复用 |
| 生成设置 | SDXL base 1.0，512，20 步 Euler，CFG 5，IP 1；原固定查询种子和运行配置不变 |
| 加载权重 | Conditional 最终投影检查点 SHA `712573c93a09545e2b92dcb274a30d7eb433abebc3b47ca4a11283c3c2c10ea6`；逐行部署张量 hash 与检查点一致 |
| 生成清单 | SHA `b75ce947844df92629594427dd577b8080a3c791799d99b274fa83289a90efad` |
| 采样成本 | 83,000 次 UNet 批次调用、664,000 次样本级求值；每 query 40 次样本级求值，包含 CFG |
| 计时 | 生成回执 14,635.14 秒；GPU 包装作业 14,640.13 秒，约 4.067 GPU 小时，退出码 0 |
| 评分前核查 | 16,600/16,600 行通过，十名被试各 1,660 行；实际 EEG 原生条件缓存 hash 全部核对 |
| 保存检查 | 生成路径唯一；图片字节 hash、RGB/512×512 解码通过；完全相同 JPEG hash 的重复组为零 |
| 目标与设置 | 逐行刺激、概念、数组索引、目标图片身份、编码器、种子、运行配置和实际加载投影已核对 |
| CPU 核查时间 | 审计器 27.75 秒；驱动阶段计时 30.00 秒，不使用 GPU |
| 当前未完成 | 特征提取、完整逐 query 评分、低层指标、评分后目标路径核查、正式比较报告 |

生成阶段没有读取目标图像或目标视觉特征作为推理条件。完整核查还绑定了固定真实图像图库的实际文件 hash。本阶段核查验证发布元数据、数组与刺激归属，不能替代独立采集事件重放；图片能够解码也不证明对象正确，零字节重复也不等于没有语义上的相似输出。

## 可追溯证据

- [完整生成回执](../audit/idea_interface_Conditional_V_generation_completion_v1.json)
- [实际 GPU 作业及计时](../audit/IDEA1_Conditional_generate_job.json)
- [完整评分前来源/保存核查摘要](../audit/idea_interface_Conditional_V_identity_before_v1.json)
- [原始失败记录：空列表](../audit/idea_interface_Conditional_V_identity_before_failures_v1.json)
- [阶段账本记录、逐行核查 CSV 的服务器路径与 SHA](../audit/idea_interface_Conditional_completed_full_generation_progress_v1.json)
- [此前全部固定 T40 图片、视觉负面记录和诊断信号](IDEA1_CONDITIONAL_T40_20261019.md)

完整图片、生成清单、逐行来源 CSV 和特征块均留在服务器，本地仅同步小型回执和报告。逐行来源 CSV 的 SHA 为 `0983d3046d981dfbd3f67a60d8f5a75255ebfab9f684689b5aa1562c13bd6ce3`。

## 结论边界与下一阶段

本次核查的 `score_rows_checked=0`：评分尚未完成，所以还不能声称评分使用的目标路径已经逐行核对。原队列将在完整特征和评分完成后再次运行核查，检查每条评分的生成路径、目标路径、刺激/概念、种子与 EEG 来源。

当前实际特征提取进程为 PID 120470，使用原 `extract_image_features.py`。DINO、EffNet、CapSim 与 ObjectF1 的输入覆盖完整 16,600 张；阶段进度仅表示对应特征的处理数量，不表示整体评价已完成。没有改变特征模型、指标方向、步数、分辨率、种子或评价集。

本阶段完成仅证明全量图像产物和评分前来源链条齐备。T40 中的明显错类记录继续保留，Conditional 与 RealPaired 的正式增量尚待两方法的完整结果；Clean、Iso、Moment、Frozen、EEG 干预、质量和独立确认仍需各自实际证据。不能据本阶段宣布 idea 成功、全部失败、达到 SOTA 或研究目标完成。
