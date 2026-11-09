# RealPaired 完整 V 生成与评分前核查完成

本报告记录 2026-10-19 20:39 UTC 的阶段快照。RealPaired 已于 2026-10-19T20:36:59Z 完成全部 16,600 张 V 图片，十名被试各 1,660 个 query。生成作业退出码为0；评分前来源核查全部通过。流程已进入原定评价特征提取，尚无 RealPaired 的完整逐 query 图片分数或与 Conditional 的图片增益结论。

| 项目 | 实际结果 |
|---|---|
| 训练覆盖 | 一个固定 seed1729，完整132,300训练query，全部十名被试；8,269次更新，采用预先规定的最后检查点 |
| 训练范围 | 仅适配已有预训练原生投影器8,400,896参数；图像注意力及UNet等保持冻结，不代表完整ATM/IP分支复现，也不证明收敛 |
| 正式生成 | 完整V16,600query，每query一张；没有新增种子或复用输出 |
| 生成配置 | SDXL base1.0，512，20步Euler，CFG5、IP1；原查询种子、编码器及运行配置不变 |
| 投影检查点 SHA | `d925c887948b0edccb2988d80fc20b791066c2e0b20ed4a7aa1b2128653c3359` |
| 生成清单 SHA | `2c06cd813b78ab2c5ff8d330c488f40cfd1a16dcedaeabe74418eb26efc90c22` |
| 采样成本 | 83,000次UNet批次调用，664,000次样本级求值；每query40次样本级求值，包含CFG |
| 计时 | 生成回执14619.76秒；GPU包装作业14624.35秒，约4.062GPU小时 |
| 评分前来源核查 | 16,600/16,600通过，十名被试各1,660；实际EEG原生缓存hash全部通过，失败0 |
| 保存检查 | 图片字节hash及RGB/512×512解码通过，重复生成JPEG hash组为0 |
| CPU计时 | 评分前审计30.11秒；跨方法输入对齐核查1.01秒 |
| 尚待完成 | 全部评价特征、逐query评分、低层指标、评分后目标路径核查及完整配对比较 |

生成推理未读取目标图像或目标视觉特征。原始图片、完整生成清单、逐行核查CSV和特征块均留在服务器，本地只保存小型回执、代码和报告。训练、历史及失败作业继续计入全局40GPU小时账本，不以本表的生成成本替代总成本。

## 与 Conditional 的完整输入及成本对齐

独立CPU核查实际连接两份完整生成清单的全部16,600个query，逐行核对37个字段，均相同：包括刺激、概念、数组行、真实EEG来源、原生和部署条件hash、编码器检查点、目标图像身份、查询种子、初始图像噪声、生成器版本、步数/CFG/IP和样本级NFE。两方法都使用同一个新编码器的实际原生EEG条件，各自加载已核对的投影检查点。

两方法的生成路径完全不重叠；同query完全相同的JPEG hash为0/16,600。字节不同只说明保存结果不同，不证明对象正确、感知改善或机制有效。此核查没有读取任何评价分数。

## 证据与边界

- [完整生成回执](../audit/idea_interface_RealPaired_V_generation_completion_v1.json)
- [实际GPU作业及计时](../audit/IDEA1_RealPaired_generate_job.json)
- [评分前来源/保存核查](../audit/idea_interface_RealPaired_V_identity_before_v1.json)
- [原始失败记录：空列表](../audit/idea_interface_RealPaired_V_identity_before_failures_v1.json)
- [全部16,600行跨方法输入对齐](../audit/idea_interface_RealPaired_Conditional_V_matched_inputs_v1.json)
- [阶段产物及服务器逐行CSV的SHA绑定](../audit/idea_interface_RealPaired_completed_full_generation_progress_v1.json)
- [此前固定T40实际图片与错类观察](IDEA1_REALPAIRED_T40_20261019.md)

本阶段的 `score_rows_checked=0`：实际评分尚未完成，不能声称已检查评分所用的全部目标路径。后续仍须在完整评分后逐行核查。发布元数据、数组与刺激归属核对不能替代独立采集事件重放；保存及解码通过也不证明重建成功。

阶段快照中的特征工作进程为PID131143，使用原始 `extract_image_features.py`，覆盖全部16,600张的DINO、EffNet、CapSim与ObjectF1；各阶段计数不等于整个评价已完成。指标代码、分辨率、采样步数、seed及评价分母保持原定配置。

Conditional的完整结果已保存，正式图片差值需等待RealPaired完成。Frozen、Moment、Iso、Clean、EEG干预、质量评价及独立确认仍需实际证据；缺失不填零，不当作负结果，不据此宣布独立机制增量、SOTA或研究目标完成。
