> 历史固定 patience20 链路快照：下文 CONDITIONAL 等名称属于旧链路实验身份。当前新 patience50/IDEA1 的同协议完整比较见 [IDEA_INTERFACE_COMPARISON.md](IDEA_INTERFACE_COMPARISON.md)。本页标注前的原始字节保存在 [历史副本](history/PATIENCE20_CURRENT_COMPARISON_BEFORE_IDEA1.md)。

更新时间：2026-10-19T03:37:40Z。这是有来源的阶段比较；缺少完整结果的行保持空值，不根据部分图片或训练 loss 填成绩。

本轮完整分区

| 协议 | 方法/干预 | 已保存/预期图片 | 状态 | DINO2AFC | SEED | ObjectF1 | 空白代理数 |
|---|---|---:|---|---:|---:|---:|---:|
| V/R4/gallery1660/steps20/CFG5.0 | REF_REG_correct_R4 | 16600/16600 | COMPLETE_IMAGE_METRICS | 0.758165 | 0.147336 | 0.141729 | 0 |
| V/R4/gallery1660/steps20/CFG5.0 | REF_PRIOR_V_correct_R4 | 0/16600 | FULL_IMAGE_METRICS_PENDING | — | — | — | — |

完整数值与来源哈希见 [统一 CSV](unified_comparison.csv)；已完成方法的各被试绝对分数见 [per_subject.csv](per_subject.csv)。完整逐 query 文件与图片均在服务器，CSV 中保留实际路径。

必需接口对照的未完成项

以下是任务要求的实验身份；尚未产生训练检查点或正式图片结果，不表示已有完整运行接口。

| 对照 | 已保存/预期图片 | 状态 |
|---|---:|---|
| CLEAN_BRIDGE | 0/16600 | NOT_TESTED_MATCHED_FULL_ARMS_UNFUNDED |
| ISO | 0/16600 | NOT_TESTED_MATCHED_FULL_ARMS_UNFUNDED |
| MOMENT | 0/16600 | NOT_TESTED_MATCHED_FULL_ARMS_UNFUNDED |
| CONDITIONAL | 0/16600 | NOT_TESTED_MATCHED_FULL_ARMS_UNFUNDED |
| REAL_PAIRED | 0/16600 | NOT_TESTED_MATCHED_FULL_ARMS_UNFUNDED |

先完成真实概念折外条件及独立 Dcheck 校验，再用实测吞吐核算全部匹配训练、图片、控制与最终确认成本。未运行的指标为空，不能当作负结果。

历史 V64：单被试、17 电极、旧 BrainMLP/Turbo

| 历史方法 | 当时状态 | DINO2AFC | 图数 |
|---|---|---:|---:|
| REF-REG | BASELINE | 0.727927 | 64 |
| REF-PRIOR | BASELINE | 0.748760 | 64 |
| REF-ADAPTER | UNDETERMINED | 0.730655 | 64 |
| REF-CONLY | UNDETERMINED | 0.742312 | 64 |
| A1 | REJECTED | 0.724702 | 64 |
| A2 | REJECTED | 0.729415 | 64 |
| B1 | DEPENDENCY_BLOCKED | — | 未运行 |
| B2 | DEPENDENCY_BLOCKED | — | 未运行 |
| C1 | REJECTED | 0.727927 | 64 |
| C2 | REJECTED | 0.711062 | 64 |

历史 REJECTED 只表示该固定配置没有通过当时的筛选，不是所有 A/C 机制的全数据否定。历史 gallery64 与本轮 gallery1660/200、通道、被试和生成器不同，不能跨表按原分数排名；旧 PixCorr/SSIM 与新补充定义也不直接混合。
B1/B2 的历史缺 SDXL 原因保留原文；当前标准 SDXL 已到位，当前仍需验证密度比/额外 EEG 证据等前置条件。

配对差值

尚无本轮完整配对图片比较。

资源：上限40 GPU小时、训练seed1729。作业墙钟加保守历史/S0额度累计 9.528 小时；更保守剩余授权 28.391 小时。供应商账单未独立核验，失败作业计入。

解释与未完成项

- DINO2AFC 是固定 gallery 相对偏好，不是对象正确率；独立语义同时报告 ObjectF1/SEED及检测覆盖。
- 一次训练seed；区间重采样被试/概念，不覆盖训练随机性。开发区间属于描述性；官方200曾暴露，不是独立H。
- 原始事件级 stimulus 标签不在发布 EEG 字典；已核验发布顺序、显式元数据、ZIP成员和图片SHA，不能宣称独立采集事件重放。
- 本轮encoder patience20来自CLI，官方benchmark为50；完整T覆盖不等于收敛或完整官方训练配置复现。
- 原生投影CPU诊断与检索不等于图片成功。当前没有受控扩容证据。
- 人工评分未开展；完整盲评材料不等于已有评分或画质非劣证明。正式每作业峰值显存尚未逐作业测量，T吞吐profile不替代该项。
- OOF、误差模型、完整matched接口训练及最终确认的状态以实际回执为准，不能把未运行视为假设被否定。
- 生成墙钟包含装载/I/O及嵌套诊断暂停；CSV另列诊断暂停和仅扣除该暂停的耗时。总资源账本始终计入这些诊断，不能把扣除后的吞吐当成总花费。

已否定/未确定/独立确认

- 已有不利证据：历史A1/A2/C1/C2固定配置未通过筛选；新epsilon prior在修正批处理后未通过T图就绪检查，未作全V全局否定。
- 两套新参照的完整图片比较尚未完成。 容量/停止规则的因果影响及新的误差适配机制仍需各自的实际验证。
- 本自动报告不宣布SOTA或独立机制确认；这还需要完整图片、独立语义、质量、EEG依赖、matched参照与未暴露确认数据的逐项审查。

完整假设状态见 [hypothesis_status.json](../research/hypothesis_status.json)。最终测试是否解锁：False。
