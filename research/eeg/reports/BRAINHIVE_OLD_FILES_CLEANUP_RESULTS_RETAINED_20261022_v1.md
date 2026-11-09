# 旧过程文件清理完成，结果与最终模型保留

2026-10-22 05:49，America/Los_Angeles。按用户“清理旧文件，结果保留好就行”继续处理。没有删除生成原图、逐query分数、配对清单、报告或最终/最佳模型；新增了一份经过完整核验的结果元数据副本。本轮没有训练、出图或新增科学结果。

| 实际处理 | 保留依据与结果 |
|---|---|
| 移除43个旧过程文件，共1,380,909,056 allocated字节 | 30个已完成三epoch旧RIME试验的早期epoch1/2检查点、6个重复最终模型的优化器/RNG恢复断点、7个可重建重复诊断query缓存；epoch3、latest和最终模型保留 |
| 核验6个旧恢复断点 | 全部模型张量的名称、dtype、shape和原始字节指纹均与保留的最终模型一致；旧优化器/RNG恢复状态不再保留，不能据此精确续跑旧训练 |
| 无损归档旧AVDE专用预处理输入 | 原3,334,466,556字节存入2,771,089,345字节持久压缩包；实际解码并核验原文件SHA，额外净增约0.563GB空间；未量化、未改变EEG数值 |
| 核验既有BrainHIVE结果 | 7份生成清单、55832张原图逐SHA通过：49800张V、6000张原生R80、32张旧oracle诊断图；这些均为既有图片 |
| 核验最终权重 | 登记的40个当前EEG最终检查点与清单SHA一致，2个当前组件最终检查点保留；额外10个历史EEG最终检查点也保留，共登记52个文件SHA |
| 复制结果元数据 | 66812个分数、manifest、配置及报告文件，压缩副本210,792,167字节；CRC及逐成员SHA与原文件一致。原图和权重仍在原持久位置，这份元数据副本不是第二套图片备份 |

清理前，对旧退役清单中仍在保留范围的113711个大文件核验SHA，删除后核对文件身份未变；该阶段在后续旧AVDE输入无损归档前完成。所有逐文件删除、保留最终模型及来源指纹均有回执。未整体执行原22.14GB退役提案，未清除旧负结果或重置候选次数。

旧AVDE输入的持久归档在服务器：

`/root/autodl-tmp/retained_legacy_exact_archives_20261022_v1/AVDE_preprocessed_eeg_training.npy.EXACT_BS4_ZLIB_v1`

原文件SHA仍为`b7494ebe01fad73c61433b388388fb9cde7284a0eeb6055b519a24bad9b65fd4`。原路径现在链接到已从持久归档实际解码并验证过的RAM副本；RAM清空后，用[恢复程序](../archive_old_AVDE_input_losslessly_CPU_20261022_v1.py)的`--restore`重新建立原字节副本。持久压缩包是保留依据，RAM不计作持久容量。

结果元数据额外副本在服务器`audit/retained_result_metadata_20261022_v1.zip`，SHA为`e183b7daef563a02805b8e06321822aa5798b3f490118442d47a4fe044b10964`。完整核验清单另有[逐字节保留的gzip副本](../audit/retained_results_verification_20261022_v1.json.gz)，其原始JSON与ZIP保留在服务器。副本反映闭合核验时的快照；随后本轮状态更新另以当前报告和Git记录保留，原分数未改。

本轮保留流程有两处工程更正：最初宽泛扫描把额外10个历史EEG模型计入当前42个模型的断言，已改为逐项核验登记的40个EEG角色及两个组件，并保留历史模型；历史元数据备份实测约211MB，超过最初128MiB估算，单独登记256MiB备份上限并核验已有完整ZIP。未修改新四臂metadata预算、原八个核心源、评分方法或任何模型配方。失败来源和日志保留，没有把早期拒绝写成完成。

05:49快照：`/root`空闲5,159,194,624字节，`/root/autodl-tmp`空闲2,214,932,480字节，合计约7.37GB。扣除新增结果副本、完整核验清单和回执，本轮实际净空闲增加约1.71GB。完整新四臂含最低缓冲仍需26,655,048,192字节，聚合缺口约19.28GB；分盘条件也尚未满足，正式功能v2拟合未启动。

另外，本轮清理前已发现`/dev/shm`为空：既定配方有20个生成器/作者先验RAM副本及派生T latent输入缺失，现存17个配方输入SHA匹配。原因未判定，缺失不是本轮删除动作造成。历史GPU工程profile成功仍有效作为历史证据，当前运行环境须先恢复缺失资产；不能把它写成当前已可正式开训。

证据：[总体实际回执](../audit/legacy_cleanup_results_retained_actual_20261022_v1.json)、[过程清理回执](../audit/legacy_process_files_cleanup_20261022_v1/receipt.json)、[删除清单](../audit/legacy_process_files_cleanup_20261022_v1/plan.json)、[旧输入归档回执](../audit/old_AVDE_input_lossless_archive_20261022_v1.json)、[归档闭合](../audit/old_AVDE_input_lossless_archive_complete_20261022_v1.json)、[结果核验摘要](../audit/retained_results_verification_summary_20261022_v1.json)。
