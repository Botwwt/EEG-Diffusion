# 旧实验退役方案：已按基线保留要求缩小执行范围

**2026-10-22 05:49 America/Los_Angeles追加更新**：按用户继续清理指令，又移除43个旧过程文件约1.38GB，保留最终模型、原图和分数；旧AVDE输入改为有逐字节恢复核验的持久无损归档，未丢弃原始数值。结果另有完整元数据副本与图像/权重核验。此前“113754文件全部保持原位置”的状态仅适用于04:12时刻；最新逐文件处理以[本轮报告](BRAINHIVE_OLD_FILES_CLEANUP_RESULTS_RETAINED_20261022_v1.md)为准，原整体提案仍未执行。

**2026-10-22 04:12 America/Los_Angeles更新**：用户要求保留当前实验或后续基线比较所需资产，未授予整体删除许可。原113820条清单中仅66个可重建训练/诊断缓存已删除，915,976,192 allocated字节；其余113754个文件全部核验并保留，包括唯一模型、原图和旧AVDE专用预处理输入。详见[实际清理与基线保留报告](BRAINHIVE_BASELINE_PRESERVING_CLEANUP_20261022_v1.md)。下方为原提案快照，不能再作为整体删除范围或可用空间预测。

2026-10-22。常规清理已移除1,755,705,344字节重复压缩包和可重建特征分片；约1.73GB净空间增加，差额为保留核验清单的元数据占用。原始EEG、所有生成图片、权重、分数及当前BrainHIVE特征缓存保持不变。

当前两个持久盘合计约4.76GB空闲。新四臂完整生命周期小计26,118,177,280字节，加两盘各256MiB缓冲至少26,655,048,192字节。现有空闲不足以正式拟合后保存完整产物。

以下是另行准备的不可恢复退役方案。已核验每个候选文件的SHA、设备、inode、实际占用及链接数；去除别名重复，按全部链接均在清单内的实际inode计算，不把表面目录大小当作可释放量。

| 待确认的退役范围 | 预计实际释放 |
|---|---:|
| 旧VAR/AVDE/ATMS及更早适配实验的生成原图、原始输出数组、旧检查点和中间产物 | 18.127GB |
| 旧AVDE专用预处理训练文件，见下方完整路径 | 3.334GB |
| 旧AVDE oracle缓存中的大文件 | 0.682GB |
| 合计 | **22.144GB** |

主要位置包括：

- `/root/rime_var_20261015/artifacts/`中登记的旧输出；不含trial_slots、trial_metadata、full_cache。
- `/root/mced_*`中登记的旧samples、reference、training_outputs和persistent_checkpoints文件。
- `/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/images/IDEA1_interface_v1/`与`images/S1_full_native_batchsafe_v2/`的旧输出，以及该目录下登记的旧`runs`、`error_validation`产物。
- `/root/autodl-tmp/avde_native680_stage2_20261016_v1/training/ar-ckpt-last.pth.EXACT_BS4_ZLIB_v1`等登记的旧AVDE/ATM检查点。
- **`/root/avde_author_training_cache_20261016/preprocessed_bc/sub-01/preprocessed_eeg_training.npy`**：仅供旧AVDE复现的专用预处理输入；该文件本身目前只有一份。
- `/root/autodl-tmp/avde_rime_oracle_latent_20261015/cache_v1/`中登记的大缓存。

这不是对上述整目录的递归删除授权请求。实际范围限于[113820条逐文件SHA清单](../audit/BrainHIVE_legacy_retirement_exact_file_plan_20261022_v2.json.gz)，服务器原件及SHA绑定见[计划汇总](../audit/BrainHIVE_legacy_retirement_exact_plan_summary_20261022_v2.json)。

退役会失去这些旧实验的原始生成像素与唯一旧模型，无法再直接复用它们重放旧实验；旧AVDE专用输入也需重新获取或重建。小型源码、配置、日志、指标、manifest和来源索引将保留，但它们不能代替已删除的大产物。

当前BrainHIVE的整条运行链、全部R4/R80图片和评价、全部当前学习权重、当前使用的规范EEG输入和生成器/评价器资产不在清单中。尚未执行这份方案，也没有由常规“清理一下”推定删除这些唯一产物的许可。

若明确同意，执行前仍重新核对SHA、实际路径、当前进程占用及当前BrainHIVE输入引用。预计合计空闲约26.90GB；这是物理占用预测，正式运行仍须按两个实际文件系统分配并通过完整生命周期保护，不能把汇总预测当作已准入。
