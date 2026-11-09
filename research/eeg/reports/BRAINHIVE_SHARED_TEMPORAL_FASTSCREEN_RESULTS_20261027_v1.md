# D07 共享时序编码器：完整快筛结果

**固定实例停止，不晋级400张。** 80条query使用同一400目标图库，原生与候选DINO2AFC为0.808208与0.757049，差−0.051159，描述性概念重采样区间[−0.063850,−0.037182]，8个概念全部负向，触发事前v2的大幅损害规则。目标余弦差−0.021082；同概念Top1由12/80到13/80，区间跨零，不能推翻主结果。[实际完整评分](../audit/shared_temporal_eeg_20261027_v1/actual_score80_summary_v2.json)、[停止回执](../audit/shared_temporal_eeg_20261027_v1/actual_score80_guard_v2.json)

模型为十个已知被试共同训练的3305940参数时序卷积编码器，两折概念隔离，每折固定10轮。660次更新、1323000次暴露全部完成，拟合主体32.96秒。条件端全13230条比较中，同概念F检索Top1由原生0.337113降至0.280121；全F图库2AFC由0.886979降至0.840722。不能把持续下降的训练loss等同于图片改善。[真实拟合](../audit/shared_temporal_eeg_20261027_v1/actual_fit_summary.json)、[匹配原生读数](../audit/shared_temporal_eeg_20261027_v1/actual_CPU_matched_native_readout.json)

完整20条epoch日志显示训练loss约7.10降至5.51，但后期held loss略升，held目标余弦下降，同概念检索趋平。训练与held的batch构成不同，不能把绝对loss差当成泛化差估计。仍使用固定最终10轮，没有按日志挑早期检查点。[完整训练分析](../audit/shared_temporal_eeg_20261027_v1/CPU_training_analysis_final/analysis.json)

首次生成因程序在总预算中预留120秒而提前耗尽工作时间，只生成67条，属于工程未完成。原模型、拟合和67张图片均未重跑；逐项核验后补13张及一次零模块像素回放，完整80条才提特征和评分。续跑阶段65.999秒闭合，后续400阶段未启动。首次失败、实现修复和等待时间均须计入整轮周转，不能把D07宣称为首次启动后15分钟内完成。[实际生成闭合](../audit/shared_temporal_eeg_20261027_v1/actual_images80_completion_v2.json)、[续跑执行器](../audit/shared_temporal_eeg_20261027_v1/actual_continuation_completion_v2.json)

这只否定当前共享时序实例。主干、被试联合训练、标准化、batch与教师缓存行为一起变化，不能由此否定所有时序模型或跨被试学习。旧ATMS已经使用时空编码与十被试数据；本轮未建立算法原创或SOTA。D正式臂保持停止。

随后再次CPU核验全部80条query/目标/种子与初始噪声字段、两组文件及像素SHA，零不一致。固定每概念第一张刺激的八行图已实际查看，例如候选将马车生成成彩色星形图案，未把小幅Top1变化解释成对象恢复成功。[核验](../audit/shared_temporal_eeg_20261027_v1/CPU_complete_image_inspection/summary.json)、[固定八行图](../audit/shared_temporal_eeg_20261027_v1/CPU_complete_image_inspection/fixed_all_concepts_first_stimulus_page1.jpg)
