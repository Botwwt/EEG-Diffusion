# Conditional / RealPaired 完整配对材料

服务器目录：`/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/galleries/IDEA1_Conditional_RealPaired_V_complete_20261019T231408Z_v1`。

两列均包含全部十被试、每被试1,660张、全部16,600条相同query，列顺序为RealPaired、Conditional。没有新增生成或评分，也没有选择生成种子。图片结果及限制见[完整比较报告](IDEA1_CONDITIONAL_VS_REALPAIRED_20261019.md)。

## 已实际完成

- 415个静态缩略图分页、交互索引、按query固定哈希分配A/B的完整盲评材料。
- 33,200条揭盲/配对记录；全部EEG刺激、目标、生成路径、评分目标、原始JPEG哈希、噪声种子与已完成源结果一致。
- 34,860个图像资产：1,660张目标及33,200张生成图。完整图与已审计源图为相同设备/inode的硬链接；源JPEG已由生成后身份审计和画廊构建完整核验。
- 所有显示分数逐行与原评分CSV一致；两个完整差值案例索引分别覆盖全部16,600条query，诊断排序不改变评价分母。
- 独立材料检查用CPU3.604秒完成；全部query唯一、十被试与166概念完整、415页齐全，盲评列无方法名/分数/揭盲表。人工评分CSV只有表头，没有评分。

存储采用完全相同缩略图字节的硬链接复用：18,260个旧缩略图复用、16,600个新缩略图写入，共324,923,118字节新缩略图。原构建器与先前画廊保持不变，未降低存储预留，未向C/D复制完整图片或模型。

证据：[构建实际作业](../audit/idea_Conditional_RealPaired_gallery_cached_CPU_v1.json)、[完成凭据](../audit/idea_matched_gallery_completion_v1.json)、[完整上下文](../audit/idea_matched_gallery_context_v1.json)、[逐字节复用凭据](../audit/idea_matched_gallery_thumbnail_reuse_v1.json)、[独立全量材料检查](../audit/IDEA1_Conditional_RealPaired_gallery_materials_check_v1.json)。

## 尚未完成与失败记录

新双列画廊的浏览器UI核验失败：本地已安装Chrome在Playwright启动阶段关闭，报`TargetClosedError`；没有打开页面、没有截图，不能声称已完成本版本视觉QA。保留[失败报告](../audit/IDEA1_Conditional_RealPaired_gallery_browser_QA_failed_v1.json)，未自动安装/修复浏览器。

检查所用服务仅绑定服务器127.0.0.1，经临时加密SSH转发，没有公开站点。失败后服务已停止，核对进程不存在，见[服务生命周期](../audit/idea_matched_gallery_private_HTTP_v1.json)。

Frozen、Moment、Iso、Clean、Conditional打乱/缺失EEG列尚未具备全部生成与评价，当前画廊如实标缺。Frozen仍在独立的完整生成/评价流程中；其完成后需新版本合并，不覆盖当前材料。没有人工评分、画质非劣结论、独立确认或SOTA结论。
